import os
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import time
import learn2learn as l2l
from .networks.FT_net.network import OFENet  # Assuming OFENet is implemented and imported from another module
from .networks.policy_net.PPO import GaussianActor, CriticV, PPO, RolloutBuffer  # PPO components

from .blocks.misc import get_target_dim, get_default_steps, get_eval_steps


class New_maml_agent(nn.Module):
    def __init__(self, env, config):
        super(New_maml_agent, self).__init__()
        self.env = env
        self.eval_env = env #将评估环境设置为与训练环境一样

        self.args = config
        #从环境中提取状态和动作空间的维度
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        #初始化回放缓存区
        self.replay_buffer = RolloutBuffer(obs_dim=state_dim, act_dim=action_dim, capacity=config.steps_per_epoch, gamma=config.discount, lam=config.lam)
        
        #设置特征提取器的参数
        self.extractor_kwargs = {
            "dim_state": state_dim, 
            "dim_action": action_dim, 
            "dim_output": 11,
            "dim_discretize": config.dim_discretize, 
            "fourier_type": config.fourier_type, 
            "discount": config.discount, 
            "use_projection": config.use_projection, 
            "projection_dim": config.projection_dim, 
            "cosine_similarity": config.cosine_similarity,
            "normalizer": config.normalizer
        }
        #初始化特征提取器
        self.extractor = OFENet(**self.extractor_kwargs, skip_action_branch = False)
        self.target_extractor = OFENet(**self.extractor_kwargs, skip_action_branch = False)
        self.soft_update(self.target_extractor, self.extractor, tau=1)

        #Use learn2learn to wrap the extractor network for future updates
        self.extractor = l2l.algorithms.MAML(self.extractor, lr=config.lr, first_order=False) # TODO: lr
        self.target_extractor = l2l.algorithms.MAML(self.target_extractor, lr=config.lr, first_order=False)

        self.actor = GaussianActor(
            self.extractor.dim_state_features,
            action_dim, max_action, actor_units=(64, 64)).to(device)
        self.critic = CriticV(
            self.extractor.dim_state_features, critic_units=(64, 64)).to(device)
        
        self.critic = l2l.algorithms.MAML(self.critic, lr = config.lr, first_order=False) # TODO: lr

        #初始化PPO策略网络
        self.policy = PPO(
            feature_extractor = self.extractor,
            actor=self.actor,
            critic=self.critic,
            pi_lr=3e-4,
            vf_lr=1e-3,
            clip_ratio=0.2,
            batch_size=64,
            discount=0.99,
            n_epoch=10,
            horizon=2048,
            gpu=0)
        
    #更新目标网络的参数，tau=1时为复制
    def soft_update(target, source, tau = 1):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

    def get_action(self, x, action=None ,return_distribution=False):
        state_features = self.extractor.features_from_states(x)

        # Get the action mean from the PPO actor
        action_mean, action_logstd = self.actor(state_features)
        action_std = torch.exp(action_logstd)
        distribution = Normal(action_mean, action_std)

        # Sample action or return log probability
        if action is None:
            action = distribution.sample()
            logprob = distribution.log_prob(action).sum(1)
        else:
            logprob = distribution.log_prob(action).sum(1)

        if not return_distribution:
            return action, logprob, distribution.entropy().sum(1)
        else:
            return action, logprob, distribution.entropy().sum(1), distribution

    def get_deterministic_action(self, x):
        x = self.extractor(x)
        action_mean = self.actor.mean_action(x)
        return action_mean

    def adapt(self, num_steps, information, config, lifetime_buffer, mean_reward_for_baseline, device='cpu'):

        max_steps = self.args.steps = get_default_steps(self.args.env)
        epochs = max_steps // self.args.steps_per_epoch + 1

        total_steps = self.args.steps_per_epoch*epochs
        steps_per_epoch = self.args.steps_per_epoch

        batch_size = self.args.batch_size

        state, _ = self.env.reset()

        print("Initialization: I am collecting samples randomly!")
        round = 0
        for i in range(self.args.random_collect):
            print("Collect Round: ", round)
            round += 1
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            episode_return += reward
            episode_timesteps += 1

            done_flag = done
            if episode_timesteps == self.env.spec.max_episode_steps:
                done_flag = False

            self.replay_buffer.add(obs=state, act=action, obs2=next_state, rew=reward, done=done_flag, val=0, logp=0)
            state = next_state

            if done:
                state, _ = self.env.reset()
                episode_timesteps = 0
                episode_return = 0

        self.policy.clone_for_adaptation() # clone extractor and critic network

        print("Pretrain: I am pretraining the extractor!")
        pretrain_step = 0
        for i in range(self.args.pre_train_step):
            print("Pretrain Step: ", pretrain_step)
            pretrain_step += 1
            sample_states, sample_actions, sample_next_states, _, sample_dones = self.replay_buffer.sample(batch_size=self.args.batch_size)
            sample_next_actions, _ = self.policy.get_action(sample_next_states)

            #adapt extractor
            pred_loss = self.policy.extractor_loss_for_adaptation(self.target_extractor, sample_states, sample_actions, sample_next_states, sample_next_actions, sample_dones)
            self.policy.extractor_adapt(pred_loss)


        state = np.array(state, dtype = np.float32)

        self.replay_buffer.get()

        done_lifetime = information['prev_done']
        episodes_returns_lifetime=[]
        episodes_successes_lifetime=[]
        episode_return_lifetime = information['current_episode_return']
        succeeded_in_episode_lifetime=information['current_episode_success']
        current_lifetime_step = information['current_lifetime_step']

        print("Train: I am starting to train myself!")
        step = 0
        for cur_steps in range(total_steps):
            print("Train Step: ", step)
            prev_done_lifetime = done_lifetime

            step += 1
            action, logp, v_t = self.policy.get_action_and_val(state)

            # Step the env
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            done_lifetime = torch.Tensor(done, dtype=torch.float32)
            episode_timesteps += 1
            episode_return += reward

            done_flag = done

            # # done is valid, when an episode is not finished by max_step.
            # if episode_timesteps == self.env.spec.max_episode_steps:
            #     done_flag = False

            self.replay_buffer.add(obs=state, act=action, obs2=next_state, rew=reward, done=done_flag, val=v_t, logp=logp)

            lifetime_buffer.store_step_data(global_step=current_lifetime_step, obs=state.to(device), act=action.to(device),
                                            reward=reward.to(device), logp=logp.to(device), prev_done=prev_done_lifetime.to(device))

            state = next_state
            current_lifetime_step += 1
            episode_return_lifetime += torch.as_tensor(reward, dtype=torch.float32).to(device)
            if info['success'] == 1.0:
                succeeded_in_episode_lifetime = True

            if done or (episode_timesteps == self.env.spec.max_episode_steps) or (cur_steps + 1) % steps_per_epoch == 0:
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0 if done else self.policy.get_val(state)
                self.replay_buffer.finish_path(last_val)
                state, _ = self.env.reset()
                episode_timesteps = 0
                episode_return = 0

                episodes_returns_lifetime.append(episode_return_lifetime)
                episode_return_lifetime = 0
                if succeeded_in_episode_lifetime == True:
                    episodes_successes_lifetime.append(1.0)
                else:
                    episodes_successes_lifetime.append(0.0)


            update_every = self.args.update_every
            #Update the extractor every update_every steps
            if self.args.gin is not None and cur_steps % update_every == 0: 
                for _ in range(update_every):
                    sample_states, sample_actions, sample_next_states, _, sample_dones = self.replay_buffer.sample(
                        batch_size=batch_size)
                    sample_next_actions, _ = self.policy.get_action(sample_next_states)

                    #adapt extractor
                    pred_loss = self.policy.extractor_loss_for_adaptation(self.target_extractor, sample_states, sample_actions, sample_next_states, sample_next_actions, sample_dones)
                    self.policy.extractor_adapt(pred_loss)

            #Update the target network every target_update_freq steps
            if self.args.gin is not None and cur_steps % self.args.target_update_freq == 0: 
                self.soft_update(self.target_extractor, self.extractor, self.args.tau)


            if (cur_steps + 1) % steps_per_epoch == 0: #Train the policy every steps_per_epoch steps
                self.policy.train(self.replay_buffer)

        lifetime_buffer.episodes_returns=lifetime_buffer.episodes_returns+episodes_returns_lifetime
        lifetime_buffer.episodes_successes =lifetime_buffer.episodes_successes+ episodes_successes_lifetime

        #需要把evaluation_error反向传播以后才能reset

        information={'prev_done': done_lifetime, 'current_lifetime_step': current_lifetime_step,
                     'current_episode_return': episode_return_lifetime, 'current_episode_success': succeeded_in_episode_lifetime}
        
        return information

    def evaluate_extractor_and_critic(self):
        total_eval_steps = get_eval_steps(self.args.env) # TODO: 在misc中实现这个函数，查参数用
        batch_size = self.args.batch_size

        state, _ = self.eval_env.reset() 
        step = 0
        for cur_steps in range(total_eval_steps):
            print(f"Eval Step: {step}")  
            step += 1
            action, logp, v_t = self.policy.get_action_and_val(state)

            # Step the env
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_timesteps += 1
            episode_return += reward

            done_flag = done

            self.replay_buffer.add(obs=state, act=action, obs2=next_state, rew=reward, done=done_flag, val=v_t, logp=logp)
            state = next_state

            if done or (episode_timesteps == self.env.spec.max_episode_steps) or (cur_steps + 1) % steps_per_epoch == 0: # TODO: steps_per_epoch参数设置
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0 if done else self.policy.get_val(state)
                self.replay_buffer.finish_path(last_val)
                state, _ = self.env.reset()
                episode_timesteps = 0
                episode_return = 0

        evaluate_every = self.args.evaluate_every
        for _ in range(evaluate_every):
            sample_states, sample_actions, sample_next_states, _, sample_dones = self.replay_buffer.sample(
                batch_size=batch_size)
            sample_next_actions, _ = self.policy.get_action(sample_next_states)

            #backward propagation to the loss
            pred_loss = self.policy.extractor_loss_for_adaptation(self.target_extractor, sample_states, sample_actions, sample_next_states, sample_next_actions, sample_dones)
            pred_loss.backward()

        self.policy.reset_cloned_networks()

    