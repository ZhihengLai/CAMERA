import os
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import time
from .networks.FT_net.network import OFENet  # Assuming OFENet is implemented and imported from another module
from .networks.policy_net.PPO import GaussianActor, CriticV, PPO, RolloutBuffer  # PPO components
from .blocks.TRPO_and_adapt_loss import maml_trpo_update
from .blocks.misc import get_target_dim, get_default_steps

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Maml_agent(nn.Module):
    def __init__(self, env, config):
        super(Maml_agent, self).__init__()

        self.env = env
        # state_dim = np.array(env.observation_space.shape).prod()
        # action_dim = np.prod(env.action_space.shape)
        # max_action = float(env.action_space.high[0])

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

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

        # OFENet Feature Extractor
        # self.ofenet = OFENet(
        #     dim_state=state_dim, 
        #     dim_action=action_dim, 
        #     dim_output=11, 
        #     dim_discretize=config.dim_discretize,
        #     fourier_type=config.fourier_type,
        #     discount=config.discount,
        #     use_projection=config.use_projection,
        #     projection_dim=config.projection_dim,
        #     cosine_similarity=config.cosine_similarity,
        #     normalizer=config.normalizer
        # )

        self.ofenet = OFENet(**self.extractor_kwargs, skip_action_branch = False)


        # Actor-Critic using PPO architecture
        self.actor = GaussianActor(
            self.ofenet.dim_state_features,  # Input is the extracted OFENet features
            action_dim, max_action
        )

        self.critic = CriticV(
            self.ofenet.dim_state_features,  # Input is the extracted OFENet features
        )

    def forward(self, state):
        # First, extract features from the state using OFENet
        state_features = self.ofenet.features_from_states(state)
        return state_features

    def get_action(self, state, action=None, return_distribution=False):
        # Extract OFENet features
        state_features = self.ofenet.features_from_states(state)

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

    def get_deterministic_action(self, state):
        # Extract OFENet features
        state_features = self.ofenet.features_from_states(state)

        # Get deterministic action from the actor's mean output
        action_mean, _ = self.actor.mean_action(state_features)
        return action_mean

    def compute_value(self, state):
        # Extract OFENet features
        state_features = self.ofenet.features_from_states(state)

        # Get the value estimate from the critic
        value = self.critic(state_features)
        return value

    def adapt(self, state, action, reward, next_state, done, lr):
        """
        Adaptation step for MAML. This function will use gradient-based updates to modify
        the agent's policy (actor) and value function (critic) based on the task-specific data.
        """

        # Update the actor with policy gradient
        state_features = self.ofenet.features_from_states(state)
        action_mean, action_logstd = self.actor(state_features)
        action_std = torch.exp(action_logstd)
        distribution = Normal(action_mean, action_std)

        # Compute policy loss for the adaptation step
        log_prob = distribution.log_prob(action).sum(1)
        advantage = reward - self.compute_value(state)  # Simple advantage calculation
        actor_loss = -log_prob * advantage.detach()

        # Actor update using gradient descent with learning rate lr
        self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=lr)
        self.actor_optimizer.zero_grad()
        actor_loss.mean().backward()
        self.actor_optimizer.step()

        # Critic update with TD-error for value function
        value_loss = (reward + (1 - done) * self.compute_value(next_state) - self.compute_value(state)).pow(2).mean()
        self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=lr)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

    def maml_update(self, maml_buffers, adapted_policies_states, config):
        """
        Perform the MAML meta-update by computing the gradients over multiple tasks and 
        adjusting the base policy (and possibly OFENet) based on these gradients.
        """
        maml_trpo_update(maml_agent=self, data_buffers=maml_buffers, adapted_policies_states=adapted_policies_states, config=config)

class New_maml_agent(nn.Module):
    def __init__(self, env, config):
        super(New_maml_agent, self).__init__()
        self.env = env
        self.eval_env = env
        # state_dim = np.array(env.observation_space.shape).prod()
        # action_dim = np.prod(env.action_space.shape)
        # max_action = float(env.action_space.high[0])

        self.args = config
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        self.replay_buffer = RolloutBuffer(obs_dim=state_dim, act_dim=action_dim, capacity=config.steps_per_epoch, gamma=config.discount, lam=config.lam)
        
        
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

        self.extractor = OFENet(**self.extractor_kwargs, skip_action_branch = False)
        self.target_extractor = OFENet(**self.extractor_kwargs, skip_action_branch = False)
        self.policy = PPO(
            state_dim = state_dim,
            action_dim = action_dim,
            max_action = max_action,
            feature_extractor = self.extractor,
            actor_units=(64, 64),
            critic_units=(64, 64),
            pi_lr=3e-4,
            vf_lr=1e-3,
            clip_ratio=0.2,
            batch_size=64,
            discount=0.99,
            n_epoch=10,
            horizon=2048,
            gpu=0)
        

    def get_action(self, x, action=None ,return_distribution=False):
        state_features = self.extractor.features_from_states(x)

        # Get the action mean from the PPO actor
        action_mean, action_logstd = self.policy.actor(state_features)
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

    def get_deterministic_action(self,x):
        x = self.extractor(x)
        action_mean = self.policy.actor.mean_action(x)
        return action_mean
        
    def _train_ppo_body(self, ):

        pass

    def _pretrain_extractor_body(self, ):
        
        pass


    def train(self, ):

        
        def soft_update(target, source, tau = 1):
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

        def evaluate_policy(env, policy, eval_episodes=10):
            avg_reward = 0.
            episode_length = []

            for _ in range(eval_episodes):
                state, _ = env.reset()
                cur_length = 0

                done = False
                while not done:
                    action = policy.select_action(np.array(state))
                    state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    avg_reward += reward
                    cur_length += 1

                episode_length.append(cur_length)

            avg_reward /= eval_episodes
            avg_length = np.average(episode_length)
            return avg_reward, avg_length

        max_steps = self.args.steps = get_default_steps(self.args.env)
        epochs = max_steps // self.args.steps_per_epoch + 1

        total_steps = self.args.steps_per_epoch*epochs
        steps_per_epoch = self.args.steps_per_epoch

        batch_size = self.args.batch_size
        save_dir_root = self.args.dir_root

        print("Pretrain: I am pretraining the extractor!")
        pretrain_step = 0
        for i in range(self.args.pre_train_step):
            print("Pretrain Step: ", pretrain_step)
            pretrain_step += 1
            sample_states, sample_actions, sample_next_states, _, sample_dones = self.replay_buffer.sample(batch_size=self.args.batch_size)
            sample_next_actions, _ = self.policy.actor.get_action(sample_next_states)

            pred_loss, pred_re_loss, pred_im_loss, grads_proj, grads_pred = self.extractor.train_model(self.extractor_target, sample_states, sample_actions, sample_next_states, sample_next_actions, sample_dones)


        state = np.array(state, dtype = np.float32)

        prev_calc_step = self.args.pre_train_step
        prev_calc_time = time.time()
        self.replay_buffer.get()

        eval_rewards = []
        eval_lengths = []
        eval_steps = []

        print("Train: I am starting to train myself!")
        step = 0
        for cur_steps in range(total_steps):
            print("Train Step: ", step)
            step += 1
            action, logp, v_t = self.policy.get_action_and_val(state)

            # Step the env
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_timesteps += 1
            episode_return += reward
            total_timesteps += 1

            done_flag = done

            # # done is valid, when an episode is not finished by max_step.
            # if episode_timesteps == self.env.spec.max_episode_steps:
            #     done_flag = False

            self.replay_buffer.add(obs=state, act=action, obs2=next_state, rew=reward, done=done_flag, val=v_t, logp=logp)
            state = next_state

            if done or (episode_timesteps == self.env.spec.max_episode_steps) or (cur_steps + 1) % steps_per_epoch == 0:
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0 if done else self.policy.get_val(state)
                self.replay_buffer.finish_path(last_val)
                state, _ = self.env.reset()
                episode_timesteps = 0
                episode_return = 0

            update_every = self.args.update_every
            #Update the extractor every update_every steps
            if self.args.gin is not None and cur_steps % update_every == 0: 
                for _ in range(update_every):
                    sample_states, sample_actions, sample_next_states, _, sample_dones = self.replay_buffer.sample(
                        batch_size=batch_size)
                    sample_next_actions, _ = self.policy.get_action(sample_next_states)
                    pred_loss, pred_re_loss, pred_im_loss, grads_proj, grads_pred = self.extractor.train_model(self.extractor_target, sample_states, sample_actions, sample_next_states, sample_next_actions, sample_dones)

            #Update the target network every target_update_freq steps
            if self.args.gin is not None and cur_steps % self.args.target_update_freq == 0: 
                soft_update(self.extractor_target, self.extractor, self.args.tau)

            if (cur_steps + 1) % steps_per_epoch == 0: #Train the policy every steps_per_epoch steps
                self.policy.train(self.replay_buffer)

            if cur_steps % self.args.eval_freq == 0: #Evaluate the policy every eval_freq steps
                duration = time.time() - prev_calc_time
                duration_steps = cur_steps - prev_calc_step
                # throughput = duration_steps / float(duration)

                cur_evaluate, average_length = evaluate_policy(self.eval_env, self.policy)

                print("Average Reward at step {}: {}".format(cur_evaluate, cur_steps))

                eval_rewards.append(cur_evaluate)
                eval_lengths.append(average_length)
                eval_steps.append(cur_steps)
                    
                prev_calc_time = time.time()
                prev_calc_step = cur_steps

            # store model
            if self.args.save_model == True and cur_steps % self.args.save_freq == 0:
                model_save_dir = os.path.join(save_dir_root, 'model')
                self.policy.save(model_save_dir)

                if self.args.gin is not None:
                    self.extractor.save(os.path.join(model_save_dir,'extractor_model.pth'))
                    self.extractor_target.save(os.path.join(model_save_dir,'extractor_target_model.pth'))
                    print('Models have been saved.')    

        # After training is done, save results to file
        results_df = pd.DataFrame({
            'Step': eval_steps,
            'Average Reward': eval_rewards,
            'Average Length': eval_lengths
        })
        results_df.to_csv(os.path.join(save_dir_root, 'evaluation_results.csv'), index=False)    

        pass