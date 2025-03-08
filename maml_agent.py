import os
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.cuda.amp import GradScaler
from tqdm import tqdm

import numpy as np
import time
import learn2learn as l2l
from policy.PPO import GaussianActor, CriticV, PPO, RolloutBuffer  # PPO components
from blocks.maml_config import Metaworld

class MAML_agent(nn.Module):
    def __init__(self, env, agent_config, maml_config):
        super(MAML_agent, self).__init__()
        self.env = env
        self.eval_env = env

        self.args = agent_config

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        self.replay_buffer = RolloutBuffer(obs_dim=state_dim, act_dim=action_dim, capacity=agent_config.steps_per_epoch, gamma=agent_config.discount, lam=agent_config.lam)

        self.actor = GaussianActor(
            state_dim, action_dim, max_action, layer_units=(64, 64))
        self.critic = CriticV(
            state_dim, units=(64, 64))
        
        self.critic = l2l.algorithms.MAML(self.critic, lr = 1e-3)

        self.policy = PPO(
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
        
    def soft_update(self, target, source, tau=1):
        for target_param, source_param in zip(target.state_dict().values(), source.state_dict().values()):
            if isinstance(target_param, torch.Tensor) and isinstance(source_param, torch.Tensor):
                target_param.copy_((1 - tau) * target_param + tau * source_param)

    def get_action(self, state, action=None ,return_distribution=False):
        # Get the action mean from the PPO actor
        action_mean, action_logstd = self.actor(state) #得到了动作的均值和对数标准差
        action_std = torch.exp(action_logstd)
        distribution = Normal(action_mean, action_std)

        # Sample action or return log probability
        #如果有给定action参数，那么从刚刚计算的分布中采样一个动作并计算该动作的对数概率
        if action is None:
            action = distribution.sample()
            logprob = distribution.log_prob(action).sum(1)
        else:
            logprob = distribution.log_prob(action).sum(1)

        ##区别就是是否多返回一个动作的概率分布
        if not return_distribution:
            return action, logprob, distribution.entropy().sum(1)
        else:
            return action, logprob, distribution.entropy().sum(1), distribution

    def adapt(self, num_steps, information, config:Metaworld, lifetime_buffer, mean_reward_for_baseline, device):

        max_steps = config.num_env_steps_per_adaptation_update
        epochs = max_steps // self.args.steps_per_epoch + 1 

        total_steps = self.args.steps_per_epoch*epochs
        steps_per_epoch = self.args.steps_per_epoch

        #重置环境，初始化回合的总奖励和时间步数
        state, _ = self.env.reset()
        episode_return = 0
        episode_timesteps = 0

        # No need to pretrain the extractor!
        self.policy.clone_for_adaptation()
        self.policy.set_optimizers()

        state = np.array(state, dtype = np.float32)

        #生命周期缓冲区初始化
        done_lifetime = information['prev_done']
        episodes_returns_lifetime=[]
        episodes_successes_lifetime=[]
        episode_return_lifetime = information['current_episode_return']
        succeeded_in_episode_lifetime=information['current_episode_success']
        current_lifetime_step = information['current_lifetime_step']

        print("Train: I am starting to train myself!")
        step = 0

        #初始化actor混合精度训练所需的gradscaler
        actor_scaler = GradScaler()

        for cur_steps in tqdm(range(total_steps)):
            prev_done_lifetime = done_lifetime

            step += 1
            action, logp, v_t = self.policy.get_action_and_val(state) #从当前策略获得动作  动作概率  状态值估计

            # Step the env
            next_state, reward, terminated, truncated, info = self.env.step(action) #在环境中执行这个动作
            done = terminated or truncated
            #组合这里torch,Tensor不支持布尔型变量和float组合
            done_lifetime = torch.tensor(done, dtype=torch.float32)
            episode_timesteps += 1
            episode_return += reward

            done_flag = done

            self.replay_buffer.add(obs=state, act=action, obs2=next_state, rew=reward, done=done_flag, val=v_t, logp=logp)

            lifetime_buffer.store_step_data(global_step=current_lifetime_step,
                                            obs=torch.tensor(state, dtype=torch.float32).to(device),
                                            act=torch.tensor(action, dtype=torch.float32).to(device),
                                            reward=torch.tensor(reward, dtype=torch.float32).to(device),
                                            logp=torch.tensor(logp, dtype=torch.float32).to(device),
                                            prev_done=torch.tensor(prev_done_lifetime, dtype=torch.float32).to(device))

            state = next_state
            current_lifetime_step += 1
            episode_return_lifetime += torch.as_tensor(reward, dtype=torch.float32).to(device)
            if info['success'] == 1.0:
                succeeded_in_episode_lifetime = True

            if done or (episode_timesteps == config.max_episode_steps) or (cur_steps + 1) % steps_per_epoch == 0:
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

            if (cur_steps + 1) % steps_per_epoch == 0: #Train the policy every steps_per_epoch steps
                self.policy.train(self.replay_buffer, actor_scaler)            

        ##将本次任务适配过程中收集的奖励、成功率数据存入lifetime_buffer
        lifetime_buffer.episodes_returns=lifetime_buffer.episodes_returns+episodes_returns_lifetime
        lifetime_buffer.episodes_successes =lifetime_buffer.episodes_successes+ episodes_successes_lifetime

        #需要把evaluation_error反向传播以后才能reset

        information={'prev_done': done_lifetime, 'current_lifetime_step': current_lifetime_step,
                     'current_episode_return': episode_return_lifetime, 'current_episode_success': succeeded_in_episode_lifetime}

        return information
    
    def evaluate_critic(self, config:Metaworld):
        total_eval_steps = config.num_env_steps_for_estimating_maml_loss

        state, _ = self.eval_env.reset() 
        step = 0
        episode_timesteps = 0
        episode_return = 0
        
        print("Evaluation Begins.")
        for cur_steps in tqdm(range(total_eval_steps)):
            step += 1
            #使用当前状态从策略网络获得该执行的动作和对数概率和状态价值
            action, logp, v_t = self.policy.get_action_and_val(state)

            # Step the env
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_timesteps += 1 #当前回合的时间步加1
            episode_return += reward #当前回合的总奖励加1

            done_flag = done

            #将这一回合的信息放在回放缓存区，更新状态
            self.replay_buffer.add(obs=state, act=action, obs2=next_state, rew=reward, done=done_flag, val=v_t, logp=logp)
            state = next_state

            if done or (episode_timesteps == config.max_episode_steps):
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0 if done else self.policy.get_val(state)
                self.replay_buffer.finish_path(last_val)
                state, _ = self.env.reset()
                episode_timesteps = 0
                episode_return = 0

        raw_states, actions, advantages, returns, logp_olds = self.replay_buffer.get()
        critic_loss = self.policy.critic_loss_for_adaptation(raw_states, returns)

        critic_loss.backward() #报错了
        print("大成功！！！！")

        #重置克隆网络
        self.policy.reset_cloned_networks()
    