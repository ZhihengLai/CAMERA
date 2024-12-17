import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.init as init

import torch.optim as optim
import numpy as np
import os

EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20

# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class RolloutBuffer:
    """A buffer for storing trajectories experienced by a PPO agent.
    Uses Generalized Advantage Estimation (GAE-Lambda) for calculating
    the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, capacity, gamma=0.99, lam=0.95, device='cpu'):
        self.obs_buf = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((capacity, act_dim), dtype=torch.float32, device=device)
        self.obs2_buf = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.adv_buf = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.ret_buf = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.val_buf = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.logp_buf = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.capacity = 0, 0, capacity
        self.device = device

    def add(self, obs, act, obs2, rew, done, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.capacity    # buffer has to have room so you can store
        self.obs_buf[self.ptr] = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.act_buf[self.ptr] = torch.tensor(act, dtype=torch.float32, device=self.device)
        self.obs2_buf[self.ptr] = torch.tensor(obs2, dtype=torch.float32, device=self.device)
        self.rew_buf[self.ptr] = torch.tensor(rew, dtype=torch.float32, device=self.device)
        self.done_buf[self.ptr] = torch.tensor(done, dtype=torch.float32, device=self.device)
        self.val_buf[self.ptr] = torch.tensor(val, dtype=torch.float32, device=self.device)
        self.logp_buf[self.ptr] = torch.tensor(logp, dtype=torch.float32, device=self.device)
        
        self.ptr += 1

    def discount_cumsum(self, x, discount):
        """
        Compute discounted cumulative sums of vectors using PyTorch.
        
        Input:
            vector x: [x0, x1, x2, ...]
            discount: scalar discount factor

        Output:
            discounted cumulative sum:
            [x0 + discount * x1 + discount^2 * x2 + ..., 
            x1 + discount * x2 + discount^2 * x3 + ...,
            x2 + discount * x3 + discount^2 * x4 + ...,
            ...]
        """       
        return torch.flip(torch.cumsum(torch.flip(x, dims=[0]) * discount, dim=0), dims=[0])

    def finish_path(self, last_val=0):
        """
        Use to compute returns and advantages.

        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = torch.cat((self.rew_buf[path_slice], torch.tensor([last_val], device=self.device)))
        vals = torch.cat((self.val_buf[path_slice], torch.tensor([last_val], device=self.device)))

        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)

        # Rewards-to-go for the value function targets
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Returns data stored in buffer.
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.capacity    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        # Advantage normalization trick
        adv_mean = torch.mean(self.adv_buf)
        adv_std = torch.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-5)

        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]

    def sample(self, batch_size=100):
        ind = torch.randint(0, self.ptr, (batch_size,), device=self.device)

        cur_states = self.obs_buf[ind, :]
        cur_next_states = self.obs2_buf[ind, :]
        cur_actions = self.act_buf[ind, :]
        cur_rewards = self.rew_buf[ind]
        cur_dones = self.done_buf[ind]

        return cur_states, cur_actions, cur_next_states, cur_rewards.unsqueeze(-1), cur_dones.unsqueeze(-1)
        

class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, layer_units=(256, 256), hidden_activation=nn.Tanh):
        super(GaussianActor, self).__init__()
        self._max_action = max_action

        # Create base layers
        layers = []
        for cur_layer_size in layer_units:
            linear_layer = nn.Linear(state_dim if len(layers) == 0 else layers[-2].out_features, cur_layer_size)
            init.orthogonal_(linear_layer.weight)  # Apply orthogonal initialization
            layers.append(linear_layer)
            layers.append(hidden_activation())
        self.base_layers = nn.Sequential(*layers)

        # Output layers for mean and log standard deviation
        self.out_mean = nn.Linear(layer_units[-1], action_dim)
        init.orthogonal_(self.out_mean.weight)  # Apply orthogonal initialization
        self.out_logstd = nn.Parameter(-0.5 * torch.ones(action_dim))

    def _dist_from_states(self, states):
        features = states

        # Forward pass through the base layers
        features = self.base_layers(features)

        # Compute mean and log standard deviation
        mu_t = self.out_mean(features)
        log_sigma_t = torch.clamp(self.out_logstd, LOG_STD_MIN, LOG_STD_MAX)

        scale_diag = torch.exp(log_sigma_t)
        cov_matrix = torch.diag(scale_diag**2)
        # Create the Gaussian distribution
        dist = MultivariateNormal(loc=mu_t, covariance_matrix=cov_matrix)

        return dist

    def forward(self, states):
        dist = self._dist_from_states(states)
        raw_actions = dist.sample()
        log_pis = dist.log_prob(raw_actions)

        return raw_actions, log_pis

    def mean_action(self, states):
        dist = self._dist_from_states(states)
        raw_actions = dist.mean
        log_pis = dist.log_prob(raw_actions)

        return raw_actions, log_pis

    def compute_log_probs(self, states, actions):
        dist = self._dist_from_states(states)
        log_pis = dist.log_prob(actions)

        return log_pis



class CriticV(nn.Module):
    def __init__(self, state_dim, units):
        super(CriticV, self).__init__()

        l1 = nn.Linear(state_dim, units[0])
        l2 = nn.Linear(units[0], units[1])
        l3 = nn.Linear(units[1], 1)
        # Apply orthogonal initialization
        init.orthogonal_(l1.weight)
        init.orthogonal_(l2.weight)
        init.orthogonal_(l3.weight)

        self.layers = nn.Sequential(l1, nn.Tanh(), l2, nn.Tanh(), l3)

    def forward(self, inputs):
        
        return self.layers(inputs).squeeze(-1)


class PPO(nn.Module):
    def __init__(
            self,
            feature_extractor,
            actor,
            critic,
            pi_lr=3e-4,
            vf_lr=1e-3,
            clip_ratio=0.2,
            batch_size=64,
            discount=0.99,
            n_epoch=10,
            horizon=2048,
            gpu=0):
        super(PPO, self).__init__()
        self.batch_size = batch_size
        self.discount = discount
        self.n_epoch = n_epoch
        self.device = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")
        self.horizon = horizon
        self.clip_ratio = clip_ratio
        assert self.horizon % self.batch_size == 0, \
            "Horizon should be divisible by batch size"

        self.actor = actor.to(self.device)
        self.critic_original = critic.to(self.device)
        self.critic_active = self.critic

        self.ofe_net_original = feature_extractor.to(self.device)
        self.ofe_net_active = self.ofe_net_original

        # Clone used in adaptation phase
        self.extractor_clone = None
        self.critic_clone = None
        self.mode = "Validation"

        # Initialize optimizer of inner loop
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=pi_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=vf_lr)

    def clone_for_adaptation(self):
        '''
        Clone the extractor and critic network for inner loop adaptation
        '''
        self.extractor_clone = self.ofe_net_original.clone()
        self.ofe_net_active = self.extractor_clone

        self.critic_clone = self.critic.clone()
        self.critic_active = self.critic_clone

        self.mode = "Adaptation"

    def reset_cloned_networks(self):
        '''
        Reset the extractor and critic network back to the version before adaptation
        '''
        self.extractor_clone = None
        self.ofe_net_active = self.ofe_net_original

        self.critic_clone = None
        self.critic_active = self.critic_original

        self.mode = "Validation"

    def extractor_loss_for_adaptation(self, target_model, states, actions, next_states, next_actions, dones, Hth_states=None):
        loss = self.ofe_net_active.model.compute_loss(target_model, states, actions, next_states, next_actions, dones, Hth_states)
        return loss
    
    def extractor_adapt(self, loss):
        self.ofe_net_active.adapt(loss)

    def get_action(self, raw_state, test=False):
        raw_state = torch.tensor(raw_state, dtype=torch.float32).to(self.device)

        is_single_input = raw_state.ndim == 1
        if is_single_input:
            raw_state = raw_state.unsqueeze(0)  # Add a batch dimension if single input

        action, logp = self._get_action_body(raw_state, test)[:2]

        if is_single_input:
            return action[0], logp
        else:
            return action, logp

    def get_action_and_val(self, raw_state, test=False):
        raw_state = torch.tensor(raw_state, dtype=torch.float32).to(self.device)

        is_single_input = raw_state.ndim == 1
        if is_single_input:
            raw_state = raw_state.unsqueeze(0)  # Add a batch dimension if single input

        action, logp, v = self._get_action_logp_v_body(raw_state, test)

        if is_single_input:
            v = v[0]
            action = action[0]

        return action.detach().numpy(), logp.detach().numpy(), v.detach().numpy()


    def get_logp_and_val(self, raw_state, action):
        is_single_input = raw_state.ndim == 1
        if is_single_input:
            raw_state = np.expand_dims(raw_state, axis=0).astype(np.float32)

        raw_state_tensor = torch.tensor(raw_state, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32).to(self.device)
        state_feature = self.ofe_net_active.features_from_states(raw_state_tensor)
        logp = self.actor.compute_log_probs(state_feature, action_tensor)
        v = self.critic_active(state_feature)

        if is_single_input:
            v = v[0]
            action = action[0]

        return logp.detach().numpy(), v.detach().numpy()

    def get_val(self, raw_state):
        is_single_input = raw_state.ndim == 1
        if is_single_input:
            raw_state = np.expand_dims(raw_state, axis=0).astype(np.float32)

        raw_state_tensor = torch.tensor(raw_state, dtype=torch.float32).to(self.device)
        state_feature = self.ofe_net_active.features_from_states(raw_state_tensor)
        v = self.critic_active(state_feature)

        if is_single_input:
            v = v[0]

        return v.detach().numpy()

    def _get_action_logp_v_body(self, raw_state, test):
        action, logp = self._get_action_body(raw_state, test)[:2]
        state_feature = self.ofe_net_active.features_from_states(raw_state)
        v = self.critic_active(state_feature)
        return action, logp, v

    def _get_action_body(self, state, test):
        state_feature = self.ofe_net_active.features_from_states(state)
        if test:
            return self.actor.mean_action(state_feature)
        else:
            return self.actor(state_feature)

    def select_action(self, raw_state):
        action, _ = self.get_action(raw_state, test=True)
        return action.detach().numpy()

    def train(self, replay_buffer, train_pi_iters=80, train_v_iters=80, target_kl=0.01):
        raw_states, actions, advantages, returns, logp_olds = replay_buffer.get()

        # Train actor and critic
        for i in range(train_pi_iters):
            actor_loss, kl, entropy, logp_news, ratio = self._train_actor_body(
                raw_states, actions, advantages, logp_olds)
            if kl > 1.5 * target_kl:
                print('Early stopping at step %d due to reaching max kl.' % i)
                break

        for _ in range(train_v_iters):
            critic_loss = self._train_critic_body(raw_states, returns)

        # Optionally: log the metrics to TensorBoard or other logging systems
        # (PyTorch does not have a built-in summary like TensorFlow)
        return actor_loss.item(), critic_loss.item()

    def _train_actor_body(self, raw_states, actions, advantages, logp_olds):
        state_features = self.ofe_net_active.features_from_states(raw_states)
        logp_news = self.actor.compute_log_probs(state_features, actions)
        ratio = torch.exp(logp_news - logp_olds.squeeze())

        min_adv = torch.where(advantages >= 0, 
                              (1 + self.clip_ratio) * advantages,
                              (1 - self.clip_ratio) * advantages)
        actor_loss = -torch.mean(torch.min(ratio * advantages, min_adv))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        kl = torch.mean(logp_olds.squeeze() - logp_news)
        entropy = -torch.mean(logp_news)

        return actor_loss, kl.item(), entropy.item(), logp_news, ratio

    def _train_critic_body(self, raw_states, returns):
        state_features = self.ofe_net_active.features_from_states(raw_states)
        current_V = self.critic_active(state_features)
        td_errors = returns.squeeze() - current_V.squeeze()
        critic_loss = torch.mean(0.5 * td_errors.pow(2))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss

    def save(self, save_dir):
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'agent_actor_model.pth'))
        torch.save(self.critic_active.model.state_dict(), os.path.join(save_dir, 'agent_critic_model.pth'))

    def load(self, load_dir):
        self.actor.load_state_dict(torch.load(os.path.join(load_dir, 'agent_actor_model.pth')))
        self.critic_active.model.load_state_dict(torch.load(os.path.join(load_dir, 'agent_critic_model.pth')))
