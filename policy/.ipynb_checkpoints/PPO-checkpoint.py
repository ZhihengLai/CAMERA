import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.init as init
from torch import autocast

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
    print("PPO: Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("PPO: Device set to : cpu")


class RolloutBuffer:
    """A buffer for storing trajectories experienced by a PPO agent.
    Uses Generalized Advantage Estimation (GAE-Lambda) for calculating
    the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, capacity, gamma=0.99, lam=0.95, device='cuda:0'):
        self.obs_buf = []  # List to store observations
        self.act_buf = []  # List to store actions
        self.obs2_buf = []  # List to store next observations
        self.rew_buf = []  # List to store rewards
        self.done_buf = []  # List to store done flags
        self.val_buf = []  # List to store value estimates
        self.logp_buf = []  # List to store log probabilities
        self.adv_buf = []  # List to store advantages
        self.ret_buf = []  # List to store returns
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.capacity = 0, 0, capacity
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def add(self, obs, act, obs2, rew, done, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.capacity  # buffer has to have room so you can store
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.obs2_buf.append(obs2)
        self.rew_buf.append(rew)
        self.done_buf.append(done)
        self.val_buf.append(val)
        self.logp_buf.append(logp)
        self.ptr += 1

    def discount_cumsum(self, x, discount):
        """
        Compute discounted cumulative sums of vectors using NumPy.
        
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
        x = np.array(x)
        return np.flip(np.cumsum(np.flip(x) * discount))

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
        rews = np.append(np.array(self.rew_buf[path_slice]), last_val)
        vals = np.append(np.array(self.val_buf[path_slice]), last_val)

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
        assert self.ptr == self.capacity  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        # Convert lists to tensors
        obs_buf = torch.tensor(np.array(self.obs_buf), dtype=torch.float32, device=self.device)
        act_buf = torch.tensor(np.array(self.act_buf), dtype=torch.float32, device=self.device)
        adv_buf = torch.tensor(np.array(self.adv_buf), dtype=torch.float32, device=self.device)
        ret_buf = torch.tensor(np.array(self.ret_buf), dtype=torch.float32, device=self.device)
        logp_buf = torch.tensor(np.array(self.logp_buf), dtype=torch.float32, device=self.device)

        # Advantage normalization trick
        adv_mean = torch.mean(adv_buf)
        adv_std = torch.std(adv_buf)
        adv_buf = (adv_buf - adv_mean) / (adv_std + 1e-5)

        # Clear buffers for next epoch
        self.obs_buf = []
        self.act_buf = []
        self.obs2_buf = []
        self.rew_buf = []
        self.done_buf = []
        self.val_buf = []
        self.logp_buf = []
        self.adv_buf = []
        self.ret_buf = []

        return [obs_buf, act_buf, adv_buf, ret_buf, logp_buf]

    def sample(self, batch_size=100):
        """
        Sample a batch of data from the buffer.
        """
        ind = np.random.randint(0, self.ptr, size=batch_size)

        cur_states = torch.tensor(np.array(self.obs_buf)[ind], dtype=torch.float32, device=self.device)
        cur_next_states = torch.tensor(np.array(self.obs2_buf)[ind], dtype=torch.float32, device=self.device)
        cur_actions = torch.tensor(np.array(self.act_buf)[ind], dtype=torch.float32, device=self.device)
        cur_rewards = torch.tensor(np.array(self.rew_buf)[ind], dtype=torch.float32, device=self.device).unsqueeze(-1)
        cur_dones = torch.tensor(np.array(self.done_buf)[ind], dtype=torch.float32, device=self.device).unsqueeze(-1)

        return cur_states, cur_actions, cur_next_states, cur_rewards, cur_dones
        

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
        self.critic_active = None

        # Clone used in adaptation phase
        self.mode = "Validation"

        # Initialize optimizer of inner loop
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=pi_lr)

        self.pi_lr = pi_lr
        self.vf_lr = vf_lr

    def clone_for_adaptation(self):
        '''
        Clone the extractor and critic network for inner loop adaptation
        '''
        self.critic_active = self.critic_original.clone()

        self.mode = "Adaptation"

    def reset_cloned_networks(self):
        '''
        Reset the extractor and critic network back to the version before adaptation
        '''
        self.critic_active = None

        self.mode = "Validation"

    def set_optimizers(self):
        '''
        Set the optimizers for the parameters of loaded state_dicts in inner loop.
        '''
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.pi_lr)

    def get_action(self, raw_state, test=False):
        raw_state = raw_state.clone().detach().to(self.device) #克隆一个张量

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

        return action.detach().cpu().numpy(), logp.detach().cpu().numpy(), v.detach().cpu().numpy()


    def get_logp_and_val(self, raw_state, action):
        is_single_input = raw_state.ndim == 1
        if is_single_input:
            raw_state = np.expand_dims(raw_state, axis=0).astype(np.float32)

        raw_state_tensor = torch.tensor(raw_state, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32).to(self.device)

        logp = self.actor.compute_log_probs(raw_state_tensor, action_tensor)
        v = self.critic_active(raw_state_tensor)

        if is_single_input:
            v = v[0]
            action = action[0]

        return logp.detach().numpy(), v.detach().numpy()

    def get_val(self, raw_state):
        is_single_input = raw_state.ndim == 1
        if is_single_input:
            raw_state = np.expand_dims(raw_state, axis=0).astype(np.float32)

        raw_state_tensor = torch.tensor(raw_state, dtype=torch.float32).to(self.device)
        v = self.critic_active(raw_state_tensor)

        if is_single_input:
            v = v[0]

        return v.detach().numpy()

    def _get_action_logp_v_body(self, raw_state, test):
        action, logp = self._get_action_body(raw_state, test)[:2]

        v = self.critic_active(raw_state)
        return action, logp, v

    def _get_action_body(self, state, test):
        if test:
            return self.actor.mean_action(state)
        else:
            return self.actor(state)

    def select_action(self, raw_state):
        action, _ = self.get_action(raw_state, test=True)
        return action.detach().numpy()

    def train(self, replay_buffer, actor_scaler:torch.cuda.amp.GradScaler, train_pi_iters=80, train_v_iters=80, target_kl=0.01):
        raw_states, actions, advantages, returns, logp_olds = replay_buffer.get()

        # Train actor and critic
        for i in range(train_pi_iters):
            actor_loss, kl, entropy, logp_news, ratio = self._train_actor_body(
                raw_states, actions, advantages, logp_olds, actor_scaler)
            if kl > 1.5 * target_kl:
                print('Early stopping at step %d due to reaching max kl.' % i)
                break

        for _ in range(train_v_iters):
            critic_loss = self._train_critic_body(raw_states, returns)

        # Optionally: log the metrics to TensorBoard or other logging systems
        # (PyTorch does not have a built-in summary like TensorFlow)
        return actor_loss.item(), critic_loss.item()

    def _train_actor_body(self, raw_states, actions, advantages, logp_olds, actor_scaler:torch.cuda.amp.GradScaler):
        self.actor_optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float16):
            logp_news = self.actor.compute_log_probs(raw_states, actions)
            ratio = torch.exp(logp_news - logp_olds.squeeze())

            min_adv = torch.where(advantages >= 0, 
                                (1 + self.clip_ratio) * advantages,
                                (1 - self.clip_ratio) * advantages)
            actor_loss = -torch.mean(torch.min(ratio * advantages, min_adv))

        actor_scaler.scale(actor_loss).backward()
        actor_scaler.step(self.actor_optimizer)
        actor_scaler.update()

        kl = torch.mean(logp_olds.squeeze() - logp_news)
        entropy = -torch.mean(logp_news)

        return actor_loss, kl.item(), entropy.item(), logp_news, ratio

    def _train_critic_body(self, raw_states, returns):
        current_V = self.critic_active(raw_states)
        td_errors = returns.squeeze() - current_V.squeeze()
        critic_loss = torch.mean(0.5 * td_errors.pow(2))

        self.critic_active.adapt(critic_loss)

        return critic_loss
    
    def critic_loss_for_adaptation(self, raw_states, returns):
        current_V = self.critic_active(raw_states)
        td_errors = returns.squeeze() - current_V.squeeze()
        critic_loss = torch.mean(0.5 * td_errors.pow(2))
        return critic_loss

    def save(self, save_dir):
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'agent_actor_model.pth'))
        torch.save(self.critic_active.module.state_dict(), os.path.join(save_dir, 'agent_critic_model.pth'))

    def load(self, load_dir):
        self.actor.load_state_dict(torch.load(os.path.join(load_dir, 'agent_actor_model.pth')))
        self.critic_active.module.load_state_dict(torch.load(os.path.join(load_dir, 'agent_critic_model.pth')))
