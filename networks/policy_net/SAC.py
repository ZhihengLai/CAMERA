import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
import os

class GaussianActor(nn.Module):
    LOG_SIG_CAP_MAX = 2
    LOG_SIG_CAP_MIN = -20
    EPS = 1e-6

    def __init__(self, state_dim, action_dim, max_action, layer_units=(256, 256)):
        super(GaussianActor, self).__init__()

        layers = []
        input_dim = state_dim
        for units in layer_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            input_dim = units

        self.base_layers = nn.Sequential(*layers)

        self.out_mean = nn.Linear(layer_units[-1], action_dim)
        self.out_sigma = nn.Linear(layer_units[-1], action_dim)
        self._max_action = max_action

    def _dist_from_states(self, states):
        features = self.base_layers(states)
        mu_t = self.out_mean(features)
        log_sigma_t = self.out_sigma(features)
        log_sigma_t = torch.clamp(log_sigma_t, min=self.LOG_SIG_CAP_MIN, max=self.LOG_SIG_CAP_MAX)

        dist = D.MultivariateNormal(loc=mu_t, scale_tril=torch.diag_embed(torch.exp(log_sigma_t)))
        return dist

    def forward(self, states):
        dist = self._dist_from_states(states)
        raw_actions = dist.sample()
        log_pis = dist.log_prob(raw_actions)

        actions = torch.tanh(raw_actions)
        diff = torch.sum(torch.log(1 - actions ** 2 + self.EPS), dim=1)
        log_pis -= diff

        actions = actions * self._max_action
        return actions, log_pis

    def mean_action(self, states):
        dist = self._dist_from_states(states)
        raw_actions = dist.mean()
        actions = torch.tanh(raw_actions) * self._max_action
        return actions


class CriticV(nn.Module):
    def __init__(self, state_dim, layer_units=(256, 256)):
        super(CriticV, self).__init__()

        layers = []
        input_dim = state_dim
        for units in layer_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            input_dim = units

        self.base_layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(layer_units[-1], 1)

    def forward(self, states):
        features = self.base_layers(states)
        values = self.out_layer(features)
        return values.squeeze(-1)

class CriticQ(nn.Module):
    def __init__(self, state_action_dim, layer_units=(256, 256)):
        super(CriticQ, self).__init__()

        layers = []
        input_dim = state_action_dim
        for units in layer_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            input_dim = units

        self.base_layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(layer_units[-1], 1)

    def forward(self, inputs):
        features = self.base_layers(inputs)
        values = self.out_layer(features)
        return values.squeeze(-1)


import torch.optim as optim

class SAC:
    def __init__(self, state_dim, action_dim, max_action, scale_reward, feature_extractor,
                 learning_rate=3e-4, actor_units=(256, 256), q_units=(256, 256), v_units=(256, 256), tau=0.005):
        self.scale_reward = scale_reward
        self.actor = GaussianActor(feature_extractor.dim_state_features, action_dim, max_action, layer_units=actor_units)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.vf = CriticV(feature_extractor.dim_state_features, layer_units=v_units)
        self.vf_target = CriticV(feature_extractor.dim_state_features, layer_units=v_units)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=learning_rate)

        self.qf1 = CriticQ(feature_extractor.dim_state_action_features, layer_units=q_units)
        self.qf2 = CriticQ(feature_extractor.dim_state_action_features, layer_units=q_units)
        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=learning_rate)
        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=learning_rate)

        # Target update initialization
        self._update_target(self.vf_target, self.vf, tau=1.0)

        # Constants
        self.tau = tau
        self.ofe_net = feature_extractor

    def _update_target(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self._select_action_body(state)
        return action.squeeze(0).detach().numpy()

    def _select_action_body(self, state):
        state_feature = self.ofe_net.features_from_states(state)
        action = self.actor.mean_action(state_feature)
        return action

    def select_action_noise(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self._select_action_noise_body(state)
        return action.squeeze(0).detach().numpy()

    def _select_action_noise_body(self, state):
        state_feature = self.ofe_net.features_from_states(state)
        action, _ = self.actor(state_feature)
        return action

    def train_for_batch(self, states, actions, rewards, next_states, done, discount=0.99):
        done = done.squeeze(-1)
        rewards = rewards.squeeze(-1)
        not_done = 1 - done.float()

        # Critic Update
        state_action_features = self.ofe_net.features_from_states_actions(states, actions)
        next_state_features = self.ofe_net.features_from_states(next_states)

        q1 = self.qf1(state_action_features)
        q2 = self.qf2(state_action_features)
        vf_next_target_t = self.vf_target(next_state_features)

        # Equation (7, 8)
        ys = (self.scale_reward * rewards + not_done * discount * vf_next_target_t).detach()

        td_loss1 = F.mse_loss(ys, q1)
        td_loss2 = F.mse_loss(ys, q2)

        # Update qf1
        self.qf1_optimizer.zero_grad()
        td_loss1.backward()
        self.qf1_optimizer.step()

        # Update qf2
        self.qf2_optimizer.zero_grad()
        td_loss2.backward()
        self.qf2_optimizer.step()

        # Actor Update
        state_features = self.ofe_net.features_from_states(states)
        vf_t = self.vf(state_features)
        sample_actions, log_pi = self.actor(state_features)

        state_action_features = self.ofe_net.features_from_states_actions(states, sample_actions)
        q1 = self.qf1(state_action_features)
        q2 = self.qf2(state_action_features)
        min_q = torch.min(q1, q2)

        # Equation (12)
        policy_loss = torch.mean(log_pi - q1)

        # Equation (5)
        target_v = (min_q - log_pi).detach()
        vf_loss_t = F.mse_loss(vf_t, target_v)

        # Update vf
        self.vf_optimizer.zero_grad()
        vf_loss_t.backward()
        self.vf_optimizer.step()

        # Update actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return policy_loss.item(), vf_loss_t.item(), td_loss1.item()

    def train(self, replay_buffer, batch_size=256, discount=0.99):
        states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size)
        policy_loss, vf_loss, td_loss = self.train_for_batch(states, actions, rewards, next_states, dones, discount)
        self._update_target(self.vf_target, self.vf, self.tau)

    def save(self, save_dir):
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'agent_actor_model.pth'))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, 'agent_critic_model.pth'))

    def load(self, load_dir):
        self.actor.load_state_dict(torch.load(os.path.join(load_dir, 'agent_actor_model.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(load_dir, 'agent_critic_model.pth')))

