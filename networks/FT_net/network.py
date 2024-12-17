import os
import gin
import torch
import torch.nn as nn
import torch.optim as optim
import math

from .blocks import DensenetBlock, MLPBlock

@gin.configurable
class OFENet(nn.Module):
    def __init__(self, dim_state, dim_action, dim_output, dim_discretize, 
                 total_units, num_layers, batchnorm, 
                 fourier_type, discount,  
                 use_projection=True, projection_dim=256, 
                 cosine_similarity=True, normalizer="batch",
                 activation=nn.ReLU(), block="densenet",
                 trainable=True, name='FeatureNet',
                 gpu = 0,
                 skip_action_branch=False):
        super(OFENet, self).__init__()
        self._skip_action_branch = skip_action_branch

        state_layer_units, action_layer_units = calculate_layer_units(dim_state, dim_action, block, total_units, num_layers)
        self.act = activation
        self.batchnorm = batchnorm
        self.block = block

        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_discretize = dim_discretize
        self.dim_output = dim_output

        self.device = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        if block not in ["densenet", "mlp", "mlp_cat"]:
            raise ValueError(f"Invalid block: {block}")

        block_class = DensenetBlock if block == "densenet" else MLPBlock

        state_blocks = []
        for cur_layer_size in state_layer_units:
            cur_state_block = block_class(units_in=dim_state if len(state_blocks) == 0 else state_blocks[-1].out_features, units_out=cur_layer_size, activation=activation, batchnorm=batchnorm, normalizer=normalizer, trainable=trainable)
            state_blocks.append(cur_state_block)
        self.state_blocks = nn.ModuleList(state_blocks)

        action_blocks = []
        self.dim_state_features = state_blocks[-1].out_features
        
        dim_feature_and_action = dim_action + self.dim_state_features
        for cur_layer_size in action_layer_units:
            cur_action_block = block_class(units_in=dim_feature_and_action if len(action_blocks) == 0 else action_blocks[-1].out_features, units_out=cur_layer_size, activation=activation, batchnorm=batchnorm, normalizer=normalizer, trainable=trainable)
            action_blocks.append(cur_action_block)
        self.action_blocks = nn.ModuleList(action_blocks)

        self.end = int(dim_discretize * 0.5 + 1)
        self.prediction = Prediction(dim_input=action_blocks[-1].out_features , dim_discretize=self.end, dim_state=dim_state, normalizer=normalizer)

        if use_projection:
            self.projection = Projection(self.end, self.dim_state, output_dim=projection_dim, normalizer=normalizer)
            self.projection2 = Projection2(output_dim=projection_dim, normalizer=normalizer)

        self.fourier_type = fourier_type
        self.use_projection = use_projection
        self.projection_dim = projection_dim
        self.cosine_similarity = cosine_similarity

        if trainable:
            self.aux_optimizer = optim.Adam(self.parameters(), lr=3e-4)
        else:
            self.aux_optimizer = None

        ratio = 2 * math.pi / dim_discretize
        con = torch.tensor([k * ratio for k in range(self.end)], device=self.device) #something is wrong about GPU
        self.Gamma_re = discount * torch.diag(torch.cos(con))
        self.Gamma_im = -discount * torch.diag(torch.sin(con))

        ratio2 = 1.0 / (2 * math.pi * dim_discretize)
        con_re = torch.tensor([1] + [2 * math.cos(k * ratio) for k in range(1, self.end - 1)] + [-1], device=self.device)
        self.con_re = ratio2 * con_re.unsqueeze(0)
        con_im = torch.tensor([0] + [-2 * math.sin(k * ratio) for k in range(1, self.end - 1)] + [0], device=self.device)
        self.con_im = ratio2 * con_im.unsqueeze(0)

        # if fourier_type == 'dft':
        #     ratio = 2 * math.pi * (dim_discretize - 1) / dim_discretize
        #     con = torch.tensor([k * ratio for k in range(dim_discretize)], device=f'cuda:{self._gpu}')
        #     self.con_re = torch.cos(con)
        #     self.con_im = torch.sin(con)
        #     self.coef = pow(discount, dim_discretize - 1)

    def forward(self, states, actions=None):
        batch_size = states.size(0)

        features = states
        for cur_block in self.state_blocks:
            features = cur_block(features)

        if self.block == "mlp_cat":
            features = torch.cat([features, states], dim=1)

        if actions is not None and not self._skip_action_branch:
            features = torch.cat([features, actions], dim=1)

            for cur_block in self.action_blocks:
                features = cur_block(features)

            if self.block == "mlp_cat":
                features = torch.cat([features, states, actions], dim=1)

        predictor_re, predictor_im = self.prediction(features)
        predictor_re = predictor_re.view(batch_size, self.end, self.dim_state)
        predictor_im = predictor_im.view(batch_size, self.end, self.dim_state)

        return predictor_re, predictor_im

    def features_from_states(self, states):
        features = states
        for cur_block in self.state_blocks:
            features = cur_block(features)

        if self.block == "mlp_cat":
            features = torch.cat([features, states], dim=1)

        return features

    def features_from_states_actions(self, states, actions):
        state_features = self.features_from_states(states)
        features = torch.cat([state_features, actions], dim=1)

        for cur_block in self.action_blocks:
            features = cur_block(features)

        if self.block == "mlp_cat":
            features = torch.cat([features, states, actions], dim=1)

        return features

    def loss(self, y_target, y, target_model=None):
        trun = 15

        if self.use_projection and target_model is not None:
            y_target2 = target_model.projection(y_target[:, trun:self.end-trun, :])
            _ = target_model.projection2(y_target2)
            y2 = self.projection(y[:, trun:self.end-trun, :])
            y2 = self.projection2(y2)

        if self.cosine_similarity:
            loss_fun = nn.CosineSimilarity(dim=-1)
            loss1 = torch.mean(loss_fun(y_target[:, :trun, :], y[:, :trun, :]))
            loss2 = torch.mean(loss_fun(y_target2, y2))
            loss3 = torch.mean(loss_fun(y_target[:, self.end-trun:self.end, :], y[:, self.end-trun:self.end, :]))

            loss = loss1 + loss2 + loss3
        else:
            loss = nn.functional.mse_loss(y_target2, y2)

        return loss

    # def train_model(self, target_model, states, actions, next_states, next_actions, dones, Hth_states=None):
        
    #     dones = dones.unsqueeze(-1).repeat(1, self.end, self.dim_state).float()
    #     O = next_states[:, :self.dim_state].unsqueeze(1).repeat(1, self.end, 1)
    #     predicted_re, predicted_im = self(states, actions)
    #     next_predicted_re, next_predicted_im = target_model(next_states, next_actions)

    #     if self.fourier_type == 'dtft':
    #         with torch.no_grad():
    #             y_target_re = O + (torch.matmul(self.Gamma_re, next_predicted_re) - torch.matmul(self.Gamma_im, next_predicted_im)) * (1 - dones)
    #         y_re = predicted_re
    #         pred_re_loss = self.loss(y_target_re.detach(), y_re, target_model)
    #         with torch.no_grad():
    #             y_target_im = (torch.matmul(self.Gamma_im, next_predicted_re) + torch.matmul(self.Gamma_re, next_predicted_im)) * (1 - dones)
    #         y_im = predicted_im
    #         pred_im_loss = self.loss(y_target_im.detach(), y_im, target_model)

    #     elif self.fourier_type == 'dft':
    #         y_target_re = O - self.coef * torch.matmul(self.con_re.unsqueeze(-1), Hth_states.unsqueeze(1)) + \
    #                       (torch.matmul(self.Gamma_re, next_predicted_re) - torch.matmul(self.Gamma_im, next_predicted_im)) * (1 - dones)
    #         y_re = predicted_re
    #         pred_re_loss = self.loss(y_target_re.detach(), y_re, target_model)

    #         y_target_im = self.coef * torch.matmul(self.con_re.unsqueeze(-1), Hth_states.unsqueeze(1)) + \
    #                       (torch.matmul(self.Gamma_im, next_predicted_re) + torch.matmul(self.Gamma_re, next_predicted_im)) * (1 - dones)
    #         y_im = predicted_im
    #         pred_im_loss = self.loss(y_target_im.detach(), y_im, target_model)

    #     else:
    #         raise ValueError(f"Invalid fourier_type: {self.fourier_type}")

    #     pred_loss = pred_re_loss + pred_im_loss

    #     self.aux_optimizer.zero_grad()
    #     pred_loss.backward()
    #     self.aux_optimizer.step()

    #     if self.use_projection:
    #         grads_proj = [p.grad for p in self.projection.parameters()]

    #     grads_pred = [p.grad for p in self.prediction.parameters()]

    #     return pred_loss, pred_re_loss, pred_im_loss, grads_proj, grads_pred
    
    def compute_loss(self, target_model, states, actions, next_states, next_actions, dones, Hth_states=None):
        
        dones = dones.unsqueeze(-1).repeat(1, self.end, self.dim_state).float()
        O = next_states[:, :self.dim_state].unsqueeze(1).repeat(1, self.end, 1)
        predicted_re, predicted_im = self(states, actions)
        next_predicted_re, next_predicted_im = target_model(next_states, next_actions)

        if self.fourier_type == 'dtft':
            with torch.no_grad():
                y_target_re = O + (torch.matmul(self.Gamma_re, next_predicted_re) - torch.matmul(self.Gamma_im, next_predicted_im)) * (1 - dones)
            y_re = predicted_re
            pred_re_loss = self.loss(y_target_re.detach(), y_re, target_model)
            with torch.no_grad():
                y_target_im = (torch.matmul(self.Gamma_im, next_predicted_re) + torch.matmul(self.Gamma_re, next_predicted_im)) * (1 - dones)
            y_im = predicted_im
            pred_im_loss = self.loss(y_target_im.detach(), y_im, target_model)

        elif self.fourier_type == 'dft':
            y_target_re = O - self.coef * torch.matmul(self.con_re.unsqueeze(-1), Hth_states.unsqueeze(1)) + \
                          (torch.matmul(self.Gamma_re, next_predicted_re) - torch.matmul(self.Gamma_im, next_predicted_im)) * (1 - dones)
            y_re = predicted_re
            pred_re_loss = self.loss(y_target_re.detach(), y_re, target_model)

            y_target_im = self.coef * torch.matmul(self.con_re.unsqueeze(-1), Hth_states.unsqueeze(1)) + \
                          (torch.matmul(self.Gamma_im, next_predicted_re) + torch.matmul(self.Gamma_re, next_predicted_im)) * (1 - dones)
            y_im = predicted_im
            pred_im_loss = self.loss(y_target_im.detach(), y_im, target_model)

        else:
            raise ValueError(f"Invalid fourier_type: {self.fourier_type}")

        pred_loss = pred_re_loss + pred_im_loss

        return pred_loss


    def save(self, save_dir):
        torch.save(self.state_dict, save_dir)

    def load(self, load_dir):
        self.load_state_dict(torch.load(load_dir))

def calculate_layer_units(state_dim, action_dim, ofe_block, total_units, num_layers):
    assert total_units % num_layers == 0

    if ofe_block == "densenet":
        per_unit = total_units // num_layers
        state_layer_units = [per_unit] * num_layers
        action_layer_units = [per_unit] * num_layers

    elif ofe_block in ["mlp"]:
        state_layer_units = [total_units + state_dim] * num_layers
        action_layer_units = [total_units * 2 + state_dim + action_dim] * num_layers

    elif ofe_block in ["mlp_cat"]:
        state_layer_units = [total_units] * num_layers
        action_layer_units = [total_units * 2] * num_layers

    else:
        raise ValueError("invalid connection type")

    return state_layer_units, action_layer_units


class Projection(nn.Module):
    def __init__(self, end, dim_state, classifier_type="mlp", output_dim=256, normalizer="batch", trainable=True):
        super(Projection, self).__init__()
        self.classifier_type = classifier_type
        self.output_dim = output_dim
        self.normalizer = normalizer
        
        self.dense1 = nn.Linear((end-30) * dim_state, output_dim*2)
        self.dense2 = nn.Linear(output_dim*2, output_dim)
        
        if normalizer == 'batch':
            self.normalization = nn.BatchNorm1d(output_dim*2)
        elif normalizer == 'layer':
            self.normalization = nn.LayerNorm(output_dim*2)
        else:
            self.normalization = None
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
    def forward(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        if self.normalization:
            x = self.normalization(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x


class Projection2(nn.Module):
    def __init__(self, classifier_type="mlp", output_dim=256, normalizer="batch", trainable=True):
        super(Projection2, self).__init__()
        self.classifier_type = classifier_type
        self.output_dim = output_dim
        self.normalizer = normalizer
        
        self.dense1 = nn.Linear(output_dim, output_dim*2)
        self.dense2 = nn.Linear(output_dim*2, output_dim)
        
        if normalizer == 'batch':
            self.normalization = nn.BatchNorm1d(output_dim*2)
        elif normalizer == 'layer':
            self.normalization = nn.LayerNorm(output_dim*2)
        else:
            self.normalization = None
        
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.dense1(inputs)
        if self.normalization:
            x = self.normalization(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x


class Prediction(nn.Module):
    def __init__(self, dim_input, dim_discretize, dim_state, normalizer="batch", trainable=True):
        super(Prediction, self).__init__()
        self.output_dim = dim_discretize * dim_state #output dimension of Prediction module
        self.normalizer = normalizer
        
        self.pred_layer = nn.Linear(dim_input, 1024)  # Adjust input dimension as needed
        self.out_layer_re = nn.Linear(1024, self.output_dim)
        self.out_layer_im = nn.Linear(1024, self.output_dim)
        
        if normalizer == 'batch':
            self.normalization = nn.BatchNorm1d(1024)
        elif normalizer == 'layer':
            self.normalization = nn.LayerNorm(1024)
        else:
            self.normalization = None
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, inputs):
        x = self.pred_layer(inputs)
        if self.normalization:
            x = self.normalization(x)
        x = self.relu(x)
        return self.out_layer_re(x), self.out_layer_im(x)
