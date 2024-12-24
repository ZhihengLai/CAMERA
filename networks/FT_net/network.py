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
                 total_units, num_layers, batchnorm, normalizer="batch",
                 activation=nn.ReLU(), block="densenet",
                 trainable=False, gpu = 0,
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
