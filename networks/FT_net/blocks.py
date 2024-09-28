import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self,units_in, units_out, activation, kernel_initializer='glorot_uniform', batchnorm=False, normalizer="batch", trainable=True, name="mlpblock"):
        super(MLPBlock, self).__init__()

        self.act = activation
        self.fc = nn.Linear(units_in, units_out)
        self.out_features = units_out

        if kernel_initializer == 'glorot_uniform':
            nn.init.xavier_uniform_(self.fc.weight)        

        if not trainable:
            for param in self.fc.parameters():
                param.requires_grad = False

        self.batchnorm = batchnorm
        if self.batchnorm:
            self.normalizer = nn.BatchNorm1d(units_out) if normalizer == "batch" else nn.LayerNorm(units_out)

    def forward(self, inputs, training = False):
        features = self.fc(inputs)

        if self.batchnorm:
            features = self.normalizer(features)
        features = self.act(features)

        return features


# class ResnetBlock(nn.Module):
#     def __init__(self, units1, units2, activation, kernel_initializer='glorot_uniform', batchnorm=False, normalizer="batch", trainable=True, name="resblock"):
#         super(ResnetBlock, self).__init__()

#         self.act = activation

#         self.fc1 = nn.Linear(units1, units1)
#         self.fc2 = nn.Linear(units2, units2)

#         if kernel_initializer == 'glorot_uniform':
#             nn.init.xavier_uniform_(self.fc1.weight)
#             nn.init.xavier_uniform_(self.fc2.weight)

#         if not trainable:
#             for param in self.fc1.parameters():
#                 param.requires_grad = False
#             for param in self.fc2.parameters():
#                 param.requires_grad = False

#         self.batchnorm = batchnorm
#         if self.batchnorm:
#             self.normalizer1 = nn.BatchNorm1d(units1) if normalizer == "batch" else nn.LayerNorm(units1)
#             self.normalizer2 = nn.BatchNorm1d(units2) if normalizer == "batch" else nn.LayerNorm(units2)

#     def forward(self, inputs, training = False):
#         identity_map = inputs

#         features = self.fc1(inputs)

#         if self.batchnorm:
#             features = self.normalizer1(features)

#         features = self.act(features)
#         features = self.fc2(features)

#         if self.batchnorm:
#             features = self.normalizer2(features)

#         if features.shape[1] > identity_map.shape[1]:
#             identity_map = F.pad(identity_map, (0, features.shape[1] - identity_map.shape[1]))
#         elif features.shape[1] < identity_map.shape[1]:
#             features = F.pad(features, (0, identity_map.shape[1] - features.shape[1]))

#         features = features + identity_map
#         features = self.act(features)

#         return features


class DensenetBlock(nn.Module):
    def __init__(self, units_in, units_out, activation, kernel_initializer='glorot_uniform', batchnorm=False, normalizer="batch", trainable=True, name="denseblock"):
        super(DensenetBlock, self).__init__()

        self.act = activation
        self.fc = nn.Linear(units_in, units_out)
        self.out_features = units_out + units_in

        # Apply He initialization
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')

        if not trainable:
            for param in self.fc.parameters():
                param.requires_grad = False

        self.batchnorm = batchnorm
        self.normalizer = normalizer
        if batchnorm:
            self.BatchNorm1d = nn.BatchNorm1d(units_out)
            self.LayerNorm = nn.LayerNorm(units_out, eps=1e-5)

    def forward(self, inputs):
        identity_map = inputs

        features = self.fc(inputs)

        if self.batchnorm:
            if self.normalizer == 'batch':
                features = self.BatchNorm1d(features)
            elif self.normalizer == 'layer':
                features = self.LayerNorm(features)

        features = self.act(features)
        features = torch.cat([features, identity_map], dim=1)

        return features
