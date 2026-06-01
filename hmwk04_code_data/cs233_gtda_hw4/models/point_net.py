"""
Point-Net.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, init_feat_dim=3, conv_dims=[32, 64, 64, 128, 128], student_optional_hyper_param=None):
        """
        Students:
        You can make a generic function that instantiates a point-net with arbitrary hyper-parameters,
        or go for an implemetnations working only with the hyper-params of the HW.
        Do not use batch-norm, drop-out and other not requested features.
        Just nn.Linear/Conv1D/ReLUs and the (max) poolings.
        
        :param init_feat_dim: input point dimensionality (default 3 for xyz)
        :param conv_dims: output point dimensionality of each layer
        """
        super(PointNet, self).__init__()
        self.model = nn.Sequential()
        for i in range(len(conv_dims)):
            prev_dim = init_feat_dim if i == 0 else conv_dims[i - 1]
            self.model.append(nn.Conv1d(prev_dim, conv_dims[i], 1)) # What to set for kernel size?
            self.model.append(nn.ReLU())
        # self.model.append(nn.MaxPool1d(conv_dims[-1]))

        
        
    def forward(self, pointclouds):
        """
        Run forward pass of the PointNet model on a given point cloud.
        :param pointclouds: (B x N x 3) point cloud
        """
        pointclouds = pointclouds.transpose(1, 2)
        model_output = self.model(pointclouds)
        max_pooling = torch.max(model_output, dim=-1).values
        return max_pooling