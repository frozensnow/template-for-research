import torch
import numpy as np
import torch.nn as nn


"""
input the name of activation and output the function.
"""
def activation_helper(activation=None):
    if activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'relu':
        act = nn.ReLU()
    elif activation == 'leakyrelu':
        act = nn.LeakyReLU()
    elif activation == 'glu':
        act = nn.GLU(dim=1)
    elif activation is None:
        def act(x):
            return x
    else:
        raise ValueError('unsupported activation: %s' % activation)
    return act
