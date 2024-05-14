import torch
import numpy as np
import torch.nn as nn

from utils.attention import FullAttention


def activation_helper(activation=None):
    """
    input the name of activation and output the function.
    :param activation: activation function name
    :return: nn.
    """
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


class TransformerModule(nn.Module):
    def __init__(self, output_attention, n_heads, d_ff, d_model=20,
                 mask_flag=True, factor=None, activation='relu', dropout=0.2, d_keys=None, d_values=None):
        super(TransformerModule, self).__init__()
        self.output_attention = output_attention
        self.dropout = nn.Dropout(p=dropout)

        self.attention = FullAttention(d_model, n_heads, d_keys, d_values,  # multi-heads settings
                                       mask_flag, factor, dropout, output_attention)  # attention settings

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,))
        self.activation = activation_helper(activation)

    def forward(self, queries, keys, values, attn_mask):
        x, attn = self.attention(queries, keys, values, attn_mask)
        x = x + self.dropout(x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class MLPModule(nn.Module):
    def __init__(self, c_in, len_in, c_out, len_out, hidden, activation='relu', dropout=0.5):
        """
        Conv1d, kernel size is adaptive to hidden length and (len_in, len_out)
        :param c_in: input channel number, for example equal to series number
        :param len_in: input series length,
        :param c_out: output channel number
        :param len_out: output vector dimension
        :param hidden: hidden layer parameters, the number of neural cells
        :param activation: activation function
        """
        super(MLPModule, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.len_in = len_in
        self.len_out = len_out
        self.activation = activation_helper(activation)
        self.dropout = nn.Dropout(p=dropout)

        layer_num = len(hidden) + 1
        kernel_size = ((len_in - len_out) // (layer_num - 1) + 1,)
        module = []
        for d_in, d_out in zip([c_in] + hidden, hidden + [c_out]):
            if d_out != c_out:
                layer = nn.Conv1d(d_in, d_out, kernel_size)
            else:
                kernel_size = (len_in - len_out - len(hidden) * (kernel_size[0] - 1) + 1, )
                layer = nn.Conv1d(d_in, d_out, kernel_size)
            module.append(layer)
        self.layers = nn.ModuleList(module)

    def forward(self, x):
        """
        (batch x length x channel) -> (batch x length x channel)
        :param x: input (batch x length x channel)
        :return:
        """
        x = x.transpose(2, 1)
        for i, fc in enumerate(self.layers):
            if i != 0 and i != len(self.layers) - 1:
                x = self.activation(x)
            x = fc(x)
        # x = self.dropout(x)

        return x.transpose(2, 1).contiguous()


class CNN2dModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3, h_model=20, w_model=20,
                 activation='relu', dropout=0.5, init_weight=True):
        super(CNN2dModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.h_model = h_model
        self.w_model = w_model

        ks1h = (h_model - 1) // (2 * (num_layers - 1)) + 1
        ks1w = (w_model - 1) // (2 * (num_layers - 1)) + 1
        ks2h = h_model - (ks1h - 1) * 2 * (num_layers - 1)
        ks2w = w_model - (ks1w - 1) * 2 * (num_layers - 1)
        self.kernel_size1 = (ks1h, ks1w)
        self.kernel_size2 = (ks2h, ks2w)

        self.activation = activation_helper(activation)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.BatchNorm2d(out_channels)
        self.maxPool = nn.MaxPool2d(self.kernel_size1, stride=(1, 1))

        module = []
        for i in range(num_layers):
            if i == 0:
                module.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, self.kernel_size1),
                        nn.BatchNorm2d(out_channels),
                        activation_helper(activation),
                        nn.MaxPool2d(self.kernel_size1, stride=(1, 1))
                    )
                )
            elif i == num_layers - 1:
                module.append(
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, self.kernel_size2),
                        nn.BatchNorm2d(out_channels),
                    )
                )
            else:
                module.append(
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, self.kernel_size1),
                        nn.BatchNorm2d(out_channels),
                        activation_helper(activation),
                        nn.MaxPool2d(self.kernel_size1, stride=(1, 1))
                    )
                )
        self.cnn = nn.ModuleList(module)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.cnn[i](x)
        x = self.dropout(x)
        return x

