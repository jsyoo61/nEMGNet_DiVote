import torch
import torch.nn as nn
import itertools as it

import tools as T

class ConvBlock(nn.Module):
    def __init__(self, in_channel=1, out_channel=64, kernel_size=3, stride=1):
        super().__init__()
        layers=[
        nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=1),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ResBlock(nn.Module):
    def __init__(self, in_channel, kernel_size=5):
        super().__init__()
        layers=[
        nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size, stride=1, padding='same'),
        nn.BatchNorm1d(in_channel),
        nn.ReLU(),
        nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size, stride=1, padding='same'),
        nn.BatchNorm1d(in_channel),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def residual_blocks(in_channel, n_repeat):
    return nn.Sequential(*[ResBlock(in_channel) for _ in range(n_repeat)])

class nEMGNet(nn.Module):
    def __init__(self, n_classes, n_channel_list = [64, 128, 256, 512, 1024], n_repeat=2):
        super().__init__()
        _n_channel_list = [1] + n_channel_list

        cnn = nn.Sequential(
            ConvBlock(in_channel=_n_channel_list[0], out_channel=_n_channel_list[1], kernel_size=11, stride=2),
            ConvBlock(in_channel=_n_channel_list[1], out_channel=_n_channel_list[1], kernel_size=7, stride=2),
            ConvBlock(in_channel=_n_channel_list[1], out_channel=_n_channel_list[1], kernel_size=5, stride=2),
            T.torch.model.Residual(blocks=residual_blocks(_n_channel_list[1], n_repeat)),
            *it.chain(*[
            [ConvBlock(in_channel=c1, out_channel=c2, kernel_size=3), T.torch.model.Residual(blocks=residual_blocks(c2, n_repeat))]
            for c1, c2 in zip(_n_channel_list[1:-1], _n_channel_list[2:])
            ])
            )

        layers = [
            cnn,
            nn.Flatten(1),
            T.torch.model.FNN(n_hidden_list=[512,256,64,16,n_classes], activation_list=[nn.LeakyReLU()]*4+[nn.Identity()])
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
