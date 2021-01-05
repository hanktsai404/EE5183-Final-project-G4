# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:19:50 2020

@author: Austin Hsu
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class RNNModel(nn.Module):
    def __init__(self, cell: str = 'RNN', num_layers: int = 3, hidden_size: int = 32,
                 input_size: int = 10, dropout: float = 0.):
        super(RNNModel, self).__init__()
        self.cell = cell
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = dropout
        
        self.recurrent_cell = getattr(nn, cell)
        self.recurrent_model = self.recurrent_cell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            )
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1)
        
    def forward(self, x):
        if self.cell == 'RNN':
            _, hn = self.recurrent_model(x)
        elif self.cell == 'GRU':
            _, hn = self.recurrent_model(x)
        else:
            _, (hn, _) = self.recurrent_model(x)
        
        hn = hn[-1]
        hn = self.layernorm(hn)
        hn = self.fc(hn)
        
        return hn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_layers, hidden_size, input_size=24, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = dropout

        layers = []
        num_channels = [self.hidden_size for _ in range(self.hidden_size)]
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.layernorm = nn.LayerNorm(num_channels[-1])
        self.fc = nn.Linear(num_channels[-1],1)

    def forward(self, x):
        x = x.transpose(-1,-2)
        x = self.network(x)
        x = x[:,:,-1]
        x = self.layernorm(x)
        x = self.fc(x)
        return x