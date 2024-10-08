import torch.nn as nn
import torch
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SLFN(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, act_fn='relu'):
        super(SLFN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.act_fn = act_fn

        self.hidden_layer1 = nn.Linear(input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        activation = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh()
        }[self.act_fn]
        x = x.view(-1, self.input_size)
        x = activation(self.hidden_layer1(x))
        x = self.output_layer(x)

        return x


class TLFN(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, act_fn_1='relu', act_fn_2='sigmoid'):
        super(TLFN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.act_fn_1 = act_fn_1
        self.act_fn_2 = act_fn_2

        self.hidden_layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size, bias=False)

    def forward(self, x):
        activation_1 = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh()
        }[self.act_fn_1]
        activation_2 = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh()
        }[self.act_fn_2]
        x = x.view(-1, self.input_size)
        x = activation_1(self.hidden_layer1(x))
        x = activation_2(self.hidden_layer2(x))
        x = self.output_layer(x)

        return x