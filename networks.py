import torch
import torch.nn as nn


def init_weight(layer):
    if type(layer) == nn.Linear:
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.Hardswish(),
                                 nn.Linear(hidden_size, action_size))
        self.net.apply(init_weight)

    def forward(self, state):
        return self.net(state)
