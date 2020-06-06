import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from forge.engine.noisy_layers import NoisyLinear


def classify_Q(logits, eps=0.1, mode='boltzmann'):
    """
    https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf
    :param eps: epsilon for eps-greedy, temperature for boltzmann, dropout rate for bayes
    :param mode: eps-greedy, boltzmann, or bayes
    """
    if len(logits.shape) == 1:
        logits = logits.view(1, -1)
    if mode.lower().startswith('boltzman'):
        distribution = Categorical(1e-4 + F.softmax(logits/eps, dim=1))
        atn = distribution.sample()
    elif mode.lower().startswith('eps'):
        if np.random.random() > eps:
            atn = torch.argmax(logits).view(1, 1)
        else:
            atn = torch.randint(logits.shape[1], (1,)).view(1, 1)
    elif (mode.lower().startswith('bayes')) or (mode.lower() == 'noise') or (mode.lower().startswith('greed')):
        atn = torch.argmax(logits).view(1, 1)
    return atn


class ConstDiscrete(nn.Module):
    def __init__(self, config, h, n_attn, n_quant):
        super().__init__()
        self.fc1 = nn.Linear(h, n_attn * n_quant)
        self.n_attn = n_attn
        self.n_quant = n_quant
        self.config = config

    def forward(self, stim, eps, punish=None):
        x = self.fc1(stim).view(-1, self.n_attn, self.n_quant)
        if punish is None:
            xIdx = classify_Q(x.mean(2), eps, mode=self.config.EXPLORE_MODE)
        else:
            xIdx = classify_Q(x.mean(2) * (1 - self.config.PUNISHMENT) + punish, eps, mode=self.config.EXPLORE_MODE)
        return x, xIdx


class ValNet(nn.Module):
    def __init__(self, config, n_quant=1, noise=False):
        super().__init__()
        self.config = config
        self.h = config.HIDDEN
        self.fc = NoisyLinear(self.h, self.h) if noise else nn.Linear(self.h, self.h)
        self.valNet = nn.Linear(self.h, n_quant)
        self.n_quant = n_quant

    def forward(self, s):
        x = F.relu(self.fc(s))
        x = self.valNet(x).view(-1, 1, self.n_quant)
        return x

    def reset_noise(self):
        self.fc.reset_noise()


class Env(nn.Module):
    def __init__(self, config, noise=False):
        super().__init__()
        self.config = config
        h = config.HIDDEN

        self.conv = nn.Conv2d(3, 6, (3, 3))
        self.fc = NoisyLinear(self.config.ENT_DIM, h) if noise else nn.Linear(self.config.ENT_DIM, h)

    def forward(self, env):
        x = F.relu(self.conv(env).view(env.shape[0], -1))
        x = F.relu(self.fc(x))
        return x

    def reset_noise(self):
        self.fc.reset_noise()
