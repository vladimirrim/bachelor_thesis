import torch
from torch import nn
from torch.nn import functional as F
from forge.engine.utils import ConstDiscrete, Env
from forge.engine.noisy_layers import NoisyLinear


class ActionNet(nn.Module):
    def __init__(self, config, args, outDim=8, n_quant=1, add_features=0, noise=False):
        super().__init__()
        self.config, self.args, self.h = config, args, config.HIDDEN
        self.fc = NoisyLinear(self.h+add_features, self.h) if noise else nn.Linear(self.h+add_features, self.h)
        self.actionNet = ConstDiscrete(config, self.h, outDim, n_quant)

    def forward(self, s, eps=0, punish=None, val=None):
        if self.config.VAL_FEATURE and val is not None:
            s = torch.cat([s, val], dim=1)
        x = F.relu(self.fc(s))
        outs, idx = self.actionNet(x, eps, punish)
        return outs, idx

    def reset_noise(self):
        self.fc.reset_noise()
