import torch
from torch import nn
from torch.nn import functional as F
from forge.engine.utils import ConstDiscrete, Env
from forge.engine.noisy_layers import NoisyLinear


class ActionNet(nn.Module):
    def __init__(self, config, entDim=19, outDim=2, n_quant=1, add_features=0, noise=False):
        super().__init__()
        self.config, self.h = config, config.HIDDEN
        self.entDim = entDim
        self.envNet = Env(config, noise=noise)
        self.fc = NoisyLinear(self.h+add_features, self.h) if noise else nn.Linear(self.h+add_features, self.h)
        self.actionNet = ConstDiscrete(config, self.h, outDim, n_quant)

    def forward(self, flat, ents, eps=0, punish=None, val=None, device='cpu'):
        stim = self.envNet(flat.to(device), ents.to(device), device=device)
        if self.config.VAL_FEATURE and val is not None:
            stim = torch.cat([stim.to(device), val.to(device)], dim=1)
        if self.config.NOISE:
            x = F.relu(self.fc(stim.to(device), device=device))
        else:
            x = F.relu(self.fc(stim.to(device)))
        outs, idx = self.actionNet(x.to(device), eps, punish)
        return outs, idx

    def reset_noise(self):
        self.envNet.reset_noise()
        self.fc.reset_noise()
