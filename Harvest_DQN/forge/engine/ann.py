import numpy as np
from torch import nn
from torch.nn import functional as F

from forge.blade import entity
from forge.blade.lib import enums
from forge.blade.lib.enums import Neon
from forge.engine.actions import ActionNet
from forge.engine.utils import ValNet, Env
from forge.ethyr import torch as torchlib


class ANN(nn.Module):
    def __init__(self, config, args):
        super().__init__()

        self.config, self.args = config, args
        self.valNet = ValNet(config, n_quant=config.N_QUANT, noise=self.config.NOISE)
        self.actionNets = nn.ModuleDict()
        self.actionNets['action'] = ActionNet(config, args,  outDim=8, n_quant=config.N_QUANT,
                                              add_features=int(self.config.VAL_FEATURE), noise=self.config.NOISE)
        self.envNet = Env(config, self.config.NOISE)

    def forward(self, env, eps=0, punishmentsLm=None, v=None):
        s = self.envNet(env)
        val = self.valNet(s)
        v = val.detach().mean(2) if v is None else v

        outputs = {}
        for name in self.actionNets.keys():
            punish = punishmentsLm[name] if punishmentsLm is not None else None
            pi, actionIdx = self.actionNets[name](s, eps, punish, v)
            outputs[name] = (pi.to('cpu'), actionIdx)

        return outputs, val.to('cpu')

    def reset_noise(self):
        self.envNet.reset_noise()
        self.valNet.reset_noise()
        for _, net in self.actionNets.items():
            net.reset_noise()
