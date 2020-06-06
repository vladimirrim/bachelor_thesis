import torch
from torch import nn
from torch.nn import functional as F

from forge.engine.actions import ActionNet
from forge.engine.utils import Env
from forge.ethyr import torch as torchlib
from forge.engine.utils import classify


class LawmakerAbstract(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.config, self.args = config, args
        self.nRealm = args.nRealm

    def forward(self, flat=None, ents=None, device=None):
        s = flat.size()[0]
        dct = {'move': torch.zeros([s, 5])}
        if self.config.ATTACK:
            dct['attack'] = torch.zeros([s, 2])
        if self.config.SHARE:
            dct['shareWater'] = torch.zeros([s, 2])
            dct['shareFood'] = torch.zeros([s, 2])
        return dct, dct

    def reset_noise(self):
        pass


class Lawmaker(LawmakerAbstract):
    def __init__(self, args, config):
        super().__init__(args, config)

        self.actionNets = nn.ModuleDict()
        self.actionNets['move'] = ActionNet(config, entDim=config.ENT_DIM, outDim=5, n_quant=config.N_QUANT_LM,
                                            noise=self.config.NOISE)
        if self.config.ATTACK:
            self.actionNets['attack'] = ActionNet(config, entDim=config.ENT_DIM, outDim=2, n_quant=config.N_QUANT_LM,
                                                  noise=self.config.NOISE)
        if self.config.SHARE:
            self.actionNets['shareWater'] = ActionNet(config, entDim=config.ENT_DIM, outDim=2, n_quant=config.N_QUANT_LM,
                                                      noise=self.config.NOISE)
            self.actionNets['shareFood'] = ActionNet(config, entDim=config.ENT_DIM, outDim=2, n_quant=config.N_QUANT_LM,
                                                     noise=self.config.NOISE)

    def forward(self, flat, ents, device='cpu'):
        outputs, punishments = dict(), dict()
        for name in self.actionNets.keys():
            outputs[name] = self.actionNets[name](flat, ents, 0, device=device)[0].to('cpu')
            punishments[name] = outputs[name].mean(2).detach() * self.config.PUNISHMENT
        return outputs, punishments

    def reset_noise(self):
        for _, net in self.actionNets.items():
            net.reset_noise()
