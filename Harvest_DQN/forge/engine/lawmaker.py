import torch
from torch import nn

from forge.engine.actions import ActionNet
from forge.engine.utils import Env


class LawmakerAbstract(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.config, self.args = config, args
        self.nRealm = args.nRealm

    def forward(self, env):
        dct = {'action': torch.zeros([env.size(0), 8])}
        return dct, dct

    def reset_noise(self):
        pass


class Lawmaker(LawmakerAbstract):
    def __init__(self, args, config):
        super().__init__(args, config)

        self.actionNets = nn.ModuleDict()
        self.actionNets['action'] = ActionNet(config, args, outDim=8, n_quant=config.N_QUANT_LM, noise=self.config.NOISE)
        self.envNet = Env(config, self.config.NOISE)

    def forward(self, env):
        s = self.envNet(env)
        outputs, punishments = dict(), dict()
        for name in self.actionNets.keys():
            outputs[name] = self.actionNets[name](s, 0)[0].to('cpu')
            punishments[name] = outputs[name].mean(2).detach() * self.config.PUNISHMENT
        return outputs, punishments

    def reset_noise(self):
        self.envNet.reset_noise()
        for _, net in self.actionNets.items():
            net.reset_noise()
