from torch import nn

from forge.engine.actions import ActionNet
from forge.engine.utils import ValNet


class ANN(nn.Module):
    def __init__(self, config, args, device='cpu', batch_size=1):
        super().__init__()
        self.valNet = ValNet(config, device=device, batch_size=batch_size)

        self.device = device
        self.config, self.args = config, args
        self.actionNets = nn.ModuleDict()
        self.actionNets['move'] = ActionNet(config, args, entDim=config.ENT_DIM, outDim=10,
                                            device=device, batch_size=batch_size)

    def forward(self, flat, ents, outsLm, done=None):
        val = self.valNet(flat, ents, done)

        actions, policy = dict(), dict()
        for name in list(self.actionNets.keys()):
            actionNet = self.actionNets[name]
            punish_feat = self.config.LM_LAMBDA * outsLm['actions'][name].detach().float()
            pi, actionIdx = actionNet(flat, ents, punish_feat, done)
            actions[name] = actionIdx
            policy[name] = pi

        return {'actions': actions,
                'policy': policy,
                'val': val}
