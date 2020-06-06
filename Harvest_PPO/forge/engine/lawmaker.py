import torch
from torch import nn
from torch.nn import functional as F

from forge.engine.utils import Env, QNet


class LawmakerAbstract(nn.Module):
    def __init__(self, args, config, device='cpu', batch_size=1):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.config, self.args = config, args
        self.nRealm = args.nRealm

    def forward(self, env, isDead=False, agentID=None):
        dct = {'action': (0, torch.zeros([self.batch_size, 8]).to(self.device),
                          torch.zeros([self.batch_size, 1]).to(self.device))}
        return dct

    def get_punishment(self, outputs, outputsAgent, pi):
        return self.forward(None, None)


class Lawmaker(LawmakerAbstract):
    def __init__(self, args, config, device='cpu', batch_size=1):
        super().__init__(args, config)
        self.device = device
        self.actionNets = nn.ModuleDict()
        self.actionNets['action'] = ActionNet(config, outDim=8, device=device, batch_size=batch_size)
        self.conv = nn.Conv2d(3, 6, (3, 3))

    def forward(self, env, isDead=False, agentID=None):
        s = F.relu(self.conv(env.to(self.device)).view(env.shape[0], -1))
        outputs = dict()
        for name in list(self.actionNets.keys()):
            out, actions, entropy, Qs = self.actionNets[name](s, isDead, agentID)
            outputs[name] = (out, actions, entropy, Qs)
        return outputs

    def get_punishment(self, outputs, actionAgent, pi):
        outputs_new, punishments = dict(), dict()
        for name in outputs.keys():
            out, actions, entropy, Qs = outputs[name]
            outCh = out.gather(1, actionAgent.view(-1, 1))
            Q = Qs.gather(1, actionAgent.view(-1, 1))
            outputs_new[name] = (entropy, outCh, Q)
        return outputs_new

    def reset_noise(self):
        for net in self.actionNets.values():
            net.reset_noise()


class ActionNet(nn.Module):
    def __init__(self, config, outDim=2, device='cpu', batch_size=1):
        super().__init__()
        self.config, self.h = config, config.HIDDEN
        self.outDim = outDim
        self.actionNet = nn.Linear(self.h, self.outDim)
        self.envNet = Env(config, self.config.LSTM, True, device=device, batch_size=batch_size)
        self.fc = nn.Linear(self.h, self.h)
        self.qNet = QNet(config, outDim, device, batch_size)

    def forward(self, s, isDead=False, agentID=None):
        stim = self.envNet(s, isDead, agentID)
        Qs = self.qNet(s, isDead, agentID)

        x = F.relu(self.fc(stim))
        logits = self.actionNet(x)
        pi = F.softmax(logits, dim=1)
        entropy = (torch.log(pi + 1e-7) * pi).sum(dim=1)

        return pi, logits.detach(), entropy, Qs

