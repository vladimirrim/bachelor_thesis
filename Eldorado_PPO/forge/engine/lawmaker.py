import torch
from torch import nn
from torch.nn import functional as F
from forge.engine.utils import Env, QNet


class LawmakerAbstract(nn.Module):
    def __init__(self, args, config, device='cpu', batch_size=1):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.actions = ['move']
        self.dims = [1]

    def forward(self, flat, ents, isDead=False, agentID=None):
        dct = {'Qs':  {k: torch.zeros([self.batch_size, v]).to(self.device) for k, v in zip(self.actions, self.dims)},
               'actions': {k: torch.zeros([self.batch_size, v]).to(self.device) for k, v in zip(self.actions, self.dims)},
               'policy': {k: torch.zeros([self.batch_size, v]).to(self.device) for k, v in zip(self.actions, self.dims)},
               'entropy': {k: torch.zeros([self.batch_size, v]).to(self.device) for k, v in zip(self.actions, self.dims)}}
        return dct

    def reset_noise(self):
        pass

    def get_punishment(self, outputs, outputsAgent):
        return self.forward(None, None)


class Lawmaker(LawmakerAbstract):
    def __init__(self, args, config, device='cpu', batch_size=1):
        super().__init__(args, config)
        self.device = device
        self.actionNets = nn.ModuleDict()
        self.actionNets['move'] = ActionNet(config, entDim=config.ENT_DIM, outDim=10,
                                            device=device, batch_size=batch_size)

    def forward(self, flat, ents, isDead=False, agentID=None):
        actionsDict = dict()
        policy = dict()
        Qs = dict()
        entropyDict = dict()
        for name in list(self.actionNets.keys()):
            out, actions, entropy, Q = self.actionNets[name](flat, ents, isDead, agentID)
            policy[name] = out
            Qs[name] = Q
            actionsDict[name] = actions
            entropyDict[name] = entropy
        return {'actions': actionsDict,
                'policy': policy,
                'entropy': entropyDict,
                'Qs': Qs}

    def get_punishment(self, outputs, actionsAgent):
        policy_new, entropy_new, actions_new, Q_new = dict(), dict(), dict(), dict()
        for name in actionsAgent.keys():
            actionAgent = actionsAgent[name]
            out = outputs['policy'][name]
            outCh = out.gather(1, actionAgent.view(-1, 1))
            policy_new[name] = outCh
            Q_new[name] = outputs['Qs'][name].gather(1, actionAgent.view(-1, 1))
            entropy_new[name] = outputs['entropy'][name]
            actions_new[name] = outputs['actions'][name]
        return {'entropy': entropy_new, 'policy': policy_new, 'actions': actions_new, 'Qs': Q_new}


class ActionNet(nn.Module):
    def __init__(self, config, entDim=11, outDim=2, device='cpu', batch_size=1):
        super().__init__()
        self.config, self.h = config, config.HIDDEN
        self.entDim, self.outDim = entDim, outDim
        self.actionNet = nn.Linear(self.h, self.outDim)
        self.envNet = Env(config, self.config.LSTM, True, device=device, batch_size=batch_size)
        self.fc = nn.Linear(self.h, self.h)
        self.qNet = QNet(config, outDim, device=device, batch_size=batch_size)

    def forward(self, flat, ents, isDead=False, agentID=None):
        stim = self.envNet(flat, ents, isDead, agentID)
        Qs = self.qNet(flat, ents, isDead, agentID)

        x = F.relu(self.fc(stim))
        logits = self.actionNet(x)
        pi = F.softmax(logits, dim=1)
        entropy = (torch.log(pi + 1e-5) * pi).sum(dim=1)

        return pi, logits.detach(), entropy, Qs

