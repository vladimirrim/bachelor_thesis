import torch
from torch import nn
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn import functional as F


def classify(logits, outLm):
    if len(logits.shape) == 1:
        logits = logits.view(1, -1)
    distribution = Categorical(F.softmax(logits + outLm, dim=1))
    atn = distribution.sample()
    return atn


####### Network Modules
class ConstDiscrete(nn.Module):
    def __init__(self, config, h, nattn):
        super().__init__()
        self.fc1 = nn.Linear(h, nattn)
        self.config = config

    def forward(self, stim, outLm):
        x = self.fc1(stim)
        xIdx = classify((1 - self.config.LM_LAMBDA) * x, outLm * self.config.LM_LAMBDA)
        return x, xIdx


####### End network modules

class ValNet(nn.Module):
    def __init__(self, config, device='cpu', batch_size=1, isLm=False):
        super().__init__()
        self.fc = nn.Linear(config.HIDDEN, 1)
        self.envNet = Env(config, isLstm=True, isLm=isLm, device=device, batch_size=batch_size)

    def forward(self, s, isDead, agentID=None):
        stim = self.envNet(s, isDead, agentID)
        x = self.fc(stim)
        x = x.view(-1, 1)
        return x


class QNet(nn.Module):
    def __init__(self, config, action_size, device='cpu', batch_size=1):
        super().__init__()
        self.fc = torch.nn.Linear(config.HIDDEN, action_size)
        self.envNet = Env(config, isLstm=True, device=device, batch_size=batch_size)

    def forward(self, s, isDead, agentID):
        stim = self.envNet(s, isDead, agentID)
        x = self.fc(stim)
        return x


class Env(nn.Module):
    def __init__(self, config, isLstm=False, isLm=False, device='cpu', batch_size=1, in_dim=1014):
        super().__init__()
        h = config.HIDDEN
        self.config = config
        self.h = h
        self.batch_size = batch_size

        self.lstm = nn.LSTM(h, h, batch_first=True).to(device) if isLstm else None
        self.isLstm = isLstm
        self.isLm = isLm
        self.device = device
        if isLstm:
            if self.isLm:
                self.hiddens = [self.init_hidden(self.batch_size, h) for _ in range(config.NPOP)]
            else:
                self.hidden = self.init_hidden(self.batch_size, h)
        self.fc1 = nn.Linear(in_dim, h)

    def init_hidden(self, batch_size, h):
        hidden = Variable(next(self.parameters()).data.new(1, batch_size, h), requires_grad=False)
        cell = Variable(next(self.parameters()).data.new(1, batch_size, h), requires_grad=False)
        return hidden.zero_().to(self.device), cell.zero_().to(self.device)

    def forward(self, flat, is_done=False, agent_id=None):
        x = F.relu(self.fc1(flat))
        if self.isLstm:
            if self.batch_size != 1:
                x, _ = self.lstm(x.unsqueeze(0))
            else:
                x, hidden = self.lstm(x.unsqueeze(0), self.hidden) if not self.isLm else \
                            self.lstm(x.unsqueeze(0), self.hiddens[agent_id])
                if is_done:
                    if self.isLm:
                        self.hiddens[agent_id] = (self.init_hidden(1, self.h))
                    else:
                        self.hidden = self.init_hidden(1, self.h)
                else:
                    if self.isLm:
                        self.hiddens[agent_id] = hidden
                    else:
                        self.hidden = hidden
            x = F.relu(x.squeeze(0))
        return x

    def reset_noise(self):
        pass
