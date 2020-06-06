from torch import nn
from torch.nn import functional as F
from forge.engine.utils import ConstDiscrete, Env


class ActionNet(nn.Module):
    def __init__(self, config, args, entDim=11, outDim=2, device='cpu', batch_size=1):
        super().__init__()
        self.config, self.args, self.h = config, args, config.HIDDEN
        self.entDim, self.outDim = entDim, outDim
        self.fc = nn.Linear(self.h, self.h)
        self.actionNet = ConstDiscrete(config, self.h, self.outDim)
        self.envNet = Env(config, config.LSTM, device=device, batch_size=batch_size)

    def forward(self, s, outLm, done=False):
        s = self.envNet(s, is_done=done)
        x = F.relu(self.fc(s))
        pi, actionIdx = self.actionNet(x, outLm)
        return pi, actionIdx

    def reset_noise(self):
        self.envNet.reset_noise()
        self.fc.reset_noise()
