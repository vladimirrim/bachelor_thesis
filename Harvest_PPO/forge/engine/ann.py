from torch import nn
from torch.nn import functional as F

from forge.engine.actions import ActionNet
from forge.engine.utils import ValNet


class ANN(nn.Module):
    def __init__(self, config, args, device='cpu', batch_size=1):
        super().__init__()
        self.intValNet = ValNet(config, device=device, batch_size=batch_size)

        self.device = device
        self.config, self.args = config, args
        self.actionNets = nn.ModuleDict()
        self.actionNets['action'] = ActionNet(config, args, outDim=8, device=device, batch_size=batch_size)
        self.conv = nn.Conv2d(3, 6, (3, 3))

    def forward(self, env, outsLm, isDead=None):
        s = F.relu(self.conv(env.to(self.device)).view(env.shape[0], -1))
        intVal = self.intValNet(s, isDead)

        playerActions = []
        actionTargets = []
        outputs = {}
        actionDecisions = {}
        for name in self.actionNets.keys():
            actionNet = self.actionNets[name]
            punish_feat = outsLm[name][1].detach().float()
            pi, actionIdx = actionNet(s, punish_feat, isDead)
            outputs[name] = (pi, actionIdx)
            
        return {
            'playerActions': playerActions,
            'actionTargets': actionTargets,
            'outputs': outputs,
            'val': intVal,
            'actionDecisions': actionDecisions,
        }
