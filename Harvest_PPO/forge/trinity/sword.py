from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch

from forge.engine import Lawmaker, LawmakerAbstract
from forge.engine.ann import ANN
from forge.ethyr.rollouts import Rollout
from forge.ethyr.torch.param import setParameters, zeroGrads


class Sword:
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.nANN = config.NPOP
        self.anns = [ANN(config, args) for _ in range(self.nANN)]

        self.updates, self.rollouts = defaultdict(lambda: Rollout()), {}
        self.updates_lm, self.rollouts_lm = defaultdict(lambda: Rollout()), {}
        self.initBuffer()

        self.lawmaker = Lawmaker(args, config) if args.lm else LawmakerAbstract(args, config)

    def backward(self):
        self.rollouts_lm = {}
        self.blobs = [r.feather.blob for r in self.rollouts.values()]
        self.rollouts = {}

    def sendLogUpdate(self):
        blobs = self.blobs
        self.blobs = []
        return blobs

    def recvUpdate(self, update):
        update, update_lm = update
        for idx, paramVec in enumerate(update):
            setParameters(self.anns[idx], paramVec)
            zeroGrads(self.anns[idx])

        setParameters(self.lawmaker, update_lm[0])
        zeroGrads(self.lawmaker)

    def collectStep(self, entID, atnArgs, val, reward, stim=None):
        if self.config.TEST:
            return
        self.updates[entID].step(atnArgs, val, reward, stim)

    def collectRollout(self, entID, ent, tick, epoch):
        assert entID not in self.rollouts
        rollout = self.updates[entID]
        rollout.finish()
        rollout.feather.blob.tick = tick
        annID = ent.annID
        self.buffer[annID]['return'][(epoch + 1) * self.config.HORIZON - 1] = self.buffer[annID]['reward'][
            (epoch + 1) * self.config.HORIZON - 1]
        for i in reversed(range(epoch * self.config.HORIZON, (epoch + 1) * self.config.HORIZON - 1)):
            self.buffer[annID]['return'][i] = self.buffer[annID]['reward'][i] + 0.99 * self.buffer[annID]['return'][
                i + 1]
        self.rollouts[entID] = rollout
        del self.updates[entID]

    def initBuffer(self):
        batchSize = self.config.HORIZON * self.config.EPOCHS
        self.buffer = defaultdict(lambda: {'state': np.ndarray((batchSize, 3, 15, 15), dtype=int),
                                           'policy': np.ndarray((batchSize, 8), dtype=float),
                                           'lmPolicy': np.ndarray((batchSize, 8), dtype=float),
                                           'action': np.ndarray((batchSize,), dtype=int),
                                           'reward': np.ndarray((batchSize,), dtype=int),
                                           'return': np.ndarray((batchSize,), dtype=float),
                                           'lmAction': np.ndarray((batchSize, 8), dtype=float)})

    def dispatchBuffer(self):
        buffer = deepcopy(self.buffer)
        self.initBuffer()
        return buffer

    def decide(self, ent, stim, reward, isDead, step, epoch):
        entID, annID = ent.agent_id + str(epoch), ent.annID

        stim = np.transpose(stim, (2, 0, 1)).copy()
        stim_tensor = torch.from_numpy(stim).unsqueeze(0).float()
        outsLm = self.lawmaker(stim_tensor, isDead, annID)
        annReturns = self.anns[annID](stim_tensor, outsLm, isDead)

        self.buffer[annID]['state'][step] = stim
        self.buffer[annID]['policy'][step] = annReturns['outputs']['action'][0].detach().numpy()
        self.buffer[annID]['action'][step] = annReturns['outputs']['action'][1]
        self.buffer[annID]['lmAction'][step] = outsLm['action'][1].detach().numpy()
        action = int(annReturns['outputs']['action'][1])

        Asw = -outsLm['action'][-1].mean()
        outsLm = self.lawmaker.get_punishment(outsLm, torch.tensor(action),
                                                          annReturns['outputs']['action'][0].detach())
        self.buffer[annID]['lmPolicy'][step] = outsLm['action'][1].detach().numpy()
        Asw += float(outsLm['action'][-1])

        self.collectStep(entID, annReturns['outputs'], annReturns['val'], reward, stim)
        if not self.config.TEST:
            self.updates[entID].feather.scrawl(np.max(self.buffer[annID]['lmAction'][step]), ent,
                                               np.max(self.buffer[annID]['policy'][step]),
                                               reward, float(Asw))
        return action
