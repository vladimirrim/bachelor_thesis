from collections import defaultdict
import numpy as np
import torch

from forge.engine import Lawmaker, LawmakerAbstract
from forge.engine.ann import ANN
from forge.ethyr.rollouts import Rollout
from forge.ethyr.torch.replay import ReplayMemory, ReplayMemoryLm
from forge.ethyr.torch.forward import Forward, ForwardLm


class Sword:
    def __init__(self, config, args, idx):
        self.config, self.args = config, args
        self.nANN, self.h = config.NPOP, config.HIDDEN
        self.anns = [ANN(config, args) for _ in range(self.nANN)]

        self.init, self.nRollouts = True, 32
        self.updates = defaultdict(lambda: Rollout(config))
        self.blobs = []
        self.idx = idx
        self.ReplayMemory = [ReplayMemory(self.config) for _ in range(self.nANN)]
        self.ReplayMemoryLm = ReplayMemoryLm(self.config)
        self.forward, self.forward_lm = Forward(config), ForwardLm(config)

        self.lawmaker = Lawmaker(args, config) if args.lm else LawmakerAbstract(args, config)
        self.tick = 0

    def sendBufferUpdate(self):
        buffer = [replay.send_buffer() for replay in self.ReplayMemory]
        priorities = [self.forward.get_priorities_from_samples(buf, ann, ann, self.lawmaker) for buf, ann in
                      zip(buffer, self.anns)] if self.config.REPLAY_PRIO else [None] * len(self.anns)
        if self.args.lm:
            bufferLm = self.ReplayMemoryLm.send_buffer()
            prioritiesLm = self.forward_lm.get_priorities_from_samples(bufferLm, self.anns, self.lawmaker) if \
                self.config.REPLAY_PRIO else None
            return (buffer, priorities), (bufferLm, prioritiesLm)
        else:
            return (buffer, priorities), None

    def sendLogUpdate(self):
        blobs = self.blobs
        self.blobs = []
        return blobs

    def sendUpdate(self):
        recvs, recvs_lm = self.sendBufferUpdate()
        logs = self.sendLogUpdate()
        return recvs, recvs_lm, logs

    def recvUpdate(self, update):
        update, update_lm = update
        if update is not None:
            self.loadAnnsFrom(update)
        if self.args.lm and (update_lm is not None):
            self.loadLmFrom(update_lm)

    def collectStep(self, entID, annID, s, atnArgs, reward, dead, val):
        if self.config.TEST:
            return
        actions = {key: val[1] for key, val in atnArgs.items()}
        self.ReplayMemory[annID].append(entID, s, actions, reward, dead, val)
        if self.args.lm:
            self.ReplayMemoryLm.append(entID, annID, s, actions, reward, dead, val)

    def collectRollout(self, entID):
        rollout = self.updates[entID]
        rollout.feather.blob.tick = self.tick
        rollout.finish()
        self.blobs.append(rollout.feather.blob)
        del self.updates[entID]

    def decide(self, entID, annID, stim, reward, reward_stats, apples, isDead):
        stim_tensor = self.prepareInput(stim)
        outsLm, punishLm = self.lawmaker(stim_tensor)
        atnArgs, val = self.anns[annID](stim_tensor, self.config.EPS_CUR, punishLm)
        action = int(atnArgs['action'][1])

        self.collectStep(entID, annID, stim_tensor, atnArgs, reward, isDead, val.detach().mean(2))
        if not self.config.TEST:
            self.updates[entID].feather.scrawl(
                annID, val.detach().mean(2), reward_stats, apples,
                outsLm['action'][:, action].mean().detach().numpy() - outsLm['action'].mean().detach().numpy())
        return action

    def prepareInput(self, stim):
        stim = np.transpose(stim, (2, 0, 1)).copy()
        stim_tensor = torch.from_numpy(stim).unsqueeze(0).float()
        return stim_tensor

    def reset_noise(self):
        nets = self.anns + [self.lawmaker]
        for net in nets:
            net.reset_noise()

    def loadAnnsFrom(self, states):
        [ann.load_state_dict(state) for ann, state in zip(self.anns, states)]

    def loadLmFrom(self, state):
        self.lawmaker.load_state_dict(state)
