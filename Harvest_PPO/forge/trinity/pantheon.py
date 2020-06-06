from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from forge.blade.lib.log import Quill
from forge.engine import LawmakerAbstract, Lawmaker
from forge.engine.ann import ANN
from forge.ethyr.torch import save, optim
from forge.ethyr.torch.param import getParameters


class Model:
    def __init__(self, config, args):
        self.saver = save.Saver(config.NPOP, config.MODELDIR,
                                'models', 'bests', 'lawmaker', resetTol=256)
        self.config, self.args = config, args
        self.agentEntropies = np.repeat(self.config.ENTROPY, self.args.nRealm)

        self.init()
        if self.config.LOAD or self.config.BEST:
            self.load(self.config.BEST)

    def init(self):
        print('Initializing new model...')
        self.unshared(self.config.NPOP)

        self.opt = None
        if not self.config.TEST:
            self.opt = [Adam(ann.parameters(), lr=self.config.LR, weight_decay=0.00001) for ann in self.anns]
            self.scheduler = [StepLR(opt, 1, gamma=0.9998) for opt in self.opt]

        self.lawmaker = (Lawmaker(self.args, self.config, device=self.config.DEVICE_OPTIMIZER,
                                  batch_size=self.config.LSTM_PERIOD).to(
            self.config.DEVICE_OPTIMIZER) if self.args.lm else
                         LawmakerAbstract(self.args, self.config, device=self.config.DEVICE_OPTIMIZER,
                                          batch_size=self.config.LSTM_PERIOD))
        if self.args.lm:
            self.lmOpt = Adam(self.lawmaker.parameters(), lr=self.config.LR, weight_decay=0.00001)
            self.lmScheduler = StepLR(self.lmOpt, 1, gamma=0.9998)

    # Initialize a new network
    def initModel(self):
        return getParameters(ANN(self.config, self.args))

    def shared(self, n):
        model = self.initModel()
        self.models = [model for _ in range(n)]

    def unshared(self, n):
        self.anns = [ANN(self.config, self.args, device=self.config.DEVICE_OPTIMIZER,
                         batch_size=self.config.LSTM_PERIOD).to(self.config.DEVICE_OPTIMIZER) for _ in range(n)]

    def annealEntropy(self, idx):
        self.agentEntropies[idx] = max(self.agentEntropies[idx] * self.config.ENTROPY_ANNEALING,
                                       self.config.MIN_ENTROPY)

    def checkpoint(self, reward):
        if self.config.TEST:
            return
        self.saver.checkpoint(reward, self.anns, self.opt)
        if self.args.lm:
            self.saver.checkpointLawmaker([self.lawmaker], [self.lmOpt])

    def load(self, best=False):
        print('Loading model...')
        self.saver.load(self.opt, self.anns, self.lmOpt, self.lawmaker, best, self.args.lm)

    @property
    def nParams(self):
        nParams = sum([len(e) for e in self.model])
        print('#Params: ', str(nParams / 1000), 'K')

    def model(self):
        return [getParameters(ann) for ann in self.anns], [getParameters(self.lawmaker)]


class Pantheon:
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.net = Model(config, args)
        self.quill = Quill(config.MODELDIR)

    def gatherTrajectory(self, states, rewards, policy, actions, lmActions, lmPolicy, annID):
        trajectory = defaultdict(list)
        trajectoryLm = defaultdict(list)

        for i in range(0, len(states), self.config.LSTM_PERIOD):
            stim = torch.from_numpy(states[i: i + self.config.LSTM_PERIOD]).float().to(self.config.DEVICE_OPTIMIZER)
            ret = torch.from_numpy(rewards[i: i + self.config.LSTM_PERIOD]).float().to(self.config.DEVICE_OPTIMIZER)
            oldPolicy = torch.from_numpy(policy[i: i + self.config.LSTM_PERIOD]).to(
                self.config.DEVICE_OPTIMIZER).float()
            lmOldPolicy = torch.from_numpy(lmPolicy[i: i + self.config.LSTM_PERIOD]).float().to(
                self.config.DEVICE_OPTIMIZER)
            lmAction = torch.from_numpy(lmActions[i: i + self.config.LSTM_PERIOD]).float().to(
                self.config.DEVICE_OPTIMIZER)
            action = torch.tensor(actions[i: i + self.config.LSTM_PERIOD]).to(self.config.DEVICE_OPTIMIZER)
            oldJointPolicy = F.softmax((1 - self.config.LM_LAMBDA) * oldPolicy + self.config.LM_LAMBDA * lmAction,
                                       dim=1).gather(1, action.view(-1, 1))

            outsLm = self.net.lawmaker(stim, annID)
            annReturns = self.net.anns[annID](stim, outsLm, (i + 1) % self.config.LSTM_PERIOD == 0)

            outsLm = self.net.lawmaker.get_punishment({'action': (outsLm['action'][0], lmAction,
                                                                              outsLm['action'][-2],
                                                                              outsLm['action'][-1])}, action,
                                                                  annReturns['outputs']['action'][0].detach())
            if self.args.lm:
                entropy, pi, Q = outsLm['action']
                trajectoryLm['QVals'].append(Q)
                trajectoryLm['policy'].append(pi)
                trajectoryLm['oldPolicy'].append(lmOldPolicy.gather(1, action.view(-1, 1)))
                trajectoryLm['correction'].append((lmOldPolicy.gather(1, action.view(-1, 1)) / oldJointPolicy).clamp(0.5, 2))
                trajectoryLm['entropy'].append(entropy)

            trajectory['vals'].append(annReturns['val'])
            trajectory['returns'].append(ret)
            trajectory['oldPolicy'].append(F.softmax(oldPolicy, dim=1).gather(1, action.view(-1, 1)))
            trajectory['policy'].append(F.softmax(annReturns['outputs']['action'][0], dim=1).
                                        gather(1, action.view(-1, 1)))
            trajectory['actions'].append(action)
            trajectory['correction'].append((F.softmax(oldPolicy, dim=1).
                                             gather(1, action.view(-1, 1)) / oldJointPolicy).clamp(0.5, 2))

        return trajectory, trajectoryLm

    def offPolicyTrain(self, batch):
        step = 500
        for i in range(0, self.config.HORIZON * self.config.EPOCHS, step):
            trajectories = []
            trajectoriesLm = []
            start = i
            for annID, agentBatch in batch.items():
                trajectory, trajectoryLm = self.gatherTrajectory(
                    agentBatch['state'][start:i + step],
                    agentBatch['reward'][start:i + step],
                    agentBatch['policy'][start:i + step],
                    agentBatch['action'][start:i + step],
                    agentBatch['lmAction'][start:i + step],
                    agentBatch['lmPolicy'][start:i + step],
                    annID)
                trajectoriesLm.append(trajectoryLm)
                trajectories.append(trajectory)
            loss, outs = optim.backwardAgentOffPolicy(trajectories, entWeight=self.net.agentEntropies[0],
                                                      device=self.config.DEVICE_OPTIMIZER)
            if self.args.lm:
                lmLoss, outsLm = optim.backwardLawmaker(trajectoriesLm, outs['vals'], outs['rets'],
                                                        entWeight=self.net.agentEntropies[0],
                                                        device=self.config.DEVICE_OPTIMIZER,
                                                        mode=self.config.LM_MODE)
                lmLoss.backward()
                nn.utils.clip_grad_norm_(self.net.lawmaker.parameters(), 0.5)
                self.net.lmOpt.step()
                self.net.lmScheduler.step()
                self.net.lmOpt.zero_grad()
            loss.backward()
            [nn.utils.clip_grad_norm_(ann.parameters(), 0.5) for ann in self.net.anns]
            [opt.step() for opt in self.net.opt]
            self.net.annealEntropy(0)
            [scheduler.step() for scheduler in self.net.scheduler]
            [opt.zero_grad() for opt in self.net.opt]
        return

    def model(self):
        return self.net.model()

    def step(self, batch, logs):
        # Write logs
        reward = self.quill.scrawl(logs)

        for i in range(self.config.EPOCHS_PPO):
            self.offPolicyTrain(batch)

        self.net.checkpoint(reward)
        self.net.saver.print()

        return self.model()
