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


def orthogonal_init(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def initialize_weights(mod, scale=1.0):
    for p in mod.parameters():
        if len(p.data.shape) >= 2:
            orthogonal_init(p.data, gain=scale)
        else:
            p.data.zero_()


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
        initialize_weights(self.lawmaker)
        if self.args.lm:
            self.lmOpt = Adam(self.lawmaker.parameters(), lr=self.config.LR, weight_decay=0.00001)
            self.lmScheduler = StepLR(self.lmOpt, 1, gamma=0.9998)

    def unshared(self, n):
        self.anns = [
            ANN(self.config, self.args, device=self.config.DEVICE_OPTIMIZER, batch_size=self.config.LSTM_PERIOD).to(
                self.config.DEVICE_OPTIMIZER)
            for _ in range(n)]
        [initialize_weights(ann) for ann in self.anns]

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
        self.saver.load(self.opt, self.anns, [self.lmOpt], [self.lawmaker], best, self.args.lm)

    def model(self):
        return [getParameters(ann) for ann in self.anns], [getParameters(self.lawmaker)]


class Pantheon:
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.net = Model(config, args)
        self.quill = Quill(config.MODELDIR)

    def gatherTrajectory(self, flat_states, ents_states, returns, policy, actions, lmActions, lmPolicy, annID):
        trajectory = {'vals': [],
                      'returns': [],
                      'lmRewards': defaultdict(list),
                      'correction': defaultdict(list),
                      'oldPolicy': defaultdict(list),
                      'policy': defaultdict(list),
                      'actions': defaultdict(list)}

        trajectoryLm = defaultdict(lambda: defaultdict(list))

        for i in range(0, len(flat_states), self.config.LSTM_PERIOD):
            flat = torch.from_numpy(flat_states[i: i + self.config.LSTM_PERIOD]).float().to(
                self.config.DEVICE_OPTIMIZER)
            ents = torch.from_numpy(ents_states[i: i + self.config.LSTM_PERIOD]).float().to(
                self.config.DEVICE_OPTIMIZER)
            ret = torch.from_numpy(returns[i: i + self.config.LSTM_PERIOD]).float().to(self.config.DEVICE_OPTIMIZER)
            oldPolicy = {
                k: torch.from_numpy(v[i: i + self.config.LSTM_PERIOD]).float().to(self.config.DEVICE_OPTIMIZER)
                    .squeeze(1)
                for k, v in policy.items()}
            lmOldPolicy = {
                k: torch.from_numpy(v[i: i + self.config.LSTM_PERIOD]).float().to(self.config.DEVICE_OPTIMIZER)
                    .squeeze(1)
                for k, v in lmPolicy.items()}
            lmAction = {
                k: torch.from_numpy(v[i: i + self.config.LSTM_PERIOD]).float().to(self.config.DEVICE_OPTIMIZER)
                    .squeeze(1)
                for k, v in lmActions.items()}
            action = {k: torch.from_numpy(v[i: i + self.config.LSTM_PERIOD]).long().to(self.config.DEVICE_OPTIMIZER)
                .view(-1, 1)
                      for k, v in actions.items()}
            oldJointPolicy = {k: 1e-5 + F.softmax((1 - self.config.LM_LAMBDA) * oldPolicy[k]
                                                  + self.config.LM_LAMBDA * lmAction[k], dim=1).gather(1, action[k])
                              for k in oldPolicy.keys()}
            outsLm = self.net.lawmaker(flat, ents, annID)
            annReturns = self.net.anns[annID](flat, ents, {'actions': lmAction})

            outsLm = self.net.lawmaker.get_punishment({'actions': lmAction, 'policy': outsLm['policy'],
                                                       'entropy': outsLm['entropy'], 'Qs': outsLm['Qs']}, action)
            if self.args.lm:
                entropy, pi, Qs = outsLm['entropy'], outsLm['policy'], outsLm['Qs']
                for k in pi.keys():
                    trajectoryLm['Qs'][k].append(Qs[k])
                    trajectoryLm['policy'][k].append(pi[k])
                    trajectoryLm['oldPolicy'][k].append(lmOldPolicy[k])
                    trajectoryLm['entropy'][k].append(entropy[k])
                    corr = (lmOldPolicy[k] / oldJointPolicy[k]).clamp(0.5, 2)
                    trajectoryLm['correction'][k].append(corr)

            trajectory['vals'].append(annReturns['val'])
            trajectory['returns'].append(ret)
            for k in oldPolicy.keys():
                trajectory['oldPolicy'][k].append(oldPolicy[k])
                trajectory['policy'][k].append(annReturns['policy'][k])
                trajectory['actions'][k].append(action[k])
                corr = (F.softmax(oldPolicy[k], dim=1).gather(1, action[k]).detach()
                        / oldJointPolicy[k]).clamp(0.5, 2)
                trajectory['correction'][k].append(corr)

        return trajectory, trajectoryLm

    def offPolicyTrain(self, batch):
        step = 500
        for i in range(0, len(batch[0]['flat']), step):
            trajectories = []
            trajectoriesLm = []
            start = i
            for annID, agentBatch in batch.items():
                trajectory, trajectoryLm = self.gatherTrajectory(
                    agentBatch['flat'][start:i + step],
                    agentBatch['ents'][start:i + step],
                    agentBatch['return'][start:i + step],
                    {k: v[start:i + step] for k, v in agentBatch['policy'].items()},
                    {k: v[start:i + step] for k, v in agentBatch['action'].items()},
                    {k: v[start:i + step] for k, v in agentBatch['lmAction'].items()},
                    {k: v[start:i + step] for k, v in agentBatch['lmPolicy'].items()},
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
        lifetime = self.quill.scrawl(logs)

        for i in range(self.config.EPOCHS_PPO):
            self.offPolicyTrain(batch)

        self.net.checkpoint(lifetime)
        self.net.saver.print()

        return self.model()
