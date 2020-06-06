import time

from torch.optim import Adam
from torch.nn.utils import clip_grad_value_

from forge.engine.ann import ANN
from forge.engine import LawmakerAbstract, Lawmaker
from forge.ethyr.torch import save
from forge.ethyr.torch.optim import backwardAgent
from forge.ethyr.torch.forward import Forward


class Model:
    def __init__(self, config, args):
        self.saver = save.Saver(config.NPOP, config.MODELDIR,
                                'models', 'bests', 'lawmaker', resetTol=256)
        self.config, self.args = config, args
        self.nANN = config.NPOP
        self.envNets = []

        self.init()
        if self.config.LOAD or self.config.BEST:
            self.load(self.config.BEST)

    def init(self):
        print('Initializing new model...')
        self.anns = [ANN(self.config, self.args).to(self.config.device) for _ in range(self.nANN)]
        self.targetAnns = [ANN(self.config, self.args).to(self.config.device) for _ in range(self.nANN)]
        self.updateTargetAnns()
        self.annsOpts = None
        if not self.config.TEST:
            self.annsOpts = [Adam(ann.parameters(), lr=0.0005, weight_decay=0.00001) for ann in self.anns]

        self.lm = (Lawmaker(self.args, self.config).to(self.config.device) if self.args.lm else
                   LawmakerAbstract(self.args, self.config))
        self.lmOpt = None
        if not self.config.TEST and self.args.lm:
            self.lmOpt = Adam(self.lm.parameters(), lr=0.0005, weight_decay=0.00001)

    def backward(self, batches):
        [opt.zero_grad() for opt in self.annsOpts]
        [backwardAgent(batch, device=self.config.device, n_quant=self.config.N_QUANT) for batch in batches]
        [clip_grad_value_(ann.parameters(), 10) for ann in self.anns]
        [opt.step() for opt in self.annsOpts]

    def checkpoint(self, reward):
        if self.config.TEST:
            return
        self.saver.checkpoint(self.anns, self.annsOpts, reward)

    def load(self, best=False):
        print('Loading model...')
        self.saver.load(self.annsOpts, self.anns, best)
        self.saver.loadLawmaker(self.lmOpt, self.lm)

    def loadAnnsFrom(self, states):
        states = [self.convertStateDict(state) for state in states]
        [ann.load_state_dict(state) for ann, state in zip(self.anns, states)]

    def loadLmFrom(self, state):
        state = self.convertStateDict(state)
        self.lm.load_state_dict(state)

    def sendAnns(self):
        states = [ann.state_dict() for ann in self.anns]
        states = [self.convertStateDict(state, device='cpu') for state in states]
        return states

    def sendLm(self):
        state = self.lm.state_dict()
        state = self.convertStateDict(state, device='cpu')
        return state

    def reset_noise(self):
        nets = self.anns + self.targetAnns + [self.lm]
        for net in nets:
            net.reset_noise()

    def updateTargetAnns(self):
        [t.load_state_dict(a.state_dict()) for a, t in zip(self.anns, self.targetAnns)]

    def convertStateDict(self, state, device=None):
        if device is None:
            device = self.config.device
        for k, v in state.items():
            state[k] = v.to(device)
        return state


class Pantheon:
    def __init__(self, config, args):
        self.start, self.tick, self.nANN = time.time(), 0, config.NPOP
        self.config, self.args = config, args
        self.net = Model(config, args)
        self.forward = Forward(config)

        self.period = 1

    def step(self, samples, weights, lifetime=0):
        if self.config.NOISE:
            self.net.reset_noise()
        batches, priorities = self.forward.forward_multi(samples, weights, self.net.anns, self.net.targetAnns,
                                                         self.net.lm, device=self.config.device)
        self.net.backward(batches)

        self.tick += 1
        if not self.config.TEST:
            if (self.tick + 1) % 256 == 0:
                self.net.checkpoint(lifetime)
                self.net.saver.print()

            if (self.tick + 1) % self.config.TARGET_PERIOD == 0:
                self.net.updateTargetAnns()

        return self.net.sendAnns(), priorities
