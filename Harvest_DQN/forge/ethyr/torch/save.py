import time

import torch

from forge.blade.lib.utils import EDA


class Resetter:
    def __init__(self, resetTol):
        self.resetTicks, self.resetTol = 0, resetTol

    def step(self, best=False):
        if best:
            self.resetTicks = 0
        elif self.resetTicks < self.resetTol:
            self.resetTicks += 1
        else:
            self.resetTicks = 0
            return True
        return False


class Saver:
    def __init__(self, nANN, root, savef, bestf, lmSavef, resetTol):
        self.bestf, self.savef, self.lmSavef = bestf, savef, lmSavef,
        self.root, self.extn = root, '.pth'
        self.nANN = nANN

        self.resetter = Resetter(resetTol)
        self.rewardAvg, self.best = EDA(), 0
        self.start, self.epoch = time.time(), 0
        self.resetTol = resetTol

    def save(self, params, opts, fname):
        data = {'param': params,
                'opt': opts,
                'epoch': self.epoch}
        torch.save(data, self.root + fname + self.extn)

    def checkpoint(self, anns, opts, reward):
        self.save([ann.state_dict() for ann in anns], [opt.state_dict() for opt in opts], self.savef)
        best = reward > self.best
        if best:
            self.best = reward
            self.save([ann.state_dict() for ann in anns], [opt.state_dict() for opt in opts], self.bestf)

        self.time = time.time() - self.start
        self.start = time.time()
        self.reward = reward
        self.epoch += 1

        if self.epoch % 100 == 0:
            self.save([ann.state_dict() for ann in anns], [opt.state_dict() for opt in opts], 'model' + str(self.epoch))

        return self.resetter.step(best)

    def checkpointLawmaker(self, lm, opt):
        self.save(lm.state_dict(), opt.state_dict(), 'lawmaker')
        if self.epoch % 100 == 0:
            self.save(lm.state_dict(), opt.state_dict(), 'lawmaker' + str(self.epoch))

    def load(self, opts, anns, best=False):
        fname = self.bestf if best else self.savef
        data = torch.load(self.root + fname + self.extn)
        [anns[i].load_state_dict(data['param'][i]) for i in range(len(anns))]
        if opts is not None:
            [opt.load_state_dict(s) for opt, s in zip(opts, data['opt'])]
        epoch = data['epoch']
        return epoch

    def loadLawmaker(self, lmOpt, lm):
        fname = self.lmSavef
        try:
            data = torch.load(self.root + fname + self.extn)
            lm.load_state_dict(data['param'])
            if lmOpt is not None:
                lmOpt.load_state_dict(data['opt'])
        except:
            print('lawmaker didn\'t load')

    def print(self):
        print(
            'Tick: ', self.epoch,
            ', Time: ', str(self.time)[:5],
            ', Lifetime: ', str(self.reward)[:5],
            ', Best: ', str(self.best)[:5])
