from forge.blade.core.config import Config


class Experiment(Config):
    def defaults(self):
        super().defaults()
        self.MODELDIR = 'resource/logs/'
        self.HIDDEN = 128
        self.TEST = False
        self.LOAD = False
        self.BEST = False
        self.EPOCHS = 1
        self.EPOCHS_PPO = 3
        self.LM_LAMBDA = 0
        self.ATTACK = True
        self.SHARE = False
        self.STEPREWARD = 0.1
        self.DEADREWARD = -1
        self.NATTN = 2
        self.NPOP = 2
        self.ENTROPY = 0.05
        self.ENTROPY_ANNEALING = 0.998
        self.MIN_ENTROPY = 0.00176
        self.VAMPYR = 1
        self.RNG = 1
        self.DMG = 1
        self.MOVEFEAT = 5
        self.SHAREFEAT = 2 * int(self.SHARE)
        self.ATTACKFEAT = 1 * int(self.ATTACK)
        self.EMBEDSIZE = 32
        self.TIMEOUT = 0
        self.DEVICE_OPTIMIZER = 'cpu'
        self.MAP = '_eldorado'
        self.stepsPerEpoch = 2 ** 12
        self.ENT_DIM = 11 + self.MOVEFEAT + self.ATTACKFEAT + self.SHAREFEAT
        self.LR = 0.001
        self.EVOLVE = False
        self.LSTM = False
        self.LSTM_PERIOD = 500
        self.LM_MODE = 'min'  # sum, min
        self.GAMMA = 0.99
        self.COMMON = False
        self.REPLAY_NSTEP = 5


class SimpleMap(Experiment):
    def defaults(self):
        super().defaults()
        self.MELEERANGE = self.RNG
        self.RANGERANGE = 0
        self.MAGERANGE = 0

    def vamp(self, ent, targ, frac, dmg):
        dmg = int(frac * dmg)
        n_food = min(targ.food.val, dmg)
        n_water = min(targ.water.val, dmg)
        targ.food.decrement(amt=n_food)
        targ.water.decrement(amt=n_water)
        ent.food.increment(amt=n_food)
        ent.water.increment(amt=n_water)

    def MELEEDAMAGE(self, ent, targ):
        dmg = self.DMG
        targ.applyDamage(dmg)
        self.vamp(ent, targ, self.VAMPYR, dmg)
        return dmg

    def RANGEDAMAGE(self, ent, targ):
        return 0

    def MAGEDAMAGE(self, ent, targ):
        return 0

    def SHAREWATER(self, ent, targ):
        ent.giveResources(targ, ent.shareWater, 0)
        return ent.shareWater

    def SHAREFOOD(self, ent, targ):
        ent.giveResources(targ, 0, ent.shareFood)
        return ent.shareFood
