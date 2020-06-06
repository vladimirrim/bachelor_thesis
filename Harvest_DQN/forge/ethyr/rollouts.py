import numpy as np

from forge.blade.lib.log import Blob


# Untested function
def discountRewards(rewards, gamma=0.9999):
    rets, N = [], len(rewards)
    discounts = np.array([gamma ** i for i in range(N)])
    rewards = np.array(rewards)
    for idx in range(N): rets.append(sum(rewards[idx:] * discounts[:N - idx]))
    return rets


class Rollout:
    def __init__(self, config, returnf=discountRewards):
        self.atnArgs = []
        self.vals = []
        self.rewards = []
        self.states = []
        self.pop_rewards = []
        self.returnf = returnf
        self.feather = Feather(config)
        self.config = config

    def step(self, atnArgs, reward, val=None, s=None):
        self.atnArgs.append(atnArgs)
        self.rewards.append(reward)
        if val is not None:
            self.vals.append(val)
        if s is not None:
            self.states.append(s)

    def finish(self):
        self.returns = self.returnf(self.rewards, gamma=self.config.GAMMA)
        self.lifespan = len(self.rewards)
        self.feather.finish()


# Rollout logger
class Feather:
    def __init__(self, config):
        #self.expMap = set()
        self.blob = Blob(config)

    def scrawl(self, annID, val, reward, apples, lmPunishment):
        self.blob.annID = annID
        self.stats(val, reward, apples, lmPunishment)

    def stats(self, value, reward, apples, lmPunishment):
        self.blob.reward.append(reward)
        self.blob.apples.append(apples)
        self.blob.value.append(float(value))
        self.blob.lmPunishment.append(float(lmPunishment))

    def finish(self):
        self.blob.finish()
