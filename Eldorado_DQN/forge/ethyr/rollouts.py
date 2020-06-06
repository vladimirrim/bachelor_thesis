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

    def scrawl(self, stim, ent, val, reward, lmPunishment, punishAttack, punishWater, punishFood,
               attack, shareFood, shareWater, contact):
        self.blob.annID = ent.annID
        # tile = self.tile(stim, ent)
        # self.move(tile, ent.pos)
        # self.action(arguments, atnArgs)
        self.stats(val, reward, lmPunishment, punishAttack, punishWater, punishFood,
                   attack, shareFood, shareWater, contact)

    def tile(self, stim, ent):
        R, C = stim.shape
        rCent, cCent = ent.pos
        tile = stim[rCent, cCent]
        return tile

    def action(self, arguments, atnArgs):
        move, attk = arguments
        moveArgs, attkArgs, _ = atnArgs
        moveLogits, moveIdx = moveArgs
        attkLogits, attkIdx = attkArgs

    def move(self, tile, pos):
        tile = type(tile.state)
        r, c = pos
        self.blob.expMap[r][c] += 1
        if pos not in self.expMap:
            self.expMap.add(pos)
            self.blob.unique[tile] += 1
        self.blob.counts[tile] += 1

    def stats(self, value, reward, lmPunishment, punishAttack, punishWater, punishFood,
              attack, shareFood, shareWater, contact):
        self.blob.reward.append(reward)
        self.blob.value.append(float(value))
        self.blob.lmPunishment.append(float(lmPunishment))
        self.blob.contact.append(float(contact))
        if attack is not None:
            self.blob.attack.append(float(attack))
        if shareFood is not None:
            self.blob.shareFood.append(float(shareFood))
        if shareWater is not None:
            self.blob.shareWater.append(float(shareWater))
        if punishAttack is not None:
            self.blob.punishAttack.append(float(punishAttack))
        if punishWater is not None:
            self.blob.punishWater.append(float(punishWater))
        if punishFood is not None:
            self.blob.punishFood.append(float(punishFood))

    def finish(self):
        self.blob.finish()
