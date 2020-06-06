import numpy as np

from forge.blade.lib.log import Blob


# Untested function
def discountRewards(rewards, gamma=0.99):
    rets, N = [], len(rewards)
    discounts = np.array([gamma ** i for i in range(N)])
    rewards = np.array(rewards)
    for idx in range(N): rets.append(sum(rewards[idx:] * discounts[:N - idx]))
    return rets


def discountRewardsTD(rewards, vals, gamma=0.99, nSteps=1):
    N = len(rewards) - 1
    rets = np.zeros(N)
    rets[-1] = rewards[-1]
    for idx in reversed(range(N - 1)):
        nStepsCur = min(nSteps, N - 1 - idx)
        ret = [rewards[idx + i + 1] * gamma ** i for i in range(nStepsCur)]
        rets[idx] = sum(ret) + gamma ** nStepsCur * vals[idx + nStepsCur]
    return list(rets)


class Rollout:
    def __init__(self, config):
        self.config = config
        self.actions = []
        self.policy = []
        self.flat_states = []
        self.ent_states = []
        self.rewards = []
        self.rets = []
        self.vals = []
        self.states = []
        self.contacts = []
        self.feather = Feather(config)

    def step(self, action, policy, flat=None, ents=None, reward=None, contact=None, val=None):
        self.actions.append(action)
        self.policy.append(policy)
        self.flat_states.append(flat)
        self.ent_states.append(ents)
        self.rewards.append(reward)
        self.vals.append(val)
        self.contacts.append(contact)

    def finish(self):
        self.lifespan = len(self.rewards)
        self.rets = discountRewardsTD(self.rewards, self.vals, self.config.GAMMA, self.config.REPLAY_NSTEP)
        self.feather.finish()


# Rollout logger
class Feather:
    def __init__(self, config):
        self.blob = Blob(config)

    def scrawl(self, ent, val, reward, lmPunishment, attack, contact):
        self.blob.annID = ent.annID
        self.stats(val, reward, lmPunishment, attack, contact)

    def stats(self, value, reward, lmPunishment, attack, contact):
        self.blob.reward.append(reward)
        self.blob.value.append(float(value))
        self.blob.lmPunishment.append(float(lmPunishment))
        self.blob.contact.append(float(contact))
        if attack is not None:
            self.blob.attack.append(float(attack))

    def finish(self):
        self.blob.finish()
