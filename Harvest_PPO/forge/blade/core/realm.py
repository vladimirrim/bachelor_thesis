import ray

from forge.blade.core.env import HarvestEnv
from forge.trinity import Sword


@ray.remote#(memory=10 ** 10)
class Realm:
    def __init__(self, config, args, idx):
        self.env = HarvestEnv(num_agents=5)
        self.horizon = config.HORIZON
        self.config = config
        self.sword = Sword(config, args)
        self.idx = idx
        self.step = 0

    def recvSwordUpdate(self, update):
        if update is None:
            return
        self.sword.recvUpdate(update)

    def run(self, update):
        self.recvSwordUpdate(update)
        for epoch in range(self.config.EPOCHS):
            obs = self.env.reset()
            rewards = self.env.getInitialRewards()
            for i in range(self.horizon):
                self.step += 1
                agents = self.env.agents
                actions = {key: self.sword.decide(agents[key], obs[key], rewards[key],
                                                  (i + 1) % self.config.LSTM_PERIOD == 0,
                                                  i + epoch * self.horizon, epoch) for key in agents.keys()}
                obs, rewards, dones, info, = self.env.step(actions)
              #  commonReward = 0
              #  for key in obs.keys():
              #      commonReward += rewards[key]

                for key in obs.keys():
                    annID = agents[key].annID
                    self.sword.buffer[annID]['reward'][i + epoch * self.horizon] = rewards[key]
            for agent in self.env.agents.values():
                self.sword.collectRollout(agent.agent_id + str(epoch), agent, self.step, epoch)
        self.sword.backward()
        logs = self.sword.sendLogUpdate()
        buf = self.sword.dispatchBuffer()
        return self.idx, buf, logs

