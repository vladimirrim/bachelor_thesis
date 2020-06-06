import ray

from forge.blade.core.env import HarvestEnv
from forge.trinity import Sword
from copy import deepcopy


@ray.remote
class Realm:
    def __init__(self, config, args, idx):
        self.env = HarvestEnv(num_agents=config.NPOP)
        self.horizon = config.HORIZON
        self.config, self.args = config, args
        self.sword = Sword(config, args, idx)
        self.idx = idx
        self.epoch = 0

    def recvSwordUpdate(self, update):
        if update is None:
            return
        self.sword.recvUpdate(update)

    def run(self, update):
        self.recvSwordUpdate(update)
        """ Rollout several timesteps of an episode of the environment.
               Args:
                   horizon: The number of timesteps to roll out.
                   save_path: If provided, will save each frame to disk at this
                       location.
               """
        obs = self.env.reset()
        rewards = self.env.getInitialRewards()
        apples, rewards_stats = rewards, rewards
        for i in range(self.horizon):
            agents = self.env.agents
            actions = {key: self.sword.decide(self.agentName(agents[key].entID), agents[key].annID, obs[key],
                                              rewards[key], rewards_stats[key],
                                              apples[key], i == (self.horizon - 1)) for key in agents.keys()}
            obs, rewards, dones, info, = self.env.step(actions)
            apples = {k: max(v, 0) for k, v in rewards.items()}
            rewards_stats = deepcopy(rewards)

            if self.args.lm:
                self.sword.ReplayMemoryLm.add_agents_to_buffer()

            self.sword.config.EPS_CUR = max(self.sword.config.EPS_MIN,
                                            self.sword.config.EPS_CUR * self.sword.config.EPS_STEP)
            if self.config.NOISE:
                [self.sword.reset_noise() for _ in range(self.idx + 1)]
            self.sword.tick += 1

            if self.config.COMMON:
                commonReward = eval(self.config.COMMON_FUN)(list(rewards.values()))
                for key in rewards.keys():
                    rewards[key] = commonReward

        for ent in self.env.agents.values():
            self.sword.collectRollout(self.agentName(ent.entID))

        self.epoch += 1

        updates, updates_lm, logs = self.sword.sendUpdate()
        return self.idx, updates, updates_lm, logs

    def agentName(self, name):
        return name + str(self.idx) + str(self.epoch)
