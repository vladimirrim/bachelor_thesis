import ray

from forge.blade import entity, core


class ActionArgs:
    def __init__(self, action, args):
        self.action = action
        self.args = args


class Realm:
    def __init__(self, config, args, idx):
        self.world, self.desciples = core.Env(config, idx), {}
        self.config, self.args = config, args
        self.npop = config.NPOP

        self.env = self.world.env
        self.values = None
        self.idx = idx

    def clientData(self):
        if self.values is None and hasattr(self, 'sword'):
            self.values = self.sword.anns[0].visVals()

        ret = {
            'environment': self.world.env,
            'entities': dict((k, v.packet()) for k, v in self.desciples.items()),
            'values': self.values
        }
        return ret

    def spawn(self):
        if len(self.desciples) >= self.config.NENT:
            return

        spawned_lst = self.god.spawn()
        for entID, color in spawned_lst:
            ent = entity.Player(entID, color, self.config)
            self.desciples[ent.entID] = ent

            r, c = ent.pos
            self.world.env.tiles[r, c].addEnt(entID, ent)
            self.world.env.tiles[r, c].counts[ent.colorInd] += 1

    def cullDead(self, dead):
        self.god.decrementTimeout()
        for ent in dead:
            entID = ent.entID
            r, c = ent.pos
            self.world.env.tiles[r, c].delEnt(entID)
            self.god.cull(ent.annID)
            del self.desciples[entID]

    def stepEnv(self):
        self.world.env.step()
        self.env = self.world.env.np()

    def stepEnt(self, ent, action, arguments):
        if self.config.ATTACK:
            if self.config.SHARE:
                move, attack, _, _ = action
                moveArgs, attackArgs, _, _ = arguments
            else:
                move, attack = action
                moveArgs, attackArgs = arguments
            ent.move = ActionArgs(move, moveArgs)
            ent.attack = ActionArgs(attack, attackArgs[0])
        else:
            ent.move = ActionArgs(action[0], arguments[0])

    def getStim(self, ent):
        return self.world.env.stim(ent.pos, self.config.STIM)


@ray.remote
class NativeRealm(Realm):
    def __init__(self, trinity, config, args, idx):
        super().__init__(config, args, idx)
        self.god = trinity.god(config, args)
        self.sword = trinity.sword(config, args, idx)
        self.sword.anns[0].world = self.world
        self.logs = []
        self.stepCount = 0

    def stepEnts(self):
        returns = []
        desciples = self.desciples.values()

        for ent in desciples:
            ent.step(self.world)

        dead = self.funeral(desciples)
        n_dead = len(dead) if self.config.COMMON else 0
        self.cullDead(dead)
        self.spawn()

        desciples = self.desciples.values()

        for ent in desciples:
            stim = self.getStim(ent)
            action, arguments = self.sword.decide(ent, stim, False, n_dead)
            returns.append((action, arguments))

        for i, ent in enumerate(desciples):
            action, arguments = returns[i]
            ent.act(self.world, action, arguments)
            self.stepEnt(ent, action, arguments)

        if self.args.lm:
            self.sword.ReplayMemoryLm.add_agents_to_buffer()

        self.sword.config.EPS_CUR = max(self.sword.config.EPS_MIN, self.sword.config.EPS_CUR * self.sword.config.EPS_STEP)
        if self.config.NOISE:
            [self.sword.reset_noise() for _ in range(self.idx + 1)]

        if not self.config.TEST:
            for ent in dead:
                self.sword.collectRollout(ent.entID, self.sword.tick)

    def funeral(self, desciples):
        dead = []
        for i, ent in enumerate(desciples):
            if not ent.alive or ent.kill:
                dead.append(ent)
        n_dead = len(dead) if self.config.COMMON else 1
        for ent in dead:
            stim = self.getStim(ent)
            self.sword.decide(ent, stim, True, n_dead)
        return dead

    def postmortem(self, ent, dead):
        entID = ent.entID
        if not ent.alive or ent.kill:
            dead.append(entID)
            if not self.config.TEST:
                self.sword.collectRollout(entID, self.sword.tick)
            return True
        return False

    def step(self):
        self.spawn()
        self.stepEnv()
        self.stepEnts()
        self.sword.tick += 1

    def run(self, swordUpdate=None):
        self.recvSwordUpdate(swordUpdate)

        updates, updates_lm, logs = None, None, None
        self.stepCount = 0
        while updates is None:
            self.stepCount += 1
            self.step()
            if self.config.TEST:
                updates, updates_lm, logs = (None, None), (None, None), None
            else:
                updates, updates_lm, logs = self.sword.sendUpdate()
        return self.idx, updates, updates_lm, logs

    def recvSwordUpdate(self, update):
        if update is None:
            return
        self.sword.recvUpdate(update)
