import ray

from forge.blade import entity, core


class ActionArgs:
    def __init__(self, action, args):
        self.action = action
        self.args = args


class Realm:
    def __init__(self, config, args, idx):
        self.world, self.desciples = core.Env(config, idx), {}
        self.config, self.args, self.tick = config, args, 0
        self.npop = config.NPOP

        self.env = self.world.env
        self.values = None
        self.idx = idx

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
            ent = self.desciples[entID]
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
        self.sword = trinity.sword(config, args)
        self.sword.anns[0].world = self.world
        self.logs = []
        self.stepCount = 0

    def stepEnts(self):
        desciples = self.desciples.values()
        returns = []

        for ent in desciples:
            ent.step(self.world)

        dead = self.funeral(desciples)
        n_dead = len(dead) if self.config.COMMON else 0
        self.cullDead(dead)
        self.spawn()

        desciples = self.desciples.values()

        for ent in desciples:
            stim = self.getStim(ent)
            playerActions, playerTargets = self.sword.decide(ent, stim, False, n_dead)
            returns.append((playerActions, playerTargets))

        for i, ent in enumerate(desciples):
            ent.act(self.world, returns[i][0], returns[i][1])

            self.stepEnt(ent, returns[i][0], returns[i][1])

        dead_ = self.funeral(desciples)
        dead.extend(dead_)
        self.cullDead(dead_)

        if not self.config.TEST:
            for ent in dead:
                self.sword.collectRollout(ent.entID, ent, self.tick)

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

    def step(self):
        self.spawn()
        self.stepEnv()
        self.stepEnts()
        self.tick += 1

    def run(self, swordUpdate=None):
        self.recvSwordUpdate(swordUpdate)

        buffer = None
        self.stepCount = 0
        while buffer is None:
            self.stepCount += 1
            self.step()
            if self.config.TEST:
                updates, updates_lm, logs = self.sword.sendLmLogUpdate(), [], []
                if updates is not None:
                    updates = (updates, self.interVector.reset())
            else:
                buffer, logs = self.sword.sendUpdate()
        return self.idx, buffer, logs

    def recvSwordUpdate(self, update):
        if update is None:
            return
        self.sword.recvUpdate(update)
