import ray

from forge.blade import core, lib


class NativeServer:
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.envs = {i: core.Realm.remote(config, args, i)
                     for i in range(args.nRealm)}
        self.tasks = [e.run.remote(None) for e in self.envs.values()]

    def clientData(self):
        return self.envs[0].clientData.remote()

    def step(self):
        recvs = [e.step.remote() for e in self.envs]
        return ray.get(recvs)

    def append(self, idx, update):
       # self.envs[idx] = core.Realm.remote(self.config, self.args, idx)
        self.tasks.append(self.envs[idx].run.remote(update))

    # Use native api (runs full trajectories)
    def run(self):
        done_id, tasks = ray.wait(self.tasks)
        self.tasks = tasks
        recvs = ray.get(done_id)[0]
        return recvs

    def send(self, swordUpdate):
        [e.recvSwordUpdate.remote(swordUpdate) for e in self.envs]


# Example base runner class
class Blacksmith:
    def __init__(self, args):
        lib.ray.init(args.ray)

    def render(self):
        from forge.embyr.twistedserver import Application
        Application(self.env, self.renderStep)


# Example runner using the (faster) native api
# Use the /forge/trinity/ spec for model code
class Native(Blacksmith):
    def __init__(self, config, args, trinity):
        super().__init__(args)
        self.pantheon = trinity.pantheon(config, args)
        self.env = NativeServer(config, args)

    # Runs full trajectories on each environment
    # With no communication -- all on the env cores.
    def run(self):
        while True:
            recvs = self.env.run()
            idx = recvs[0]
            recvs = recvs[1:]
            self.pantheon.step(*recvs)
            self.env.append(idx, self.pantheon.model())

    def step(self):
        self.env.step()
