import ray
from time import sleep

from forge.ethyr.torch.replay import ReplayMemoryLmMaster, ReplayMemoryMaster
from forge.blade.lib.log import Quill
from forge.blade import core, lib


@ray.remote
class RemoteStateDict:
    def __init__(self, config):
        self.config = config
        self.stateDict = None
        self.stateDictLm = None
        self.counter = -config.MIN_BUFFER
        self.collectFlag = True

    def updateAnns(self, state):
        self.stateDict = state

    def updateLm(self, state):
        self.stateDictLm = state

    def sendAnns(self):
        return self.stateDict

    def sendLm(self):
        return self.stateDictLm

    def send(self):
        return self.sendAnns(), self.sendLm()

    def increaseCounter(self, n):
        self.counter += n
        return self.counter

    def decreaseCounter(self, n):
        self.counter -= n / 4
        if self.counter > 0:
            self.collectFlag = False
        else:
            self.collectFlag = True
        return self.counter

    def getFlag(self):
        return self.collectFlag

    def count(self):
        return self.counter


class NativeServer:
    def __init__(self, config, args, trinity):
        self.envs = [core.NativeRealm.remote(trinity, config, args, i)
                     for i in range(args.nRealm)]
        self.tasks = [e.run.remote(None) for e in self.envs]

    def clientData(self):
        return self.envs[0].clientData.remote()

    def step(self):
        recvs = [e.step.remote() for e in self.envs]
        return ray.get(recvs)

    def run(self):
        done_id, self.tasks = ray.wait(self.tasks)
        recvs = ray.get(done_id)[0]
        return recvs

    def append(self, idx, update):
        self.tasks.append(self.envs[idx].run.remote(update))


class Blacksmith:
    def __init__(self, config, args):
        if args.render:
            print('Enabling local test mode for render')
            args.ray = 'local'
            args.nRealm = 1

        lib.ray.init(args.ray)

    def render(self):
        from forge.embyr.twistedserver import Application
        Application(self.env, self.renderStep)


class Native(Blacksmith):
    def __init__(self, config, args, trinity):
        super().__init__(config, args)
        self.config, self.args = config, args
        self.trinity = trinity
        self.nRealm = args.nRealm
        self.lmPeriod = config.LMPERIOD

        self.renderStep = self.step
        self.idx = 0

    def run(self):
        sharedReplay = ReplayMemoryMaster.remote(self.args, self.config)
        sharedReplayLm = ReplayMemoryLmMaster.remote(self.args, self.config) if self.args.lm else None
        sharedQuill = Quill.remote(self.config.MODELDIR)
        sharedStateDict = RemoteStateDict.remote(self.config)

        self.run_actors.remote(self, sharedReplay, sharedReplayLm, sharedQuill, sharedStateDict)

        pantheonProcessId = self.run_pantheon.remote(self, sharedReplay, sharedQuill, sharedStateDict)
        if self.args.lm:
            self.run_pantheon_lm.remote(self, sharedReplayLm, sharedStateDict)
        ray.get(pantheonProcessId)

    @ray.remote(num_gpus=0)
    def run_actors(self, sharedReplay, sharedReplayLm, sharedQuill, sharedStateDict):
        env = NativeServer(self.config, self.args, self.trinity)

        while True:
            if not ray.get(sharedStateDict.getFlag.remote()):
                sleep(5)
                continue

            idx, buffer, bufferLm, logs = env.run()

            env.append(idx, ray.get(sharedStateDict.send.remote()))

            if buffer is not None:
                sharedReplay.update.remote(buffer)
            if bufferLm is not None and self.args.lm:
                sharedReplayLm.update.remote(bufferLm, idx)

            if logs is not None:
                sharedQuill.scrawl.remote([logs])

            sharedStateDict.increaseCounter.remote(len(buffer[0][0]))

    @ray.remote
    def run_pantheon(self, sharedReplay, sharedQuill, sharedStateDict):
        pantheon = self.trinity.pantheon(self.config, self.args)
        sharedStateDict.updateAnns.remote(pantheon.net.sendAnns())

        while ray.get(sharedReplay.len.remote()) < self.config.MIN_BUFFER:
            sleep(5)

        while True:
            if self.args.lm:
                pantheon.net.loadLmFrom(ray.get(sharedStateDict.sendLm.remote()))

            lifetimeID = sharedQuill.latest.remote()
            idx, sample, weights = ray.get(sharedReplay.sample.remote())
            lifetime = ray.get(lifetimeID)
            states, priorities = pantheon.step(sample, weights, lifetime)
            sharedReplay.update_priorities.remote(idx, priorities)

            sharedStateDict.updateAnns.remote(states)

            sharedStateDict.decreaseCounter.remote(self.config.BATCH_SIZE)

    @ray.remote
    def run_pantheon_lm(self, sharedReplayLm, sharedStateDict):
        pantheonLm = self.trinity.pantheon_lm(self.config, self.args)
        sharedStateDict.updateLm.remote(pantheonLm.net.sendLm())

        while ray.get(sharedReplayLm.len.remote()) < self.config.MIN_BUFFER / self.args.nRealm:
            sleep(5)

        while True:
            pantheonLm.net.loadAnnsFrom(ray.get(sharedStateDict.sendAnns.remote()))

            (idx, sample, weights), i = ray.get(sharedReplayLm.sample.remote())
            stateLm, priorities = pantheonLm.step(sample, weights)
            sharedReplayLm.update_priorities.remote(idx, priorities, i)

            sharedStateDict.updateLm.remote(stateLm)

    def step(self):
        self.env.step()
