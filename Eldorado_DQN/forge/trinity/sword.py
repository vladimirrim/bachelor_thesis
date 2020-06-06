from collections import defaultdict

from forge.engine import Lawmaker, LawmakerAbstract
from forge.engine.ann import ANN
from forge.ethyr.rollouts import Rollout
from forge.ethyr.torch.replay import ReplayMemory, ReplayMemoryLm
from forge.ethyr.torch.forward import Forward, ForwardLm

from forge.ethyr import torch as torchlib
from forge.blade.action.tree import ActionTree
from forge.blade.action.v2 import ActionV2
from forge.engine.utils import checkTile


class Sword:
    def __init__(self, config, args, idx):
        self.config, self.args = config, args
        self.nANN, self.h = config.NPOP, config.HIDDEN
        self.anns = [ANN(config, args) for _ in range(self.nANN)]

        self.init, self.nRollouts = True, 32
        self.updates = defaultdict(lambda: Rollout(config))
        self.blobs = []
        self.idx = idx
        self.ReplayMemory = [ReplayMemory(self.config) for _ in range(self.nANN)]
        self.ReplayMemoryLm = ReplayMemoryLm(self.config)
        self.forward, self.forward_lm = Forward(config), ForwardLm(config)
        self.buffer_size_to_send = 2 ** 8

        self.lawmaker = Lawmaker(args, config) if args.lm else LawmakerAbstract(args, config)
        self.tick = 0
        self.logTick = 0

    def sendBufferUpdate(self):
        for replay in self.ReplayMemory:
            if len(replay) < self.buffer_size_to_send:
                return None, None
        buffer = [replay.send_buffer() for replay in self.ReplayMemory]
        priorities = [self.forward.get_priorities_from_samples(buf, ann, ann, self.lawmaker) for buf, ann in
                      zip(buffer, self.anns)] if self.config.REPLAY_PRIO else [None] * len(self.anns)
        if self.args.lm:
            bufferLm = self.ReplayMemoryLm.send_buffer()
            prioritiesLm = self.forward_lm.get_priorities_from_samples(bufferLm, self.anns, self.lawmaker) if \
                self.config.REPLAY_PRIO else None
            return (buffer, priorities), (bufferLm, prioritiesLm)
        else:
            return (buffer, priorities), None

    def sendLogUpdate(self):
        self.logTick += 1
        if ((self.logTick + 1) % 2 ** 6) == 0:
            blobs = self.blobs
            self.blobs = []
            return blobs
        return None

    def sendUpdate(self):
        recvs, recvs_lm = self.sendBufferUpdate()
        logs = self.sendLogUpdate() if recvs is not None else None
        return recvs, recvs_lm, logs

    def recvUpdate(self, update):
        update, update_lm = update
        if update is not None:
            self.loadAnnsFrom(update)
        if self.args.lm and (update_lm is not None):
            self.loadLmFrom(update_lm)

    def collectStep(self, entID, annID, s, atnArgs, reward, dead, val):
        if self.config.TEST:
            return
        actions = {key: val[1] for key, val in atnArgs.items()}
        self.ReplayMemory[annID].append(entID, s, actions, reward, dead, val)
        if self.args.lm:
            self.ReplayMemoryLm.append(entID, annID, s, actions, reward, dead, val)

    def collectRollout(self, entID, tick):
        rollout = self.updates[entID]
        rollout.feather.blob.tick = tick
        rollout.finish()
        self.blobs.append(rollout.feather.blob)
        del self.updates[entID]

    def decide(self, ent, stim, isDead=False, n_dead=0):
        entID, annID = ent.entID, ent.annID
        reward = self.config.DEADREWARD * n_dead + self.config.STEPREWARD

        flat, ents = self.prepareInput(ent, stim)
        outputsLm, punishmentsLm = self.lawmaker(flat, ents)
        atnArgs, val = self.anns[annID](flat, ents, self.config.EPS_CUR, punishmentsLm)
        action, arguments, decs = self.actionTree(ent, stim, atnArgs)

        attack = decs.get('attack', None)
        shareFood = decs.get('shareFood', None)
        shareWater = decs.get('shareWater', None)
        ent.moveDec = int(atnArgs['move'][1])
        contact = int(attack is not None)
        if not contact:
            punishAttack, punishWater, punishFood = None, None, None
        else:
            punishAttack = punishmentsLm.get('attack', [[None, None]])[0][1]
            punishWater = punishmentsLm.get('shareWater', [[None, None]])[0][1]
            punishFood = punishmentsLm.get('shareFood', [[None, None]])[0][1]
            ent.shareFoodDec = shareFood
            ent.shareWaterDec = shareWater
            ent.attackDec = attack

        self.collectStep(entID, annID, {'flat': flat, 'ents': ents}, atnArgs, reward, isDead, val.detach().mean(2))
        avgPunishmentLm = calcAvgPunishment(atnArgs, outputsLm)
        if not self.config.TEST:
            self.updates[entID].feather.scrawl(
                stim, ent, val.detach().mean(2), reward, avgPunishmentLm, punishAttack, punishWater, punishFood,
                attack, shareFood, shareWater, contact)
        return action, arguments

    def prepareInput(self, ent, env):
        s = torchlib.Stim(ent, env, self.config)
        return s.flat.unsqueeze(0), s.ents.unsqueeze(0)

    def actionTree(self, ent, env, outputs):
        actions = ActionTree(env, ent, ActionV2).actions()
        _, move, attkShare = actions

        playerActions = [move]
        actionTargets = [move.args(env, ent, self.config)[int(outputs['move'][1])]]

        actionDecisions = {}
        for name in ['attack', 'shareWater', 'shareFood']:
            if name not in outputs.keys():
                continue
            action = attkShare.args(env, ent, self.config)[name]
            targets = action.args(env, ent, self.config)
            target, decision = checkTile(ent, int(outputs[name][1]), targets)
            playerActions.append(action), actionTargets.append([target])
            actionDecisions[name] = decision
        return playerActions, actionTargets, actionDecisions

    def reset_noise(self):
        nets = self.anns + [self.lawmaker]
        for net in nets:
            net.reset_noise()

    def loadAnnsFrom(self, states):
        [ann.load_state_dict(state) for ann, state in zip(self.anns, states)]

    def loadLmFrom(self, state):
        self.lawmaker.load_state_dict(state)


def calcAvgPunishment(atnArgs, punishmentsLm):
    sumPunishment = 0
    cntr = 0
    for key in atnArgs.keys():
        out, punish = atnArgs[key], punishmentsLm[key].mean(2)
        if out[1] is not None:
            sumPunishment += punish.view((-1,))[int(out[1])].detach().numpy() - punish.mean().detach().numpy()
            cntr += 1
    return sumPunishment / cntr
