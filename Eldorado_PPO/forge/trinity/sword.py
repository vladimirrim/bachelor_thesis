from collections import defaultdict, deque

import numpy as np

from forge.blade.action.tree import ActionTree
from forge.blade.action.v2 import ActionV2
from forge.engine import Lawmaker, LawmakerAbstract
from forge.engine.ann import ANN
from forge.engine.utils import checkTile
from forge.ethyr import torch as torchlib
from forge.ethyr.rollouts import Rollout
from forge.ethyr.torch.param import setParameters, zeroGrads


class Sword:
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.nANN, self.h = config.NPOP, config.HIDDEN
        self.anns = [ANN(config, args) for _ in range(self.nANN)]

        self.updates, self.rollouts = defaultdict(lambda: Rollout(config)), {}
        self.updates_lm, self.rollouts_lm = defaultdict(lambda: Rollout(config)), {}
        self.nGrads = 0
        self.rets = defaultdict(deque)
        self.flat_states = defaultdict(deque)
        self.ent_states = defaultdict(deque)
        self.lmActions = defaultdict(deque)
        self.lmPolicy = defaultdict(deque)
        self.policy = defaultdict(deque)
        self.actions = defaultdict(deque)
        self.contacts = defaultdict(deque)
        self.buffer = None

        self.lawmaker = Lawmaker(args, config) if args.lm else LawmakerAbstract(args, config)

    def backward(self):
        self.rollouts_lm = {}
        self.blobs = [r.feather.blob for r in self.rollouts.values()]
        self.rollouts = {}
        length = min([len(self.rets[i]) for i in range(self.nANN)])
        if length == 0:
            return
        self.initBuffer()
        buffer = defaultdict(lambda: {'policy': defaultdict(list),
                                      'action': defaultdict(list),
                                      'lmPolicy': defaultdict(list),
                                      'lmAction': defaultdict(list),
                                      'flat': [], 'ents': [], 'return': []})
        for _ in range(length):
            for i in range(self.nANN):
                buffer[i]['flat'].append(self.flat_states[i].popleft())
                buffer[i]['ents'].append(self.ent_states[i].popleft())
                buffer[i]['return'].append(self.rets[i].popleft())
                for (k, v), action in zip(self.policy[i].popleft().items(), self.actions[i].popleft().values()):
                    buffer[i]['policy'][k].append(v.detach().numpy())
                    buffer[i]['action'][k].append(action.detach().numpy())
                for (k, v), action in zip(self.lmPolicy[i].popleft().items(), self.lmActions[i].popleft().values()):
                    buffer[i]['lmPolicy'][k].append(v.detach().numpy())
                    buffer[i]['lmAction'][k].append(action.detach().numpy())
        for i in range(self.nANN):
            self.buffer[i]['flat'] = np.asarray(buffer[i]['flat'], dtype=np.float32)
            self.buffer[i]['ents'] = np.asarray(buffer[i]['ents'], dtype=np.float32)
            self.buffer[i]['return'] = np.asarray(buffer[i]['return'], dtype=np.float32)
            self.buffer[i]['policy'] = {k: np.asarray(v, dtype=np.float32) for k, v in buffer[i]['policy'].items()}
            self.buffer[i]['action'] = {k: np.asarray(v, dtype=np.float32) for k, v in buffer[i]['action'].items()}
            self.buffer[i]['lmPolicy'] = {k: np.asarray(v, dtype=np.float32) for k, v in buffer[i]['lmPolicy'].items()}
            self.buffer[i]['lmAction'] = {k: np.asarray(v, dtype=np.float32) for k, v in buffer[i]['lmAction'].items()}

    def sendLogUpdate(self):
        blobs = self.blobs
        self.blobs = []
        return blobs

    def sendUpdate(self):
        if self.buffer is None:
            return None, None
        buffer = self.dispatchBuffer()
        return buffer, self.sendLogUpdate()

    def recvUpdate(self, update):
        update, update_lm = update
        for idx, paramVec in enumerate(update):
            setParameters(self.anns[idx], paramVec)
            zeroGrads(self.anns[idx])

        setParameters(self.lawmaker, update_lm[0])
        zeroGrads(self.lawmaker)

    def collectStep(self, entID, action, policy, flat, ents, reward, contact, val):
        if self.config.TEST:
            return
        self.updates[entID].step(action, policy, flat, ents, reward, contact, val)

    def collectStepLm(self, entID, action, policy):
        if self.config.TEST:
            return
        self.updates_lm[entID].step(action, policy)

    def collectRollout(self, entID, ent, tick):
        assert entID not in self.rollouts
        rollout = self.updates[entID]
        rollout.finish()
        nGrads = rollout.lifespan
        self.rets[ent.annID] += rollout.rets
        self.ent_states[ent.annID] += rollout.ent_states[:-1]
        self.flat_states[ent.annID] += rollout.flat_states[:-1]
        self.policy[ent.annID] += rollout.policy[:-1]
        self.actions[ent.annID] += rollout.actions[:-1]
        self.contacts[ent.annID] += rollout.contacts[:-1]
        rollout_lm = self.updates_lm[entID]
        self.lmPolicy[ent.annID] += rollout_lm.policy[:-1]
        self.lmActions[ent.annID] += rollout_lm.actions[:-1]
        del self.updates_lm[entID]
        rollout.feather.blob.tick = tick
        self.rollouts[entID] = rollout
        del self.updates[entID]
        self.nGrads += nGrads

        if self.nGrads >= self.config.stepsPerEpoch:
            self.nGrads = 0
            self.backward()

    def initBuffer(self):
        self.buffer = defaultdict(dict)

    def dispatchBuffer(self):
        buffer = self.buffer
        self.buffer = None
        return buffer

    def getActionArguments(self, annReturns, stim, ent):
        actions = ActionTree(stim, ent, ActionV2).actions()
        move, attkShare = actions
        playerActions = [move]
        actionDecisions = {}
        moveAction = int(annReturns['actions']['move'])
        attack = moveAction > 4
        if attack:
            moveAction -= 5
        actionTargets = [move.args(stim, ent, self.config)[moveAction]]

        action = attkShare.args(stim, ent, self.config)['attack']
        targets = action.args(stim, ent, self.config)
        target, decision = checkTile(ent, int(attack), targets)
        playerActions.append(action), actionTargets.append([target])
        actionDecisions['attack'] = decision

        return playerActions, actionTargets, actionDecisions

    def decide(self, ent, stim, isDead, n_dead=0):
        entID, annID = ent.entID, ent.annID
        reward = self.config.STEPREWARD + self.config.DEADREWARD * n_dead

        stim_tensor = torchlib.Stim(ent, stim, self.config)
        outsLm = self.lawmaker(stim_tensor.flat.view(1, -1), stim_tensor.ents.unsqueeze(0), isDead, annID)
        annReturns = self.anns[annID](stim_tensor.flat.view(1, -1), stim_tensor.ents.unsqueeze(0), outsLm, isDead)

        playerActions, actionTargets, actionDecisions = self.getActionArguments(annReturns, stim, ent)

        moveAction = int(annReturns['actions']['move'])
        attack = actionDecisions.get('attack', None)
        if moveAction > 4:
            moveAction -= 5
        ent.moveDec = moveAction
        contact = int(attack is not None)

        Asw = -np.mean([float(t.mean()) for t in outsLm['Qs'].values()])
        outsLm = self.lawmaker.get_punishment(outsLm, annReturns['actions'])
        Asw += np.mean([float(t) for t in outsLm['Qs'].values()])

        self.collectStep(entID, annReturns['actions'], annReturns['policy'],
                         stim_tensor.flat.numpy(), stim_tensor.ents.numpy(), reward, contact, float(annReturns['val']))
        self.collectStepLm(entID, outsLm['actions'], outsLm['policy'])
        if not self.config.TEST:
            self.updates[entID].feather.scrawl(ent, float(annReturns['val']),
                                               reward, Asw,
                                               attack, contact)
        return playerActions, actionTargets
