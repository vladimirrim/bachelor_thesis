import numpy as np
from torch import nn

from forge.blade import entity
from forge.blade.lib import enums
from forge.blade.lib.enums import Neon
from forge.engine.actions import ActionNet
from forge.engine.utils import ValNet
from forge.ethyr import torch as torchlib


class ANN(nn.Module):
    def __init__(self, config, args):
        super().__init__()

        self.config, self.args = config, args
        self.valNet = ValNet(config, n_quant=config.N_QUANT, noise=self.config.NOISE)
        self.actionNets = nn.ModuleDict()
        self.actionNets['move'] = ActionNet(config, entDim=config.ENT_DIM, outDim=5, n_quant=config.N_QUANT,
                                            add_features=int(self.config.VAL_FEATURE), noise=self.config.NOISE)
        if self.config.ATTACK:
            self.actionNets['attack'] = ActionNet(config, entDim=config.ENT_DIM, outDim=2, n_quant=config.N_QUANT,
                                                  add_features=int(self.config.VAL_FEATURE), noise=self.config.NOISE)
        if self.config.SHARE:
            self.actionNets['shareWater'] = ActionNet(config, entDim=config.ENT_DIM, outDim=2, n_quant=config.N_QUANT,
                                                      add_features=int(self.config.VAL_FEATURE), noise=self.config.NOISE)
            self.actionNets['shareFood'] = ActionNet(config, entDim=config.ENT_DIM, outDim=2, n_quant=config.N_QUANT,
                                                     add_features=int(self.config.VAL_FEATURE), noise=self.config.NOISE)

    def forward(self, flat, ents, eps=0, punishmentsLm=None, v=None, device='cpu'):
        val = self.valNet(flat, ents, device=device)
        v = val.detach().mean(2) if v is None else v

        outputs = {}
        for name in self.actionNets.keys():
            punish = punishmentsLm[name] if punishmentsLm is not None else None
            pi, actionIdx = self.actionNets[name](flat, ents, eps, punish, v, device=device)
            outputs[name] = (pi.to('cpu'), actionIdx)

        return outputs, val.to('cpu')

    def reset_noise(self):
        self.valNet.reset_noise()
        for _, net in self.actionNets.items():
            net.reset_noise()

    # Messy hooks for visualizers
    def visDeps(self):
        from forge.blade.core import realm
        from forge.blade.core.tile import Tile
        colorInd = int(12 * np.random.rand())
        color = Neon.color12()[colorInd]
        color = (colorInd, color)
        ent = realm.Desciple(-1, self.config, color).server
        targ = realm.Desciple(-1, self.config, color).server

        sz = 15
        tiles = np.zeros((sz, sz), dtype=object)
        for r in range(sz):
            for c in range(sz):
                tiles[r, c] = Tile(enums.Grass, r, c, 1, None)

        targ.pos = (7, 7)
        tiles[7, 7].addEnt(0, targ)
        posList, vals = [], []
        for r in range(sz):
            for c in range(sz):
                ent.pos = (r, c)
                tiles[r, c].addEnt(1, ent)
                s = torchlib.Stim(ent, tiles, self.config)
                conv, flat, ents = s.conv, s.flat, s.ents
                val = self.valNet(s)
                vals.append(float(val))
                tiles[r, c].delEnt(1)
                posList.append((r, c))
        vals = list(zip(posList, vals))
        return vals

    def visVals(self, food='max', water='max'):
        posList, vals = [], []
        R, C = self.world.shape
        for r in range(self.config.BORDER, R - self.config.BORDER):
            for c in range(self.config.BORDER, C - self.config.BORDER):
                colorInd = int(12 * np.random.rand())
                color = Neon.color12()[colorInd]
                color = (colorInd, color)
                ent = entity.Player(-1, color, self.config)
                ent._pos = (r, c)

                if food != 'max':
                    ent._food = food
                if water != 'max':
                    ent._water = water
                posList.append(ent.pos)

                self.world.env.tiles[r, c].addEnt(ent.entID, ent)
                stim = self.world.env.stim(ent.pos, self.config.STIM)
                s = torchlib.Stim(ent, stim, self.config)
                val = self.valNet(s).detach()
                self.world.env.tiles[r, c].delEnt(ent.entID)
                vals.append(float(val))

        vals = list(zip(posList, vals))
        return vals
