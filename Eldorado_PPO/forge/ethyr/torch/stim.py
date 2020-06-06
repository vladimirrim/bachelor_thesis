import numpy as np

from forge.ethyr import stim
from forge.ethyr.torch import utils as tu


class Stim:
    def __init__(self, ent, env, config):
        sz = config.STIM
        flat = stim.entity(ent, ent, config)
        conv, ents = stim.environment(env, ent, sz, config)

        self.flat = tu.var(flat)
        self.conv = tu.var(conv)
        self.ents = tu.var(ents)
        self.color = tu.var(np.array([ent._colorInd]))
