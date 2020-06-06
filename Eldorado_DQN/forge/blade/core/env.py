# Main world definition. Defines and manages entity handlers,
# Defines behavior of the world under all circumstances and handles
# interaction by agents. Also defines an accurate stimulus that
# encapsulates the world as seen by a particular agent

import numpy as np

from forge.blade.core import Map
from forge.blade.lib import utils


class Env:
    def __init__(self, config, idx):
        # Load the world file
        self.env = Map(config, idx)
        self.shape = self.env.shape
        self.spawn = config.SPAWN
        self.config = config

        # Entity handlers
        self.stimSize = 3
        self.worldDim = 2 * self.stimSize + 1

        self.tick = 0
        self.envTimer = utils.BenchmarkTimer()
        self.entTimer = utils.BenchmarkTimer()
        self.cpuTimer = utils.BenchmarkTimer()
        self.handlerTimer = utils.BenchmarkTimer()
        self.statTimer = utils.BenchmarkTimer()

    def stim(self, pos):
        return self.env.getPadded(self.env.tiles, pos,
                                  self.stimSize, key=lambda e: e.index).astype(np.int8)

    # Hook for render
    def graphicsData(self):
        return self.env
