# Various high level routines and tools for building quick
# NPC AIs. Returns only action arguments where possible. More
# complex routines return (action, args) pairs where required.

from forge.blade.lib import utils


# Adjacency functions
def adjacentDeltas():
    return [(-1, 0), (1, 0), (0, 1), (0, -1)]


def l1Deltas(s):
    rets = []
    for r in range(-s, s + 1):
        for c in range(-s, s + 1):
            rets.append((r, c))
    return rets


def adjacentPos(pos):
    return [posSum(pos, delta) for delta in adjacentDeltas()]


def adjacentEmptyPos(env, pos):
    return [p for p in adjacentPos(pos)
            if utils.inBounds(*p, env.size)]


def adjacentTiles(env, pos):
    return [env.tiles[p] for p in adjacentPos(pos)
            if utils.inBounds(*p, env.size)]


def adjacentMats(env, pos):
    return [type(env.tiles[p].mat) for p in adjacentPos(pos)
            if utils.inBounds(*p, env.shape)]


def adjacencyDelMatPairs(env, pos):
    return zip(adjacentDeltas(), adjacentMats(env, pos))


def posSum(pos1, pos2):
    return pos1[0] + pos2[0], pos1[1] + pos2[1]

