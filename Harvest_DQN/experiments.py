import os

from configs import Experiment

remote = False
exps = {}


def makeExp(name, conf):
    ROOT = 'resource/exps/' + name + '/'
    try:
        os.makedirs(ROOT)
        os.makedirs(ROOT + 'model')
        os.makedirs(ROOT + 'train')
        os.makedirs(ROOT + 'test')
    except FileExistsError:
        pass

    exp = conf()
    exps[name] = exp
    print(name, ', NPOP: ', exp.NPOP)


makeExp('exploreIntensifiesAuto', Experiment)
