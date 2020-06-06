import os

from configs import SimpleMap

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
    MODELDIR = ROOT + 'model/'

    exp = conf(remote, MODELDIR=MODELDIR)
    exps[name] = exp
    print(name, ', NENT: ', exp.NENT, ', NPOP: ', exp.NPOP)


makeExp('exploreIntensifiesAuto', SimpleMap)
