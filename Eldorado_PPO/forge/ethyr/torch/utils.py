import torch
from torch.autograd import Variable


# Variable wrapper
def var(xNp, volatile=False, cuda=False):
    x = Variable(torch.from_numpy(xNp), volatile=volatile).float()
    if cuda:
        x = x.cuda()
    return x
