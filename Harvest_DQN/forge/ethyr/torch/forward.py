import numpy as np
import torch
from collections import defaultdict


class Forward:
    def __init__(self, config):
        self.config = config
        self.prio = self.config.REPLAY_PRIO
        self.nSteps = self.config.REPLAY_NSTEP
        self.a = 0.6
        self.e = 0.001

    def forward_multi(self, samples, weights, anns, targetAnns, lawmaker, device='cpu'):
        batches = [self.forward(samples[i], weights[i], anns[i], targetAnns[i], lawmaker, device=device)
                   for i in range(len(samples))]
        batches, priorities = [batch[0] for batch in batches], [batch[1] for batch in batches]
        return batches, priorities

    def forward(self, sample_of_samples, weights, ann, targetAnn, lawmaker, device='cpu'):
        Qs, returns, priorities = [], [], []

        stim = torch.cat([samples[0][0] for samples in sample_of_samples], dim=0)
        v = torch.cat([samples[0][4] for samples in sample_of_samples], dim=0)
        atnArgs, val = ann(stim.to(device), v=v.to(device))

        stim = torch.cat([samples[-1][0] for samples in sample_of_samples], dim=0)
        v = torch.cat([samples[-1][4] for samples in sample_of_samples], dim=0)
        atnArgs_n, val_n = ann(stim.to(device), v=v.to(device))
        atnArgs_nt, val_nt = targetAnn(stim.to(device),  v=v.to(device))
        punishments = lawmaker(stim.to(device))[1]

        for j, samples in enumerate(sample_of_samples):
            Qs.append(torch.zeros((self.config.N_QUANT,), dtype=torch.float32))
            returns.append(torch.zeros((self.config.N_QUANT,), dtype=torch.float32))

            s, a, r, d, v = samples[0]

            Qs[-1] += val[j][0]
            for action in atnArgs.keys():
                Qs[-1] += self.calculate_A(atnArgs[action][0][j], a[action])

            if d:
                if self.prio:
                    priorities.append(self.calculate_priority(Qs[-1].detach().numpy()))
                continue

            for step, sample in enumerate(samples[1:]):
                s, a, r, d, v = sample
                returns[-1] += self.config.GAMMA ** step * r
                if d:
                    break

            if not d:
                gamma = self.config.GAMMA ** (len(samples) - 1)

                returns[-1] += gamma * val_nt[j][0].detach()
                for action in atnArgs_n.keys():
                    returns[-1] += self.calculate_return(atnArgs_n[action][0][j], punishments[action][j],
                                                         atnArgs_nt[action][0][j], gamma)

            if self.prio:
                priorities.append(self.calculate_priority(Qs[-1].detach().numpy() - returns[-1].numpy()))
        return {'Qs': Qs, 'returns': returns, 'weights': weights}, priorities

    def calculate_A(self, out, action):
        return out[action.view(1)].view(-1,) - out.mean(0).view(-1)

    def calculate_return(self, out_n, punishments, out_nt, gamma):
        return gamma * (out_nt[torch.argmax(out_n.mean(1).view(-1,) + punishments.view(-1,)), :].view(-1) -
                        out_nt.mean(0).view(-1)).detach()

    def calculate_priority(self, td):
        return (np.abs(td).mean() + self.e) ** self.a

    def rolling(self, samples):
        sample_of_samples = []
        for i in range(len(samples) - 1):
            sample_of_samples.append(samples[i: min(i + 1 + self.nSteps, len(samples))])

        return sample_of_samples

    def get_priorities_from_samples(self, samples, ann, targetAnn, lawmaker, device='cpu'):
        sample_of_samples = self.rolling(samples)
        _, priorities = self.forward(sample_of_samples, None, ann, targetAnn, lawmaker, device=device)
        priorities = np.append(priorities, np.ones(1) * np.mean(priorities))
        return priorities


class ForwardLm(Forward):
    def __init__(self, config):
        super(ForwardLm, self).__init__(config)
        self.nSteps = 1

    def forward(self, sample_of_samples, weights, anns, lawmaker, device='cpu'):
        Qs, returns, priorities = [], [], []

        for j, samples in enumerate(sample_of_samples):
            Qs.append(torch.zeros((self.config.N_QUANT,), dtype=torch.float32))
            returns.append(torch.zeros((self.config.N_QUANT,), dtype=torch.float32))

            ents = list(samples[0].keys())
            vals = []
            for ent in ents:
                lst = samples[0][ent]
                vals.append(anns[lst[1]](lst[0].to(device))[1].view(-1).detach())

            val = self.combine_ents(vals)

            target = []
            for ent in ents:
                val_n = torch.zeros(self.config.N_QUANT)
                if ent in samples[-1].keys():
                    lst = samples[-1][ent]
                    if not lst[4]:
                        val_n = anns[lst[1]](lst[0].to(device))[1].view(-1).detach()
                reward = sum([sample[ent][3] * self.config.GAMMA ** step if ent in sample.keys() else 0
                              for step, sample in enumerate(samples[1:])])
                target.append(val_n * self.config.GAMMA ** len(samples) + reward)

            target = self.combine_ents(target)

            Qs[-1] += val
            returns[-1] += target

            for ent in ents:
                s, annID, a, r, d, v = samples[0][ent]
                atnArgs = lawmaker(s.to(device))[0]
                for key in a.keys():
                    if a[key] is not None:
                        Qs[-1] += self.calculate_A(atnArgs[key], a[key])

            if self.config.N_QUANT_LM == 1:
                Qs[-1] = Qs[-1].mean().view(1)
                returns[-1] = returns[-1].mean().view(1)

            if self.prio:
                priorities.append(
                    self.calculate_priority(Qs[-1].detach().to('cpu').numpy() - returns[-1].to('cpu').numpy()))

        return {'Qs': Qs, 'returns': returns, 'weights': weights}, priorities

    def calculate_A(self, out, action):
        return out[:, action].view(-1)

    def combine_ents(self, lst):
        if self.config.LM_FUNCTION == 'sum':
            return sum(lst)
        else:
            if self.config.LM_FUNCTION == 'min':
                idx = np.argmin([t.mean().numpy() for t in lst]).item()
            elif self.config.LM_FUNCTION == 'max':
                idx = np.argmax([t.mean().numpy() for t in lst]).item()
            return lst[idx]

    def get_priorities_from_samples(self, samples, ann, lawmaker, device='cpu'):
        sample_of_samples = self.rolling(samples)
        _, priorities = self.forward(sample_of_samples, None, ann, lawmaker, device=device)
        priorities = np.append(priorities, np.ones(1) * np.mean(priorities))
        return priorities


def aggTensorBy(tensor, by, fun):
    """
    Group by analogue for pytorch tensor
    :param tensor: tensor to aggregate by
    :param by: 1d tensor with sorted (!) indexes to aggregate the tensor by
    :param fun: aggregation function
    :return: tuple (unique indexes, aggregated tensor by by)
    """
    idxs, vals = torch.unique(by, return_counts=True)
    vs = torch.split_with_sizes(tensor, tuple(vals))
    return idxs, torch.stack([fun(v) for v in vs])


def avgTensorBy(tensor, by):
    return aggTensorBy(tensor, by, lambda x: torch.mean(x, dim=0))


def sumTensorBy(tensor, by):
    return aggTensorBy(tensor, by, lambda x: torch.sum(x, dim=0))


def minTensorBy(tensor, by):
    return aggTensorBy(tensor, by, lambda x: torch.min(x, dim=0)[0])


def maxTensorBy(tensor, by):
    return aggTensorBy(tensor, by, lambda x: torch.max(x, dim=0)[0])
