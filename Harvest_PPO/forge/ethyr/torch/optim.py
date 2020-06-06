from collections import defaultdict
import torch
from forge.ethyr.torch import loss


def mergeTrajectories(trajectories, device):
    outs = {'vals': [], 'rewards': [], 'policy': [], 'correction': [], 'rho': [], 'rets': [],
            'action': defaultdict(lambda: defaultdict(list))}
    for trajectory in trajectories:
        vals = torch.cat(trajectory['vals']).view(-1, 1)
        rets = torch.cat(trajectory['returns']).view(-1, 1)
        outs['rets'].append(rets + 0.99 * torch.cat((vals[1:].detach(), torch.zeros(1, 1).to(device))))
        outs['vals'].append(vals)
        policy = torch.cat(trajectory['policy'])
        oldPolicy = torch.cat(trajectory['oldPolicy'])
        outs['policy'].append(policy)
        outs['correction'].append(torch.cat(trajectory['correction']).view(-1, 1))
        outs['rho'].append(policy / (1e-5 + oldPolicy))
    return outs


def backwardActorOffPolicy(rho, adv, pi):
    pg, entropy = loss.PG(pi, rho, adv)
    return pg, entropy


def backwardAgentOffPolicy(trajectories, valWeight=0.5, entWeight=0.01, device='cuda'):
    outs = mergeTrajectories(trajectories, device)
    vals = torch.stack(outs['vals']).to(device).view(-1, 1).float()
    rets = torch.stack(outs['rets']).to(device).view(-1, 1).float()
    rho = torch.stack(outs['rho']).to(device).view(-1, 1).float()
    policy = torch.cat(outs['policy']).to(device).float()
    corr = torch.stack(outs['correction']).to(device).view(-1, 1).float()

    pg, entropy = backwardActorOffPolicy(rho, corr * loss.advantage(rets - vals), policy)
    valLoss = loss.valueLoss(rets, vals)
    # wandb.log({"Value loss": valLoss, "PPO Loss": pg, "Entropy loss": entWeight * entropy})
    return pg + valWeight * valLoss + entWeight * entropy, outs


def welfareFunction(vals, mode):
    if mode == 'min':
        return torch.stack(vals).view(5, -1).min(0)[0]
    else:
        return torch.stack(vals).view(5, -1).sum(0)


def mergeLm(trajectories, vals, rets, mode):
    outs = {'vals': torch.stack(vals).view(5, -1),
            'rets': torch.stack(rets).view(5, -1),
            'sumVals': welfareFunction(vals, mode),
            'sumRets': welfareFunction(rets, mode),
            'rho': [],
            'correction': [],
            'QVals': [],
            'entropy': []}
    for i, trajectory in enumerate(trajectories):
        outs['correction'].append(torch.cat(trajectory['correction']).view(-1, 1))
        outs['QVals'].append(torch.cat(trajectory['QVals']).view(-1, 1))
        outs['entropy'].append(torch.cat(trajectory['entropy']).view(-1, 1))
        outs['rho'].append(
            ((torch.cat(trajectory['policy']) + 1e-7).log() - (torch.cat(trajectory['oldPolicy']) + 1e-7).log()).exp().
                view(-1, 1))
    return outs


def backwardLawmaker(rolls, vals, rets, entWeight=0.01, device='cpu', mode='min'):
    outs = mergeLm(rolls, vals, rets, mode)
    sumVals = outs['sumVals'].to(device).reshape(-1, 1).float()
    sumRets = outs['sumRets'].to(device).reshape(-1, 1).float()
    qs = torch.cat(outs['QVals']).to(device).view(5, -1, 1).float()
    totLoss = 0
    totLoss += loss.valueLoss(qs.sum(0).reshape(-1, 1), (sumRets - sumVals.detach()))
    for i, (ent, rho, corr) in enumerate(zip(outs['entropy'], outs['rho'], outs['correction'])):
        entropy = entWeight * ent.mean()
        pg = loss.ppo_loss(corr * loss.advantage(qs[i]), rho)
        totLoss += pg + entropy
    return totLoss, outs
