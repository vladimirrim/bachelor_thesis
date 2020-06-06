from collections import defaultdict

import torch
from torch.nn import functional as F

from forge.ethyr.torch import loss


def mergeTrajectories(trajectories):
    outs = {'vals': [], 'rewards': [], 'rets': [],
            'correction': defaultdict(list),
            'policy': defaultdict(list),
            'rho': defaultdict(list),
            'action': defaultdict(list)}
    for trajectory in trajectories:
        vals = torch.cat(trajectory['vals']).view(-1, 1)
        rets = torch.cat(trajectory['returns']).view(-1, 1)
        outs['rets'].append(rets)
        outs['vals'].append(vals)
        for k in trajectory['correction'].keys():
            outs['correction'][k].append(torch.cat(trajectory['correction'][k]).view(-1, 1))
            actions = torch.cat(trajectory['actions'][k])
            policy = torch.cat(trajectory['policy'][k])
            outs['policy'][k].append(policy)
            oldPolicy = torch.cat(trajectory['oldPolicy'][k])
            outs['rho'][k].append(F.softmax(policy, dim=1).gather(1, actions.view(-1, 1)) /
                                  (1e-5 + F.softmax(oldPolicy, dim=1).gather(1, actions.view(-1, 1))))
    return outs


def backwardActorOffPolicy(rho, adv, pi):
    pg, entropy = loss.PG(pi, rho, adv)
    return pg, entropy


def backwardAgentOffPolicy(trajectories, valWeight=0.5, entWeight=0.01, device='cuda'):
    outs = mergeTrajectories(trajectories)
    vals = torch.stack(outs['vals']).to(device).view(-1, 1).float()
    rets = torch.stack(outs['rets']).to(device).view(-1, 1).float()
    rho = {k: torch.stack(v).to(device).view(-1, 1).float() for k, v in outs['rho'].items()}
    corr = {k: torch.stack(v).to(device).view(-1, 1).float().detach() for k, v in outs['correction'].items()}
    policy = {k: torch.stack(v).to(device).view(-1, 1).float() for k, v in outs['policy'].items()}
    pgTotal, entropyTotal = 0, 0
    for k in policy.keys():
        pg, entropy = backwardActorOffPolicy(rho[k], corr * loss.advantage(rets - vals), policy[k])
        pgTotal += pg
        entropyTotal += entropy
    valLoss = loss.valueLoss(rets, vals)
    # wandb.log({"Value loss": valLoss, "PPO Loss": pg, "Entropy loss": entWeight * entropy})
    return pgTotal + valWeight * valLoss + entWeight * entropyTotal, outs


def welfareFunction(vals, mode):
    if mode == 'min':
        return torch.stack(vals).view(2, -1).min(0)[0]
    else:
        return torch.stack(vals).view(2, -1).sum(0)


def mergeLm(trajectories, vals, rets, mode):
    outs = {'vals': welfareFunction(vals, mode),
            'rets': welfareFunction(rets, mode),
            'sumRets': torch.stack(rets).sum(0),
            'rho': defaultdict(lambda: defaultdict(list)),
            'Qs': defaultdict(lambda: defaultdict(list)),
            'correction': defaultdict(lambda: defaultdict(list)),
            'entropy': defaultdict(lambda: defaultdict(list))}
    for i, trajectory in enumerate(trajectories):
        for k in trajectory['entropy'].keys():
            outs['entropy'][i][k].append(torch.cat(trajectory['entropy'][k]).view(-1, 1))
            outs['Qs'][i][k].append(torch.cat(trajectory['Qs'][k]).view(-1, 1))
            outs['correction'][i][k].append(torch.cat(trajectory['correction'][k]).view(-1, 1).detach())
            policy = torch.cat(trajectory['policy'][k])
            oldPolicy = (1e-5 + torch.cat(trajectory['oldPolicy'][k]))
            outs['rho'][i][k].append((policy / oldPolicy).view(-1, 1))
    return outs


def backwardLawmaker(rolls, vals, rets, entWeight=0.01, device='cpu', mode='sum'):
    outs = mergeLm(rolls, vals, rets, mode)
    vals = outs['vals'].to(device).reshape(-1, 1).float()
    rets = outs['rets'].to(device).reshape(-1, 1).float()
    totLoss = 0
    totQs = 0
    for Qs in outs['Qs'].values():
        for k in Qs.keys():
            totQs += torch.cat(Qs[k]).view(-1, 1)
    totLoss += loss.valueLoss(totQs, (rets - vals).detach())
    for i, (ent, rho, corr, Qs) in enumerate(zip(outs['entropy'].values(),
                                                 outs['rho'].values(),
                                                 outs['correction'].values(), outs['Qs'].values())):
        for k in rho.keys():
            entropy = entWeight * torch.cat(ent[k]).view(-1, 1).mean()
            pg = loss.ppo_loss(torch.cat(corr[k]).view(-1, 1) * loss.advantage(torch.cat(Qs[k]).view(-1, 1)),
                               torch.cat(rho[k]).view(-1, 1))
            #   wandb.log({'LmLoss': pg, 'LmEntropy': entropy})
            totLoss += pg + entropy
    return totLoss / 2, outs
