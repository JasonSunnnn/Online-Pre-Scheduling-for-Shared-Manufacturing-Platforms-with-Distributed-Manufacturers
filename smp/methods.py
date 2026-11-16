# -*- coding: utf-8 -*-
"""方法注册表：按名字训练 / 载入任何一种方法。模型存放在 runs/<方法名>/（多随机种子时为 runs/<方法名>_seed<k>/）。"""
from __future__ import print_function

import os

import numpy as np

from . import case as C
from . import core as K

# 论文表5的超参数（GIN-DPPO-E 及其变体共用）
NET = dict(layers=3, d=64, hid_a=64, hid_c=128, fc=2)
QNET = dict(d=64, heads=8, layers=2, hidden=128)    # QIHGRL
ROOT = os.environ.get('SMP_RUNS', 'runs')          # 模型目录（自检时改到别处）
if os.environ.get('SMP_TINY'):                     # 小网络（仅用于快速测试）
    NET = dict(layers=2, d=16, hid_a=16, hid_c=16, fc=2)
    QNET = dict(d=16, heads=4, layers=1, hidden=16)

GRAPH = {   # 名称: (arch, enc, pool, edge, 熵系数方式, 训练环境的估计方式)
    'GIN-DPPO-E': ('dual', 'gin', 'mean', True, 'adaptive', 'pess'),
    'GIN-DPPO-EE': ('dual', 'gin', 'mean', True, 'adaptive', 'exp'),      # 期望值估计
    'DPPO-E': ('dual', 'mlp', 'mean', True, 'adaptive', 'pess'),          # 去掉 GIN
    'GIN-PPO-E': ('single', 'gin', 'mean', True, 'adaptive', 'pess'),     # 单 actor（联合动作空间）
    'GIN-DPPO-E-Sum': ('dual', 'gin', 'sum', True, 'adaptive', 'pess'),   # 图级读出改为 sum pooling
    'GIN-DPPO-E-NE': ('dual', 'gin', 'mean', False, 'adaptive', 'pess'),  # 去掉边特征
    'GIN-DPPO-E-FE': ('dual', 'gin', 'mean', True, 'fixed', 'pess'),      # 固定熵系数 0.01
    'GCN-DPPO-E': ('dual', 'gcn', 'mean', True, 'adaptive', 'pess'),      # GIN 换成 GCN
    'PBMP': ('task', 'mlp', 'mean', True, 'fixed', 'pess'),               # 策略学习选任务 + SPT 选制造商
    'GIN-DPPO-E-NL': ('dual', 'gin', 'mean', True, 'adaptive', 'pess'),   # 去掉负载感知估计（未派任务按无等待下界估计）
}
ABLATION = ['GIN-DPPO-E', 'GIN-DPPO-E-NL', 'GIN-DPPO-EE', 'DPPO-E', 'GIN-PPO-E', 'GIN-DPPO-E-Sum', 'GIN-DPPO-E-NE', 'GIN-DPPO-E-FE',
            'GCN-DPPO-E']
DRL_BASELINES = ['MPDDQN', 'R-PPO', 'PBMP', 'D-Q', 'QIHGRL']
OPTIONAL = []
LEARNED = ABLATION + DRL_BASELINES
EST_LOAD = set(ABLATION) - {'GIN-DPPO-E-NL'} | set(OPTIONAL)
# “无负载感知估计”一族：与 GIN-DPPO-E-NL 只差一个因素的各消融变体。
# 它们不在 train.py --methods all 里；有模型就会自动出现在表6、动态消融和曲线图中。
NL_FAMILY = ['GIN-DPPO-E-NL', 'GIN-DPPO-EE-NL', 'DPPO-E-NL', 'GIN-PPO-E-NL', 'GIN-DPPO-E-Sum-NL', 'GIN-DPPO-E-NE-NL',
             'GIN-DPPO-E-FE-NL', 'GCN-DPPO-E-NL']
for _n in NL_FAMILY[1:]:
    GRAPH[_n] = GRAPH[_n[:-3]]
ABLATION_ALL = ABLATION + NL_FAMILY[1:]
META = ['GA', 'KTPOA']
ALL = LEARNED + META + K.RULE_NAMES


def est_of(name):
    """该方法对“尚未派出的任务”完工时间的估计方式（core.St 的 est）：
    'load'   负载感知估计 —— 本文方法及其消融变体；
    'nowait' 无等待下界 —— GIN-DPPO-E-NL（去掉负载感知估计的消融）和各对比方法。
    试验时可用环境变量 SMP_EST 覆盖图网络类方法的设置。"""
    if name in EST_LOAD:
        return os.environ.get('SMP_EST', 'load')
    return 'nowait'


def have(name, seed=0, root=None):
    """该方法是否已有训练好的模型"""
    return os.path.exists(os.path.join(run_dir(name, seed, root), 'best.npz' if name == 'D-Q' else 'best.pth'))


def run_dir(name, seed=0, root=None):
    return os.path.join(root or ROOT, name if seed == 0 else '%s_seed%d' % (name, seed))


def device(name=''):
    import torch
    return torch.device(name or ('cuda' if torch.cuda.is_available() else 'cpu'))


def build(name, dev=None):
    """构造（未训练的）学习型方法。返回 (agent, 训练环境估计方式)"""
    import torch
    from . import nets, agents
    dev = dev or device()
    if name in GRAPH:
        arch, enc, pool, edge, _, mode = GRAPH[name]
        ag = agents.GraphAgent(nets.GraphNet(arch=arch, enc=enc, pool=pool, edge=edge, **NET).to(dev), dev)
    elif name == 'R-PPO':
        ag, mode = agents.RuleAgent(nets.RuleNet(len(K.RULE_NAMES)).to(dev), dev), 'pess'
    elif name == 'QIHGRL':
        ag, mode = agents.QIHGRLAgent(nets.QIHGRLNet(**QNET).to(dev), dev), 'pess'
    elif name == 'MPDDQN':
        from . import ddqn
        ag, mode = ddqn.DDQNAgent(nets.RuleNet(len(K.RULE_NAMES), value=False).to(dev), dev), 'pess'
    else:
        raise ValueError('not a network-based method: ' + name)
    ag.name = name
    ag.est = est_of(name)
    return ag, mode


def train(name, epochs=None, train_size=None, batch=None, seed=0, resume=False, dev=None, root=None, log=print, **over):
    """训练一个方法。epochs / train_size / batch 为 None 时用默认值（40 / 12800 / 256）"""
    out = run_dir(name, seed, root)
    tsz = train_size or 12800
    train_set = C.gen_dataset(tsz, C.SEED_TRAIN)
    val_set = C.gen_dataset(over.pop('val_size', 300), C.SEED_VALID)
    if name == 'D-Q':
        from . import dq
        return dq.train(out, train_set, val_set, epochs=min(epochs or 30, 30), batch=batch or 256, seed=seed, log=log)
    import torch
    from . import ppo
    ag, mode = build(name, dev)
    if name == 'MPDDQN':
        from . import ddqn, nets
        cfg = dict(ddqn.DEFAULT)
        cfg.update(seed=seed, epochs=epochs or cfg['epochs'], batch=batch or cfg['batch'])
        cfg.update(over)
        target = nets.RuleNet(len(K.RULE_NAMES), value=False).to(ag.dev)
        return ddqn.train(ag, target, cfg, out, train_set, val_set, log=log)
    cfg = dict(ppo.DEFAULT)
    cfg.update(seed=seed, mode=mode)
    if name in GRAPH:
        cfg['ent'] = GRAPH[name][4]
    elif name == 'R-PPO':
        cfg['ent'] = 'fixed'
    elif name == 'QIHGRL':      # 原文 Table II 的超参数
        cfg.update(lr=2e-4, gamma=1.0, k_epochs=3, c_v=0.5, ent='fixed', beta0=0.01, mb=256)
    if epochs:
        cfg['epochs'] = epochs
    if batch:
        cfg['batch'] = batch
    cfg.update(over)
    return ppo.train(ag, cfg, out, train_set, val_set, resume=resume, log=log)


def load(name, seed=0, dev=None, root=None, which='best', **kw):
    """载入训练好的方法（或构造不需要训练的方法），返回 core.Policy"""
    if name in K.COMBINED_RULES:
        return K.RulePolicy(*K.COMBINED_RULES[name], name=name)
    if name in META:
        from . import meta
        return meta.make_policy(name, **kw)
    out = run_dir(name, seed, root)
    if name == 'D-Q':
        from . import dq
        return dq.DQPolicy().load(os.path.join(out, 'best.npz'))
    import torch
    ag, mode = build(name, dev)
    ck = torch.load(os.path.join(out, which + '.pth'), map_location=ag.dev)
    if ck.get('est', 'nowait') != ag.est:
        raise RuntimeError('%s 里的模型是按 est=%s 训练的，而 %s 现在应使用 est=%s，请重新训练该方法。' % (out, ck.get('est', 'nowait'), name, ag.est))
    ag.net.load_state_dict(ck['net'])
    ag.net.eval()
    ag.greedy, ag.store = True, False
    ag.mode = mode
    return ag


def solve(policy, instances, batch=100):
    """用策略求解算例，返回 (悲观估计口径的订单完成时间, 决策, 每个算例平均用时)。
    GIN-DPPO-EE 在期望值估计环境中决策，其决策在悲观估计下重放后再计分。"""
    mode = getattr(policy, 'mode', 'pess')
    costs, decs, t = K.evaluate(policy, instances, mode, batch)
    if mode != 'pess':
        costs = np.array([K.replay(K.Inst(x), d, 'pess')[0] for x, d in zip(instances, decs)])
    for x, d in zip(instances, decs):
        K.check_feasible(K.Inst(x), d)
    return costs, decs, t
