# -*- coding: utf-8 -*-
"""
MPDDQN 改造版（X. Yang et al., 2025）：Double DQN，动作 = 8 个组合调度规则，状态 = 手工全局状态向量。
经验回放 + 目标网络（软更新），epsilon 线性衰减。奖励与 GIN-DPPO-E 相同。
"""
from __future__ import print_function

import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import core as K
from . import feat as Ft
from .agents import RS
from .core import Policy, COMBINED_RULES, RULE_NAMES, choose_task, choose_shop

DEFAULT = dict(lr=3e-4, gamma=0.99, batch=256, epochs=40, train_size=12800, mb=256, buffer=200000, tau=0.01, eps0=1.0, eps1=0.05,
               eps_frac=0.6, grad_clip=2.5, patience=10, seed=0, updates_per_batch=50)


class DDQNAgent(Policy):
    def __init__(self, net, device):
        self.net, self.dev = net, device
        self.eps, self.store = 0.0, False
        self.mem = []
        self.pos = 0
        self.cap = DEFAULT['buffer']
        self.n_dec, self.t_dec = 0, 0.0

    def begin(self, eps):
        self.pend = [None] * len(eps)

    def _push(self, tr):
        if len(self.mem) < self.cap:
            self.mem.append(tr)
        else:
            self.mem[self.pos] = tr
            self.pos = (self.pos + 1) % self.cap

    def decide(self, eps, idxs):
        t0 = time.time()
        S = np.stack([Ft.global_state(eps[i][0], eps[i][1]) for i in idxs])
        with torch.no_grad():
            a = self.net(torch.from_numpy(S).to(self.dev)).argmax(-1).cpu().numpy()
        out = []
        for j, i in enumerate(idxs):
            inst, st = eps[i]
            act = int(a[j])
            if self.store:
                if np.random.rand() < self.eps:
                    act = int(np.random.randint(len(RULE_NAMES)))
                if self.pend[i] is not None:
                    s0, a0, r0 = self.pend[i]
                    self._push((s0, a0, r0, S[j], 0.0))
                self.pend[i] = [S[j], act, 0.0]
            sr, ar = COMBINED_RULES[RULE_NAMES[act]]
            r, c = choose_task(sr, inst, st, st.candidates(inst))
            out.append((r, c, choose_shop(ar, inst, st, r, c)))
        self.n_dec += len(idxs)
        self.t_dec += time.time() - t0
        return out

    def after_step(self, eps, idxs, choices, rewards):
        if self.store:
            for i, rw in zip(idxs, rewards):
                self.pend[i][2] = rw / RS

    def end(self, eps):
        if self.store:
            for p in self.pend:
                if p is not None:
                    self._push((p[0], p[1], p[2], p[0], 1.0))


def _learn(agent, target, opt, cfg):
    idx = np.random.randint(0, len(agent.mem), size=cfg['mb'])
    b = [agent.mem[k] for k in idx]
    dev = agent.dev
    s = torch.from_numpy(np.stack([x[0] for x in b])).to(dev)
    a = torch.from_numpy(np.array([[x[1]] for x in b], dtype=np.int64)).to(dev)
    r = torch.from_numpy(np.array([x[2] for x in b], dtype=np.float32)).to(dev)
    s2 = torch.from_numpy(np.stack([x[3] for x in b])).to(dev)
    d = torch.from_numpy(np.array([x[4] for x in b], dtype=np.float32)).to(dev)
    with torch.no_grad():
        a2 = agent.net(s2).argmax(-1).unsqueeze(-1)                       # Double DQN：在线网络选动作
        y = r + cfg['gamma'] * (1.0 - d) * target(s2).gather(1, a2).squeeze(-1)   # 目标网络估值
    q = agent.net(s).gather(1, a).squeeze(-1)
    loss = Fn.smooth_l1_loss(q, y)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.net.parameters(), cfg['grad_clip'])
    opt.step()
    with torch.no_grad():
        sd, td = agent.net.state_dict(), target.state_dict()
        target.load_state_dict({k: td[k] * (1.0 - cfg['tau']) + sd[k] * cfg['tau'] for k in td})
    return loss.item()


def train(agent, target, cfg, out_dir, train_set, val_set, log=print):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    target.load_state_dict(agent.net.state_dict())
    opt = torch.optim.Adam(agent.net.parameters(), lr=cfg['lr'])
    n_it = cfg['epochs'] * int(np.ceil(len(train_set) / float(cfg['batch'])))
    it, best, bad, rows, t_start = 0, np.inf, 0, [], time.time()
    for ep in range(cfg['epochs']):
        t0 = time.time()
        order = np.random.permutation(len(train_set))
        ls, cs = [], []
        for b0 in range(0, len(order), cfg['batch']):
            agent.store = True
            agent.eps = max(cfg['eps1'], cfg['eps0'] + (cfg['eps1'] - cfg['eps0']) * it / (cfg['eps_frac'] * n_it))
            c, _, _ = K.rollout(agent, [K.Inst(train_set[k]) for k in order[b0:b0 + cfg['batch']]])
            cs.append(c.mean())
            it += 1
            if len(agent.mem) >= cfg['mb']:
                ls.extend(_learn(agent, target, opt, cfg) for _ in range(cfg['updates_per_batch']))
        agent.store = False
        vc = K.evaluate(agent, val_set)[0]
        flag = ''
        if vc.mean() < best - 1e-9:
            best, bad, flag = float(vc.mean()), 0, ' *'
            torch.save({'net': agent.net.state_dict(), 'epoch': ep + 1, 'val': best}, os.path.join(out_dir, 'best.pth'))
        else:
            bad += 1
        rows.append([ep + 1, float(np.mean(ls)) if ls else 0.0, float(agent.eps), float(np.mean(cs)), float(vc.mean()), best,
                     time.time() - t0])
        with open(os.path.join(out_dir, 'train_log.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'loss', 'epsilon', 'train_cost', 'val_mean', 'best_val_mean', 'epoch_time_s'])
            w.writerows(rows)
        log('epoch %3d  loss %.4f  eps %.2f  train %.2f  valid %.3f%s  (%.0fs)' % (ep + 1, rows[-1][1], agent.eps, np.mean(cs),
                                                                                   vc.mean(), flag, time.time() - t0))
        if bad >= cfg['patience'] and agent.eps <= cfg['eps1'] + 1e-9:      # 探索率降到最小值之后才允许提前停止
            log('early stop')
            break
    with open(os.path.join(out_dir, 'train_time.txt'), 'a') as f:
        f.write('%.1f s for %d epochs\n' % (time.time() - t_start, len(rows)))
    return best
