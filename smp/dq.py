# -*- coding: utf-8 -*-
"""
D-Q：双层 Q-learning（Wang, Sarker, Li & Li, 2021, IJPR 59(19): 5867-5883）的改造版，只用 numpy。

  * 下层（全局）agent：动作 = 任务排序规则 {FIFO, MOPNR, LWKR, MWKR}，目标 = 订单完成时间；
    奖励在回合结束时给出：-(订单完成时间 - 初始下界)/初始下界（等于 GIN-DPPO-E 逐步奖励之和按初始下界归一化）
  * 上层（局部）agent：动作 = 制造商分配规则 {SPT, EET}；
    奖励 = -(所选制造商上新产生的机器空闲时间 + 制造商平均负载标准差的增量)/10（原文上层的局部目标）
  * 状态离散化：下层 (进度 4 档, 候选产品剩余工作量变异系数 3 档)；
                上层 (进度 4 档, 制造商平均负载变异系数 3 档, 所选任务的运输敏感度 3 档)
  * 表格 Q-learning，gamma=1，学习率 max(alpha_min, 1/n^0.7)，epsilon 从 eps0 线性衰减到 eps1
"""
from __future__ import print_function

import os
import time

import numpy as np

from . import core as K
from .core import Policy, choose_task, choose_shop, ready_time, trans_in, deliver_time

P_EDGES, CV_EDGES, LCV_EDGES, TS_EDGES = (0.25, 0.5, 0.75), (0.3, 0.6), (0.2, 0.5), (0.2, 0.5)
SEQ, ASG = ('FIFO', 'MOPNR', 'LWKR', 'MWKR'), ('SPT', 'EET')


def _bin(x, edges):
    return sum(1 for e in edges if x >= e)


def _cv(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size < 2 or v.mean() < 1e-9:
        return 0.0
    return float(v.std() / v.mean())


class DQPolicy(Policy):
    name = 'D-Q'

    def __init__(self, seed=0, alpha_omega=0.7, alpha_min=0.005, scale=10.0):
        self.Qb, self.Qt = np.zeros((12, len(SEQ))), np.zeros((36, len(ASG)))
        self.Nb, self.Nt = np.zeros_like(self.Qb), np.zeros_like(self.Qt)
        self.alpha_omega, self.alpha_min, self.scale = alpha_omega, alpha_min, scale
        self.eps, self.training = 0.0, False
        self.rng = np.random.RandomState(seed)

    def save(self, path):
        np.savez(path, Qb=self.Qb, Qt=self.Qt, Nb=self.Nb, Nt=self.Nt)

    def load(self, path):
        d = np.load(path)
        self.Qb, self.Qt, self.Nb, self.Nt = d['Qb'], d['Qt'], d['Nb'], d['Nt']
        return self

    def begin(self, eps):
        self.lb0 = [max(st.cest - st.t0, 1e-6) for _, st in eps]
        self.pb, self.pt_ = [None] * len(eps), [None] * len(eps)

    def end(self, eps):
        if not self.training:
            return
        for i, (inst, st) in enumerate(eps):
            if self.pb[i] is not None:
                self.pb[i][2] += -(K.cost(st) - self.lb0[i]) / self.lb0[i]
                self._upd(self.Qb, self.Nb, self.pb[i], None)
            if self.pt_[i] is not None:
                self._upd(self.Qt, self.Nt, self.pt_[i], None)

    def _upd(self, Q, Nn, p, s_next):
        s, a, r = p
        Nn[s, a] += 1
        al = max(self.alpha_min, 1.0 / Nn[s, a] ** self.alpha_omega)
        Q[s, a] += al * ((r if s_next is None else r + Q[s_next].max()) - Q[s, a])

    def _pick(self, q):
        if self.training and self.rng.rand() < self.eps:
            return int(self.rng.randint(len(q)))
        return int(np.flatnonzero(q >= q.max() - 1e-12)[0])

    def _local(self, inst, st, r, c, s):
        jt = ready_time(inst, st, r, c, s)
        L = st.loads[s][:]
        idle = 0.0
        for ms, p in zip(inst.msets[r][c][s], inst.PT[r][c]):
            best = min(ms, key=lambda k: (L[k], k))
            idle += max(jt - max(L[best], st.t0), 0.0)
            jt = max(L[best], jt) + p
            L[best] = jt
        before = [sum(x) / len(x) for x in st.loads]
        after = list(before)
        after[s] = sum(L) / len(L)
        return -(idle + float(np.std(after) - np.std(before))) / self.scale

    def decide(self, eps, idxs):
        out = []
        for i in idxs:
            inst, st = eps[i]
            cands = st.candidates(inst)
            prog = _bin(st.ndone / float(inst.n_real), P_EDGES)
            sb = prog * 3 + _bin(_cv([inst.rem_work[r][c] for (r, c) in cands]), CV_EDGES)
            if self.training and self.pb[i] is not None:
                self._upd(self.Qb, self.Nb, self.pb[i], sb)
            ab = self._pick(self.Qb[sb])
            r, c = choose_task(SEQ[ab], inst, st, cands)
            vals = [trans_in(inst, st, r, c, s) + deliver_time(inst, r, c, s) for s in inst.elig[r][c]]
            ts = (max(vals) - min(vals)) / (inst.pt[r][c] + 1e-6)
            stt = (prog * 3 + _bin(_cv([sum(x) / len(x) - st.t0 for x in st.loads]), LCV_EDGES)) * 3 + _bin(ts, TS_EDGES)
            if self.training and self.pt_[i] is not None:
                self._upd(self.Qt, self.Nt, self.pt_[i], stt)
            at = self._pick(self.Qt[stt])
            s = choose_shop(ASG[at], inst, st, r, c)
            if self.training:
                self.pb[i] = [sb, ab, 0.0]
                self.pt_[i] = [stt, at, self._local(inst, st, r, c, s)]
            out.append((r, c, s))
        return out


def train(out_dir, train_set, val_set, epochs=30, batch=256, eps0=0.3, eps1=0.02, seed=0, log=print):
    os.makedirs(out_dir, exist_ok=True)
    pol = DQPolicy(seed)
    n_it = epochs * int(np.ceil(len(train_set) / float(batch)))
    it, best, rows, t0 = 0, np.inf, [], time.time()
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        order = rng.permutation(len(train_set))
        cs = []
        for b0 in range(0, len(order), batch):
            pol.training = True
            pol.eps = eps0 + (eps1 - eps0) * it / float(max(n_it - 1, 1))
            c, _, _ = K.rollout(pol, [K.Inst(train_set[k]) for k in order[b0:b0 + batch]])
            cs.append(c.mean())
            it += 1
        pol.training = False
        vc = K.evaluate(pol, val_set)[0].mean()
        flag = ''
        if vc < best - 1e-9:
            best, flag = vc, ' *'
            pol.save(os.path.join(out_dir, 'best.npz'))
        rows.append([ep + 1, np.mean(cs), vc, best])
        np.savetxt(os.path.join(out_dir, 'train_log.csv'), np.array(rows), delimiter=',', header='epoch,train_cost,val_mean,best_val_mean',
                   comments='')
        log('epoch %3d  train %.2f  valid %.3f%s' % (ep + 1, np.mean(cs), vc, flag))
    with open(os.path.join(out_dir, 'train_time.txt'), 'a') as f:
        f.write('%.1f s for %d epochs\n' % (time.time() - t0, epochs))
    return best
