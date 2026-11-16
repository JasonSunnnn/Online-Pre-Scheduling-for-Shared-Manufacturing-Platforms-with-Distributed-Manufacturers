# -*- coding: utf-8 -*-
"""
学习型方法的策略类（统一接口 core.Policy）：
GraphAgent   GIN-DPPO-E 及其全部变体、GIN-PPO-E、PBMP 改造版
RuleAgent    R-PPO：PPO 选择组合调度规则
QIHGRLAgent  QIHGRL：异构图 + QNA + IPA，单 actor 选（任务, 制造商）对
store=True 时把每一步的状态、动作、旧策略对数概率、状态价值记录到 buf，供 ppo.py 更新。
"""
import time

import numpy as np
import torch
from torch.distributions import Categorical

from . import case as C
from . import feat as Ft
from .core import Policy, choose_shop, choose_task, COMBINED_RULES, RULE_NAMES

RS = 10.0                      # 奖励缩放：r / RS


def to_t(d, dev):
    out = {}
    for k, v in d.items():
        if v.dtype == np.bool_:
            out[k] = torch.from_numpy(v).to(dev)
        elif v.dtype.kind == 'i':
            out[k] = torch.from_numpy(v.astype(np.int64)).to(dev)
        else:
            out[k] = torch.from_numpy(v.astype(np.float32)).to(dev)
    return out


class Agent(Policy):
    heads = 1

    def __init__(self, net, device):
        self.net, self.dev = net, device
        self.greedy, self.store = True, False
        self.buf, self.traj = [], []
        self.n_dec, self.t_dec, self.t_enc = 0, 0.0, 0.0

    def begin(self, eps):
        self.ctx = [self.init_ctx(inst) for inst, _ in eps]
        if self.store:
            self.buf, self.traj = [], [[] for _ in eps]

    def init_ctx(self, inst):
        return None

    def update_ctx(self, ctx, inst, st, r, c, s):
        pass

    def after_step(self, eps, idxs, choices, rewards):
        for i, (r, c, s), rw in zip(idxs, choices, rewards):
            self.update_ctx(self.ctx[i], eps[i][0], eps[i][1], r, c, s)
            if self.store:
                self.buf[self.traj[i][-1]]['r'] = rw / RS

    def _sample(self, logits):
        a = logits.argmax(-1) if self.greedy else Categorical(logits=logits).sample()
        lp = torch.log_softmax(logits, -1).gather(1, a.unsqueeze(-1)).squeeze(-1)
        return a.cpu().numpy(), lp.cpu().numpy()

    def _keep(self, i, tr):
        self.traj[i].append(len(self.buf))
        self.buf.append(tr)

    def pack(self, trs):
        """把若干条记录堆叠成 numpy 批"""
        return {k: np.stack([t[k] for t in trs]) for k in trs[0] if k != 'r'}


def _ent(logits, mask):
    """各样本的熵及其上界 log(可选动作数)"""
    hmax = torch.log(mask.float().sum(-1).clamp(min=1.0))
    return Categorical(logits=logits).entropy(), hmax


GKEYS = ('x', 'A', 'ew', 'deg', 'cidx', 'cmask', 'gm')


class GraphAgent(Agent):
    def __init__(self, net, device):
        Agent.__init__(self, net, device)
        self.B = Ft.GraphBuilder()
        self.arch = net.arch
        self.heads = 2 if self.arch == 'dual' else 1

    def init_ctx(self, inst):
        return self.B.init(inst)

    def update_ctx(self, ctx, inst, st, r, c, s):
        self.B.update(ctx, inst, st, r, c, s)

    def decide(self, eps, idxs):
        t0 = time.time()
        obs = [self.B.build(eps[i][0], eps[i][1], self.ctx[i]) for i in idxs]
        out = []
        with torch.no_grad():
            ob = to_t(Ft.stack(obs, GKEYS), self.dev)
            t1 = time.time()
            hc, hg = self.net.encode(ob)
            self.t_enc += time.time() - t1
            v = self.net.value(hg, ob['gm']).cpu().numpy()
            if self.arch == 'single':
                fm = [self.B.mfr_all(eps[i][0], eps[i][1]) for i in idxs]
                mfa, mma = np.stack([f for f, _ in fm]), np.stack([m for _, m in fm])
                a, lp = self._sample(self.net.joint_logits(hc, hg, torch.from_numpy(mfa).to(self.dev), torch.from_numpy(mma).to(self.dev)))
                for j, i in enumerate(idxs):
                    r, s = int(a[j]) // Ft.M, int(a[j]) % Ft.M
                    out.append((r, eps[i][1].nxt[r], s))
                    if self.store:
                        tr = dict(obs[j]); tr.update(mfa=mfa[j], mma=mma[j], a=np.array([a[j]], dtype=np.int64),
                                                     logp=np.array([lp[j]], dtype=np.float32), v=np.float32(v[j]), r=0.0)
                        self._keep(i, tr)
            else:
                at, lpt = self._sample(self.net.task_logits(hc, hg, ob['cmask']))
                rc = [(int(at[j]), eps[i][1].nxt[int(at[j])]) for j, i in enumerate(idxs)]
                if self.arch == 'dual':
                    fm = [self.B.mfr(eps[i][0], eps[i][1], r, c) for i, (r, c) in zip(idxs, rc)]
                    mf, mm = np.stack([f for f, _ in fm]), np.stack([m for _, m in fm])
                    am, lpm = self._sample(self.net.mfr_logits(torch.from_numpy(mf).to(self.dev), hg, torch.from_numpy(mm).to(self.dev)))
                for j, i in enumerate(idxs):
                    r, c = rc[j]
                    if self.arch == 'dual':
                        s = int(am[j])
                    else:
                        s = choose_shop('SPT', eps[i][0], eps[i][1], r, c)
                    out.append((r, c, s))
                    if self.store:
                        tr = dict(obs[j])
                        if self.arch == 'dual':
                            tr.update(mf=mf[j], mm=mm[j], a=np.array([at[j], am[j]], dtype=np.int64),
                                      logp=np.array([lpt[j], lpm[j]], dtype=np.float32), v=np.float32(v[j]), r=0.0)
                        else:
                            tr.update(a=np.array([at[j]], dtype=np.int64), logp=np.array([lpt[j]], dtype=np.float32),
                                      v=np.float32(v[j]), r=0.0)
                        self._keep(i, tr)
        self.n_dec += len(idxs)
        self.t_dec += time.time() - t0
        return out

    def evaluate(self, mb):
        """mb：to_t(pack(...))。返回 新策略对数概率 [b,H]、熵 [b,H]、熵上界 [b,H]、状态价值 [b]"""
        hc, hg = self.net.encode(mb)
        v = self.net.value(hg, mb['gm'])
        if self.arch == 'single':
            lg = self.net.joint_logits(hc, hg, mb['mfa'], mb['mma'])
            e, hm = _ent(lg, mb['mma'].reshape(lg.shape[0], -1))
            lp = torch.log_softmax(lg, -1).gather(1, mb['a'])
            return lp, e.unsqueeze(-1), hm.unsqueeze(-1), v
        lt = self.net.task_logits(hc, hg, mb['cmask'])
        et, ht = _ent(lt, mb['cmask'])
        lpt = torch.log_softmax(lt, -1).gather(1, mb['a'][:, 0:1])
        if self.arch != 'dual':
            return lpt, et.unsqueeze(-1), ht.unsqueeze(-1), v
        lm = self.net.mfr_logits(mb['mf'], hg, mb['mm'])
        em, hm = _ent(lm, mb['mm'])
        lpm = torch.log_softmax(lm, -1).gather(1, mb['a'][:, 1:2])
        return torch.cat([lpt, lpm], -1), torch.stack([et, em], -1), torch.stack([ht, hm], -1), v


class RuleAgent(Agent):
    """R-PPO：动作 = 8 个组合调度规则"""

    def decide(self, eps, idxs):
        t0 = time.time()
        S = np.stack([Ft.global_state(eps[i][0], eps[i][1]) for i in idxs])
        with torch.no_grad():
            s = torch.from_numpy(S).to(self.dev)
            a, lp = self._sample(self.net(s))
            v = self.net.value(s).cpu().numpy()
        out = []
        for j, i in enumerate(idxs):
            inst, st = eps[i]
            sr, ar = COMBINED_RULES[RULE_NAMES[int(a[j])]]
            r, c = choose_task(sr, inst, st, st.candidates(inst))
            out.append((r, c, choose_shop(ar, inst, st, r, c)))
            if self.store:
                self._keep(i, {'s': S[j], 'a': np.array([a[j]], dtype=np.int64), 'logp': np.array([lp[j]], dtype=np.float32),
                               'v': np.float32(v[j]), 'r': 0.0})
        self.n_dec += len(idxs)
        self.t_dec += time.time() - t0
        return out

    def evaluate(self, mb):
        lg = self.net(mb['s'])
        e = Categorical(logits=lg).entropy()
        lp = torch.log_softmax(lg, -1).gather(1, mb['a'])
        hm = torch.zeros(lg.shape[0]).to(self.dev) + float(np.log(lg.shape[1]))
        return lp, e.unsqueeze(-1), hm.unsqueeze(-1), self.net.value(mb['s'])


class QIHGRLAgent(Agent):
    def __init__(self, net, device):
        Agent.__init__(self, net, device)
        self.B = Ft.HeteroBuilder()

    def init_ctx(self, inst):
        return self.B.init(inst)

    def update_ctx(self, ctx, inst, st, r, c, s):
        self.B.update(ctx, inst, st, r, c, s)

    def decide(self, eps, idxs):
        t0 = time.time()
        built = [self.B.build(eps[i][0], eps[i][1], self.ctx[i]) for i in idxs]
        with torch.no_grad():
            logits, v = self.net(to_t(Ft.hetero_stack([o for o, _ in built]), self.dev))
            a, lp = self._sample(logits)
            v = v.cpu().numpy()
        out = []
        for j, i in enumerate(idxs):
            out.append(built[j][1][int(a[j])])
            if self.store:
                tr = dict(built[j][0])
                tr.update(a=np.array([a[j]], dtype=np.int64), logp=np.array([lp[j]], dtype=np.float32), v=np.float32(v[j]), r=0.0)
                self._keep(i, tr)
        self.n_dec += len(idxs)
        self.t_dec += time.time() - t0
        return out

    def pack(self, trs):
        d = Ft.hetero_stack(trs)
        for k in ('a', 'logp', 'v'):
            d[k] = np.stack([t[k] for t in trs])
        return d

    def evaluate(self, mb):
        lg, v = self.net(mb)
        e, hm = _ent(lg, mb['opt_mask'])
        lp = torch.log_softmax(lg, -1).gather(1, mb['a'])
        return lp, e.unsqueeze(-1), hm.unsqueeze(-1), v
