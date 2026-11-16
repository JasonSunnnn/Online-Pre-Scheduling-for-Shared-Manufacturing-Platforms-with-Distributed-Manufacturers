# -*- coding: utf-8 -*-
"""
元启发式对比算法：GA 与 KTPOA。两者编码相同（任务序列 OS：基于产品的重复编码；制造商分配 MS），
目标函数相同（悲观估计下的订单完成时间；适应度另加一个很小的次要项用于区分完成时间相同的解，见 core.decode_fit），
终止条件相同（目标函数评价次数 max_evals）。

GA（Sun et al., 2024 的两层编码遗传算法，按本问题改造）：随机初始化，二元锦标赛选择，POX + 均匀交叉，
    变异（任务序列以 0.5 的概率交换或插入一次；制造商分配每个基因以 1.5/L 的概率重新分配），(mu+lambda) 精英保留。
    种群 100，交叉概率 0.9。
KTPOA（Pan, Lei & Wang, 2022, IEEE TCYB 52(6): 5051-5063 的改造版）：
    * 启发式（8 个组合规则的解）+ 随机初始化
    * 两个种群协同进化：种群 A 用 GA 型算子，种群 B 用离散 DE（DE/best/1 + 交叉 + 一对一贪婪选择），每代互换最优个体
    * 两个基于知识的局部搜索（作用于当前最优解）：LS1 关键产品的任务改派到其他可选制造商；
      LS2 关键产品的任务在序列中前移
"""
from __future__ import print_function

import random
import time

from .core import COMBINED_RULES, construct, decode_fit, to_decisions, apply, FixedPolicy


class Ind(object):
    __slots__ = ('os', 'ms', 'f')

    def __init__(self, os_, ms, f=None):
        self.os, self.ms, self.f = os_, ms, f

    def key(self):
        return (tuple(self.os), tuple(self.ms))


class KTPOA(object):
    def __init__(self, inst, st0, pop_size=100, max_evals=20000, pc=0.9, pm=1.5, pm_os=0.5, F=0.5, CR=0.5,
                 n_exchange=2, ls_tries=10, time_limit=None, seed=0):
        self.inst, self.st0 = inst, st0
        self.pop_size = max(6, int(pop_size))
        self.max_evals = int(max_evals)
        self.pc, self.pm, self.pm_os, self.F, self.CR = pc, pm, pm_os, F, CR
        self.n_exchange, self.ls_tries = n_exchange, ls_tries
        self.time_limit = time_limit
        self.rng = random.Random(seed)
        self.evals = 0
        self.tasks = [inst.flat(r, c) for (r, c) in inst.real_tasks]
        self.rows = [r for r in range(inst.n_j) if inst.nsub[r] > 0]
        self.base_os = [r for r in range(inst.n_j) for _ in range(inst.nsub[r])]
        self.history = []

    # ------------------------------------------------------------------ basics
    def f(self, ind):
        if ind.f is None:
            ind.f = decode_fit(self.inst, self.st0, ind.os, ind.ms)
            self.evals += 1
        return ind.f

    def budget_left(self):
        if self.evals >= self.max_evals:
            return False
        if self.time_limit is not None and time.time() - self.t0 > self.time_limit:
            return False
        return True

    def random_ind(self):
        os_ = self.base_os[:]
        self.rng.shuffle(os_)
        ms = [-1] * (self.inst.n_j * self.inst.n_o)
        for (r, c) in self.inst.real_tasks:
            ms[self.inst.flat(r, c)] = self.rng.choice(self.inst.elig[r][c])
        return Ind(os_, ms)

    def tournament(self, pop):
        a, b = self.rng.sample(pop, 2)
        return a if a.f <= b.f else b

    # ------------------------------------------------------------------ operators
    def pox(self, a, b):
        if len(self.rows) < 2:
            return a[:]
        J1 = set(r for r in self.rows if self.rng.random() < 0.5)
        if not J1 or len(J1) == len(self.rows):
            J1 = set(self.rng.sample(self.rows, max(1, len(self.rows) // 2)))
        fill = iter([g for g in b if g not in J1])
        return [g if g in J1 else next(fill) for g in a]

    def ms_uniform(self, a, b):
        x, y = a[:], b[:]
        for t in self.tasks:
            if self.rng.random() < 0.5:
                x[t], y[t] = y[t], x[t]
        return x, y

    def mutate(self, ind):
        os_, ms = ind.os[:], ind.ms[:]
        if self.rng.random() < self.pm_os and len(os_) > 1:          # 任务序列：交换或插入
            i, j = self.rng.sample(range(len(os_)), 2)
            if self.rng.random() < 0.5:
                os_[i], os_[j] = os_[j], os_[i]
            else:
                g = os_.pop(i)
                os_.insert(j, g)
        pg = self.pm / float(max(len(self.tasks), 1))                # 制造商分配：每个基因以 pm/L 的概率重新分配
        for t in self.tasks:
            if self.rng.random() < pg:
                r, c = divmod(t, self.inst.n_o)
                ms[t] = self.rng.choice(self.inst.elig[r][c])
        return Ind(os_, ms)

    # ------------------------------------------------------------------ population A: GA / NSGA-II type
    def evolve_A(self, A):
        n = len(A)
        off = []
        while len(off) < n and self.budget_left():
            p1, p2 = self.tournament(A), self.tournament(A)
            if self.rng.random() < self.pc:
                o1 = self.pox(p1.os, p2.os)
                o2 = self.pox(p2.os, p1.os)
                m1, m2 = self.ms_uniform(p1.ms, p2.ms)
            else:
                o1, o2, m1, m2 = p1.os[:], p2.os[:], p1.ms[:], p2.ms[:]
            for o, m in ((o1, m1), (o2, m2)):
                child = self.mutate(Ind(o, m))
                self.f(child)
                off.append(child)
        return self.select(A + off, n)

    def select(self, pool, n):
        seen, out = set(), []
        for ind in sorted(pool, key=lambda z: z.f):
            k = ind.key()
            if k in seen:
                continue
            seen.add(k)
            out.append(ind)
            if len(out) == n:
                break
        while len(out) < n:                     # 去重后不足时用随机解补齐
            z = self.random_ind()
            self.f(z)
            out.append(z)
        return out

    # ------------------------------------------------------------------ population B: DE type
    def evolve_B(self, B, gbest):
        n = len(B)
        out = list(B)
        L = len(gbest.os)
        for k in range(n):
            if not self.budget_left():
                break
            x = out[k]
            r1, r2 = self.rng.sample([z for j, z in enumerate(out) if j != k], 2)
            vms = gbest.ms[:]
            for t in self.tasks:
                if r1.ms[t] != r2.ms[t] and self.rng.random() < self.F:
                    vms[t] = r1.ms[t]
            vos = gbest.os[:]
            D = [p for p in range(L) if r1.os[p] != r2.os[p]]
            for _ in range(int(round(self.F * len(D) / 2.0))):
                if len(D) < 2:
                    break
                p, q = self.rng.sample(D, 2)
                vos[p], vos[q] = vos[q], vos[p]
            uos = self.pox(x.os, vos) if self.rng.random() < self.CR else vos
            jr = self.rng.choice(self.tasks) if self.tasks else -1
            ums = x.ms[:]
            for t in self.tasks:
                if t == jr or self.rng.random() < self.CR:
                    ums[t] = vms[t]
            u = Ind(uos, ums)
            if self.f(u) <= x.f:
                out[k] = u
        return out

    # ------------------------------------------------------------------ knowledge-based local search
    def critical_row(self, ind):
        st = self.st0.copy()
        for (r, c, s) in to_decisions(self.inst, ind.os, ind.ms):
            apply(self.inst, st, r, c, s)
        ends = {r: st.temp1[r][self.inst.nsub[r] - 1] for r in self.rows}
        return max(ends, key=ends.get)

    def ls_reassign(self, ind):
        """LS1：关键产品的任务改派到其他可选制造商（从最后一个任务往前），首次改进即接受"""
        inst = self.inst
        r = self.critical_row(ind)
        tries = 0
        for c in range(inst.nsub[r] - 1, -1, -1):
            t = inst.flat(r, c)
            for s in inst.elig[r][c]:
                if s == ind.ms[t]:
                    continue
                if tries >= self.ls_tries or not self.budget_left():
                    return ind
                ms = ind.ms[:]
                ms[t] = s
                cand = Ind(ind.os[:], ms)
                tries += 1
                if self.f(cand) < ind.f - 1e-9:
                    return cand
        return ind

    def ls_advance(self, ind):
        """LS2：关键产品的任务在 OS 中前移（保持同产品任务的先后顺序），首次改进即接受"""
        r = self.critical_row(ind)
        pos = [p for p, g in enumerate(ind.os) if g == r]
        order = list(range(len(pos)))
        self.rng.shuffle(order)
        tries = 0
        for k in order:
            lo = pos[k - 1] + 1 if k > 0 else 0
            p = pos[k]
            if p - lo < 1:
                continue
            for _ in range(2):
                if tries >= self.ls_tries or not self.budget_left():
                    return ind
                q = self.rng.randint(lo, p - 1)
                os_ = ind.os[:]
                g = os_.pop(p)
                os_.insert(q, g)
                cand = Ind(os_, ind.ms[:])
                tries += 1
                if self.f(cand) < ind.f - 1e-9:
                    return cand
        return ind

    # ------------------------------------------------------------------ main loop
    def run(self):
        self.t0 = time.time()
        heur = []
        for name, (sr, ar) in sorted(COMBINED_RULES.items()):
            os_, ms, _ = construct(self.inst, self.st0, sr, ar)
            ind = Ind(os_, ms)
            self.f(ind)
            heur.append(ind)
        pop = heur + [self.random_ind() for _ in range(max(0, self.pop_size - len(heur)))]
        for z in pop:
            self.f(z)
        self.rng.shuffle(pop)
        nA = self.pop_size // 2
        A, B = self.select(pop[:nA], nA), self.select(pop[nA:], self.pop_size - nA)
        best = min(A + B, key=lambda z: z.f)
        gen = 0
        while self.budget_left():
            A = self.evolve_A(A)
            B = self.evolve_B(B, best)
            A.sort(key=lambda z: z.f)
            B.sort(key=lambda z: z.f)
            ne = min(self.n_exchange, len(A) // 2, len(B) // 2)      # 种群间信息交换
            if ne > 0:
                ea, eb = A[:ne], B[:ne]
                A = A[:len(A) - ne] + [Ind(z.os[:], z.ms[:], z.f) for z in eb]
                B = B[:len(B) - ne] + [Ind(z.os[:], z.ms[:], z.f) for z in ea]
            cur = min(A + B, key=lambda z: z.f)
            if cur.f < best.f:
                best = cur
            for ls in (self.ls_reassign, self.ls_advance):              # 知识驱动局部搜索
                if not self.budget_left():
                    break
                nb = ls(best)
                if nb.f < best.f - 1e-9:
                    best = nb
                    A.sort(key=lambda z: z.f)
                    A[-1] = Ind(nb.os[:], nb.ms[:], nb.f)                # 替换种群 A 的最差个体
            gen += 1
            self.history.append((gen, self.evals, best.f))
        self.time = time.time() - self.t0
        return best


class GA(KTPOA):
    def run(self):
        self.t0 = time.time()
        pop = [self.random_ind() for _ in range(self.pop_size)]
        for z in pop:
            self.f(z)
        gen = 0
        best = min(pop, key=lambda z: z.f)
        while self.budget_left():
            pop = self.evolve_A(pop)
            best = pop[0]
            gen += 1
            self.history.append((gen, self.evals, best.f))
        self.time = time.time() - self.t0
        return best


def make_policy(kind, max_evals=20000, pop_size=100, seed=0, **kw):
    """kind = 'GA' 或 'KTPOA'；返回可用于 core.rollout / core.evaluate 的策略"""
    cls = GA if kind == 'GA' else KTPOA
    cnt = [0]

    def solver(inst, st):
        cnt[0] += 1
        best = cls(inst, st, pop_size=pop_size, max_evals=max_evals, seed=seed + cnt[0], **kw).run()
        return best.os, best.ms
    return FixedPolicy(solver, kind)
