# -*- coding: utf-8 -*-
"""
模型验证（论文 2.3.2 节 “Model validation”）。
  1) 制造商车间重调度模型 (10)-(19)：小规模车间算例用 MILP（HiGHS 求解器）求最优 makespan，
     与 Rules 1-3 生成的调度比较（并检查 Rules 1-3 的调度满足全部约束）
  2) 平台派工模型 (1)-(9)：小规模订单在悲观估计去模糊化后用分支定界求最优订单完成时间，
     与 GIN-DPPO-E（及 KTPOA、最好的调度规则）比较
输出 results/validation_shop.csv、results/validation_order.csv、results/validation_summary.csv
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # Windows 下 PyTorch 与 numpy/matplotlib/scipy 各带一份 OpenMP 库时避免 OMP Error #15
import argparse
import os

import numpy as np

from smp import case as C, core as K, methods as M, exact
from smp.util import write_csv, available


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shop_jobs', default='4,6,8')
    p.add_argument('--shop_rep', type=int, default=5)
    p.add_argument('--time_limit', type=float, default=120.0)
    p.add_argument('--order_nb', default='2,3,4,5')
    p.add_argument('--order_rep', type=int, default=10)
    p.add_argument('--node_limit', type=int, default=3000000)
    p.add_argument('--out', default='results')
    a = p.parse_args()
    rng = np.random.RandomState(99)
    rows, gaps = [], []
    try:
        from scipy.optimize import milp  # noqa: F401
        sizes = [int(x) for x in a.shop_jobs.split(',')]
    except Exception as e:
        print('[skip] the shop MILP needs scipy >= 1.9 (pip install -U scipy): %s' % e)
        sizes = []
    for s in (0, 3, 6, 10):                         # 四种服务类型各取一个制造商
        for n in sizes:
            for k in range(a.shop_rep):
                jobs = exact.shop_jobs(rng, s, n)
                mk, sched = exact.shop_rules(s, jobs)
                exact.check_shop_schedule(s, jobs, sched)
                opt, ok, t = exact.shop_milp(s, jobs, a.time_limit)
                gap = 100.0 * (mk - opt) / opt
                rows.append(['S%d' % (s + 1), C.SHOP_TYPE[s], n, sum(len(C.OPS[p_][c_]) for p_, c_ in jobs), '%.2f' % opt,
                             'yes' if ok else 'no (time limit)', '%.2f' % mk, '%.2f' % gap, '%.1f' % t])
                if ok:
                    gaps.append(gap)
                print('shop S%d jobs %d: MILP %.2f (%s), Rules 1-3 %.2f, gap %.2f%%' % (s + 1, n, opt, 'optimal' if ok else 'limit', mk, gap),
                      flush=True)
    write_csv(os.path.join(a.out, 'validation_shop.csv'), ['manufacturer', 'service type', 'jobs', 'operations', 'MILP makespan',
                                                           'proved optimal', 'Rules 1-3 makespan', 'gap(%)', 'MILP time(s)'], rows)
    summary = [['shop model: instances solved to optimality', '%d / %d' % (len(gaps), len(rows))],
               ['shop model: mean gap of Rules 1-3 to the optimum (%)', '%.2f' % (np.mean(gaps) if gaps else float('nan'))],
               ['shop model: max gap (%)', '%.2f' % (np.max(gaps) if gaps else float('nan'))],
               ['shop model: instances where Rules 1-3 are optimal', '%d' % sum(1 for g in gaps if g < 1e-6)]]
    # ---------------- platform model
    pols = {n: M.load(n) for n in available(['GIN-DPPO-E'])}
    pols['KTPOA'] = M.load('KTPOA', max_evals=5000)
    rows, g = [], {n: [] for n in list(pols) + ['best rule']}
    for nb in [int(x) for x in a.order_nb.split(',')]:
        for k in range(a.order_rep):
            ins = C.gen_instance(rng, nb=nb)
            inst = K.Inst(ins)
            opt, ok, nodes = exact.order_optimum(inst, node_limit=a.node_limit)
            vals = {n: float(M.solve(pols[n], [ins])[0][0]) for n in pols}
            vals['best rule'] = min(K.construct(inst, K.St.initial(inst), *K.COMBINED_RULES[r])[2] for r in K.RULE_NAMES)
            opt = min([opt] + list(vals.values()))
            rows.append([nb, inst.n_real, '%.2f' % opt, 'yes' if ok else 'no (node limit)', nodes] + ['%.2f' % vals[n] for n in g])
            if ok:
                for n in g:
                    g[n].append(100.0 * (vals[n] - opt) / opt)
            print('order nb=%d tasks=%d: optimum %.2f (%s) %s' % (nb, inst.n_real, opt, 'proved' if ok else 'limit',
                  {n: round(v, 2) for n, v in vals.items()}), flush=True)
    write_csv(os.path.join(a.out, 'validation_order.csv'), ['batches', 'tasks', 'optimum', 'proved optimal', 'B&B nodes'] + list(g), rows)
    for n in g:
        if g[n]:
            summary.append(['platform model: mean gap of %s to the optimum (%%), %d proved instances' % (n, len(g[n])), '%.2f' % np.mean(g[n])])
            summary.append(['platform model: instances where %s is optimal' % n, '%d' % sum(1 for x in g[n] if x < 1e-6)])
    write_csv(os.path.join(a.out, 'validation_summary.csv'), ['item', 'value'], summary)


if __name__ == '__main__':
    main()
