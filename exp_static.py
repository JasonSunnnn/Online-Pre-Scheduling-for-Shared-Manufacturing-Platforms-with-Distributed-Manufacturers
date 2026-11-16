# -*- coding: utf-8 -*-
"""
静态算例实验（16 个测试算例 ins-1..ins-16）：
    results/table6.csv  消融：GIN-DPPO-E 及 7 个变体
    results/table7.csv  GIN-DPPO-E、GA、KTPOA、8 个组合调度规则
    results/table8.csv  GIN-DPPO-E 与 MPDDQN、R-PPO、PBMP、D-Q、QIHGRL
    results/static_time.csv        各方法平均每个算例的求解时间
    results/static_realized.csv    各方法的调度在插入式仿真（真实车间）下的订单完成时间
    results/checkpoints.csv        GIN-DPPO-E 最优验证轮次的模型 与 最后一轮模型 的对比；多随机种子的最优验证分数
表中数值 = 悲观估计下的订单完成时间（h），与训练/验证口径一致。
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # Windows 下 PyTorch 与 numpy/matplotlib/scipy 各带一份 OpenMP 库时避免 OMP Error #15
import argparse
import os

import numpy as np

from smp import case as C, core as K, methods as M
from smp.util import write_csv, available, table, pm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evals', type=int, default=20000, help='GA 和 KTPOA 的目标函数评价次数（= 种群 100 × 200 代）')
    p.add_argument('--pop', type=int, default=100)
    p.add_argument('--skip_meta', action='store_true')
    p.add_argument('--out', default='results')
    a = p.parse_args()
    test = C.test_set()
    nb = [len(x['products']) for x in test]
    opt = [n for n in M.OPTIONAL if os.path.exists(os.path.join(M.run_dir(n), 'best.pth'))]
    abl = available(['GIN-DPPO-E']) + [n for n in M.ABLATION_ALL[1:] if M.have(n)]
    names = abl + available(M.DRL_BASELINES + opt + ([] if a.skip_meta else M.META) + K.RULE_NAMES)
    cost, real, tm = {}, {}, {}
    for n in names:
        pol = M.load(n, max_evals=a.evals, pop_size=a.pop) if n in M.META else M.load(n)
        if n not in M.META and n not in K.COMBINED_RULES:
            M.solve(pol, test[:2])                          # 预热（首次调用含 CUDA 初始化，不计入求解时间）
        cost[n], decs, tm[n] = M.solve(pol, test)
        real[n] = np.array([K.replay(K.Inst(x), d, 'sim')[0] for x, d in zip(test, decs)])
        if n in M.META:
            tm[n] = float(np.mean(pol.solve_times))
        print('%-16s mean %.3f   %.4f s/instance' % (n, cost[n].mean(), tm[n]), flush=True)
    ex = {'Batches per order': nb}
    ref = 'GIN-DPPO-E' if 'GIN-DPPO-E' in cost else None
    for fn, group in (('table6.csv', abl), ('table7.csv', ['GIN-DPPO-E'] + M.META + K.RULE_NAMES),
                      ('table8.csv', ['GIN-DPPO-E'] + M.DRL_BASELINES + opt)):
        g = [n for n in group if n in cost]
        if g:
            write_csv(os.path.join(a.out, fn), *table(g, cost, ex, ref))
    if cost:
        write_csv(os.path.join(a.out, 'static_realized.csv'), *table(list(cost), real, ex, ref))
        write_csv(os.path.join(a.out, 'static_time.csv'), ['method', 'mean_cost', 'time_per_instance_s'],
                  [[n, '%.3f' % cost[n].mean(), '%.4f' % tm[n]] for n in cost])
    rules = [n for n in K.RULE_NAMES if n in cost]
    if ref and rules:
        best_rule = np.min(np.stack([cost[n] for n in rules]), 0)
        print('GIN-DPPO-E relative to the best rule of each instance: %+.2f%% (negative = GIN-DPPO-E is better)'
              % (100.0 * np.mean((cost[ref] - best_rule) / best_rule)))
    for est in ('nowait', 'load'):
        la = K.evaluate(K.LookAheadPolicy(est), test)[0]
        print('reference only (not a method of the paper): greedy look-ahead rule (%s estimate), mean %.3f' % (est, la.mean()))
    # 最优验证轮次 vs 最后一轮；多随机种子
    rows = []
    if ref and os.path.exists(os.path.join(M.run_dir(ref), 'last.pth')):
        import torch
        for which in ('best', 'last'):
            ck = torch.load(os.path.join(M.run_dir(ref), which + '.pth'), map_location='cpu')
            c, _, _ = M.solve(M.load(ref, which=which), test)
            rows.append(['seed 0, %s checkpoint (epoch %d)' % (which, ck['epoch']), '%.3f' % c.mean()])
    vals = []
    for seed in range(0, 20):
        f = os.path.join(M.run_dir('GIN-DPPO-E', seed), 'train_log.csv')
        if os.path.exists(f):
            log = np.loadtxt(f, delimiter=',', skiprows=1, ndmin=2)
            vals.append(log[:, 9].min())
            rows.append(['seed %d best validation mean (sum over 300 = %.1f)' % (seed, 300 * vals[-1]), '%.3f' % vals[-1]])
    if len(vals) > 1:
        rows.append(['best validation mean over %d seeds' % len(vals), pm(vals, 3)])
    if rows:
        write_csv(os.path.join(a.out, 'checkpoints.csv'), ['item', 'value'], rows)


if __name__ == '__main__':
    main()
