# -*- coding: utf-8 -*-
"""
图8：计算时间随同时调度的订单数的变化（GIN-DPPO-E、GA、KTPOA、调度规则）；
以及 GIN 编码器的计算开销：每步编码器前向时间、占决策时间的比例、
GIN-DPPO-E 与 DPPO-E 的每订单决策时间和训练时间。
GA/KTPOA 默认只实际求解 --meta_sample 个订单，再按订单数线性外推（各订单相互独立，耗时与订单数成正比）；
加 --meta_full 则全部实际求解。输出 results/fig8.csv、results/fig8.png、results/gin_cost.csv
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # Windows 下 PyTorch 与 numpy/matplotlib/scipy 各带一份 OpenMP 库时避免 OMP Error #15
import argparse
import os
import time

import numpy as np

from smp import case as C, core as K, methods as M
from smp.util import write_csv, available


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sizes', default='50,100,150,200,250,300')
    p.add_argument('--evals', type=int, default=20000)
    p.add_argument('--meta_sample', type=int, default=5)
    p.add_argument('--meta_full', action='store_true')
    p.add_argument('--batch', type=int, default=100)
    p.add_argument('--out', default='results')
    a = p.parse_args()
    sizes = [int(x) for x in a.sizes.split(',')]
    ins = C.gen_dataset(max(sizes), 777)
    names = available(['GIN-DPPO-E']) + ['GA', 'KTPOA', 'FEET']
    T = {n: [] for n in names}
    per = {}
    for n in names:
        if n in M.META:
            sub = ins[:max(sizes)] if a.meta_full else ins[:a.meta_sample]
            pol = M.load(n, max_evals=a.evals)
            K.evaluate(pol, sub)
            per[n] = float(np.mean(pol.solve_times))
            T[n] = [per[n] * s for s in sizes]
        else:
            pol = M.load(n)
            K.evaluate(pol, ins[:a.batch], getattr(pol, 'mode', 'pess'), a.batch)   # 预热（CUDA 初始化不计入）
            for s in sizes:
                t0 = time.time()
                K.evaluate(pol, ins[:s], getattr(pol, 'mode', 'pess'), a.batch)
                T[n].append(time.time() - t0)
        print('%-12s %s' % (n, ['%.2f' % t for t in T[n]]), flush=True)
    write_csv(os.path.join(a.out, 'fig8.csv'), ['orders'] + names, [[s] + ['%.3f' % T[n][k] for n in names] for k, s in enumerate(sizes)])
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for n, mk in zip(names, ['o-', 's-', '^-', 'd-']):
            ax.plot(sizes, T[n], mk, label='Dispatching rule' if n == 'FEET' else n)
        ax.set_yscale('log'); ax.set_xlabel('Number of orders'); ax.set_ylabel('Computation time (s)'); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(a.out, 'fig8.png'), dpi=300); plt.close(fig)
        print('  saved', os.path.join(a.out, 'fig8.png'))
    except Exception as e:
        print('  (figure skipped: %s)' % e)
    # GIN 编码器开销
    rows = []
    sub = ins[:100]
    for n in [x for x in ('GIN-DPPO-E', 'DPPO-E', 'GIN-PPO-E', 'GIN-DPPO-E-NL', 'DPPO-E-NL', 'GIN-PPO-E-NL') if M.have(x)]:
        pol = M.load(n)
        K.evaluate(pol, sub[:5], batch=1)                  # 预热
        pol.n_dec, pol.t_dec, pol.t_enc = 0, 0.0, 0.0
        t0 = time.time()
        K.evaluate(pol, sub, batch=1)                      # 逐个订单在线调度
        tot = time.time() - t0
        tt = ''
        f = os.path.join(M.run_dir(n), 'train_time.txt')
        if os.path.exists(f):
            tt = '%.2f' % (sum(float(l.split()[0]) for l in open(f) if l.strip()) / 3600.0)
        else:                                              # 训练被手动中断时没有 train_time.txt：用 train_log.csv 各轮耗时之和
            f = os.path.join(M.run_dir(n), 'train_log.csv')
            if os.path.exists(f):
                tt = '%.2f' % (np.loadtxt(f, delimiter=',', skiprows=1, ndmin=2)[:, 12].sum() / 3600.0)
        te = ''
        f = os.path.join(M.run_dir(n), 'train_log.csv')
        if os.path.exists(f):
            lg = np.loadtxt(f, delimiter=',', skiprows=1, ndmin=2)
            te = '%.2f (%d epochs)' % (lg[:, 12].mean() / 60.0, len(lg))
        rows.append([n, '%.3f' % (1000.0 * pol.t_enc / pol.n_dec), '%.3f' % (1000.0 * tot / pol.n_dec),
                     '%.1f' % (100.0 * pol.t_enc / tot), '%.4f' % (tot / len(sub)), tt, te])
    if rows:
        write_csv(os.path.join(a.out, 'gin_cost.csv'), ['method', 'encoder_forward_ms_per_step', 'decision_ms_per_step',
                                                        'encoder_share(%)', 'decision_time_per_order_s', 'training_time_h',
                                                        'training_time_per_epoch_min'], rows)


if __name__ == '__main__':
    main()
