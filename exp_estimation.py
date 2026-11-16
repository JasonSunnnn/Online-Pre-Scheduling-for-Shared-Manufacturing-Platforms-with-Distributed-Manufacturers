# -*- coding: utf-8 -*-
"""
图6：三种订单完成时间计算方式的精度与耗时（300 个验证算例）。
以插入式仿真（Rules 1-3）为基准，比较期望值估计和悲观估计的相对偏差，以及三者的平均计算时间。
调度方案取自训练好的 GIN-DPPO-E（没有训练好的模型时用 WEET 规则）。
输出 results/fig6.csv、results/fig6_summary.csv、results/fig6.png
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
    p.add_argument('--repeat', type=int, default=20)
    p.add_argument('--out', default='results')
    a = p.parse_args()
    ins = C.gen_dataset(300, C.SEED_VALID)
    name = 'GIN-DPPO-E' if available(['GIN-DPPO-E']) else 'WEET'
    _, decs, _ = M.solve(M.load(name), ins)
    insts = [K.Inst(x) for x in ins]
    cost, tm = {}, {}
    for mode in ('sim', 'exp', 'pess'):
        t0 = time.time()
        for _ in range(a.repeat):
            c = [K.replay(i, d, mode)[0] for i, d in zip(insts, decs)]
        tm[mode] = (time.time() - t0) / (a.repeat * len(insts))
        cost[mode] = np.array(c)
    dev_e = (cost['exp'] - cost['sim']) / cost['sim'] * 100.0
    dev_p = (cost['pess'] - cost['sim']) / cost['sim'] * 100.0
    write_csv(os.path.join(a.out, 'fig6.csv'), ['instance', 'batches', 'simulation', 'expected', 'pessimistic', 'dev_expected(%)',
                                                'dev_pessimistic(%)'],
              [[k + 1, insts[k].nb, '%.3f' % cost['sim'][k], '%.3f' % cost['exp'][k], '%.3f' % cost['pess'][k], '%.3f' % dev_e[k],
                '%.3f' % dev_p[k]] for k in range(len(insts))])
    write_csv(os.path.join(a.out, 'fig6_summary.csv'), ['item', 'insertion-based simulation', 'expected value', 'pessimistic'],
              [['schedules from', name, '', ''],
               ['mean relative deviation (%)', 0, '%.2f' % dev_e.mean(), '%.2f' % dev_p.mean()],
               ['mean absolute relative deviation (%)', 0, '%.2f' % np.abs(dev_e).mean(), '%.2f' % np.abs(dev_p).mean()],
               ['time per instance (s)', '%.5f' % tm['sim'], '%.5f' % tm['exp'], '%.5f' % tm['pess']]])
    print('deviation: expected %.2f%%, pessimistic %.2f%%; time sim %.5f exp %.5f pess %.5f s/instance'
          % (dev_e.mean(), dev_p.mean(), tm['sim'], tm['exp'], tm['pess']))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        x = np.arange(1, len(insts) + 1)
        ax[0].plot(x, dev_e, lw=0.8, label='Expected value estimation')
        ax[0].plot(x, dev_p, lw=0.8, label='Pessimistic estimation')
        ax[0].axhline(0, color='k', lw=0.5)
        ax[0].set_xlabel('Instance'); ax[0].set_ylabel('Relative deviation (%)'); ax[0].legend()
        ax[1].bar(['Insertion-based simulation', 'Expected value estimation', 'Pessimistic estimation'],
                  [tm['sim'], tm['exp'], tm['pess']], color=['#7f7f7f', '#1f77b4', '#ff7f0e'], width=0.5)
        ax[1].set_ylabel('Time (s/instance)')
        fig.tight_layout(); fig.savefig(os.path.join(a.out, 'fig6.png'), dpi=300); plt.close(fig)
        print('  saved', os.path.join(a.out, 'fig6.png'))
    except Exception as e:
        print('  (figure skipped: %s)' % e)


if __name__ == '__main__':
    main()
