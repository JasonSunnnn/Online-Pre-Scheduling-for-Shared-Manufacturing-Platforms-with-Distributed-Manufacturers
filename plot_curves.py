# -*- coding: utf-8 -*-
"""图7：GIN-DPPO-E 的训练曲线（损失、clip fraction、训练奖励、验证分数）。读 runs/GIN-DPPO-E/train_log.csv，
输出 results/fig7.png；另画各消融变体的验证曲线 results/fig_ablation_curves.png。"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # Windows 下 PyTorch 与 numpy/matplotlib/scipy 各带一份 OpenMP 库时避免 OMP Error #15
import argparse
import os

import numpy as np

from smp import methods as M


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', default='GIN-DPPO-E')
    p.add_argument('--out', default='results')
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(M.run_dir(a.method), 'train_log.csv')
    log = np.loadtxt(f, delimiter=',', skiprows=1, ndmin=2)
    ep = log[:, 0]
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.5))
    for x, col, name in ((ax[0, 0], 1, 'Loss'), (ax[0, 1], 6, 'Clip fraction'), (ax[1, 0], 7, 'Training reward'),
                         (ax[1, 1], 10, 'Validation score (sum over 300 instances)')):
        x.plot(ep, log[:, col], lw=1.2)
        x.set_xlabel('Epoch'); x.set_ylabel(name); x.grid(alpha=0.3)
    k = int(np.argmin(log[:, 9]))
    ax[1, 1].plot(ep[k], log[k, 10], 'r*', ms=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'fig7.png'), dpi=300); plt.close(fig)
    print('saved', os.path.join(a.out, 'fig7.png'))
    print('best validation: epoch %d, mean %.3f, sum %.1f; last epoch %d, mean %.3f' % (ep[k], log[k, 9], log[k, 10], ep[-1], log[-1, 9]))
    fig, x = plt.subplots(figsize=(7, 4.5))
    n = 0
    for name in M.ABLATION_ALL:
        f = os.path.join(M.run_dir(name), 'train_log.csv')
        if os.path.exists(f):
            lg = np.loadtxt(f, delimiter=',', skiprows=1, ndmin=2)
            x.plot(lg[:, 0], lg[:, 9], lw=1.0, label=name)
            n += 1
    if n > 1:
        x.set_xlabel('Epoch'); x.set_ylabel('Validation score (mean order completion time)'); x.legend(fontsize=7); x.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, 'fig_ablation_curves.png'), dpi=300)
        print('saved', os.path.join(a.out, 'fig_ablation_curves.png'))
    plt.close(fig)


if __name__ == '__main__':
    main()
