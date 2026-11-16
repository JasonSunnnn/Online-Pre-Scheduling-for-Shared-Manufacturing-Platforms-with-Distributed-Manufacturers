# -*- coding: utf-8 -*-
import csv
import os

import numpy as np

from . import core as K
from . import methods as M


def write_csv(path, header, rows):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:   # utf-8-sig：Windows 下 Excel 能正确显示 ±
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print('  saved', path)


def available(names, root=None, seed=0):
    """返回已经训练好（或不需要训练）的方法；缺模型的方法给出提示并跳过"""
    out = []
    for n in names:
        if n in K.COMBINED_RULES or n in M.META:
            out.append(n)
            continue
        f = os.path.join(M.run_dir(n, seed, root), 'best.npz' if n == 'D-Q' else 'best.pth')
        if os.path.exists(f):
            out.append(n)
        else:
            print('  [skip] %s is not trained yet (missing %s); run: python train.py --methods %s' % (n, f, n))
    return out


def pm(x, nd=2):
    x = np.asarray(x, dtype=np.float64)
    return ('%%.%df ± %%.%df' % (nd, nd)) % (x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0)


def table(names, costs, extra_cols=None, ref=None):
    """costs: {方法: 数组}。返回 (表头, 行)，末尾加 Avg. 行；给定 ref 时再加 Gap(%) 行（相对 ref 的平均值）"""
    n = len(costs[names[0]])
    header = ['Instance'] + (list(extra_cols.keys()) if extra_cols else []) + names
    rows = []
    for k in range(n):
        rows.append(['ins-%d' % (k + 1)] + ([extra_cols[c][k] for c in extra_cols] if extra_cols else []) +
                    ['%.2f' % costs[m][k] for m in names])
    pad = [''] * (len(extra_cols) if extra_cols else 0)
    rows.append(['Avg.'] + pad + ['%.2f' % costs[m].mean() for m in names])
    if ref is not None:
        rows.append(['Gap(%)'] + pad + ['%.2f' % (100.0 * (costs[m].mean() - costs[ref].mean()) / costs[ref].mean()) for m in names])
    return header, rows
