# -*- coding: utf-8 -*-
"""
只重画图9、图10（10 订单场景中的订单 7），不重跑 D-30/50/100，约 1 分钟：

    python fig_order7.py

输出（都在 results/ 下）：
  fig9.png             订单 7 主要制造商车间内的甘特图（灰色 = 其他订单的任务）
  fig9_info.txt        图9 画的是哪个制造商，以及该车间每台机器上每段作业的时间和归属
  fig10.png            订单 7 各产品批次的生产进度（车间内加工、制造商间运输、配送）
  order7_schedule.csv  图10 的时间数据
用的是 runs/GIN-DPPO-E 里训练好的模型。
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import argparse

import numpy as np

from smp import case as C, core as K, methods as M, dynamic as D
from smp.util import write_csv


def gantt(kept, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cnt = {}
    for (r, c, s, ready, end, ops) in kept['rec']:
        cnt[s] = cnt.get(s, 0) + 1
    s = max(cnt, key=cnt.get)
    mine = {}
    for (r, c, sh, ready, end, ops) in kept['rec']:
        if sh == s:
            for (m, a, b) in ops:
                mine[(m, round(a, 6), round(b, 6))] = r
    lo = min(a for (m, a, b) in mine)
    hi = max(b for (m, a, b) in mine)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cmap = plt.get_cmap('tab20')
    for m, ivs in enumerate(kept['after'][s]):
        for (a, b) in ivs:
            if b < lo - 5 or a > hi + 5:
                continue
            r = mine.get((m, round(a, 6), round(b, 6)))
            ax.barh(m, b - a, left=a, height=0.7, color='#bdbdbd' if r is None else cmap(r % 20), edgecolor='k', linewidth=0.3)
            if r is not None:
                ax.text((a + b) / 2, m, 'J%d' % (r + 1), ha='center', va='center', fontsize=6)
    ax.set_yticks(range(len(kept['after'][s])))
    ax.set_yticklabels(['M%d' % (m + 1) for m in range(len(kept['after'][s]))], fontsize=7)
    ax.set_xlabel('Time (h)')
    ax.set_title('Manufacturer S%d (grey: tasks of other orders)' % (s + 1), fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    print('  saved', path, '(manufacturer index %d)' % (s + 1))
    with open(path[:-4] + '_info.txt', 'w') as f:        # 图9 画的是哪个制造商、其中哪些产品批次属于订单 7
        f.write('manufacturer S%d\n' % (s + 1))
        for (r, c, sh, ready, end, ops) in sorted(kept['rec']):
            if sh == s:
                f.write('J%d = task %d of product batch %d\n' % (r + 1, c + 1, r + 1))


def gantt_order(kept, ins, path, csv_path):
    """图10：订单 7 各产品批次的生产进度（以订单为中心）。横条 = 任务在制造商车间内的时间（从首道工序开工到末道工序完工），
    细条 = 制造商之间的运输和最终配送；同时把每个任务的时间写入 csv。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    inst = K.Inst(ins)
    info, rows = {}, []
    for (r, c, s, ready, end, ops) in kept['rec']:
        st, fin = min(a for (m, a, b) in ops), max(b for (m, a, b) in ops)
        info[(r, c)] = (s, ready, st, fin, end)
        rows.append([r + 1, ins['products'][r] + 1, c + 1, s + 1, '%.2f' % ready, '%.2f' % st, '%.2f' % fin, '%.2f' % end])
    rows.sort(key=lambda x: (x[0], x[2]))
    write_csv(csv_path, ['product batch', 'product type', 'task', 'manufacturer', 'arrival at manufacturer', 'start of first operation',
                         'end of last operation', 'completion (incl. delivery for the last task)'], rows)
    cmap = plt.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(9, 0.42 * inst.nb + 1.6))
    used = set()
    # 任务 1→2（PCB、SMT）与任务 3（机加工）并行进行，分别画在该批次的上、下半行；任务 4、5 占整行
    lane = lambda r, c: (r - 0.18, 0.32) if c in (0, 1) else ((r + 0.18, 0.32) if c == 2 else (r, 0.68))
    for (r, c), (s, ready, st, fin, end) in info.items():
        y, h = lane(r, c)
        ax.barh(y, fin - st, left=st, height=h, color=cmap(s % 20), edgecolor='k', linewidth=0.3)
        ax.text((st + fin) / 2, y, 'p%d,%d' % (r + 1, c + 1), ha='center', va='center', fontsize=4.5)
        used.add(s)
        for q in inst.preds[r][c]:                       # 前序任务所在制造商到本制造商的运输
            sq, _, _, fq, _ = info[(r, q)]
            if inst.ss[sq][s] > 0:
                ax.barh(lane(r, q)[0], inst.ss[sq][s], left=fq, height=0.1, color='#555555')
        if c == inst.nsub[r] - 1 and end > fin:          # 配送到客户
            ax.barh(y, end - fin, left=fin, height=0.1, color='k', hatch='////', fill=False, linewidth=0.5)
    ax.axvline(kept['RO'], color='k', ls=':', lw=0.8)
    ax.set_yticks(range(inst.nb))
    ax.set_yticklabels(['Batch %d (P%d)' % (r + 1, ins['products'][r] + 1) for r in range(inst.nb)], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Time (h)')
    hs = [Patch(facecolor=cmap(s % 20), edgecolor='k', label='S%d' % (s + 1)) for s in sorted(used)]
    hs += [Patch(facecolor='#555555', label='Transportation between manufacturers'), Patch(facecolor='w', edgecolor='k', hatch='////', label='Delivery to customer')]
    ax.legend(handles=hs, fontsize=6, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print('  saved', path, 'and', csv_path)


def detail(kept, ins, path):
    """把图9所画车间每台机器上的作业时间写进 fig9_info.txt（便于在正文中准确描述图9）"""
    cnt = {}
    for (r, c, s, ready, end, ops) in kept['rec']:
        cnt[s] = cnt.get(s, 0) + 1
    s = max(cnt, key=cnt.get)
    mine = {}
    for (r, c, sh, ready, end, ops) in kept['rec']:
        if sh == s:
            for (m, a, b) in ops:
                mine[(m, round(a, 6), round(b, 6))] = (r, c)
    with open(path, 'a') as f:
        f.write('\nRO of order 7 = %.2f h; tasks of order 7 at S%d: %d of %d\n' % (kept['RO'], s + 1, cnt[s], len(kept['rec'])))
        f.write('machine, start, end, owner (Jr.c = task c of product batch r of order 7; other = other orders / background)\n')
        for m, ivs in enumerate(kept['after'][s]):
            for (a, b) in ivs:
                o = mine.get((m, round(a, 6), round(b, 6)))
                f.write('M%d, %.2f, %.2f, %s\n' % (m + 1, a, b, 'J%d.%d' % (o[0] + 1, o[1] + 1) if o else 'other'))
        bef = kept['before'][s]
        f.write('\nintervals on S%d before order 7 was scheduled: %d; after: %d\n'
                % (s + 1, sum(len(x) for x in bef), sum(len(x) for x in kept['after'][s])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', default='GIN-DPPO-E')
    p.add_argument('--out', default='results')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bg, orders = C.scenario10()
    pol = M.load(a.method)
    rows, kept = D.run_sequence(pol, orders, bg, keep=7)
    print('order completion times:', ' '.join('%.2f' % r['completion'] for r in rows))
    gantt(kept, os.path.join(a.out, 'fig9.png'))
    detail(kept, orders[6][0], os.path.join(a.out, 'fig9_info.txt'))
    gantt_order(kept, orders[6][0], os.path.join(a.out, 'fig10.png'), os.path.join(a.out, 'order7_schedule.csv'))


if __name__ == '__main__':
    main()
