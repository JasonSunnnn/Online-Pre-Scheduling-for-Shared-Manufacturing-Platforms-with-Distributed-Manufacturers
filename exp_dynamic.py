# -*- coding: utf-8 -*-
"""
动态场景实验（论文 4.3 节，表9、表10、图9）。
  1) 10 个订单分 5 批到达的场景（订单的产品组成和到达时间 RO 与论文表9相同）
  2) 更大规模：D-30、D-50、D-100，每个场景 --n_seq 条随机到达序列（到达间隔服从均值 --mean_iat 小时的指数分布，
     每个订单 4-14 个产品批次），报告 mean ± std
对比方法：GIN-DPPO-E、MPDDQN、R-PPO、PBMP、D-Q、QIHGRL（缺模型的自动跳过）。
输出：
  results/table9.csv            各订单的订单完成时间（平台在订单到达时按悲观估计报出的完成时间）
  results/table9_realized.csv   各订单在插入式仿真（真实车间）中的实际完成时间
  results/table10.csv           各订单的平均车间空闲率（%）和总运输时间
  results/dynamic_summary.csv   D-30/50/100：三项指标的 mean ± std 以及平均每订单决策时间
  results/dynamic_detail.csv    D-30/50/100 每个订单一行
  results/fig9.png              订单 7 在其主要制造商车间内的甘特图（GIN-DPPO-E）；fig9_info.txt 说明画的是哪个制造商
  results/fig10.png             订单 7 各产品批次的生产进度图；order7_schedule.csv 为对应的时间数据
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # Windows 下 PyTorch 与 numpy/matplotlib/scipy 各带一份 OpenMP 库时避免 OMP Error #15
import argparse
import os

import numpy as np

from smp import case as C, core as K, methods as M, dynamic as D
from smp.util import write_csv, available, pm


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--methods', default='', help='逗号分隔；默认 GIN-DPPO-E 与各 DRL 对比方法；写 ablation 则为全部消融变体')
    p.add_argument('--scenarios', default='30,50,100')
    p.add_argument('--n_seq', type=int, default=10)
    p.add_argument('--mean_iat', type=float, default=6.0, help='D-30/50/100 的平均到达间隔（h）')
    p.add_argument('--out', default='results')
    a = p.parse_args()
    if a.methods == 'ablation':
        want = ['GIN-DPPO-E'] + [n for n in M.ABLATION_ALL[1:] if M.have(n)]
    elif a.methods:
        want = a.methods.split(',')
    else:
        want = ['GIN-DPPO-E'] + M.DRL_BASELINES + [n for n in M.OPTIONAL if os.path.exists(os.path.join(M.run_dir(n), 'best.pth'))]
    names = available(want)
    os.makedirs(a.out, exist_ok=True)
    pols = {n: M.load(n) for n in names}
    # ---------------- 10 个订单
    bg, orders = C.scenario10()
    res = {}
    for n in names:
        if n == 'GIN-DPPO-E':
            res[n], kept = D.run_sequence(pols[n], orders, bg, keep=7)
            for fn in (lambda: gantt(kept, os.path.join(a.out, 'fig9.png')),
                       lambda: gantt_order(kept, orders[6][0], os.path.join(a.out, 'fig10.png'), os.path.join(a.out, 'order7_schedule.csv'))):
                try:
                    fn()
                except Exception as e:
                    print('  (figure skipped: %s)' % e)
        else:
            res[n] = D.run_sequence(pols[n], orders, bg)
        print('%-12s 10 orders: completion %.2f  idle %.2f%%  transport %.1f' % (n, np.mean([r['completion'] for r in res[n]]),
              100 * np.mean([r['idle_rate'] for r in res[n]]), np.mean([r['transport'] for r in res[n]])), flush=True)
    head = ['Order', 'Required product', 'RO']
    desc = [[k + 1, ' '.join(str(x) for x in C.ORDERS10[k][0]), C.ORDERS10[k][1]] for k in range(len(orders))]
    for fn, key, fmt in (('table9.csv', 'completion', '%.2f'), ('table9_realized.csv', 'realized', '%.2f')):
        rows = [desc[k] + [fmt % res[n][k][key] for n in names] for k in range(len(orders))]
        rows.append(['Avg.', '', ''] + [fmt % np.mean([r[key] for r in res[n]]) for n in names])
        write_csv(os.path.join(a.out, fn), head + names, rows)
    rows = [desc[k] + ['%.2f' % (100 * res[n][k]['idle_rate']) for n in names] + ['%.1f' % res[n][k]['transport'] for n in names]
            for k in range(len(orders))]
    rows.append(['Avg.', '', ''] + ['%.2f' % (100 * np.mean([r['idle_rate'] for r in res[n]])) for n in names] +
                ['%.1f' % np.mean([r['transport'] for r in res[n]]) for n in names])
    write_csv(os.path.join(a.out, 'table10.csv'), head + ['idle rate %% | %s' % n for n in names] + ['transport | %s' % n for n in names], rows)
    # ---------------- D-30 / D-50 / D-100
    detail, summ, tests = [], [], []
    for n_orders in [int(x) for x in a.scenarios.split(',') if x]:
        agg = {n: [] for n in names}
        for q in range(a.n_seq):
            bg, orders = C.scenario_random(n_orders, C.SEED_DYN + 1000 * n_orders + q, a.mean_iat)
            for n in names:
                rows = D.run_sequence(pols[n], orders, bg)
                for r in rows:
                    detail.append(['D-%d' % n_orders, q, n, r['order'], '%.3f' % r['RO'], r['nb'], '%.3f' % r['completion'],
                                   '%.3f' % r['realized'], '%.5f' % r['idle_rate'], '%.2f' % r['transport'], '%.4f' % r['time']])
                agg[n].append([np.mean([r[k] for r in rows]) for k in ('completion', 'realized', 'idle_rate', 'transport', 'time')])
            print('D-%d sequence %d done' % (n_orders, q), flush=True)
        ref = names[0]
        for n in names:
            v = np.array(agg[n])
            if n != ref:                # 与第一个方法（默认 GIN-DPPO-E）按序列配对比较
                d = np.array(agg[ref])[:, 0] - v[:, 0]
                pv = ''
                try:
                    from scipy import stats
                    pv = '%.4f' % stats.wilcoxon(np.array(agg[ref])[:, 0], v[:, 0]).pvalue if np.any(np.abs(d) > 1e-12) else '1.0000'
                except Exception:
                    pass
                tests.append(['D-%d' % n_orders, ref, n, '%.3f' % np.array(agg[ref])[:, 0].mean(), '%.3f' % v[:, 0].mean(),
                              '%.2f' % (100.0 * (np.array(agg[ref])[:, 0].mean() - v[:, 0].mean()) / v[:, 0].mean()),
                              '%d / %d' % (int((d < -1e-9).sum()), len(d)), pv])
            summ.append(['D-%d' % n_orders, n, pm(v[:, 0]), pm(v[:, 1]), pm(100 * v[:, 2]), pm(v[:, 3], 1), '%.4f' % v[:, 4].mean()])
            print('D-%-4d %-12s completion %s | idle %s %% | transport %s | %.3f s/order'
                  % (n_orders, n, pm(v[:, 0]), pm(100 * v[:, 2]), pm(v[:, 3], 1), v[:, 4].mean()))
    if summ:
        write_csv(os.path.join(a.out, 'dynamic_summary.csv'), ['scenario', 'method', 'completion (mean ± std)', 'realized completion',
                                                               'idle rate % (mean ± std)', 'transport (mean ± std)',
                                                               'decision time per order (s)'], summ)
        if tests:
            write_csv(os.path.join(a.out, 'dynamic_tests.csv'), ['scenario', 'reference', 'method', 'reference mean completion',
                                                                'method mean completion', 'difference of reference (%)',
                                                                'sequences where reference is better', 'Wilcoxon p (paired by sequence)'], tests)
        write_csv(os.path.join(a.out, 'dynamic_detail.csv'), ['scenario', 'sequence', 'method', 'order', 'RO', 'batches', 'completion',
                                                              'realized', 'idle_rate', 'transport', 'time_s'], detail)


if __name__ == '__main__':
    main()
