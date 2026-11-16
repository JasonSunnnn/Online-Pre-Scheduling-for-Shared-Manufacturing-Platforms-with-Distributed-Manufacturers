# -*- coding: utf-8 -*-
"""
论文图6、图9、图10 的绘图（图7、图8 仍用 plot_curves.py、exp_time.py 的原图）。
图9、图10 沿用原图的配色方式（图10 按制造商着色，图9 按产品批次着色，20 色成对调色板），只是换成饱和度更均衡的 Tableau 20 配色。
只读结果文件、不需要模型，也不重新计算任何数据：

    python plot_paper_figs.py                     # 读 results/，图写回 results/（覆盖 fig6.png、fig9.png、fig10.png）
    python plot_paper_figs.py --out figs          # 写到别的文件夹
    python plot_paper_figs.py --figs 9,10         # 只画部分图

需要的文件：results/fig6.csv、results/fig6_summary.csv、results/fig9_info.txt、results/order7_schedule.csv。三张图宽度、字号相同，插入 Word 后字号一致。
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import argparse
import csv
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ---------------------------------------------------------------- palette
TEAL, OLIVE = '#1F8FB8', '#6A9A2E'                                     # lines and bars of Fig. 6
GREY, GREY_L, GREY_D, INK = '#8C8C8C', '#D9D9D9', '#4D4D4D', '#262626'  # baseline, other orders, transport/axes, text
# Figs. 9 and 10 keep the colour assignment of the original figures (20 paired colours as in matplotlib 'tab20':
# manufacturer S_k -> colour k-1 in Fig. 10, product batch r -> colour r-1 in Fig. 9), with the saturation toned down:
#   'soft'    : tab20 itself, HSL saturation of the strong colours capped (dark members <= 0.50, light members <= 0.75)
#   'tableau' : the re-balanced Tableau 20 palette (same colour pairs as tab20, muted and balanced saturation)
PALETTES = {
    'soft': ['#408BBF', '#AEC7E8', '#C3834A', '#EEBB89', '#40BF40', '#98DF8A', '#BF4040', '#F2A5A3', '#9467BD', '#C5B0D5',
             '#955C50', '#C49C94', '#D684BD', '#F5B8D2', '#7F7F7F', '#C7C7C7', '#BEBF40', '#DBDB8D', '#40B3BF', '#9EDAE5'],
    'tableau': ['#4E79A7', '#A0CBE8', '#F28E2B', '#FFBE7D', '#59A14F', '#8CD17D', '#E15759', '#FF9D9A', '#B07AA1', '#D4A6C8',
                '#9D7660', '#D7B5A6', '#D37295', '#FABFD2', '#79706E', '#BAB0AC', '#B6992D', '#F1CE63', '#499894', '#86BCB6']}
PAL = PALETTES['tableau']
EDGE = '#404040'
WIDTH = 7.0


class C:
    """Case data needed for Figs. 9 and 10 (as published in the paper): service type of manufacturers S1..S12 (Table 3),
    predecessors of tasks 1..5 of product types P1..P7 (Fig. 5, 0-based task indices) and transportation time (h) between
    the manufacturers (Table 4 and the manufacturer locations in Table 3)."""
    SHOP_TYPE = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4]
    PREDS = [[[], [0], [], [1, 2], [3]], [[], [0], [], [1, 2]], [[], [0], [], [1, 2]], [[], [0], [], [1, 2]], [[], [0], [], [1, 2]], [[], [0], [], [1, 2], [3]], [[], [0], [], [1, 2]]]
    SS = [[0.0, 2.0, 12.0, 12.0, 23.0, 2.0, 2.0, 2.0, 23.0, 25.0, 1.0, 12.0], [2.0, 0.0, 24.0, 24.0, 20.0, 1.0, 1.0, 1.0, 20.0, 24.0, 2.0, 24.0], [12.0, 24.0, 0.0, 1.0, 1.0, 25.0, 24.0, 24.0, 1.0, 2.5, 12.0, 1.0], [12.0, 24.0, 1.0, 0.0, 1.0, 25.0, 24.0, 24.0, 1.0, 2.5, 12.0, 1.0], [23.0, 20.0, 1.0, 1.0, 0.0, 20.0, 20.0, 20.0, 1.0, 2.0, 23.0, 1.0], [2.0, 1.0, 25.0, 25.0, 20.0, 0.0, 1.0, 1.0, 20.0, 24.0, 2.0, 25.0], [2.0, 1.0, 24.0, 24.0, 20.0, 1.0, 0.0, 1.0, 20.0, 24.0, 2.0, 24.0], [2.0, 1.0, 24.0, 24.0, 20.0, 1.0, 1.0, 0.0, 20.0, 24.0, 2.0, 24.0], [23.0, 20.0, 1.0, 1.0, 1.0, 20.0, 20.0, 20.0, 0.0, 2.0, 23.0, 1.0], [25.0, 24.0, 2.5, 2.5, 2.0, 24.0, 24.0, 24.0, 2.0, 0.0, 25.0, 2.5], [1.0, 2.0, 12.0, 12.0, 23.0, 2.0, 2.0, 2.0, 23.0, 25.0, 0.0, 12.0], [12.0, 24.0, 1.0, 1.0, 1.0, 25.0, 24.0, 24.0, 1.0, 2.5, 12.0, 0.0]]



def _lum(hx):
    c = [int(hx[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def label_colour(fill):
    """black or white text, whichever has the higher contrast on the fill"""
    lf = _lum(fill)
    return INK if (lf + 0.05) / (_lum(INK) + 0.05) >= 1.05 / (lf + 0.05) else 'white'


def style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10, 'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': GREY_D, 'axes.linewidth': 0.8,
        'xtick.color': GREY_D, 'ytick.color': GREY_D, 'axes.labelcolor': INK, 'text.color': INK,
        'axes.axisbelow': True, 'grid.color': '#E3E3E3', 'grid.linewidth': 0.6,
        'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04})


def rows(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.reader(f))


def top_legend(ax, ncol, **kw):
    return ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=ncol, handlelength=1.8, columnspacing=1.6, **kw)


# ---------------------------------------------------------------- Fig. 6
def fig6(res, out):
    r = rows(os.path.join(res, 'fig6.csv'))
    h = r[0]
    de = np.array([float(x[h.index('dev_expected(%)')]) for x in r[1:]])
    dp = np.array([float(x[h.index('dev_pessimistic(%)')]) for x in r[1:]])
    s = {x[0]: x[1:] for x in rows(os.path.join(res, 'fig6_summary.csv'))}
    tm = [float(v) * 1000 for v in s['time per instance (s)']]          # ms
    x = np.arange(1, len(de) + 1)
    fig, ax = plt.subplots(2, 1, figsize=(WIDTH, 5.6), gridspec_kw={'height_ratios': [1.35, 1]})
    ax[0].axhline(0, color=GREY_D, lw=0.6)
    ax[0].plot(x, de, lw=0.9, color=TEAL, label='Expected value estimation')
    ax[0].plot(x, dp, lw=0.9, color=OLIVE, label='Pessimistic estimation')
    ax[0].set_xlabel('Instance'); ax[0].set_ylabel('Relative deviation (%)')
    ax[0].set_xlim(0, len(de) + 1); ax[0].grid(axis='y')
    top_legend(ax[0], 2)
    names = ['Insertion-based\nsimulation', 'Expected value\nestimation', 'Pessimistic\nestimation']
    b = ax[1].bar(names, tm, color=[GREY, TEAL, OLIVE], width=0.5)
    for rect, v in zip(b, tm):
        ax[1].text(rect.get_x() + rect.get_width() / 2, v, '%.2f' % v, ha='center', va='bottom', fontsize=10)
    ax[1].set_ylabel('Time (ms/instance)'); ax[1].set_ylim(0, max(tm) * 1.18); ax[1].grid(axis='y')
    fig.tight_layout(h_pad=1.2); fig.savefig(os.path.join(out, 'fig6.png')); plt.close(fig)


# ---------------------------------------------------------------- Fig. 9
def fig9(res, out):
    from matplotlib.legend_handler import HandlerTuple
    txt = open(os.path.join(res, 'fig9_info.txt')).read().splitlines()
    shop = int(re.match(r'manufacturer S(\d+)', txt[0]).group(1))
    typ = C.SHOP_TYPE[shop - 1]
    ops = []
    for ln in txt:
        m = re.match(r'M(\d+), ([\d.]+), ([\d.]+), (\S+)$', ln)
        if m:
            ops.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4)))
    mine = [o for o in ops if o[3] != 'other']
    lo, hi = min(o[1] for o in mine), max(o[2] for o in mine)
    nm = max(o[0] for o in ops)
    fig, ax = plt.subplots(figsize=(WIDTH, 4.4))
    used = []
    for (m, a, b, w) in ops:
        if b < lo - 5 or a > hi + 5:
            continue
        if w == 'other':
            ax.barh(m, b - a, left=a, height=0.7, color=GREY_L, edgecolor=EDGE, linewidth=0.4)
        else:
            r = int(w[1:].split('.')[0])                 # product batch r -> colour r-1 (as in the original figure)
            fc = PAL[(r - 1) % 20]
            used.append(r)
            ax.barh(m, b - a, left=a, height=0.7, color=fc, edgecolor=EDGE, linewidth=0.4)
            ax.text((a + b) / 2, m, 'J%d' % r, ha='center', va='center', fontsize=8, color=label_colour(fc))
    ax.set_yticks(range(1, nm + 1)); ax.set_yticklabels(['M%d' % k for k in range(1, nm + 1)], fontsize=9)
    ax.set_ylim(0.3, nm + 0.7)
    ax.set_ylabel(r'Machines of manufacturer $S_{%d,%d}$' % (shop, typ))
    ax.set_xlabel('Time (h)'); ax.grid(axis='x')
    ax.xaxis.set_major_locator(MultipleLocator(2)); ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    own = tuple(Patch(facecolor=PAL[(r - 1) % 20], edgecolor=EDGE, linewidth=0.4) for r in sorted(set(used))[:4])
    ax.legend(handles=[own, Patch(facecolor=GREY_L, edgecolor=EDGE, linewidth=0.4)],
              labels=['Order 7 (J$r$: job of product batch $r$)', 'Other orders'],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, handlelength=3.2,
              loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=2.0)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig9.png')); plt.close(fig)


# ---------------------------------------------------------------- Fig. 10
def fig10(res, out):
    r = rows(os.path.join(res, 'order7_schedule.csv'))[1:]
    S = {}
    for x in r:
        S[(int(x[0]), int(x[2]))] = dict(pt=int(x[1]), shop=int(x[3]), arr=float(x[4]), st=float(x[5]), en=float(x[6]), comp=float(x[7]))
    nb = max(b for (b, c) in S)
    ro = min(v['arr'] for v in S.values())
    # PCB fabrication and SMT (tasks 1, 2) in the upper half, machining (task 3) in the lower half, assembly / testing full height
    lane = lambda b, c: (b - 0.19, 0.34) if c in (1, 2) else ((b + 0.19, 0.34) if c == 3 else (b, 0.74))
    fig, ax = plt.subplots(figsize=(WIDTH, 6.3))
    for (b, c), v in S.items():
        s = v['shop']
        fc = PAL[(s - 1) % 20]                            # manufacturer S_k -> colour k-1 (as in the original figure)
        y, h = lane(b, c)
        ax.barh(y, v['en'] - v['st'], left=v['st'], height=h, color=fc, edgecolor=EDGE, linewidth=0.4, zorder=3)
        ax.text((v['st'] + v['en']) / 2, y, r'$p_{%d,%d}$' % (b, c), ha='center', va='center', fontsize=8.5 if h < 0.5 else 9.5,
                color=label_colour(fc), zorder=4)
        for q in C.PREDS[v['pt'] - 1][c - 1]:
            u = S[(b, q + 1)]
            t = C.SS[u['shop'] - 1][s - 1]
            if t > 0:
                ax.barh(lane(b, q + 1)[0], t, left=u['en'], height=0.09, color=GREY_D, zorder=2)
        last = max(cc for (bb, cc) in S if bb == b)
        if c == last and v['comp'] > v['en']:
            ax.barh(y, v['comp'] - v['en'], left=v['en'], height=0.16, facecolor='white', edgecolor=GREY_D, hatch='////',
                    linewidth=0.6, zorder=2)
    end = max(v['comp'] for v in S.values())
    ax.axvline(ro, color=GREY_D, ls=':', lw=1.0, zorder=1)
    ax.text(ro - 0.3, (nb + 1) / 2, 'Release of Order 7 (%.1f h)' % ro, rotation=90, ha='right', va='center', fontsize=9, color=GREY_D)
    ax.set_xlim(ro - 2.8, end + 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_yticks(range(1, nb + 1))
    ax.set_yticklabels(['Batch %d (P%d)' % (b, S[(b, 1)]['pt']) for b in range(1, nb + 1)])
    ax.set_ylim(nb + 0.6, 0.4)
    ax.set_xlabel('Time (h)'); ax.grid(axis='x')
    # legend above the plot: transportation / delivery in the lower row, the manufacturers in the rows above it
    lg2 = ax.legend(handles=[Line2D([0], [0], color=GREY_D, lw=3, label='Transportation between manufacturers'),
                             Patch(facecolor='white', edgecolor=GREY_D, hatch='////', label='Delivery to customer')],
                    loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.8, columnspacing=2.0)
    ax.add_artist(lg2)
    fig.tight_layout()
    fig.canvas.draw()
    top = ax.transAxes.inverted().transform((0, lg2.get_window_extent().y1))[1]
    shops = sorted(set(v['shop'] for v in S.values()))
    hs = [Patch(facecolor=PAL[(k - 1) % 20], edgecolor=EDGE, linewidth=0.4, label=r'$S_{%d,%d}$' % (k, C.SHOP_TYPE[k - 1])) for k in shops]
    ax.legend(handles=hs, loc='lower center', bbox_to_anchor=(0.5, top - 0.005), ncol=(len(hs) + 1) // 2, handlelength=1.6,
              columnspacing=1.4, title='Manufacturer', title_fontsize=10)
    fig.savefig(os.path.join(out, 'fig10.png')); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results')
    p.add_argument('--out', default=None, help='output folder (default: the results folder)')
    p.add_argument('--figs', default='6,9,10')
    p.add_argument('--palette', default='tableau', choices=sorted(PALETTES), help='colours of Figs. 9 and 10')
    a = p.parse_args()
    global PAL
    PAL = PALETTES[a.palette]
    out = a.out or a.results
    os.makedirs(out, exist_ok=True)
    style()
    jobs = {'6': lambda: fig6(a.results, out), '9': lambda: fig9(a.results, out), '10': lambda: fig10(a.results, out)}
    for k in a.figs.split(','):
        try:
            jobs[k.strip()]()
            print('saved', os.path.join(out, 'fig%s.png' % k.strip()))
        except FileNotFoundError as e:
            print('fig%s skipped: %s' % (k, e))


if __name__ == '__main__':
    main()
