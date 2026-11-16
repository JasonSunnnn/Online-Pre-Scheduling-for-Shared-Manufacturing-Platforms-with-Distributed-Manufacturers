# -*- coding: utf-8 -*-
"""
一键运行全部实验（可中断，重新运行时已完成的训练会自动跳过）。

    python run_all.py                    # 默认：每个方法最多训练 40 轮（验证分数 10 轮不改善则提前停止）
    python run_all.py --epochs 60 --patience 15    # 改变最多训练轮数和提前停止的耐心
    python run_all.py --skip_train       # 只跑实验（用已有模型）
    python run_all.py --seeds 1,2        # 另外用这些随机种子再训练 GIN-DPPO-E（可选）

步骤：1 训练全部学习型方法  2 静态算例（表6-8）  3 估计精度（图6）  4 计算时间（图8、GIN 开销）
      5 动态场景（表9、10、图9）  6 模型验证  7 训练曲线（图7）
全部结果在 results/ 下，results/README_results.txt 说明每个文件对应论文的哪一处。
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # Windows 下 PyTorch 与 numpy/matplotlib/scipy 各带一份 OpenMP 库时避免 OMP Error #15
import argparse
import subprocess
import sys
import time


def run(args):
    print('\n' + '#' * 100 + '\n# python ' + ' '.join(args) + '\n' + '#' * 100, flush=True)
    t0 = time.time()
    rc = subprocess.call([sys.executable] + args)
    print('# finished in %.1f min (exit code %d)' % ((time.time() - t0) / 60.0, rc), flush=True)
    return rc


GUIDE = """结果文件与论文的对应关系
table6.csv               表6  消融实验（含 NL：去掉负载感知估计，以及 Sum / NE / FE / GCN）及 Avg.、Gap 行
table7.csv               表7  GIN-DPPO-E、GA、KTPOA、8 个组合调度规则
table8.csv               表8  GIN-DPPO-E 与 MPDDQN、R-PPO、PBMP、D-Q、QIHGRL
static_time.csv          4.2 节各方法平均每个算例的求解时间（KTPOA、GA 的计算时间）
static_realized.csv      各方法的调度在插入式仿真下的实际完成时间（备查）
checkpoints.csv          最优验证轮次模型 vs 最后一轮模型；多随机种子的最优验证分数
fig6.csv / fig6_summary.csv / fig6.png     图6 及 2.3.2、4.2 节中三种估计方式的偏差和耗时
fig8.csv / fig8.png      图8 计算时间随订单数的变化
gin_cost.csv             GIN 编码器每步前向时间、占比、每订单决策时间、训练时间（3.2.2 节）
table9.csv               表9  10 个订单的订单完成时间
table10.csv              表10 10 个订单的平均车间空闲率和总运输时间
dynamic_summary.csv      表9、表10 中 D-30 / D-50 / D-100 各行（mean ± std）和 4.3 节的平均每订单决策时间
dynamic_tests.csv        动态场景下 GIN-DPPO-E 与其他方法按序列配对的比较（胜出序列数、Wilcoxon 检验 p 值）
results_ablation_dynamic/  消融变体在动态场景下的同样一套结果（含 GIN-DPPO-E-NL：去掉负载感知估计）
fig9.png                 图9  订单 7 的甘特图
validation_*.csv         2.3.2 节 Model validation 段
fig7.png                 图7  训练曲线；runs/GIN-DPPO-E/train_log.csv 中有每轮的数值
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=0)
    p.add_argument('--train_size', type=int, default=0)
    p.add_argument('--batch', type=int, default=0)
    p.add_argument('--mb', type=int, default=0)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seeds', default='')
    p.add_argument('--device', default='')
    p.add_argument('--skip_train', action='store_true')
    p.add_argument('--evals', type=int, default=20000)
    p.add_argument('--out', default='results')
    a = p.parse_args()
    common = ['--epochs', str(a.epochs), '--train_size', str(a.train_size), '--batch', str(a.batch), '--patience', str(a.patience),
              '--mb', str(a.mb)]
    if a.device:
        common += ['--device', a.device]
    if not a.skip_train:
        run(['train.py', '--methods', 'all'] + common)
        if a.seeds:
            run(['train.py', '--methods', 'GIN-DPPO-E', '--seeds', a.seeds] + common)
    run(['exp_static.py', '--evals', str(a.evals), '--out', a.out])
    run(['exp_estimation.py', '--out', a.out])
    run(['exp_time.py', '--evals', str(a.evals), '--out', a.out])
    run(['exp_dynamic.py', '--out', a.out])
    run(['exp_dynamic.py', '--methods', 'ablation', '--out', a.out + '_ablation_dynamic'])
    run(['exp_validation.py', '--out', a.out])
    run(['plot_curves.py', '--out', a.out])
    with open(a.out + '/README_results.txt', 'w', encoding='utf8') as f:
        f.write(GUIDE)
    print('\nALL DONE. Results are in %s/' % a.out)


if __name__ == '__main__':
    main()
