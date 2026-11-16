# -*- coding: utf-8 -*-
"""
训练。例：
    python train.py --methods GIN-DPPO-E                      # 主方法（12800 个训练算例、批大小 256、最多 40 轮）
    python train.py --methods ablation                        # 全部消融变体
    python train.py --methods baselines                       # MPDDQN, R-PPO, PBMP, D-Q, QIHGRL
    python train.py --methods all --epochs 60 --patience 15   # 改变最多训练轮数和提前停止的耐心
    python train.py --methods GIN-DPPO-E --seeds 1,2,3        # 多随机种子
    python train.py --methods GIN-DPPO-E --resume             # 接着上次中断的地方继续
已有 runs/<方法>/best.pth 的方法默认跳过（加 --force 重新训练）。Ctrl+C 中断后已保存的最优模型不受影响。
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # Windows 下 PyTorch 与 numpy/matplotlib/scipy 各带一份 OpenMP 库时避免 OMP Error #15
import argparse
import os
import time

from smp import methods as M


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--methods', default='GIN-DPPO-E', help='逗号分隔的方法名，或 all / ablation / baselines')
    p.add_argument('--epochs', type=int, default=0, help='最多训练轮数，0 = 默认 40（D-Q 最多 30）')
    p.add_argument('--train_size', type=int, default=0, help='0 = 12800')
    p.add_argument('--batch', type=int, default=0, help='每次采样的订单数，0 = 256（只占内存，不占显存）')
    p.add_argument('--mb', type=int, default=0, help='PPO 更新的小批量大小，0 = 默认（GIN 类 1024，QIHGRL 256）；显存不够时会自动减半')
    p.add_argument('--patience', type=int, default=10, help='验证分数连续这么多轮不改善就提前停止')
    p.add_argument('--seeds', default='0')
    p.add_argument('--device', default='', help='cuda / cpu，默认自动')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    names = {'all': M.LEARNED, 'ablation': M.ABLATION, 'baselines': M.DRL_BASELINES}.get(a.methods, a.methods.split(','))
    for seed in [int(s) for s in a.seeds.split(',')]:
        for n in names:
            out = M.run_dir(n, seed)
            done = os.path.exists(os.path.join(out, 'best.npz' if n == 'D-Q' else 'best.pth'))
            if done and not (a.force or a.resume):
                print('[skip] %s already trained (%s)' % (n, out))
                continue
            print('=' * 20, 'training %s (seed %d) ->' % (n, seed), out, '=' * 20, flush=True)
            t0 = time.time()
            over = {} if n == 'D-Q' else {'patience': a.patience}
            if a.mb and n not in ('D-Q', 'MPDDQN'):
                over['mb'] = a.mb
            best = M.train(n, epochs=a.epochs or None, train_size=a.train_size or None, batch=a.batch or None, seed=seed,
                           resume=a.resume, dev=M.device(a.device) if n != 'D-Q' else None,
                           log=lambda s: print(s, flush=True), **over)
            print('%s finished: best validation mean %.3f, %.2f h' % (n, best, (time.time() - t0) / 3600.0), flush=True)


if __name__ == '__main__':
    main()
