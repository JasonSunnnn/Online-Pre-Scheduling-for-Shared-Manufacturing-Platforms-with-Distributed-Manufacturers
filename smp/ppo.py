# -*- coding: utf-8 -*-
"""
PPO 训练（论文 3.4 节 / 算法 1）。对 GraphAgent、RuleAgent、QIHGRLAgent 通用。

  * 优势函数 A_t = 折扣累计奖励 - V(s_t)（式 (34)），按批标准化
  * 每个 actor 各自的裁剪代理损失（式 (35)），共享 critic；总损失 = c_p*L_clip + c_v*MSE - c_e*beta*H
  * 自适应熵系数（式 (36)）：beta <- clip(beta * exp(kappa * (H_target - H_norm)), beta_min, beta_max)，
    其中 H_norm 为归一化策略熵（熵 / log 可选动作数，两个 actor 合计），H_target 随训练从 ent_t0 线性降到 ent_t1；
    ent='fixed' 时 beta 恒为初值（GIN-DPPO-E-FE）
  * clip fraction = 一次更新中 |ratio-1| > eps 的样本比例（对所有 actor、所有更新步取平均）
每个 epoch 遍历一次训练集；每个 epoch 结束后在验证集上贪婪解码，保存验证分数最好的模型。
"""
from __future__ import print_function

import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import core as K
from .agents import to_t, RS

DEFAULT = dict(lr=3e-4, gamma=0.99, clip=0.2, k_epochs=4, beta0=0.01, c_p=1.0, c_e=1.0, c_v=1.0, batch=256, grad_clip=2.5,
               epochs=40, train_size=12800, val_size=300, mb=1024, ent='adaptive', kappa=0.05, ent_t0=0.5, ent_t1=0.1,
               beta_min=1e-4, beta_max=0.05, patience=10, mode='pess', seed=0)


def returns(buf, traj, gamma):
    for idxs in traj:
        g = 0.0
        for t in reversed(idxs):
            g = buf[t]['r'] + gamma * g
            buf[t]['ret'] = np.float32(g)


def update(agent, opt, cfg, beta):
    buf = agent.buf
    n = len(buf)
    ret = np.array([b['ret'] for b in buf], dtype=np.float32)
    adv = ret - np.array([b['v'] for b in buf], dtype=np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = []
    data = agent.pack(buf)                 # 整批样本只堆叠一次，小批量直接按下标取（比每个小批量重新堆叠快）

    def step(idx):
        mb = to_t({k: v[idx] for k, v in data.items()}, agent.dev)
        lp, ent, hmax, v = agent.evaluate(mb)
        A = torch.from_numpy(adv[idx]).to(agent.dev).unsqueeze(-1)
        R = torch.from_numpy(ret[idx]).to(agent.dev)
        ratio = torch.exp(lp - mb['logp'])
        surr = torch.min(ratio * A, torch.clamp(ratio, 1.0 - cfg['clip'], 1.0 + cfg['clip']) * A)
        p_loss = -surr.sum(-1).mean()
        v_loss = Fn.mse_loss(v, R)
        e = ent.sum(-1).mean()
        loss = cfg['c_p'] * p_loss + cfg['c_v'] * v_loss - cfg['c_e'] * beta * e
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.net.parameters(), cfg['grad_clip'])
        opt.step()
        rr = ratio.detach().cpu().numpy()
        hn = float(ent.detach().cpu().numpy().sum() / max(float(hmax.detach().cpu().numpy().sum()), 1e-8))
        return [loss.item(), p_loss.item(), v_loss.item(), hn, float((np.abs(rr - 1.0) > cfg['clip']).mean())]

    for _ in range(cfg['k_epochs']):
        perm = np.random.permutation(n)
        s0 = 0
        while s0 < n:
            idx = perm[s0:s0 + cfg['mb']]
            oom = False
            try:
                stats.append(step(idx))
            except RuntimeError as err:          # 显存不够：把小批量减半后重试（之后一直用减半后的大小）
                if 'out of memory' not in str(err).lower() or cfg['mb'] <= 32:
                    raise
                oom = True
            if oom:
                opt.zero_grad()
                torch.cuda.empty_cache()
                cfg['mb'] = max(32, cfg['mb'] // 2)
                print('[memory] GPU out of memory -> minibatch size reduced to %d' % cfg['mb'], flush=True)
                continue
            s0 += len(idx)
    return np.mean(stats, 0)


def score(agent, instances, mode='pess', batch=100):
    """贪婪解码的订单完成时间（悲观估计口径）。训练环境为期望值估计时，把决策在悲观估计下重放"""
    g, s = agent.greedy, agent.store
    agent.greedy, agent.store = True, False
    agent.net.eval()
    costs, decs, _ = K.evaluate(agent, instances, mode, batch)
    if mode != 'pess':
        costs = np.array([K.replay(K.Inst(x), d, 'pess')[0] for x, d in zip(instances, decs)])
    agent.greedy, agent.store = g, s
    return costs


def train(agent, cfg, out_dir, train_set, val_set, resume=False, log=print):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    opt = torch.optim.Adam(agent.net.parameters(), lr=cfg['lr'])
    beta, best, start, bad, rows = cfg['beta0'], np.inf, 0, 0, []
    last = os.path.join(out_dir, 'last.pth')
    if resume and os.path.exists(last):
        ck = torch.load(last, map_location=agent.dev)
        if ck.get('est', 'nowait') != getattr(agent, 'est', 'nowait'):
            raise RuntimeError('%s 是按 est=%s 训练的，不能接着用 est=%s 续训'
                               % (last, ck.get('est', 'nowait'), getattr(agent, 'est', 'nowait')))
        agent.net.load_state_dict(ck['net'])
        opt.load_state_dict(ck['opt'])
        beta, best, start, bad, rows = ck['beta'], ck['best'], ck['epoch'], ck['bad'], ck['rows']
        log('resumed from epoch %d (best validation %.3f)' % (start, best))
    t_start = time.time()
    # 细粒度验证（可选）：每 val_every 次 PPO 迭代在验证集上贪婪解码一次，写 train_log_fine.csv。
    # 贪婪解码不消耗随机数，所以不改变训练过程本身
    ve, fine, seen, it = int(cfg.get('val_every', 0)), [], 0, 0

    def fine_point(train_cost):
        fine.append([it, seen, float(score(agent, val_set, cfg['mode']).mean()), train_cost, time.time() - t_start])
        with open(os.path.join(out_dir, 'train_log_fine.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['iteration', 'orders_seen', 'val_mean', 'train_cost', 'elapsed_s'])
            w.writerows(fine)
        log('  iteration %4d  orders %6d  valid %.3f' % (it, seen, fine[-1][2]))

    if ve and start == 0:
        fine_point('')
    for ep in range(start, cfg['epochs']):
        if cfg.get('stop_epoch') and ep >= cfg['stop_epoch']:
            break
        t0 = time.time()
        order = np.random.permutation(len(train_set))
        st, rws, cs = [], [], []
        for b0 in range(0, len(order), cfg['batch']):
            insts = [K.Inst(train_set[k]) for k in order[b0:b0 + cfg['batch']]]
            agent.net.eval()
            agent.greedy, agent.store = False, True
            c, _, _ = K.rollout(agent, insts, cfg['mode'])
            returns(agent.buf, agent.traj, cfg['gamma'])
            rws.append(np.mean([sum(agent.buf[t]['r'] for t in tr) * RS for tr in agent.traj]))
            cs.append(c.mean())
            agent.net.train()
            s = update(agent, opt, cfg, beta)
            st.append(s)
            if cfg['ent'] == 'adaptive':
                tgt = cfg['ent_t0'] + (cfg['ent_t1'] - cfg['ent_t0']) * ep / float(max(cfg['epochs'] - 1, 1))
                beta = float(np.clip(beta * np.exp(cfg['kappa'] * (tgt - s[3])), cfg['beta_min'], cfg['beta_max']))
            agent.buf, agent.traj = [], []
            it += 1
            seen += len(insts)
            if ve and it % ve == 0:
                fine_point(float(c.mean()))
        st = np.mean(st, 0)
        vc = score(agent, val_set, cfg['mode'])
        flag = ''
        if vc.mean() < best - 1e-9:
            best, bad, flag = float(vc.mean()), 0, ' *'
            torch.save({'net': agent.net.state_dict(), 'epoch': ep + 1, 'val': best, 'est': getattr(agent, 'est', 'nowait')},
                       os.path.join(out_dir, 'best.pth'))
        else:
            bad += 1
        rows.append([ep + 1] + [float(x) for x in (st[0], st[1], st[2], st[3], beta, st[4], np.mean(rws), np.mean(cs), vc.mean(),
                                                   vc.sum(), best, time.time() - t0)])
        torch.save({'net': agent.net.state_dict(), 'opt': opt.state_dict(), 'beta': float(beta), 'best': float(best), 'epoch': ep + 1,
                    'bad': bad, 'rows': rows, 'est': getattr(agent, 'est', 'nowait')}, last)
        with open(os.path.join(out_dir, 'train_log.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'loss', 'policy_loss', 'value_loss', 'entropy_norm', 'beta', 'clip_fraction', 'train_reward',
                        'train_cost', 'val_mean', 'val_sum', 'best_val_mean', 'epoch_time_s'])
            w.writerows(rows)
        log('epoch %3d  loss %.4f  clip %.3f  ent %.3f  beta %.4f  train %.2f  valid %.3f%s  (%.0fs)'
            % (ep + 1, st[0], st[4], st[3], beta, np.mean(cs), vc.mean(), flag, time.time() - t0))
        if bad >= cfg['patience']:
            log('early stop: no improvement on the validation set for %d epochs' % bad)
            break
    with open(os.path.join(out_dir, 'train_time.txt'), 'a') as f:
        f.write('%.1f s for epochs %d-%d\n' % (time.time() - t_start, start + 1, len(rows)))
    return best
