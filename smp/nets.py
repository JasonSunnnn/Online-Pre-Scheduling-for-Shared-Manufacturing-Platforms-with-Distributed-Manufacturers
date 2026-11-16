# -*- coding: utf-8 -*-
"""网络结构（PyTorch）。只使用最基本的张量运算，便于在不同版本的 PyTorch 上运行。"""
import math

import torch
import torch.nn as nn

from . import feat as Ft

NEG = -1e9


def mlp(i, h, o, layers=2, act=nn.Tanh):
    """layers 个隐藏层的全连接网络"""
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, h), act()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class GINLayer(nn.Module):
    """式 (31)：h_v' = MLP((1+eps) h_v + sum_{u in N(v)} (h_u + Linear(w_uv)))"""

    def __init__(self, d, edge=True):
        super(GINLayer, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge = edge
        if edge:                              # Linear(w)：1 维边权 -> d 维；对邻居求和后 = (sum w)*weight + deg*bias
            self.ew = nn.Parameter(torch.randn(d) * 0.1)
            self.eb = nn.Parameter(torch.zeros(d))

    def forward(self, h, A, ew, deg):
        m = torch.matmul(A, h)
        if self.edge:
            m = m + ew.unsqueeze(-1) * self.ew + deg.unsqueeze(-1) * self.eb
        return torch.relu(self.norm(self.mlp((1.0 + self.eps) * h + m)))


class GCNLayer(nn.Module):
    """h' = ReLU(D^-1/2 (A+I) D^-1/2 h W)"""

    def __init__(self, d):
        super(GCNLayer, self).__init__()
        self.lin = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, A, ew, deg):
        dinv = 1.0 / torch.sqrt(deg + 1.0).unsqueeze(-1)
        hn = h * dinv
        return torch.relu(self.norm(self.lin((torch.matmul(A, hn) + hn) * dinv)))


class MLPLayer(nn.Module):
    """无图结构（DPPO-E）：只对节点自身特征做变换"""

    def __init__(self, d):
        super(MLPLayer, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, A, ew, deg):
        return torch.relu(self.norm(self.mlp(h)))


class Encoder(nn.Module):
    def __init__(self, kind='gin', layers=3, d=64, pool='mean', edge=True):
        super(Encoder, self).__init__()
        self.inp = nn.Linear(Ft.F_NODE, d)
        if kind == 'gin':
            self.layers = nn.ModuleList([GINLayer(d, edge) for _ in range(layers)])
        elif kind == 'gcn':
            self.layers = nn.ModuleList([GCNLayer(d) for _ in range(layers)])
        else:
            self.layers = nn.ModuleList([MLPLayer(d) for _ in range(layers)])
        self.pool = pool

    def forward(self, x, A, ew, deg):
        h = self.inp(x)
        for layer in self.layers:
            h = layer(h, A, ew, deg)
        hg = h.mean(1) if self.pool == 'mean' else h.sum(1)       # 对固定的 N 个节点做读出
        return h, hg


class GraphNet(nn.Module):
    """arch = 'dual'   任务 actor + 制造商 actor + 共享 critic（GIN-DPPO-E）
             'single' 单个 actor 在（候选任务 × 制造商）联合动作空间上打分（GIN-PPO-E）
             'task'   只有任务 actor，制造商由 SPT 规则确定（PBMP 改造版）"""

    def __init__(self, arch='dual', enc='gin', layers=3, d=64, pool='mean', edge=True, hid_a=64, hid_c=128, fc=2):
        super(GraphNet, self).__init__()
        self.arch, self.d = arch, d
        self.enc = Encoder(enc, layers, d, pool, edge)
        if arch == 'single':
            self.joint = mlp(2 * d + Ft.F_MFR, hid_a, 1, fc)
        else:
            self.task = mlp(2 * d, hid_a, 1, fc)
            if arch == 'dual':
                self.mfr = mlp(Ft.F_MFR + d, hid_a, 1, fc)
        self.critic = mlp(d + 2 * Ft.M, hid_c, 1, fc)

    def encode(self, ob):
        h, hg = self.enc(ob['x'], ob['A'], ob['ew'], ob['deg'])
        B, d = hg.shape
        hc = h.gather(1, ob['cidx'].unsqueeze(-1).expand(B, Ft.NJ, d))            # 候选任务的嵌入 [B,NJ,d]
        return hc, hg

    def value(self, hg, gm):
        return self.critic(torch.cat([hg, gm], -1)).squeeze(-1)

    def task_logits(self, hc, hg, cmask):
        B, d = hg.shape
        z = torch.cat([hc, hg.unsqueeze(1).expand(B, Ft.NJ, d)], -1)
        return self.task(z).squeeze(-1).masked_fill(~cmask, NEG)

    def mfr_logits(self, mf, hg, mmask):
        B, d = hg.shape
        z = torch.cat([mf, hg.unsqueeze(1).expand(B, Ft.M, d)], -1)
        return self.mfr(z).squeeze(-1).masked_fill(~mmask, NEG)

    def joint_logits(self, hc, hg, mfa, mma):
        B, d = hg.shape
        z = torch.cat([hc.unsqueeze(2).expand(B, Ft.NJ, Ft.M, d), hg.unsqueeze(1).unsqueeze(1).expand(B, Ft.NJ, Ft.M, d), mfa], -1)
        return self.joint(z).squeeze(-1).reshape(B, Ft.NJ * Ft.M).masked_fill(~mma.reshape(B, Ft.NJ * Ft.M), NEG)


class RuleNet(nn.Module):
    """R-PPO / MPDDQN：全局状态向量 -> n_act 个规则的 logits（或 Q 值）；R-PPO 另有价值头"""

    def __init__(self, n_act, hid=128, value=True):
        super(RuleNet, self).__init__()
        self.body = mlp(Ft.F_GLOBAL, hid, n_act, 2)
        self.v = mlp(Ft.F_GLOBAL, hid, 1, 2) if value else None

    def forward(self, s):
        return self.body(s)

    def value(self, s):
        return self.v(s).squeeze(-1)


# --------------------------------------------------------------------------- QIHGRL (Zhao et al., 2026)
class QNALayer(nn.Module):
    """Question-aware neighbourhood aggregation（原文式 (12)-(16)）"""

    def __init__(self, d, heads=8, gamma2=0.95):
        super(QNALayer, self).__init__()
        assert d % heads == 0
        self.h, self.dk, self.g2 = heads, d // heads, gamma2
        self.WQ = nn.Linear(d, d, bias=False)
        self.WK = nn.Linear(d, d, bias=False)
        self.WH = nn.Linear(d, d, bias=False)
        self.WQq = nn.Linear(d, d, bias=False)
        self.WKq = nn.Linear(d, d, bias=False)
        self.W0 = nn.Linear(d, d)

    def forward(self, h, adj, eq):
        B, V, d = h.shape
        H, dk = self.h, self.dk
        sc = 1.0 / math.sqrt(dk)
        Q = self.WQ(h).reshape(B, V, H, dk).transpose(1, 2)
        K = self.WK(h).reshape(B, V, H, dk).transpose(1, 2)
        Vv = self.WH(h).reshape(B, V, H, dk).transpose(1, 2)
        Qq = self.WQq(eq).reshape(1, H, 1, dk)
        Kq = self.WKq(eq).reshape(1, H, 1, dk)
        n_ij = torch.matmul(Q, K.transpose(2, 3)) * sc
        h_iq = (Q * Kq).sum(-1) * sc
        t_qj = (K * Qq).sum(-1) * sc
        A = (1.0 - 2.0 * self.g2) * n_ij + self.g2 * h_iq.unsqueeze(-1) + self.g2 * t_qj.unsqueeze(-2)
        A = A.masked_fill(~adj.unsqueeze(1), NEG)
        z = torch.matmul(torch.softmax(A, -1), Vv).transpose(1, 2).reshape(B, V, d)
        return self.W0(z) + h


class IPA(nn.Module):
    """Inter-option prompt attention（原文式 (17)-(20)）"""

    def __init__(self, d, heads=8):
        super(IPA, self).__init__()
        self.h, self.dk = heads, d // heads
        self.LM = nn.Linear(2 * d, d)
        self.WQ = nn.Linear(d, d, bias=False)
        self.WK = nn.Linear(d, d, bias=False)
        self.WV = nn.Linear(d, d, bias=False)
        self.FFN = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, Hk, h_ar, opt_mask, eq):
        B, K, Nk, d = Hk.shape
        H, dk = self.h, self.dk
        T = self.LM(torch.cat([eq.reshape(1, 1, d).expand(B, K, d), h_ar], -1))
        q = self.WQ(Hk).reshape(B, K * Nk, H, dk).transpose(1, 2)
        k = self.WK(T).reshape(B, K, H, dk).transpose(1, 2)
        v = self.WV(T).reshape(B, K, H, dk).transpose(1, 2)
        sc = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(dk)
        sc = sc.masked_fill(~opt_mask.reshape(B, 1, 1, K), NEG)
        o = torch.matmul(torch.softmax(sc, -1), v).transpose(1, 2).reshape(B, K, Nk, d)
        return (self.FFN(o) + Hk).max(dim=2)[0]


class QIHGRLNet(nn.Module):
    def __init__(self, d=64, heads=8, layers=2, gamma2=0.95, hidden=128):
        super(QIHGRLNet, self).__init__()
        self.d = d
        self.in_t = nn.Linear(Ft.HF_T, d)
        self.in_m = nn.Linear(Ft.HF_M, d)
        self.eq = nn.Parameter(torch.randn(d) * 0.1)             # 可学习的调度目标代理向量 e_q
        self.qna = nn.ModuleList([QNALayer(d, heads, gamma2) for _ in range(layers)])
        self.opt_enc = nn.Sequential(nn.Linear(2 * d + Ft.HF_O, d), nn.ReLU(), nn.Linear(d, d))
        self.ipa = IPA(d, heads)
        self.actor = mlp(3 * d, hidden, 1, 2)
        self.critic = mlp(2 * d, hidden, 1, 2)

    def forward(self, ob):
        tx, mx, adj, valid = ob['task_x'], ob['shop_x'], ob['adj'], ob['valid']
        B, Nn, _ = tx.shape
        d = self.d
        h = torch.cat([self.in_t(tx), self.in_m(mx)], 1)
        for layer in self.qna:
            h = layer(h, adj, self.eq)
        Ht, Hm = h[:, :Nn], h[:, Nn:]
        vt = valid[:, :Nn].float().unsqueeze(-1)
        g = torch.cat([(Ht * vt).sum(1) / vt.sum(1).clamp(min=1.0), Hm.mean(1)], -1)
        K = ob['opt_task'].shape[1]
        P = ob['opt_pidx'].shape[2]
        h_o = Ht.gather(1, ob['opt_task'].unsqueeze(-1).expand(B, K, d))
        h_s = Hm.gather(1, ob['opt_shop'].unsqueeze(-1).expand(B, K, d))
        hp = Ht.gather(1, ob['opt_pidx'].reshape(B, K * P, 1).expand(B, K * P, d)).reshape(B, K, P, d)
        pm = ob['opt_pmask'].float().unsqueeze(-1)
        h_p = (hp * pm).sum(2) / pm.sum(2).clamp(min=1.0)
        h_ar = self.opt_enc(torch.cat([h_o, h_s, ob['opt_x']], -1))
        # 候选动作的子图：任务节点、制造商节点、前序任务节点，以及任务-制造商这条边自身的特征编码 h_ar
        sp = self.ipa(torch.stack([h_o, h_s, h_p, h_ar], 2), h_ar, ob['opt_mask'], self.eq)
        logits = self.actor(torch.cat([g.unsqueeze(1).expand(B, K, 2 * d), sp], -1)).squeeze(-1)
        return logits.masked_fill(~ob['opt_mask'], NEG), self.critic(g).squeeze(-1)
