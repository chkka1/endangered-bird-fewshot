# -*- coding: utf-8 -*-
"""
核实 5-shot 的两个可写点：
  ① 落差极差是否"远超噪声"
  ② 落差与基线准确率是否正相关

要点（避免误判）：
  · 必须先确认 acc_std 是"episode 间标准差"还是"均值的标准误"——
    若为前者，则均值的标准误 SE = std/√n_episode，噪声尺度要小得多；
  · n=7 的相关系数极易受单点影响，必须做留一法（leave-one-out）稳健性检查，
    并给出 p 值而非只报 r。
"""
from __future__ import annotations

import csv
import io
import os

import numpy as np

FO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_out")
CSV = os.path.join(FO, "final_main_table_calibrated.csv")
CA, CB = ("cub", "clean_multitask"), ("cubunseen", "unseen_multitask")
METHODS = [("nn", "NN"), ("protonet", "ProtoNet"), ("simpleshot", "SimpleShot"),
           ("fusion_only", "融合"), ("protoprior_GWP", "GWP"),
           ("baseaug", "Base 增广"), ("linearprobe", "线性探针")]


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d else float("nan")


def spearman(x, y):
    def rank(a):
        o = np.argsort(a)
        r = np.empty(len(a), float)
        r[o] = np.arange(len(a), dtype=float)
        return r
    return pearson(rank(np.asarray(x, float)), rank(np.asarray(y, float)))


def t_p(r, n):
    """双侧 p 值（t 检验，df=n-2）"""
    if n < 3 or abs(r) >= 1:
        return float("nan")
    t = r * np.sqrt((n - 2) / (1 - r * r))
    # 用近似式算 t 分布尾概率，避免依赖 scipy
    from math import lgamma, log, exp
    df = n - 2
    x = df / (df + t * t)
    # 正则化不完全贝塔 I_x(df/2, 1/2) 的连分式近似
    a, b = df / 2.0, 0.5

    def betacf(a, b, x, itmax=200, eps=3e-16):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        h = d
        for m in range(1, itmax + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < eps:
                break
        return h

    lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    front = exp(a * log(x) + b * log(1 - x) - lbeta) / a
    return float(min(1.0, front * betacf(a, b, x)))


def main():
    rows = list(csv.DictReader(io.open(CSV, encoding="utf-8-sig")))
    A, SD, EP = {}, {}, {}
    for r in rows:
        try:
            k = (r["dataset"], r["backbone"], r["shot"], r["key"])
            A[k] = float(r["acc_mean"])
            SD[k] = float(r["acc_std"]) if r.get("acc_std") else 0.0
            EP[k] = int(r["episodes"]) if r.get("episodes") else 0
        except (ValueError, KeyError):
            pass

    base, drop, sd, ep = [], [], [], []
    print("%-12s %10s %10s %9s %8s" % ("方法", "基线", "落差", "acc_std", "轮数"))
    print("-" * 54)
    for k, n in METHODS:
        a = A.get(CA + ("5", k))
        b = A.get(CB + ("5", k))
        if a is None or b is None:
            continue
        base.append(a)
        drop.append(a - b)
        sd.append(SD.get(CA + ("5", k), 0.0))
        ep.append(EP.get(CA + ("5", k), 0))
        print("%-12s %10.2f %10.2f %9.2f %8d" % (n, a, a - b, sd[-1], ep[-1]))

    base, drop, sd = np.array(base), np.array(drop), np.array(sd)
    ne = max(ep) if ep else 50
    se = sd / np.sqrt(ne)                 # 均值的标准误

    print()
    print("=" * 66)
    print("① 极差 vs 噪声")
    print("=" * 66)
    print("  acc_std 的均值            %.3f   （episode 间标准差）" % sd.mean())
    print("  轮数 n_episode            %d" % ne)
    print("  均值的标准误 SE = std/√n  %.3f" % se.mean())
    print("  落差极差                  %.2f" % (drop.max() - drop.min()))
    print()
    print("  口径说明：论断是『两个极值之差是否可由抽样波动解释』，")
    print("           故正确基准是两个极值之差的合并标准误，而非单点 SE。")
    # 两两比较：最远两方法的落差之差是否超过其合并 SE（正确口径）
    i, j = int(np.argmax(drop)), int(np.argmin(drop))
    se_d = np.sqrt(sd[i] ** 2 + sd[j] ** 2) / np.sqrt(ne)
    gap = drop[i] - drop[j]
    print("  最大落差 %.2f（%s）− 最小落差 %.2f（%s）= %.2f"
          % (drop[i], METHODS[i][1], drop[j], METHODS[j][1], gap))
    print("  两者之差的合并 SE ≈ %.3f" % se_d)
    print("  ★ 主判据  差值 / 差值SE = %.1f 倍" % (gap / se_d))
    print("  参考值    极差 / 单点SE = %.1f 倍（口径不当，仅列出以便对照）"
          % ((drop.max() - drop.min()) / se.mean()))
    print("  参考值    极差 / acc_std = %.1f 倍" % ((drop.max() - drop.min()) / sd.mean()))
    print("  判定：%s" % ("极差远超噪声（>5 倍差值SE）" if gap / se_d > 5
                        else "未超过噪声尺度，需谨慎"))

    print()
    print("=" * 66)
    print("② 落差 vs 基线准确率")
    print("=" * 66)
    r = pearson(base, drop)
    rho = spearman(base, drop)
    p_r = t_p(r, len(base))
    p_s = t_p(rho, len(base))
    print("  n = %d" % len(base))
    print("  Pearson  r = %+.3f   p = %.4f" % (r, p_r))
    print("  Spearman ρ = %+.3f   p = %.4f" % (rho, p_s))
    print()
    print("  留一法（逐一剔除，看 r 是否稳定）：")
    loo = []
    for m in range(len(base)):
        msk = np.ones(len(base), bool)
        msk[m] = False
        rr = pearson(base[msk], drop[msk])
        loo.append(rr)
        print("    剔除 %-11s r = %+.3f" % (METHODS[m][1], rr))
    loo = np.array(loo)
    print("  留一法 r 范围 %+.3f ~ %+.3f（跨度 %.3f）" % (loo.min(), loo.max(), loo.max() - loo.min()))
    print("  判定：%s" % ("相关稳健（留一法同号且跨度小）"
                        if (loo > 0).all() and (loo.max() - loo.min()) < 0.4
                        else "相关不稳健（受单点影响大）"))


if __name__ == "__main__":
    main()
