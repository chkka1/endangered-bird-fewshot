# -*- coding: utf-8 -*-
"""
核实图 4 下图的 r = +0.61 是否稳健（数据来自 criterion_design_points.json）。

与 5-shot 落差同样的问题：n 很小、点分布不均，相关系数可能由个别点驱动。
本脚本给出 Pearson / Spearman / 留一法。
"""
from __future__ import annotations

import io
import json
import os

import numpy as np

FO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_out")
CDP = os.path.join(FO, "criterion_design_points.json")


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


def main():
    if not os.path.exists(CDP):
        print("缺少 %s" % CDP)
        return
    D = json.load(io.open(CDP, encoding="utf-8"))
    print("设定数 n = %d" % len(D))
    hs = np.array([r["summary"]["sample_heldout_mean"] for r in D])
    ls = np.array([r["summary"]["novel_loss_sample_split_pp"] for r in D])
    print()
    print("%-4s %-10s %-24s %12s %12s" % ("#", "dataset", "backbone", "留出准确率", "Novel 损失"))
    print("-" * 68)
    for i, r in enumerate(D):
        print("%-4d %-10s %-24s %12.2f %12.2f"
              % (i, r.get("dataset", "?"), r.get("backbone", "?"), hs[i], ls[i]))

    r_p = pearson(hs, ls)
    r_s = spearman(hs, ls)

    # 重复的留出值：endb 与 cub 共享 cub_base，留出准确率成对相同。
    # 论文以"不同留出值"口径报告，故这里同时给出两种口径。
    from collections import defaultdict
    agg = defaultdict(list)
    for h, l in zip(hs, ls):
        agg[round(float(h), 2)].append(float(l))
    xa = sorted(agg)
    ya = [sum(agg[k]) / len(agg[k]) for k in xa]
    r_dedup = pearson(xa, ya)

    print()
    print("=== 相关系数（两种口径）===")
    print("  [原始 %d 行]           Pearson = %+.2f   Spearman = %+.3f" % (len(hs), r_p, r_s))
    print("  [不同留出值 %d 个]      Pearson = %+.2f   ← 论文采用" % (len(xa), r_dedup))
    dup = {k: v for k, v in agg.items() if len(v) > 1}
    if dup:
        print("  重复来源：endb 与 cub 共享 cub_base，故留出准确率成对相同：")
        for k in sorted(dup):
            print("     留出 %.2f 出现 %d 次" % (k, len(dup[k])))
    print()
    print("留一法 A：原始 %d 行（逐一剔除一行）——供对照" % len(hs))
    loo = []
    for m in range(len(hs)):
        msk = np.ones(len(hs), bool)
        msk[m] = False
        rr = pearson(hs[msk], ls[msk])
        loo.append(rr)
        tag = D[m].get("dataset", "?") + "/" + D[m].get("backbone", "?")
        print("  剔除 %-30s r = %+.3f" % (tag, rr))
    loo = np.array(loo)
    print()
    print("留一法 A 范围 %+.3f ~ %+.3f，跨度 %.3f" % (loo.min(), loo.max(), loo.max() - loo.min()))
    print("同号：%s" % ("是" if (loo > 0).all() else "**否**"))

    # 留一法 B：与论文报告的 r 同口径（去重后逐一剔除一个留出值）
    loo_d = []
    for m in range(len(xa)):
        idx = [q for q in range(len(xa)) if q != m]
        loo_d.append(pearson([xa[q] for q in idx], [ya[q] for q in idx]))
    loo_d = np.array(loo_d)
    print()
    print("留一法 B：不同留出值 %d 个（逐一剔除一个留出值）——与论文 r 同口径" % len(xa))
    for m in range(len(xa)):
        print("    剔除留出 %6.2f（含 %d 行）    r = %+.3f"
              % (xa[m], len(agg[xa[m]]), loo_d[m]))
    print("    范围 %+.3f ~ %+.3f，跨度 %.3f，同号：%s"
          % (loo_d.min(), loo_d.max(), loo_d.max() - loo_d.min(),
             "是" if (loo_d > 0).all() else "**否**"))
    print()
    # 判定口径（与论文一致）：
    #   主判据 = 符号是否稳定（翻转才说明"由单点驱动"）；
    #   跨度只作描述，不单独判死 —— 任意阈值都会把"中等稳定"误判为"不稳健"。
    same_sign = bool((loo_d > 0).all() or (loo_d < 0).all())
    span = float(loo_d.max() - loo_d.min())
    # 不给"小/中等/偏大"这类定性标签 —— 任何阈值在临界点都毫无意义
    # （曾用 0.3、0.4 两次，两次都在边界上产生误导性判定）。改为直接陈述事实。
    if not same_sign:
        print("判定：**方向不稳定** —— 剔除个别留出值后相关系数变号，说明该 r 由单点驱动，")
        print("      不宜作为结论；建议只报告分层比较（高/低饱和两组的均值差）。")
    else:
        print("判定：**方向稳定** —— 剔除任一留出值后符号不变（6 个留出值全部为正）。")
        print("      强度不稳：留一法 r 跨度 %+.3f ~ %+.3f。" % (loo_d.min(), loo_d.max()))
        print("      论文据此以『不同留出值』口径报告 r = %+.2f（%d 行 → %d 个不同留出值），"
              % (r_dedup, len(hs), len(xa)))
        print("      给出上述范围并声明仅作描述性统计；")
        print("      主要依据为分层比较（高/低饱和两组的均值差），该比较不依赖 r 的强度。")


if __name__ == "__main__":
    main()
