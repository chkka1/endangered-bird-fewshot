# -*- coding: utf-8 -*-
"""
查判据在 clean_species 上的选择是否稳定。

做法：完整复现 CAL.select() 的内层循环，但**不打擂台、把每个 (成员, eps) 的
留出分数全部打出来**，从而看出最优点的领先幅度。

CAL.select() 的判定规则（见 calibration.py）：
    key = (score, -eps)     # 同分时偏向更小的 eps
    best = max(key)

稳定性判据：
    · 若最优与次优的差 < 0.3 个百分点 → 选择基本任意（网格区分度不足）
    · 若某个 eps 的平台很宽 → 该 eps 附近的选择不应被解读为"精确最优"

只读缓存，不写任何数据。
"""
from __future__ import annotations

import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
os.chdir(TOOLS)

import calibration as CAL      # noqa: E402
import improve_methods as IM   # noqa: E402

BB = "clean_species"
SHOT = 5
GROUPS = {"cub": "cub_base", "nab": "nab_base", "endb": "endb"}


def curve(fb, yb, scorer="ncc", k=SHOT, reps=5, seed=0):
    """复现 CAL.select 的打分，但不提前取最优 —— 返回全部 (kind, eps, score)。"""
    rng = np.random.RandomState(seed)
    cls = np.unique(yb)
    perm = rng.permutation(len(cls))
    half = len(cls) // 2
    fit_m = np.isin(yb, cls[perm[:half]])
    ev_m = np.isin(yb, cls[perm[half:]])
    out = []
    byc = None
    draws = None
    for kind in ["none"] + CAL.MEMBERS:
        for eps in ([0.0] if kind == "none" else CAL.EPS_GRID):
            W = CAL.fit(kind, fb[fit_m], yb[fit_m], eps)
            Fev = CAL.apply_W(W, fb[ev_m])
            yh = yb[ev_m]
            if byc is None:
                byc = {int(c): np.where(yh == c)[0] for c in np.unique(yh)}
                keep = [c for c, idx in byc.items() if len(idx) >= 2 * k]
                draws = []
                for _ in range(reps):
                    sup, qry = [], []
                    for c in keep:
                        idx = byc[c].copy()
                        rng.shuffle(idx)
                        sup += idx[:k].tolist()
                        qry += idx[k:2 * k].tolist()
                    draws.append((np.array(sup), np.array(qry)))
            accs = [CAL._ncc(Fev[s], yh[s], Fev[q], yh[q]) if scorer == "ncc"
                    else CAL._lr(Fev[s], yh[s], Fev[q], yh[q]) for s, q in draws]
            out.append((kind, eps, float(np.mean(accs))))
    return out, len(cls)


def main():
    print("主干 %s | %d-shot | scorer=ncc（与论文的 ProtoNet 行一致）" % (BB, SHOT))
    print()
    for ds, group in GROUPS.items():
        try:
            # IM.load 返回 (feats, feats_flip, species, genus)
            F, _Ff, y, _g = IM.load(group, BB)
        except Exception as e:
            print("  %-8s 载入失败：%s" % (ds, e))
            continue
        rows, ncls = curve(F, y)
        rows.sort(key=lambda r: (-r[2], r[1]))
        best = rows[0]
        second = rows[1] if len(rows) > 1 else best
        print("=== %s（%d 个 Base 类）===" % (ds, ncls))
        print("  %-8s %-6s %8s" % ("成员", "eps", "留出分"))
        for kind, eps, sc in rows:
            mark = "  ← 最优" if (kind, eps, sc) == best else ""
            print("  %-8s %-6.2f %8.2f%s" % (kind, eps, sc, mark))
        gap = best[2] - second[2]
        print("  最优 %s/eps=%.2f = %.2f ；次优 %s/eps=%.2f = %.2f ；差 %.2f"
              % (best[0], best[1], best[2], second[0], second[1], second[2], gap))
        print("  判定：%s" % ("选择稳定（领先 >0.3）" if gap > 0.3 else
                            "⚠️ 选择不稳定（领先 ≤0.3，网格区分度不足）"))
        print()


if __name__ == "__main__":
    main()
