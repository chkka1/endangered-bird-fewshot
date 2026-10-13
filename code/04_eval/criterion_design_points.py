# -*- coding: utf-8 -*-
"""
判据的两个设计点实验 —— 用当前管线重跑
=======================================

方法：逐行照抄 calibration.select()，只改两处：
    设计点一  fit/ev 的划分方式   class（按类） vs sample（按样本）
    设计点二  并列时的取向       small（取更小 eps） vs large（取更大 eps）

评分器一律使用 calibration._ncc（select 内部用的那个），不自写采样器——
此前的失败正源于自写 k-shot 采样。

输出：final_out/criterion_design_points.json
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
os.chdir(TOOLS)

import calibration as CAL                                    # noqa: E402
from improve_methods import load, build_episodes, EVAL_SEEDS  # noqa: E402
from _paths import find_root, FINAL_OUT  # noqa: E402

FO = FINAL_OUT
OUT = os.path.join(FO, "criterion_design_points.json")

SETS = [("cub", "cub_novel", "cub_base",
         ["clean_multitask", "clean_species", "clip_vitb16", "dinov2_vitb14"]),
        ("nab", "nab_novel", "nab_base", ["clean_multitask", "clip_vitb16"]),
        ("endb", "endb", "cub_base", ["clean_multitask", "clip_vitb16"])]
SHOT = 5
K = 5
REPS = 5
SEEDS = list(range(10))

MEMBERS = CAL.MEMBERS
EPS_GRID = CAL.EPS_GRID


def criterion(fb, yb, scorer="ncc", split="class", tie="small", seed=0):
    """照抄 calibration.select()，只改划分方式与并列取向。返回 (kind, eps, score)。"""
    rng = np.random.RandomState(seed)
    cls = np.unique(yb)
    if split == "class":
        perm = rng.permutation(len(cls))
        half = len(cls) // 2
        fit_m = np.isin(yb, cls[perm[:half]])
        ev_m = np.isin(yb, cls[perm[half:]])
    else:                                  # sample：同类样本随机对半
        m = rng.rand(len(yb)) < 0.5
        fit_m, ev_m = m, ~m

    byc = None
    draws = []
    best = None
    for kind in ["none"] + MEMBERS:
        for eps in ([0.0] if kind == "none" else EPS_GRID):
            W = CAL.fit(kind, fb[fit_m], yb[fit_m], eps)
            Fev = CAL.apply_W(W, fb[ev_m])
            yh = yb[ev_m]
            if byc is None:
                byc = {int(c): np.where(yh == c)[0] for c in np.unique(yh)}
                keep = [c for c, idx in byc.items() if len(idx) >= 2 * K]
                for _ in range(REPS):
                    sup, qry = [], []
                    for c in keep:
                        idx = byc[c].copy(); rng.shuffle(idx)
                        sup += idx[:K].tolist(); qry += idx[K:2 * K].tolist()
                    draws.append((np.array(sup), np.array(qry)))
            accs = [CAL._ncc(Fev[s], yh[s], Fev[q], yh[q]) if scorer == "ncc"
                    else CAL._lr(Fev[s], yh[s], Fev[q], yh[q]) for s, q in draws]
            sc = float(np.mean(accs))
            key = (sc, -eps) if tie == "small" else (sc, eps)
            if best is None or key > best[0]:
                best = (key, kind, eps, sc)
    return best[1], best[2], best[3]


def novel_acc(kind, eps, fb, yb, fn, yn, idx):
    """判据选出的成员在全 Base 上重拟合，再在 Novel 上评测（论文协议）。"""
    W = CAL.fit(kind, fb, yb, eps)
    F = CAL.apply_W(W, fn)
    accs = []
    for sup, qry in idx:
        C = np.unique(yn[sup])
        P = CAL.l2(np.stack([F[sup][yn[sup] == c].mean(0) for c in C]))
        accs.append(100.0 * float((C[np.argmax(CAL.l2(F[qry]) @ P.T, axis=1)] == yn[qry]).mean()))
    return float(np.mean(accs))


def run_setting(ds, ng, bg, bb):
    fn, _, yn, _ = load(ng, bb)
    fb, _, yb, _ = load(bg, bb)
    idx = build_episodes(yn, SHOT, EVAL_SEEDS)
    rows = []
    for sd in SEEDS:
        ck, ce, cs = criterion(fb, yb, "ncc", "class", "small", sd)
        sk, se, ss = criterion(fb, yb, "ncc", "sample", "small", sd)
        lk, le, ls = criterion(fb, yb, "ncc", "class", "large", sd)
        rows.append(dict(seed=sd, cls_held=cs, cls_pick="%s/%.2f" % (ck, ce),
                         cls_novel=novel_acc(ck, ce, fb, yb, fn, yn, idx),
                         smp_held=ss, smp_pick="%s/%.2f" % (sk, se),
                         smp_novel=novel_acc(sk, se, fb, yb, fn, yn, idx),
                         large_pick="%s/%.2f" % (lk, le), large_eps=le,
                         large_novel=novel_acc(lk, le, fb, yb, fn, yn, idx)))
    return dict(dataset=ds, backbone=bb, rows=rows)


def main():
    t0 = time.time()
    n = sum(len(b) for _, _, _, b in SETS)
    print("=" * 92)
    print("判据设计点实验（照抄 select()，只改划分方式与并列取向）")
    print("  %d 个设定 x %d 个种子 | 评分器 calibration._ncc" % (n, len(SEEDS)))
    print("=" * 92, flush=True)

    out = []
    for ds, ng, bg, bbs in SETS:
        for bb in bbs:
            print("\n%s / %s ..." % (ds, bb), flush=True)
            try:
                r = run_setting(ds, ng, bg, bb)
            except Exception as e:
                print("   失败 %s: %s" % (type(e).__name__, e), flush=True)
                continue
            rows = r["rows"]
            hs = float(np.mean([x["smp_held"] for x in rows]))
            hc = float(np.mean([x["cls_held"] for x in rows]))
            loss = float(np.mean([x["cls_novel"] - x["smp_novel"] for x in rows]))
            pl = float(np.mean([x["cls_novel"] - x["large_novel"] for x in rows]))
            r["summary"] = {"sample_heldout_mean": round(hs, 2),
                            "class_heldout_mean": round(hc, 2),
                            "novel_loss_sample_split_pp": round(loss, 2),
                            "novel_loss_large_tiebreak_pp": round(pl, 2)}
            out.append(r)
            print("   样本二分留出 %.2f（类二分 %.2f）| 其 Novel 损失 %+.2f pp | 并列取大损失 %+.2f pp"
                  % (hs, hc, loss, pl), flush=True)

    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    if out:
        hs = [r["summary"]["sample_heldout_mean"] for r in out]
        ls = [r["summary"]["novel_loss_sample_split_pp"] for r in out]
        pl = [r["summary"]["novel_loss_large_tiebreak_pp"] for r in out]
        print("\n" + "=" * 92)
        print("汇总（%d 个设定，用时 %.0fs）" % (len(out), time.time() - t0))
        print("  设计点一  样本二分留出饱和率  %.2f +- %.2f（最大 %.2f）"
              % (np.mean(hs), np.std(hs), max(hs)))
        print("            其 Novel 损失        %+.2f +- %.2f pp（最大 %.2f）"
              % (np.mean(ls), np.std(ls), max(ls)))
        print("  设计点二  并列取大的损失      %+.2f +- %.2f pp（最大 %.2f）"
              % (np.mean(pl), np.std(pl), max(pl)))
        print("  稿件原值：留出 98.7 / 损失 12.3 / 并列 4.19")
        print("\n  saved: %s" % OUT)
        print("=" * 92)


if __name__ == "__main__":
    main()
