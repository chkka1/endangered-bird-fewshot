# -*- coding: utf-8 -*-
"""判据稳定性实验：20 次随机类二分。

对应审稿意见：
  「在至少 20 次随机类二分下，报告每个设定选出的成员分布、选择频率」

做法：
  对每个 (数据集, 主干, 判别规则)：
    for seed in 0..19:
        kind, eps, _, hsc = CAL.select(fb, yb, scorer=..., k=5, seed=seed)
        记录选中的成员 kind、收缩系数 eps、留出分数 hsc
  汇总：成员选择频率、留出分数的均值±标准差/95%CI、选中成员的一致性

用法：
  FE_STAB_SETS=cub          # 限定数据集（可选）
  FE_STAB_SEEDS=20          # 二分次数（默认 20）
  FE_STAB_OUT=<csv 路径>
"""
import csv
import io
import os
import statistics
import sys
import time
from collections import Counter, defaultdict

import numpy as np

T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, T)
import calibration as CAL          # noqa: E402

SETS = [("cub", "cub_novel", "cub_base"),
        ("nab", "nab_novel", "nab_base"),
        ("endb", "endb", "cub_base")]  # 濒危集复用 CUB Base 拟合校准（见 4.1 节）
BACKBONES = ["clean_multitask", "clean_species", "clip_vitb16", "dinov2_vitb14"]
SHOT = 5
SEEDS = int(os.environ.get("FE_STAB_SEEDS") or 20)
WANT = {x.strip() for x in (os.environ.get("FE_STAB_SETS") or "").split(",") if x.strip()}
OUT = os.environ.get("FE_STAB_OUT") or r"D:\soft\bird_project\release\results\criterion_stability.csv"


def load(group, backbone):
    p = os.path.join(T, "feat_cache", "%s__%s.npz" % (group, backbone))
    z = np.load(p, allow_pickle=True)
    keys = {k.lower(): k for k in z.files}
    F = z[keys.get("feats", z.files[0])]
    y = z[keys.get("species", keys.get("y", z.files[1]))]
    return np.asarray(F, dtype=np.float32), np.asarray(y)


rows = []
print("  判据稳定性实验：%d 次随机类二分" % SEEDS)
for ds, ngroup, bgroup in SETS:
    if WANT and ds not in WANT:
        continue
    for bb in BACKBONES:
        fp = os.path.join(T, "feat_cache", "%s__%s.npz" % (bgroup, bb))
        if not os.path.exists(fp):
            print("  skip %s/%s (no base cache)" % (ds, bb))
            continue
        fb, yb = load(bgroup, bb)
        print("\n  [%s / %s] base %d 张, %d 类"
              % (ds, bb, len(fb), len(np.unique(yb))), flush=True)

        for scorer in ("ncc", "lr"):
            t0 = time.time()
            kinds, epss, hscs = [], [], []
            for sd in range(SEEDS):
                try:
                    kind, eps, _, hsc = CAL.select(fb, yb, scorer=scorer,
                                                   k=SHOT, seed=sd)
                    kinds.append(kind); epss.append(float(eps)); hscs.append(float(hsc))
                except Exception as e:
                    print("      seed %d 失败: %s" % (sd, e), flush=True)
            if not kinds:
                continue
            cnt = Counter(kinds)
            top, topn = cnt.most_common(1)[0]
            h = np.array(hscs)
            ci = (statistics.mean(hscs) - 1.96 * statistics.stdev(hscs) / np.sqrt(len(h)),
                  statistics.mean(hscs) + 1.96 * statistics.stdev(hscs) / np.sqrt(len(h)))
            rows.append([ds, bb, scorer, SEEDS,
                         top, topn, "%.1f%%" % (100.0 * topn / len(kinds)),
                         "|".join("%s:%d" % kv for kv in cnt.most_common()),
                         "%.3f" % statistics.mean(epss),
                         "%.3f" % statistics.stdev(epss) if len(epss) > 1 else "0",
                         "%.3f" % statistics.mean(hscs),
                         "%.3f" % statistics.stdev(hscs) if len(hscs) > 1 else "0",
                         "[%.3f,%.3f]" % ci,
                         "%.0fs" % (time.time() - t0)])
            print("      %-4s 首选=%-8s %d/%d (%.0f%%) eps=%.2f±%.2f 留出=%.2f±%.2f (%.0fs)"
                  % (scorer, top, topn, len(kinds), 100.0 * topn / len(kinds),
                     statistics.mean(epss),
                     statistics.stdev(epss) if len(epss) > 1 else 0,
                     statistics.mean(hscs),
                     statistics.stdev(hscs) if len(hscs) > 1 else 0,
                     time.time() - t0), flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with io.open(OUT, "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset", "backbone", "scorer", "n_bisections",
                "top_member", "top_count", "top_freq",
                "member_distribution", "eps_mean", "eps_std",
                "holdout_mean", "holdout_std", "holdout_ci95", "elapsed"])
    w.writerows(rows)
print("\n  写出 %s（%d 行）" % (OUT, len(rows)))
