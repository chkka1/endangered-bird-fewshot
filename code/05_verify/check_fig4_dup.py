# -*- coding: utf-8 -*-
"""
核查：图 4 下图的 8 个"设定"中，留出准确率存在重复值。

事实：
  final_eval.py 的 DATASETS 把濒危集的 base 组设为 cub_base
      ("endb", "endb", "cub_base", [5])
  因此 endb/* 的留出准确率与 cub/* 逐位相同：
      cub/clean_multitask 98.31  ==  endb/clean_multitask 98.31
      cub/clip_vitb16     77.42  ==  endb/clip_vitb16     77.42
  但 Novel 损失不同（Novel 集合不同），故是"同一 x、不同 y"。

问题：相关系数建立在含重复 x 的 8 个点上，有效独立样本数小于 8。
本脚本给出：
  · 原始 8 点 r
  · 按重复 x 聚合（取 y 均值）后的 6 点 r
  · 只保留 cub/* 与 nab/*（剔除 endb 行）的 6 点 r
  · 逐点留一（已在上一个脚本给出）
"""
from __future__ import annotations

import io
import json
import os

import numpy as np

FO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_out")


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
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
    D = json.load(io.open(os.path.join(FO, "criterion_design_points.json"), encoding="utf-8"))
    hs = np.array([r["summary"]["sample_heldout_mean"] for r in D])
    ls = np.array([r["summary"]["novel_loss_sample_split_pp"] for r in D])
    tags = ["%s/%s" % (r.get("dataset", "?"), r.get("backbone", "?")) for r in D]

    print("=== 重复的留出准确率 ===")
    from collections import defaultdict
    g = defaultdict(list)
    for i, h in enumerate(hs):
        g[round(float(h), 2)].append(i)
    dup = {k: v for k, v in g.items() if len(v) > 1}
    for k, v in sorted(dup.items()):
        print("  留出 %.2f 出现 %d 次：%s" % (k, len(v), ", ".join(tags[i] for i in v)))
        print("     对应 Novel 损失：%s" % ", ".join("%.2f" % ls[i] for i in v))
    print("  不同留出值个数：%d / %d" % (len(g), len(hs)))
    print()

    r8 = pearson(hs, ls)
    print("=== 三种口径的相关 ===")
    print("  [A] 原始 8 点                  n=%d  Pearson r = %+.3f  Spearman = %+.3f"
          % (len(hs), r8, spearman(hs, ls)))

    # 按重复 x 聚合（y 取均值）
    xa, ya = [], []
    for k, v in sorted(g.items()):
        xa.append(k)
        ya.append(float(np.mean([ls[i] for i in v])))
    print("  [B] 重复 x 聚合为 6 点          n=%d  Pearson r = %+.3f  Spearman = %+.3f"
          % (len(xa), pearson(xa, ya), spearman(xa, ya)))

    # 只用 cub 与 nab（剔除 endb 的重复行）
    msk = np.array([not t.startswith("endb") for t in tags])
    print("  [C] 剔除 endb 行（6 点）        n=%d  Pearson r = %+.3f  Spearman = %+.3f"
          % (msk.sum(), pearson(hs[msk], ls[msk]), spearman(hs[msk], ls[msk])))
    print()
    print("说明：[B] 与 [C] 都消除了由共享 cub_base 带来的重复，")
    print("      若二者与 [A] 差异较大，说明 r 值对重复点敏感。")


if __name__ == "__main__":
    main()
