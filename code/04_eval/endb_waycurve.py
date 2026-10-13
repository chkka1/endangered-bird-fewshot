# -*- coding: utf-8 -*-
"""
B4 方案③：分路数曲线——把"是否有 CUB 同属"与"路数"两个因素解耦。

问题：全类口径下两组路数不同（有同属 48 类 vs 无同属 22 类），因此
      "有同属组增益更大"无法归因——路数越多，校准的收益空间通常越大。

做法：两组都在**相同的路数**下评测（5 / 10 / 20 路），比较"增益-路数"曲线。
      若两条曲线重合，说明增益与"是否有 CUB 同属"无关；
      若在相同路数下仍分离，则该因素有独立影响。
      末尾另附全类端点（48 / 22 路）供参照，但明确标注其路数不同。

协议细节：
  · 校准矩阵 W 与论文一致，在 CUB Base 上由判据选出并拟合，所有条件共用；
  · 每轮从该组的类池中无放回抽 n_way 类；每类 5 个支持样本，其余为查询；
  · 每个种子重复 reps 次不同的类子集，以降低类抽样方差；
  · 支持集与查询集共用同一批 (seed, rep) 构造，两组严格配对。

输出：final_out/endb_waycurve.csv
"""
import csv
import io
import json
import os
import random
import sys

import numpy as np

T = os.path.join(ROOT, "_tools")
ROOT = find_root()
sys.path.insert(0, T)
import calibration as CAL                                     # noqa: E402
from improve_methods import load, EVAL_SEEDS                  # noqa: E402
from _paths import find_root  # noqa: E402

FO = os.path.join(T, "final_out")
SHOT = 5
WAYS = [5, 10, 20]
REPS = 3          # 每个种子抽 3 个不同的类子集


def way_episodes(y, n_way, seeds, reps):
    """按路数抽类的 episode：返回 [(sup, qry), ...]。

    与论文 build_episodes 的唯一区别是"每轮只抽 n_way 类"（论文为全类），
    其余（每类 5 支持、其余查询）保持一致。
    """
    by_cls = {}
    for i, v in enumerate(y):
        by_cls.setdefault(int(v), []).append(i)
    classes = sorted(by_cls)
    eps = []
    for sd in seeds:
        for r in range(reps):
            rng = random.Random((sd, r, n_way))
            if n_way >= len(classes):
                pick = classes
            else:
                pick = rng.sample(classes, n_way)
            sup, qry = [], []
            for c in pick:
                idxs = by_cls[c][:]
                rng.shuffle(idxs)
                # 每类至少留 1 个查询样本
                k = max(1, min(SHOT, len(idxs) - 1))
                sup.extend(idxs[:k])
                qry.extend(idxs[k:])
            if sup and qry:
                eps.append((np.array(sup), np.array(qry)))
    return eps


def acc_of(F, y, idx):
    per = []
    for sup, qry in idx:
        C = np.unique(y[sup])
        P = CAL.l2(np.stack([F[sup][y[sup] == c].mean(0) for c in C]))
        pred = C[np.argmax(CAL.l2(F[qry]) @ P.T, axis=1)]
        per.append(100.0 * float((pred == y[qry]).mean()))
    return np.array(per)


def main():
    meta = json.load(io.open(os.path.join(ROOT, "cub_parsed", "cub_metadata.json"),
                             encoding="utf-8"))
    sp = json.load(io.open(os.path.join(ROOT, "cub_parsed",
                                        "dataset_split_by_genus.json"), encoding="utf-8"))
    g2 = {int(m["species_id"]): m["genus"] for m in meta}
    cub_base_g = {g2[s] for s in (int(x) for x in sp["base_species_ids"]) if s in g2}

    fn, _, yn, gn = load("endb", "clean_multitask")
    fb, _, yb, _ = load("cub_base", "clean_multitask")

    st = list(csv.DictReader(io.open(os.path.join(ROOT, "data", "endangered_birds",
                                                  "species_table.csv"), encoding="utf-8-sig")))
    name2gid, gid2name = {}, {}
    for r in st:
        if r["genus"] not in name2gid:
            gid = len(name2gid)
            name2gid[r["genus"]] = gid
            gid2name[gid] = r["genus"]
    sp2gen = {int(v): int(g) for v, g in zip(yn, gn)}
    has = sorted({s for s in np.unique(yn)
                  if gid2name.get(sp2gen[int(s)]) in cub_base_g})
    no = sorted(set(int(s) for s in np.unique(yn)) - set(int(s) for s in has))

    print("有 CUB 同属 %d 种 / 无 CUB 同属 %d 种（CUB Base 属 %d）\n"
          % (len(has), len(no), len(cub_base_g)))

    kind, eps, _, hsc = CAL.select(fb, yb, scorer="ncc", k=SHOT)
    W = CAL.fit(kind, fb, yb, eps)
    Fn_cal = CAL.apply_W(W, fn)
    print("校准：判据选出 %s eps=%.2f 留出 %.2f\n" % (kind, eps, hsc))

    rows = []
    print("%-12s %5s | %8s %8s | %8s %8s | %8s %8s"
          % ("组", "路数", "基线", "基线SE", "方法族", "族SE", "增益", "增益SE"))
    print("-" * 84)
    for tag, sps in (("有 CUB 同属", has), ("无 CUB 同属", no)):
        m = np.isin(yn, sps)
        F_sub, y_sub, Fc_sub = fn[m], yn[m], Fn_cal[m]
        for nw in WAYS + [len(sps)]:
            if nw > len(sps):
                continue
            idx = way_episodes(y_sub, nw, EVAL_SEEDS, REPS)
            a = acc_of(F_sub, y_sub, idx)
            b = acc_of(Fc_sub, y_sub, idx)
            d = b - a
            allc = (nw == len(sps))
            rows.append(dict(group=tag, ways=nw, is_all_class=int(allc),
                             n_species=len(sps), n_episodes=len(idx),
                             base_acc="%.2f" % a.mean(),
                             base_se="%.2f" % (a.std() / np.sqrt(len(a))),
                             fam_acc="%.2f" % b.mean(),
                             fam_se="%.2f" % (b.std() / np.sqrt(len(b))),
                             gain="%+.2f" % d.mean(),
                             gain_se="%.2f" % (d.std() / np.sqrt(len(d)))))
            print("%-12s %5d%s| %8.2f %8.2f | %8.2f %8.2f | %+8.2f %8.2f"
                  % (tag, nw, "*" if allc else " ", a.mean(),
                     a.std() / np.sqrt(len(a)), b.mean(),
                     b.std() / np.sqrt(len(b)), d.mean(),
                     d.std() / np.sqrt(len(d))))
        print("-" * 84)

    out = os.path.join(FO, "endb_waycurve.csv")
    with io.open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "ways", "is_all_class", "n_species",
                                          "n_episodes", "base_acc", "base_se",
                                          "fam_acc", "fam_se", "gain", "gain_se"])
        w.writeheader()
        w.writerows(rows)
    print("\n* = 全类端点（两组路数不同，仅供参照，不用于组间比较）")
    print("saved %s" % out)


if __name__ == "__main__":
    main()
