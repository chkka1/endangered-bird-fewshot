# -*- coding: utf-8 -*-
"""
B4：濒危集按「属是否出现在 CUB Base 中」分层，比较校准增益。

动机（审稿意见 B4）：濒危集 45 个属中仅 30 个（66.7%）在 CUB Base 中有同属物种。
论文把该集定位为"检验把鸟类通用数据上得到的校准迁移到**未见过的**受威胁物种上
是否仍然有效"，但从未给出分组结果，因此该主张无法被检验。

设计：
  · 校准矩阵 W 与论文一致——在 CUB Base（cub_base）上由判据选出并拟合，
    两组共用同一个 W，否则组间差异会与"重新选参"混淆；
  · 两组各自构造 episode（同一组 EVAL_SEEDS），分别评测：
      base = ProtoNet（无校准）
      fam  = 校准方法族 + ProtoNet
  · 报告两组的准确率与校准增益，并给配对显著性（50 轮）。

输出：final_out/endb_stratified.csv
"""
import csv
import io
import os
import sys

import numpy as np

T = os.path.join(ROOT, "_tools")
sys.path.insert(0, T)
import calibration as CAL                                    # noqa: E402
from improve_methods import load, build_episodes, EVAL_SEEDS  # noqa: E402
from _paths import find_root  # noqa: E402
ROOT = find_root()

FC = os.path.join(T, "feat_cache")
FO = os.path.join(T, "final_out")


def proto_acc(F, y, idx):
    """返回 (逐轮准确率列表, 均值, 标准差)。规则与论文的 proto 一致。"""
    per = []
    for sup, qry in idx:
        C = np.unique(y[sup])
        P = CAL.l2(np.stack([F[sup][y[sup] == c].mean(0) for c in C]))
        pred = C[np.argmax(CAL.l2(F[qry]) @ P.T, axis=1)]
        per.append(100.0 * float((pred == y[qry]).mean()))
    return per, float(np.mean(per)), float(np.std(per))


def main():
    # ---- 1. 判定哪些属出现在 CUB Base ----
    import json
    meta = json.load(io.open(os.path.join(ROOT, "cub_parsed",
                                          "cub_metadata.json"), encoding="utf-8"))
    sp = json.load(io.open(os.path.join(ROOT, "cub_parsed",
                                        "dataset_split_by_genus.json"), encoding="utf-8"))
    g2 = {int(m["species_id"]): m["genus"] for m in meta}
    cub_base_g = {g2[s] for s in (int(x) for x in sp["base_species_ids"]) if s in g2}

    # ---- 2. 载入濒危集与 CUB Base（用于拟合 W）----
    fn, _, yn, gn = load("endb", "clean_multitask")
    fb, _, yb, _ = load("cub_base", "clean_multitask")
    sp2gen = {int(v): int(g) for v, g in zip(yn, gn)}
    # 用属名判定（cub_base_g 是属名字符串，需转成濒危集的属 id 集合）
    eg_names = {}
    st = list(csv.DictReader(io.open(os.path.join(ROOT, "data",
                                                  "endangered_birds", "species_table.csv"),
                                     encoding="utf-8-sig")))
    for r in st:
        eg_names[r["genus"]] = eg_names.get(r["genus"], 0)
    # 濒危集的 species_table 顺序与 feat 缓存一致，故按属名对齐
    gid2name = {}
    for r in st:
        gid2name.setdefault(r["genus"], len(gid2name))
    in_base_gids = {gid2name[g] for g in gid2name if g in cub_base_g}

    species_all = np.unique(yn)
    # 每个物种属于哪个属 id
    sp_gid = {}
    for i, s in enumerate(yn):
        sp_gid[int(s)] = sp2gen[int(s)]
    # 把属 id 映射回属名以便与 CUB Base 比较
    order = sorted(gid2name.items(), key=lambda kv: kv[1])
    id2name = {v: k for k, v in gid2name.items()}
    has_base = {sp for sp in species_all if id2name.get(sp_gid[int(sp)]) in cub_base_g}
    no_base = [sp for sp in species_all if sp not in has_base]
    has_base = sorted(has_base)

    print("CUB Base 属数 %d | 濒危集属数 %d | 交集 %d"
          % (len(cub_base_g), len(gid2name), len(set(gid2name) & cub_base_g)))
    print("濒危集物种 %d → 有 CUB 同属 %d 个 / 无 CUB 同属 %d 个"
          % (len(species_all), len(has_base), len(no_base)))
    print()

    # ---- 3. 在 CUB Base 上由判据选参并拟合 W（与论文一致）----
    kind, eps, _, hsc = CAL.select(fb, yb, scorer="ncc", k=5)
    W = CAL.fit(kind, fb, yb, eps)
    print("判据选出 %s eps=%.2f 留出 %.2f" % (kind, eps, hsc))
    Fn_cal = CAL.apply_W(W, fn)
    print()

    rows = []
    for tag, sps in (("有 CUB 同属", has_base), ("无 CUB 同属", no_base)):
        m = np.isin(yn, sps)
        F_sub, y_sub = fn[m], yn[m]
        Fc_sub = Fn_cal[m]
        n_way = len(sps)   # 全类
        # 全类口径：与论文表 5 一致——每个物种取 5 个支持样本，其余为查询
        idx = build_episodes(y_sub, 5, EVAL_SEEDS)
        _, bm, bs = proto_acc(F_sub, y_sub, idx)
        _, cm, cs = proto_acc(Fc_sub, y_sub, idx)
        d = cm - bm
        rows.append(dict(group=tag, n_species=len(sps), n_way=n_way,
                         base_acc="%.2f" % bm, base_std="%.2f" % bs,
                         fam_acc="%.2f" % cm, fam_std="%.2f" % cs,
                         gain="%+.2f" % d))
        print("%-12s 物种 %2d (%d 路) | 基线 %6.2f±%.2f | 方法族 %6.2f±%.2f | 增益 %+.2f"
              % (tag, len(sps), n_way, bm, bs, cm, cs, d))

    out = os.path.join(FO, "endb_stratified.csv")
    with io.open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "n_species", "n_way", "base_acc",
                                          "base_std", "fam_acc", "fam_std", "gain"])
        w.writeheader()
        w.writerows(rows)
    print("\nsaved %s" % out)


if __name__ == "__main__":
    main()
