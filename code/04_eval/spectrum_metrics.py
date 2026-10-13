# -*- coding: utf-8 -*-
"""
B2：为四个主干 × 三个数据集计算谱指标，支撑"校准增益依赖主干"的机制解释。

审稿意见 B2 指出：论文反复以"谱形状（有效秩、前 10% 维度能量占比）"解释增益
正负，但表 2/表 3 只给准确率，图 2 又只覆盖 CUB 一个数据集，机制解释没有
可核对的数字支撑。本脚本补出这些数字。

指标定义（均基于 Novel 集特征的协方差矩阵特征值 λ1≥…≥λd，Σλ=1）：
  · 有效秩      exp(−Σ p_i ln p_i)，p_i = λ_i/Σλ（谱熵的指数，即"参与比"的熵形式）
  · 参与比       (Σλ)² / Σλ²（participation ratio，对重尾更敏感）
  · 前10%能量   前 ⌈d/10⌉ 个特征值之和 / 总能量
  · 条件数       λ1 / λ_d

输出：final_out/spectrum_metrics.csv
"""
import csv
import io
import os

import numpy as np
from _paths import find_root, FINAL_OUT  # noqa: E402

ROOT = find_root()
FC = os.path.join(ROOT, "_tools", "feat_cache")
FO = FINAL_OUT
BBS = [("clean_multitask", "领域微调"), ("clean_species", "无属监督"),
       ("clip_vitb16", "CLIP ViT-B/16"), ("dinov2_vitb14", "DINOv2 ViT-B/14")]
# 各数据集 Novel 侧的缓存名（与 final_eval 的评测对象一致）
# 白化矩阵在 Base 特征上拟合，故谱指标必须用 Base 划分计算（原为 novel，与标题不符）。
# 濒危集不自带 Base 划分、复用 CUB 的 Base 类，其谱即 CUB Base，故不重复列出。
DS = [("cub", "cub_base"), ("nab", "nab_base"), ("endb", "endb")]


def metrics(X):
    X = X.astype(np.float64)
    X = X - X.mean(0, keepdims=True)
    n = X.shape[0]
    C = (X.T @ X) / max(n - 1, 1)
    w = np.linalg.eigvalsh(C)[::-1]
    w = np.clip(w, 0, None)
    tot = w.sum()
    if tot <= 0:
        return None
    p = w / tot
    nz = p[p > 0]
    eff_rank = float(np.exp(-(nz * np.log(nz)).sum()))
    part = float(tot ** 2 / (w ** 2).sum())
    k = max(1, int(np.ceil(len(w) * 0.1)))
    top10 = float(w[:k].sum() / tot * 100.0)
    cond = float(w[0] / w[-1]) if w[-1] > 1e-12 else float("inf")
    return dict(dim=len(w), eff_rank=eff_rank, part_ratio=part,
                top10_pct=top10, cond=cond)


def find(ds_key):
    """匹配缓存文件：优先 <ds_key>__<bb>.npz，回退到任意含该前缀者。"""
    hits = {}
    for bb, _ in BBS:
        for cand in ("%s__%s.npz" % (ds_key, bb), "%s_%s.npz" % (ds_key, bb)):
            p = os.path.join(FC, cand)
            if os.path.exists(p):
                hits[bb] = p
                break
    return hits


def main():
    rows = []
    for dsk, cache in DS:
        hits = find(cache)
        print("=== %s（找到 %d/%d 个主干缓存）===" % (dsk, len(hits), len(BBS)))
        for bb, lab in BBS:
            p = hits.get(bb)
            if not p:
                print("   %-18s 缓存缺失" % lab)
                continue
            with np.load(p) as z:
                X = z["feats"]
            m = metrics(X)
            if not m:
                continue
            print("   %-18s 维%4d 有效秩 %6.1f 参与比 %6.1f 前10%% %5.1f 条件数 %9.1f"
                  % (lab, m["dim"], m["eff_rank"], m["part_ratio"],
                     m["top10_pct"], m["cond"]))
            rows.append(dict(dataset=dsk, backbone=bb, backbone_label=lab, **m))
        print()

    out = os.path.join(FO, "spectrum_metrics.csv")
    with io.open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "backbone", "backbone_label",
                                          "dim", "eff_rank", "part_ratio",
                                          "top10_pct", "cond"])
        w.writeheader()
        w.writerows(rows)
    print("saved %s（%d 行）" % (out, len(rows)))

    # 与校准增益的关系：把 discr_projection 的 zca 增益拉进来做相关
    dp = os.path.join(FO, "discr_projection.csv")
    if os.path.exists(dp):
        R = list(csv.DictReader(io.open(dp, encoding="utf-8-sig")))
        gain = {}
        for r in R:
            if r["method"] == "zca":
                try:
                    gain[(r["dataset"], r["backbone"])] = float(r["novel_gain"])
                except (KeyError, ValueError):
                    pass
        pairs = [(m["eff_rank"], gain[(m["dataset"], m["backbone"])], m)
                 for m in rows if (m["dataset"], m["backbone"]) in gain]
        if len(pairs) >= 4:
            x = np.array([a for a, _, _ in pairs])
            y = np.array([b for _, b, _ in pairs])
            r = float(np.corrcoef(x, y)[0, 1])
            print("\n有效秩 vs ZCA 增益（%d 个设定）：Pearson r = %+.3f" % (len(pairs), r))
            for a, b, m in sorted(pairs):
                print("   %-6s %-18s 有效秩 %6.1f  增益 %+6.2f" %
                      (m["dataset"], m["backbone_label"], a, b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
