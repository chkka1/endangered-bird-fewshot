#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
正文聚合量核对  ——  把正文里的每个"算出来的数"重算一遍
=========================================================

追踪器报出 38 处正文数字不在任何数据源中。其中大部分不是"没有出处"，而是
**构建时算出来的聚合量**（均值、差值、计数），白名单里只有原始值，自然查不到。

本脚本把这些聚合量逐个重算，与稿件中的数字比对。这是真正能发现错误的一步：
聚合量的算式写错、或输入表换了而算式没跟着改，都会在这里暴露。

用法：
    python _tools\verify_prose_aggregates.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import sys

import numpy as np
from _paths import find_root, FINAL_OUT, find_paper  # noqa: E402

ROOT = find_root()
TOOLS = os.path.join(ROOT, "_tools")
FO = FINAL_OUT
def _default_docx():
    """未设 PAPER_DOCX 时，自动选 paper_code 下版本号最大的 v*.docx。

    原先默认指向 论文_修订稿v7_图表优化版.docx（旧稿），忘记设环境变量就会
    比对错稿件、结果无意义。改为自动选最新，避免这个陷阱。
    """
    pc = os.path.join(ROOT, "paper_code")
    if os.path.isdir(pc):
        cand = []
        for f in os.listdir(pc):
            m = re.match(r"^v(\d+)\.docx$", f)
            if m:
                cand.append((int(m.group(1)), os.path.join(pc, f)))
        if cand:
            cand.sort()
            return cand[-1][1]
    return os.path.join(pc, "论文_修订稿v7_图表优化版.docx")


DOCX = os.environ.get("PAPER_DOCX") or find_paper()

BB4 = ["clean_multitask", "clean_species", "clip_vitb16", "dinov2_vitb14"]


def load(fn):
    p = os.path.join(FO, fn)
    return list(csv.DictReader(io.open(p, encoding="utf-8-sig"))) if os.path.exists(p) else []


def docx_text():
    from docx import Document
    return "\n".join(p.text for p in Document(DOCX).paragraphs)


def main():
    DP = load("discr_projection.csv")
    F5 = load("final_5way_table.csv")
    CAL = [r for r in load("final_main_table_calibrated.csv") if r["shot"] == "5"]
    SIG = load("final_significance.csv")
    CM = load("criterion_match.csv")

    A = {(r["dataset"], r["backbone"], r["key"]): float(r["acc_mean"]) for r in CAL}
    F5A = {(r["dataset"], r["backbone"], r["key"]): float(r["acc_mean"]) for r in F5}

    text = docx_text()
    checks = []

    def chk(label, computed, claimed):
        ok = abs(float(computed) - float(claimed)) <= 0.02
        checks.append((ok, label, computed, claimed))

    # ---- 表 3 的三行聚合 ----
    def vals(key):
        return [float(r["novel_acc"]) for r in DP if r["method"] == key]

    sel = [float(r["novel_acc"]) for r in DP if r["method"].startswith("BWC-select")]
    # 本次实验实现的是对称(ZCA)白化，数据层的成员键为 "zca"（早期的 "pca" 已全部改名）
    zca = vals("zca")
    none = vals("none")
    within = vals("within")
    lda = vals("lda")
    if sel and zca:
        chk("方法族 12 设定均值（段2/98/151）", np.mean(sel), 68.29)
        chk("ZCA 白化 12 设定均值（段98）", np.mean(zca), 65.01)
        chk("方法族 − ZCA 白化（段2/98/151）", np.mean(sel) - np.mean(zca), 3.28)
        chk("固定判别投影的最差设定（段98）", min(lda), 32.11)
        chk("方法族最差 − 判别投影最差（段98）", min(sel) - min(lda), 10.84)

    # ---- 三级校准均值（段124）----
    three = {}
    for key, want in (("protonet", 61.46), ("uncond_protonet", 62.91),
                      ("cal_protonet", 68.29), ("linearprobe", 65.55),
                      ("uncond_linearprobe", 66.27), ("cal_linearprobe", 69.40)):
        v = [A[(d, b, key)] for d in ("cub", "nab", "endb") for b in BB4 if (d, b, key) in A]
        if v:
            chk("三级均值 %s（段124）" % key, np.mean(v), want)
            three[key] = np.mean(v)

    # ---- DINOv2 vs 领域微调（段125）----
    if ("cub", "dinov2_vitb14", "protonet") in A and ("cub", "clean_multitask", "protonet") in A:
        chk("DINOv2 − 领域微调（段125）",
            A[("cub", "dinov2_vitb14", "protonet")] - A[("cub", "clean_multitask", "protonet")], 31.93)

    # ---- 协议对比（段132）----
    def avg(k, tab):
        v = [tab[(d, b, k)] for d in ("cub", "nab", "endb") for b in BB4 if (d, b, k) in tab]
        return np.mean(v) if v else None
    allc = avg("cal_protonet", A), avg("protonet", A)
    if allc[0] and allc[1]:
        chk("全类 N 路：方法族 − 无校准（段132）", allc[0] - allc[1], 6.84)
    f5c = avg("cal_protonet", F5A), avg("protonet", F5A)
    if f5c[0] and f5c[1]:
        chk("5 路：方法族 − 无校准（段132）", f5c[0] - f5c[1], 1.12)

    # ---- 判据匹配（段91/92）----
    if CM:
        ndiff = sum(1 for r in CM if r["same"] != "相同")
        chk("判据分歧设定数（段91/92）", ndiff, 3)
        # 稿件算式：全部设定（含 0 与正值）的均值取绝对值，不是只对负值求平均
        d = [float(r["delta"]) for r in CM]
        if d:
            chk("判据错配平均代价（段92）", abs(np.mean(d)), 3.44)

    # ---- 显著性（5.4 节）----
    # 稿件：「以 CUB 为例，白化相对完整属先验方法提升 +9.15 个百分点，NABirds 上提升
    #        +7.06 个百分点，濒危集上提升 +7.66 个百分点，在三个数据集上均 p<0.001
    #        （50 轮配对 t 检验）」——三个数值取自 clean_multitask 主干、5-shot。
    h = [r for r in SIG if r["comparison"] == "白化GWP vs protoprior_GWP"
         and r["metric"] == "acc" and r["shot"] == "5" and r["backbone"] == "clean_multitask"]
    SIG_CLAIM = {"cub": 9.15, "nab": 7.06, "endb": 7.66}
    for ds, want in SIG_CLAIM.items():
        got = [r for r in h if r["dataset"] == ds]
        if not got:
            checks.append((False, "§5.4 显著性 %s（缺数据）" % ds, 0.0, want))
            continue
        r0 = got[0]
        chk("§5.4 白化GWP−属先验 %s" % ds, float(r0["mean_diff"]), want)
        # p<0.001 的声明：p_value 列为 0.0000 即视为满足
        p_ok = float(r0["p_value"]) < 0.001
        checks.append((p_ok, "§5.4 %s 的 p<0.001" % ds, float(r0["p_value"]), 0.001))

    # ---- 数据集统计 ----
    p = os.path.join(FO, "dataset_stats.json")
    if os.path.exists(p):
        st = json.load(io.open(p, encoding="utf-8"))
        chk("多物种属数（段104/113）", st.get("n_multi_genus", 0), 24)
        chk("同属物种对数（段113）", st.get("n_same_genus_pairs", 0), 26)
        chk("观察者数（段112）", st.get("n_observers", 0), 1014)

    # ---- 报告 ----
    print("=" * 92)
    print("正文聚合量核对  ——  重算 vs 稿件")
    print("=" * 92)
    nf = 0
    for ok, label, got, want in checks:
        nf += 0 if ok else 1
        print("  [%s] %-38s 重算 %9.2f  稿件 %9.2f  差 %+6.2f"
              % ("PASS" if ok else "FAIL", label, got, want, got - want))
    print("=" * 92)
    print("PASS %d | FAIL %d" % (len(checks) - nf, nf))
    if nf:
        print("\n不一致项需人工确认：是算式写错，还是输入表换了而算式没跟着改。")
    print("=" * 92)
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
