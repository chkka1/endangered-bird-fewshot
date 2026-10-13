# -*- coding: utf-8 -*-
"""
核查 5-shot 与 1-shot 的属级可迁移性落差（供本地复跑）。

数据来源：_tools/final_out/final_main_table_calibrated.csv
  条件 A 属分层  (dataset=cub,       backbone=clean_multitask)
  条件 B 未见属  (dataset=cubunseen, backbone=unseen_multitask)
  落差 = A − B（正数表示去掉属级可迁移性后的损失）

本脚本同时核对稿件中的三处相关表述：
  · 摘要/引言/结论：5-shot 下降 7.0~12.0 个百分点（均值 9.0）
  · 摘要/引言/结论：1-shot 下降约 10.1 个百分点
  · 5.5 节：7 种方法中有 5 种在 1-shot 下落差更大

用法：
    & "D:\\soft\\<PY_DIST>\\envs\\<PY_ENV>\\python.exe" _tools\check_causal_drops.py
"""
from __future__ import annotations

import csv
import io
import os
import sys

try:                                     # 包内：共享路径模块
    from _paths import FINAL_OUT
    FO = FINAL_OUT
except ImportError:                      # 源环境：本地推导
    _ROOT = find_root()
    FO = os.path.join(_ROOT, '_tools', 'final_out')
CSV = os.path.join(FO, "final_main_table_calibrated.csv")

A_COND = ("cub", "clean_multitask")          # 属分层（可迁移率 100%）
B_COND = ("cubunseen", "unseen_multitask")   # 未见属（可迁移率 0%）

METHODS = [("nn", "NN"), ("protonet", "ProtoNet"), ("simpleshot", "SimpleShot"),
           ("fusion_only", "融合"), ("protoprior_GWP", "GWP"),
           ("baseaug", "Base 增广"), ("linearprobe", "线性探针")]


def main() -> int:
    if not os.path.exists(CSV):
        print("缺少 %s" % CSV)
        return 2
    rows = list(csv.DictReader(io.open(CSV, encoding="utf-8-sig")))

    A, SD = {}, {}
    for r in rows:
        try:
            kk = (r["dataset"], r["backbone"], r["shot"], r["key"])
            A[kk] = float(r["acc_mean"])
            SD[kk] = float(r["acc_std"]) if r.get("acc_std") else 0.0
        except (ValueError, KeyError):
            pass

    print("=" * 74)
    print("条件 A 属分层  : dataset=%s  backbone=%s" % A_COND)
    print("条件 B 未见属  : dataset=%s  backbone=%s" % B_COND)
    print("落差 = A − B（正数 = 去掉属级可迁移性后的损失，百分点）")
    print("=" * 74)
    print()
    print("%-12s %22s %22s %10s" % ("方法", "5-shot", "1-shot", "谁更大"))
    print("%-12s %10s %11s %10s %11s %s" % ("", "准确率", "落差", "准确率", "落差", ""))
    print("-" * 74)

    d5, d1, both = [], [], []
    for k, n in METHODS:
        a5, b5 = A.get(A_COND + ("5", k)), A.get(B_COND + ("5", k))
        a1, b1 = A.get(A_COND + ("1", k)), A.get(B_COND + ("1", k))
        if None in (a5, b5, a1, b1):
            print("%-12s %s" % (n, "数据缺失"))
            continue
        D5, D1 = a5 - b5, a1 - b1
        d5.append(D5)
        d1.append(D1)
        both.append((n, D5, D1))
        print("%-12s %5.2f→%5.2f %9.2f   %5.2f→%5.2f %9.2f   %s"
              % (n, a5, b5, D5, a1, b1, D1, "1-shot" if D1 > D5 else "5-shot"))

    if not d5:
        print("没有可比方法，退出")
        return 3

    print("-" * 74)
    print("%-12s %20.2f %20.2f" % ("均值落差", sum(d5) / len(d5), sum(d1) / len(d1)))
    print("%-12s %20.2f %20.2f" % ("最小落差", min(d5), min(d1)))
    print("%-12s %20.2f %20.2f" % ("最大落差", max(d5), max(d1)))
    print("%-12s %20.2f %20.2f" % ("极差", max(d5) - min(d5), max(d1) - min(d1)))
    print()

    n1_bigger = sum(1 for _, x, y in both if y > x)
    print("1-shot 落差更大的方法数：%d / %d" % (n1_bigger, len(both)))
    print()

    # ---- 与稿件表述逐条对照 ----
    print("=" * 74)
    print("稿件表述核对")
    print("=" * 74)
    lo, hi = min(d5), max(d5)
    mean5 = sum(d5) / len(d5)
    mean1 = sum(d1) / len(d1)
    checks = [
        ("摘要/引言/结论：5-shot 下降 7.0~12.0",
         "%.1f~%.1f" % (lo, hi), abs(lo - 7.0) < 0.05 and abs(hi - 12.0) < 0.05),
        ("摘要/引言/结论：5-shot 均值 9.0",
         "%.1f" % mean5, abs(mean5 - 9.0) < 0.05),
        ("摘要/引言/结论：1-shot 约 10.1",
         "%.1f" % mean1, abs(mean1 - 10.1) < 0.05),
        ("5.5 节：7 种方法中有 5 种在 1-shot 下落差更大",
         "%d 种" % n1_bigger, n1_bigger == 5),
    ]
    ok = True
    for desc, got, good in checks:
        print("  [%s] %-46s 实测 %s" % ("OK" if good else "!!", desc, got))
        ok = ok and good
    print()
    print("总判定：%s" % ("全部与稿件一致" if ok else "有不一致，请核对"))
    print()
    print("备注：5-shot 各方法 episode 标准差 %.2f~%.2f；1-shot %.2f~%.2f"
          % (min(SD.get(A_COND + ("5", k), 0) for k, _ in METHODS),
             max(SD.get(A_COND + ("5", k), 0) for k, _ in METHODS),
             min(SD.get(A_COND + ("1", k), 0) for k, _ in METHODS),
             max(SD.get(A_COND + ("1", k), 0) for k, _ in METHODS)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
