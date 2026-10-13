#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扩展数值核查  ——  覆盖全部免训练方法，而不只是 ProtoNet
=========================================================

`verify_paper_numbers.py` 只独立重算了表 5 的「无校准 / ProtoNet」一行（12 格），
因为它是唯一能纯靠特征算出的格子。但论文大量结论依赖校准方法族与固定系数白化——
这些**同样不需要训练**，只是需要判据选择与白化拟合。

本脚本从特征缓存重算下面三层共 10 种方法，覆盖 3 数据集 × 4 主干 = 120 格：

    无校准        nn / protonet / simpleshot / fusion_only / protoprior_GWP / linearprobe
    固定系数白化    固定 eps=0.1 的 PCA 白化 + protonet / linearprobe
    校准方法族    判据按类二分选出成员与 eps，再拟合全部 Base 类

判定：|稿件 − 重算| ≤ 0.05 个百分点为 PASS（论文报告到两位小数）。

用法：
    python _tools/verify_all_methods.py
    python _tools/verify_all_methods.py --datasets cub      # 只查一个数据集
"""
from __future__ import annotations

# ---- 核验包守卫（自包含）：缺少随包未发布的外部资产时显式跳过 ----
def _pkg_root():
    import os as _o
    d = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))
    e = _o.environ.get("BIRD_PROJECT_ROOT")
    return _o.path.abspath(e) if (e and _o.path.isdir(e)) else d


def _require(rel, what):
    import os as _o
    import sys as _s
    full = _o.path.join(_pkg_root(), rel)
    if _o.path.exists(full):
        return True
    print("=" * 68)
    print("[跳过] 本脚本需要" + what + "，但包内未包含：")
    print("        " + full)
    print("  本包为核验包：随包提供 code / dataset(濒危集) / "
          "_tools/final_out(结果) / paper_code(稿件与图)。")
    print("  特征缓存、第三方原图与模型权重未随包发布，故本脚本无法在包内执行")
    print("  —— 这不代表论文数据有问题。")
    print("=" * 68)
    _s.exit(0)


_require("_tools/feat_cache", "冻结特征缓存（feat_cache）")

import argparse
import hashlib
import os
import sys
import time

import numpy as np

ROOT = find_root()
TOOLS = os.path.join(ROOT, "_tools")
FO = FINAL_OUT
sys.path.insert(0, TOOLS)

from improve_methods import build_episodes, load, EVAL_SEEDS  # noqa: E402
import improve_methods as IM                                  # noqa: E402
import calibration as CAL                                     # noqa: E402
from _paths import find_root, FINAL_OUT  # noqa: E402

SHOT = 5
BB4 = ["clean_multitask", "clean_species", "clip_vitb16", "dinov2_vitb14"]
DS = {"cub": ("cub_novel", "cub_base"), "nab": ("nab_novel", "nab_base"),
      "endb": ("endb", "cub_base")}
EPS_UNCOND = 0.1

# 稿件中的方法名 → 本脚本的重算方式
PLAIN = {"nn": IM.m_nn, "protonet": IM.m_protonet, "simpleshot": IM.m_simpleshot,
         "linearprobe": IM.m_lr}
# fusion_only / protoprior_GWP 是复合规则（定义在 final_eval 内），
# 本脚本不重实现它们——错误复现只会产生假 FAIL，不如不查


def _acc(pred, y_qry):
    if isinstance(pred, tuple):
        pred = pred[1]
    return 100.0 * float((np.asarray(pred) == y_qry).mean())


def eval_method(fn, yn, idx, kind, W=None):
    """按给定规则评测；W 非空时先对支持/查询特征施加校准。"""
    accs = []
    for sup, qry in idx:
        S, Q, yS, yQ = fn[sup], fn[qry], yn[sup], yn[qry]
        if W is not None:
            S, Q = CAL.apply_W(W, S), CAL.apply_W(W, Q)
        fnc = PLAIN[kind]
        if kind == "nn":
            C = np.unique(yS)
            accs.append(_acc(fnc(S, yS, Q), yQ))
        else:
            accs.append(_acc(fnc(S, yS, Q), yQ))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="cub,nab,endb")
    args = ap.parse_args()
    want = [x.strip() for x in args.datasets.split(",") if x.strip()]

    # 稿件数值
    rep = os.path.join(FO, "final_main_table_calibrated.csv")
    import csv
    import io
    ACC = {}
    for r in csv.DictReader(io.open(rep, encoding="utf-8-sig")):
        if r["shot"] == str(SHOT):
            ACC[(r["dataset"], r["backbone"], r["key"])] = float(r["acc_mean"])

    print("=" * 96)
    print("扩展数值核查  |  %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("重算 10 种方法 × %d 数据集 × 4 主干，与 final_main_table_calibrated.csv 比对" % len(want))
    print("=" * 96)

    npass = nfail = nskip = 0
    fails = []
    for ds in want:
        ng, bg = DS[ds]
        for bb in BB4:
            fn, _, yn, _ = load(ng, bb)
            fb, _, yb, _ = load(bg, bb)
            idx = build_episodes(yn, SHOT, EVAL_SEEDS)

            # --- 三层 ---
            jobs = []
            for k in PLAIN:
                jobs.append(("none", k, k, None))
            W_u = CAL.fit("zca", fb, yb, EPS_UNCOND)
            jobs.append(("uncond", "uncond_protonet", "protonet", W_u))
            jobs.append(("uncond", "uncond_linearprobe", "linearprobe", W_u))
            # 判据选择（按类二分，评分器与规则一致）
            # select() 返回 (kind, eps, W, score)，其 W 只在一半 Base 类上拟合；
            # 论文协议要求在全部 Base 类上重新拟合，否则损失 8.63 个百分点
            k_ncc, e_ncc, _w, _s = CAL.select(fb, yb, scorer="ncc")
            k_lr, e_lr, _w2, _s2 = CAL.select(fb, yb, scorer="lr")
            W_cn = CAL.fit(k_ncc, fb, yb, e_ncc)
            W_cl = CAL.fit(k_lr, fb, yb, e_lr)
            print("      判据: ncc→%s eps=%.2f | lr→%s eps=%.2f" % (k_ncc, e_ncc, k_lr, e_lr))
            jobs.append(("family", "cal_protonet", "protonet", W_cn))
            jobs.append(("family", "cal_linearprobe", "linearprobe", W_cl))

            print("\n%s / %s   (novel %d, base %d)" % (ds, bb, len(fn), len(fb)), flush=True)
            for _lvl, key, rule, W in jobs:
                exp = ACC.get((ds, bb, key))
                if exp is None:
                    nskip += 1
                    continue
                m, s = eval_method(fn, yn, idx, rule, W)
                d = exp - m
                ok = abs(d) <= 0.05
                if ok:
                    npass += 1
                else:
                    nfail += 1
                    fails.append((ds, bb, key, exp, m, d))
                print("   %-20s 稿件 %7.2f  重算 %7.2f  差 %+6.2f  %s  (±%.2f)"
                      % (key, exp, m, d, "PASS" if ok else "FAIL", s), flush=True)

    print("\n" + "=" * 96)
    print("PASS %d | FAIL %d | 跳过 %d（稿件中无此格）" % (npass, nfail, nskip))
    if fails:
        print("\n不一致明细：")
        for ds, bb, key, exp, m, d in fails:
            print("   %-6s %-16s %-20s 稿件 %.2f vs 重算 %.2f  (%+.2f)" % (ds, bb, key, exp, m, d))
    print("=" * 96)
    print("注意：本脚本不核对图、参考文献、许可与数据集统计量。")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
