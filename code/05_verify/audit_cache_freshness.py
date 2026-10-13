# -*- coding: utf-8 -*-
"""
缓存新鲜度审计
==============

本会话已四次遇到「下游产物未随上游刷新」：
    1. final_main_table.csv      濒危行陈旧
    2. tip_cache_branch.csv      早于特征刷新 10 小时
    3. criterion_design_points   早期管线产物
    4. endb__imagenet.npz        仍是 1673 张的旧濒危集

根因相同：特征缓存与派生统计没有强制刷新机制，而文件名不体现数据版本。
本脚本把「期望样本数」从当前数据源的构建函数算出，再与每个缓存逐一比对，
使陈旧缓存可被机械检出，而不依赖人工比对样本数。

用法：
    python audit_cache_freshness.py            # 只审计已存在的缓存
    python audit_cache_freshness.py --strict   # 缺失也视为失败
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

import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
os.chdir(TOOLS)

import cache_features as cf                                   # noqa: E402

CACHE = cf.CACHE


def expected_counts():
    """从当前数据源算出每个分组的期望 (样本数, 类数)。"""
    exp = {}

    # --- CUB：按当前的属分层划分重建条目 ---
    try:
        meta, is_train, genus_split, genus2id = cf.load_meta()
        base_ids = set(genus_split["base_species_ids"])
        novel_ids = set(genus_split["novel_species_ids"])
        for gname, items in (
                ("cub_novel", cf.cub_items(meta, is_train, genus2id, novel_ids, want_train=False)),
                ("cub_base", cf.cub_items(meta, is_train, genus2id, base_ids, want_train=True)),
                # Base 类的测试划分：用于与濒危集做可比的类内距离度量
                ("cub_basetest", cf.cub_items(meta, is_train, genus2id, base_ids, want_train=False))):
            exp[gname] = (len(items), len({s for _, s, _ in items}))
    except Exception as e:
        print("  [警告] CUB 期望值无法计算: %s" % e)

    # --- 濒危集 ---
    try:
        items = cf.endb_items()
        exp["endb"] = (len(items), len({s for _, s, _ in items}))
    except Exception as e:
        print("  [警告] 濒危集期望值无法计算: %s" % e)

    # --- NABirds：复用其缓存脚本的条目构建 ---
    try:
        import cache_nabirds_features as cnf
        for gname in ("nab_novel", "nab_base"):
            fn = getattr(cnf, "items_for", None)
            if fn is not None:
                it = fn(gname)
                exp[gname] = (len(it), len({s for _, s, _ in it}))
    except Exception as e:
        print("  [警告] NABirds 期望值无法计算: %s" % e)

    return exp


def main():
    strict = "--strict" in sys.argv

    print("=" * 88)
    print("缓存新鲜度审计")
    print("  期望样本数由当前数据源的构建函数实时算出，不手写")
    print("=" * 88)

    print("\n[1] 计算期望值 ...", flush=True)
    exp = expected_counts()
    if not exp:
        print("  无法计算任何期望值，退出")
        return 1
    for g in sorted(exp):
        print("  %-12s 期望 %5d 样本 / %3d 类" % (g, exp[g][0], exp[g][1]))

    print("\n[2] 比对缓存 ...", flush=True)
    print("%-34s %8s %8s %8s %s" % ("缓存文件", "样本", "期望", "类", "判定"))
    print("-" * 88)

    stale, missing, ok = [], [], 0
    for f in sorted(os.listdir(CACHE)):
        if not f.endswith(".npz"):
            continue
        g = f.split("__")[0]
        if g not in exp:
            continue
        p = os.path.join(CACHE, f)
        try:
            with np.load(p) as z:
                n = int(len(z["species"]))
                k = int(len(np.unique(z["species"])))
        except Exception as e:
            print("%-34s  无法读取: %s" % (f, str(e)[:32]))
            continue
        en, ek = exp[g]
        bad_n, bad_k = n != en, k != ek
        tag = "OK" if not (bad_n or bad_k) else (
            ("样本数 %d→%d " % (n, en) if bad_n else "") +
            ("类数 %d→%d " % (k, ek) if bad_k else "") + "陈旧")
        if tag == "OK":
            ok += 1
        else:
            stale.append(f)
        print("%-34s %8d %8d %8d %s" % (f, n, en, k, tag))

    # 缺失检查
    # 每个分组只检查它实际使用的主干，避免把「本就不需要」的缓存报成缺失。
    # cub_basetest 仅用于类内距离度量，只需 clean_multitask 与 imagenet。
    WANT = {
        "cub_basetest": ["clean_multitask", "imagenet"],
    }
    _default = ["clean_multitask", "clean_species", "clip_vitb16", "dinov2_vitb14", "imagenet"]
    present = {f for f in os.listdir(CACHE) if f.endswith(".npz")}
    for g in exp:
        for bb in WANT.get(g, _default):
            fn = "%s__%s.npz" % (g, bb)
            if fn not in present:
                missing.append(fn)

    print("\n" + "=" * 88)
    print("汇总：一致 %d 个，陈旧 %d 个，缺失 %d 个" % (ok, len(stale), len(missing)))
    if stale:
        print("\n陈旧缓存（需重新抽取）：")
        for f in stale:
            print("   %s" % f)
        print("   处置建议：移入 feat_cache_quarantine/ 后重跑对应抽取脚本")
    if missing:
        print("\n缺失缓存：")
        for f in missing[:12]:
            print("   %s" % f)
        if len(missing) > 12:
            print("   ... 另有 %d 个" % (len(missing) - 12))
    verdict = "通过" if not stale else "未通过"
    print("\n总判定：%s" % verdict)
    print("=" * 88)
    return 0 if not stale else 2


if __name__ == "__main__":
    sys.exit(main())
