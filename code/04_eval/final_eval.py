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

# -*- coding: utf-8 -*-
"""
Final unified evaluation: genus-prior ablation AND base-feature whitening in one
table, on any cached backbone, with the same-genus / cross-genus decomposition
and paired t-tests.

Design note: whitening is applied to the cached feature matrices *before* the
episode objects are built, so every scoring rule (including the whole genus-prior
family) can be evaluated on whitened features without duplicating code.

Protocol: 50 evaluation episodes with fresh seeds (1000, 1010, ...), disjoint
from the tuning block.  Hyper-parameters are re-used from rerun_out/tuning_grid.json
(selected on the tuning block only).  CUB and the endangered set share them.

Usage:
    python final_eval.py genus_center clip_vitb16 dinov2_vitb14
"""
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import rerun_ablation as ra            # noqa: E402  (Ep, RULES, evaluate_rule)
from improve_methods import (          # noqa: E402
    load, build_episodes, l2, fit_whitening, apply_white, EVAL_SEEDS)

OUT = os.path.join(SCRIPT_DIR, "final_out")
os.makedirs(OUT, exist_ok=True)
CACHE = ra.CACHE

# methods re-evaluated on whitened features (the rest are unaffected in spirit)
WHITE_KEYS = ["nn", "protonet", "simpleshot", "fusion_only",
              "fusion_calib", "protoprior_GWP", "linearprobe"]
WHITE_TTA_KEYS = ["simpleshot_tta", "fusion_only_tta", "protoprior_tta"]

METHOD_LABEL = {
    "nn": "NN 基线（余弦 kNN）",
    "protonet": "ProtoNet（类原型均值）",
    "simpleshot": "SimpleShot（欧氏最近类均值）",
    "matchingnet": "MatchingNet（注意力 kNN）",
    "gsingle": "全局单属提权（无掩码）",
    "ablationA_single_local": "消融A：单属＋局部掩码",
    "ablationB_dual_global": "消融B：双属（无掩码）",
    "full_masked": "Full：双属＋局部掩码",
    "full_nomask": "Full（去掩码，==消融B）",
    "full_oldparam": "Full（原硬编码超参）",
    "calibrated": "z 校准＋双属",
    "baseaug": "属先验＋Base 属原型",
    "fusion_only": "仅原型-实例融合（无属先验）",
    "fusion_calib": "融合＋z 校准（无属先验）",
    "protoprior_GWP": "GWP/ProtoPrior（属先验版）",
    "protoprior_oldparam": "GWP（原硬编码超参）",
    "protoprior_nomask": "GWP（去局部掩码）",
    "linearprobe": "线性探针",
    "simpleshot_tta": "SimpleShot＋TTA",
    "protonet_tta": "ProtoNet＋TTA",
    "fusion_only_tta": "仅融合＋TTA",
    "protoprior_tta": "GWP＋TTA",
    "protoprior_oldparam_tta": "GWP＋TTA（原超参）",
    # whitened variants
    "white_nn": "★白化＋NN",
    "white_protonet": "★白化＋ProtoNet",
    "white_simpleshot": "★白化＋SimpleShot",
    "white_fusion_only": "★白化＋仅融合",
    "white_fusion_calib": "★白化＋融合＋z校准",
    "white_protoprior_GWP": "★白化＋GWP",
    "white_linearprobe": "★白化＋线性探针",
    "white_simpleshot_tta": "★白化＋SimpleShot＋TTA",
    "white_fusion_only_tta": "★白化＋仅融合＋TTA",
    "white_protoprior_tta": "★白化＋GWP＋TTA",
}
ORDER = list(METHOD_LABEL)


def macro_prf(pred, y_true):
    """macro Precision / Recall / F1（%）。对该轮出现的每个类分别求，再取平均。

    micro 口径在单标签分类下恒等于 Accuracy，故不采用。
    """
    ps, rs, fs = [], [], []
    for c in np.unique(y_true):
        tp = int(((pred == c) & (y_true == c)).sum())
        fp = int(((pred == c) & (y_true != c)).sum())
        fn = int(((pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        ps.append(p); rs.append(r); fs.append(f)
    if not ps:
        return 0.0, 0.0, 0.0
    return 100.0 * float(np.mean(ps)), 100.0 * float(np.mean(rs)), 100.0 * float(np.mean(fs))


def load_params():
    p = os.path.join(ra.OUT, "tuning_grid.json")
    with open(p, encoding="utf-8") as f:
        tun = json.load(f)
    return tun["best_proto"], tun["best_sample"], tun


def make_params(bp, bs, has_base, base_proto, base_mu, base_sig):
    common = dict(calib=0.02, base_w=0.5, use_mask=True,
                  base_proto=base_proto if has_base else None,
                  base_mu=base_mu if has_base else None,
                  base_sigma=base_sig if has_base else None)
    p_proto = dict(common, fw=bp["fw"], p1=bp["p1"], p2=1.0 + (bp["p1"] - 1.0) * 0.8,
                   top_k=bp["top_k"], cand=bp["cand"], b1=1.0, b2=1.0)
    p_sample = dict(common, fw=0.6, p1=bs["b1"], p2=1.0 + (bs["b1"] - 1.0) * 0.8,
                    top_k=bs["top_k"], cand=bs["cand"],
                    b1=bs["b1"], b2=1.0 + (bs["b1"] - 1.0) * 0.8)
    return p_proto, p_sample


def build_base_library(group, backbone):
    """Genus prototypes / intra-genus similarity stats from a base-class feature set."""
    bf, _, bspecies, bgenus = load(group, backbone)
    proto, mu, sig = {}, {}, {}
    for g in np.unique(bgenus):
        fg = bf[bgenus == g]
        p = fg.mean(axis=0)
        proto[int(g)] = p / max(np.linalg.norm(p), 1e-12)
        n = fg.shape[0]
        if n >= 2:
            inner = fg @ fg.T
            triu = inner[np.triu_indices(n, k=1)]
            mu[int(g)] = float(triu.mean())
            sig[int(g)] = float(triu.std()) + 1e-6
        else:
            mu[int(g)], sig[int(g)] = 0.0, 1.0
    return proto, mu, sig


# dataset -> (evaluation group, base group for whitening/prototypes, shot list)
# The endangered set has no Base split of its own, so its whitening matrix is fitted
# on the CUB base features of the same backbone: that uses only base-class data and
# never touches the endangered evaluation images.
DATASETS = [
    ("cub", "cub_novel", "cub_base", [1, 5]),
    ("endb", "endb", "cub_base", [5]),
    ("nab", "nab_novel", "nab_base", [1, 5]),
    # unseen-genus control: Base 145 classes / 101 genera, Novel 55 classes with
    # genera disjoint from Base (0% genus overlap)
    ("cubunseen", "cubunseen_novel", "cubunseen_base", [1, 5]),
]


def run_backbone(backbone, bp, bs):
    rows, sig_rows = [], []
    # 可用 FE_DATASETS 限定数据集，避免为改一个数据集而重算全部
    import os as _o
    _want = {x.strip() for x in (_o.environ.get("FE_DATASETS") or "").split(",") if x.strip()}
    for ds, group, base_group, shots in DATASETS:
        if _want and ds not in _want:
            continue
        fpath = os.path.join(CACHE, "%s__%s.npz" % (group, backbone))
        if not os.path.exists(fpath):
            print("skip %s/%s (no cache)" % (ds, backbone))
            continue
        feats, feats_flip, species, genus = load(group, backbone)
        has_base = base_group is not None
        base_proto = base_mu = base_sig = None
        W = None
        try:
            bf, _, _, _ = load(base_group, backbone)
            base_proto, base_mu, base_sig = build_base_library(base_group, backbone)
            t0 = time.time()
            W = fit_whitening(bf)
            print("  [%s/%s] whitening fitted on %s (%d imgs, %d-d) %.1fs"
                  % (ds, backbone, base_group, bf.shape[0], bf.shape[1], time.time() - t0),
                  flush=True)
        except FileNotFoundError:
            print("  [%s/%s] no %s cache, no whitening" % (ds, backbone, base_group))
        p_proto, p_sample = make_params(bp, bs, has_base, base_proto, base_mu, base_sig)

        for shot in shots:
            idx = build_episodes(species, shot, EVAL_SEEDS)
            # One episode at a time, scored by every rule, keeps peak memory low:
            # holding 50 episodes for NABirds cost ~7.7 GiB and blew up the run.
            fw = fwf = None
            if W is not None:
                fw = apply_white(W, feats)
                fwf = apply_white(W, feats_flip)

            acc = defaultdict(list)
            tot = defaultdict(list)
            cro = defaultdict(list)
            sam = defaultdict(list)
            prf = defaultdict(list)      # 逐轮 (macroP, macroR, macroF1)

            for s, q in idx:
                ep = ra.Ep(feats, feats_flip, species, genus, s, q, use_tta=False)
                ep_tta = ra.Ep(feats, feats_flip, species, genus, s, q, use_tta=True)
                wep = wep_tta = None
                if fw is not None:
                    wep = ra.Ep(fw, fwf, species, genus, s, q, use_tta=False)
                    wep_tta = ra.Ep(fw, fwf, species, genus, s, q, use_tta=True)

                for name, rule in ra.RULES.items():
                    params = p_sample if name in ra.SAMPLE_KEYS else p_proto
                    _pred = rule(ep, params)
                    a, t, c, m2 = ra.score_one(_pred, ep)
                    acc[name].append(a); tot[name].append(t); cro[name].append(c); sam[name].append(m2)
                    prf[name].append(macro_prf(_pred, ep.yQ))
                for name, rule in ra.TTA_RULES.items():
                    params = p_proto if name in ra.TTA_PROTO_KEYS else p_sample
                    _pred = rule(ep_tta, params)
                    a, t, c, m2 = ra.score_one(_pred, ep_tta)
                    acc[name].append(a); tot[name].append(t); cro[name].append(c); sam[name].append(m2)
                    prf[name].append(macro_prf(_pred, ep_tta.yQ))
                if wep is not None:
                    for name in WHITE_KEYS:
                        params = p_sample if name in ra.SAMPLE_KEYS else p_proto
                        _pred = ra.RULES[name](wep, params)
                        a, t, c, m2 = ra.score_one(_pred, wep)
                        k = "white_" + name
                        acc[k].append(a); tot[k].append(t); cro[k].append(c); sam[k].append(m2)
                        prf[k].append(macro_prf(_pred, wep.yQ))
                    for name in WHITE_TTA_KEYS:
                        _pred = ra.TTA_RULES[name](wep_tta, p_proto)
                        a, t, c, m2 = ra.score_one(_pred, wep_tta)
                        k = "white_" + name
                        acc[k].append(a); tot[k].append(t); cro[k].append(c); sam[k].append(m2)
                        prf[k].append(macro_prf(_pred, wep_tta.yQ))

            res = {k: (np.array(acc[k]), np.array(tot[k]), np.array(cro[k]),
                       np.array(sam[k]), np.array(prf[k]))
                   for k in acc}

            print("  [%s/%s/%d-shot] done" % (ds, backbone, shot), flush=True)
            for name in ORDER:
                if name not in res:
                    continue
                acc, tot, cro, sam, _prf = res[name]
                _pa = np.array(_prf) if len(_prf) else np.zeros((1, 3))
                rows.append([ds, backbone, shot, name, METHOD_LABEL[name],
                             "%.3f" % acc.mean(), "%.3f" % acc.std(),
                             "%.2f" % tot.mean(), "%.2f" % cro.mean(),
                             "%.2f" % sam.mean(),
                             "%.3f" % (100.0 * cro.mean() / max(tot.mean(), 1e-9)),
                             len(EVAL_SEEDS),
                             "%.3f" % _pa[:, 0].mean(), "%.3f" % _pa[:, 1].mean(),
                             "%.3f" % _pa[:, 2].mean(),
                             "%.3f" % _pa[:, 0].std(), "%.3f" % _pa[:, 1].std(),
                             "%.3f" % _pa[:, 2].std()])

            # paired t-tests of the best whitened method vs the classical baselines
            for key, mk in (("acc", 0), ("cross", 1), ("total", 2), ("same", 3)):
                if ("white_protoprior_GWP" not in res) or ("nn" not in res):
                    continue
                a = res["white_protoprior_GWP"][mk]
                for other in ("nn", "protonet", "simpleshot", "fusion_only",
                              "protoprior_GWP", "linearprobe", "simpleshot_tta"):
                    if other not in res:
                        continue
                    b = res[other][mk]
                    t, pv = stats.ttest_rel(a, b)
                    d = a - b
                    ci = (stats.t.interval(0.95, len(d) - 1, loc=d.mean(),
                                           scale=stats.sem(d))
                          if d.std() > 0 else (np.nan, np.nan))
                    star = ("***" if pv < 0.001 else "**" if pv < 0.01
                            else "*" if pv < 0.05 else "ns")
                    sig_rows.append([ds, backbone, shot, key,
                                     "白化GWP vs " + other, "%.4f" % d.mean(),
                                     "%.3f" % t, "%.4f" % pv, star,
                                     "[%.3f,%.3f]" % (ci[0], ci[1])])
    return rows, sig_rows


def _set_determinism(seed=0):
    """固定随机性来源，使迭代型规则（线性探针）可复现。

    闭式解规则本就逐位确定，此处的意义在于消除线性探针的末位漂移。
    """
    import random as _r
    import numpy as _n
    _r.seed(seed)
    _n.random.seed(seed)
    try:
        import torch as _t
        _t.manual_seed(seed)
        if _t.cuda.is_available():
            _t.cuda.manual_seed_all(seed)
        _t.backends.cudnn.deterministic = True
        _t.backends.cudnn.benchmark = False
        # warn_only：个别算子若无确定实现，只告警不中断
        _t.use_deterministic_algorithms(True, warn_only=True)
    except Exception as _e:
        print("  [warn] 确定性设置部分失败: %s" % _e)


def main(backbones):
    _set_determinism(0)
    import os as _o2
    _want = {x.strip() for x in (_o2.environ.get("FE_DATASETS") or "").split(",") if x.strip()}
    bp, bs, tun = load_params()
    print("复用调参结果（仅在调参块上选出）: proto fw=%.2f p1=%.3f top_k=%d cand=%d | "
          "sample b1=%.2f top_k=%d cand=%d"
          % (bp["fw"], bp["p1"], bp["top_k"], bp["cand"],
             bs["b1"], bs["top_k"], bs["cand"]))
    all_rows, all_sig = [], []
    for bb in backbones:
        print("=" * 92)
        print("backbone = %s" % bb, flush=True)
        print("=" * 92)
        r, s = run_backbone(bb, bp, bs)
        all_rows += r
        all_sig += s

    # 用 FE_DATASETS 只重算部分数据集时，其余数据集的旧行必须保留，
    # 否则整文件被覆盖（本项目已因此丢失过 cubunseen 与显著性行）。
    # 保留条件必须同时看**数据集**与**主干**：
    # 只按数据集过滤会在 FE_BACKBONES 限定时丢掉被重算数据集的其余主干
    # （本项目因此把基表从 924 行截到 594 行，丢失 330 行）。
    _want_bb = set(backbones)
    _keep_ds = {ds for ds, _g, _b, _s in DATASETS if _want and ds not in _want}

    def _still_needed(row):
        """该旧行是否应保留：数据集或主干任一不在重算范围内即保留。"""
        ds_, bb_ = row[0], row[1]
        if _keep_ds and ds_ in _keep_ds:
            return True
        if _bb_want and bb_ not in _bb_want:
            return True
        return False

    _bb_want = set(x.strip() for x in (os.environ.get("FE_BACKBONES") or "").split(",")
                   if x.strip())
    if _keep_ds or _bb_want:
        _mp = os.path.join(OUT, "final_main_table.csv")
        if os.path.exists(_mp):
            with open(_mp, encoding="utf-8-sig") as f:
                kept = [r for r in list(csv.reader(f))[1:] if r and _still_needed(r)]
            all_rows = kept + all_rows
            print("保留未重算的行 %d 行" % len(kept))
        _sp = os.path.join(OUT, "final_significance.csv")
        if os.path.exists(_sp):
            with open(_sp, encoding="utf-8-sig") as f:
                ks = [r for r in list(csv.reader(f))[1:] if r and _still_needed(r)]
            all_sig = ks + all_sig
            print("保留其它数据集显著性 %d 行" % len(ks))

    with open(os.path.join(OUT, "final_main_table.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "backbone", "shot", "key", "method", "acc_mean", "acc_std",
                    "total_err", "cross_genus_err", "same_genus_err", "cross_share_pct",
                    "episodes", "precision", "recall", "f1",
                    "precision_std", "recall_std", "f1_std"])
        w.writerows(all_rows)
    with open(os.path.join(OUT, "final_significance.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "backbone", "shot", "metric", "comparison", "mean_diff",
                    "t_stat", "p_value", "sig", "ci95"])
        w.writerows(all_sig)

    # console summary for the current-genus backbones
    print()
    print("=" * 108)
    print("摘要（只列关键行；完整表见 final_out/final_main_table.csv）")
    print("=" * 108)
    for ds in ("cub", "endb", "nab"):
        for shot in ([5, 1] if ds in ("cub", "nab") else [5]):
            print("\n%s | %d-shot" % (ds, shot))
            print("%-26s %s" % ("方法", "  ".join("%-16s" % b for b in backbones)))
            keys = ["nn", "protonet", "simpleshot", "fusion_only", "protoprior_GWP",
                    "white_protonet", "white_linearprobe", "linearprobe"]
            for k in keys:
                cells = []
                for b in backbones:
                    m = [r for r in all_rows
                         if r[0] == ds and r[1] == b and r[2] == shot and r[3] == k]
                    cells.append("%-16s" % (m[0][5] + "%" if m else "-"))
                print("%-26s %s" % (METHOD_LABEL.get(k, k), "  ".join(cells)))
    print("\nsaved:", os.path.join(OUT, "final_main_table.csv"))


if __name__ == "__main__":
    # 默认主干原为 ["genus_center", ...]，与论文实际使用的清洁微调主干不符，
    # 直接跑会用错主干重建基表（曾因此丢失 594 行）。改为读环境变量。
    import os as _os
    _bb = _os.environ.get("FE_BACKBONES")
    _bb = _bb.split(",") if _bb else ["clean_multitask", "clean_species",
                                       "clip_vitb16", "dinov2_vitb14"]
    main(sys.argv[1:] or _bb)
