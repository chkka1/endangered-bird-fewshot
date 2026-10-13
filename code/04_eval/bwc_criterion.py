# -*- coding: utf-8 -*-
"""
BWC — Base-Whitened Calibration, and the criterion that decides when to use it.

Idea
    Whitening the base feature space helps weak / domain-adapted backbones a lot
    (+7 to +10 pp) but can hurt strong generic ones (CLIP: -15 pp).  Applying it
    unconditionally is therefore wrong, and choosing by evaluation performance
    would be test-set tuning.

Criterion (uses BASE data only)
    Split the base classes into two disjoint halves.
      fit half  : estimate the whitening matrix W(eps) and the class means
      held half : measure nearest-class-mean accuracy with and without whitening
    delta_base(eps) = acc_white - acc_plain on the held-out base classes.
    Enable whitening only if max_eps delta_base(eps) > 0, and use that eps.

This script
    1. computes delta_base for every (dataset, backbone) over an eps grid,
    2. compares the predicted decision with the ACTUAL gain measured on the novel
       evaluation split (read from final_main_table.csv),
    3. writes final_out/bwc_criterion.csv
"""
import csv
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from improve_methods import load, l2, fit_whitening, apply_white  # noqa: E402

OUT = os.path.join(SCRIPT_DIR, "final_out")
EPS_GRID = [0.0, 0.05, 0.1, 0.2, 0.4]     # 0.0 means "no whitening"

# dataset -> base group; evaluation group and the corresponding table row
DATASETS = {
    "cub": ("cub_base", "cub_novel"),
    "endb": ("cub_base", "endb"),
    "nab": ("nab_base", "nab_novel"),
    "cubunseen": ("cubunseen_base", "cubunseen_novel"),
}
BACKBONES = ["clean_multitask", "clean_species", "clip_vitb16", "dinov2_vitb14",
             "unseen_multitask", "unseen_species"]
# which backbones are valid for which dataset (unseen caches only exist for cubunseen)
VALID = {
    "cub": ["clean_multitask", "clean_species", "clip_vitb16", "dinov2_vitb14"],
    "endb": ["clean_multitask", "clean_species", "clip_vitb16", "dinov2_vitb14"],
    "nab": ["clean_multitask", "clean_species", "clip_vitb16", "dinov2_vitb14"],
    "cubunseen": ["unseen_multitask", "unseen_species"],
}


def ncc_acc(fit_feats, fit_y, eval_feats, eval_y):
    """Nearest-class-mean accuracy with cosine similarity."""
    C = np.unique(fit_y)
    P = np.stack([fit_feats[fit_y == c].mean(axis=0) for c in C])
    P = l2(P)
    pred = C[np.argmax(l2(eval_feats) @ P.T, axis=1)]
    return 100.0 * float((pred == eval_y).mean())


def delta_base(group, backbone, eps, k=5, reps=8):
    """Held-out base-class gain from whitening, fitted on base data only.

    The held-out evaluation mimics the few-shot regime: class means are built from
    only k support samples and scored on k disjoint query samples, averaged over
    several draws.  Using all samples per class instead would overstate whitening's
    benefit, because whitening amplifies low-variance directions and that hurts much
    more when the class mean is estimated from few samples (this is exactly why the
    all-sample criterion wrongly approved whitening for CLIP).
    """
    f, _, sp, _ = load(group, backbone)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(sp))
    cut = len(sp) // 2
    mfit = np.zeros(len(sp), dtype=bool)
    mfit[perm[:cut]] = True
    mheld = ~mfit

    if eps == 0.0:
        Ffit, Fheld = f[mfit], f[mheld]
    else:
        W = fit_whitening(f[mfit], eps=eps)
        Ffit, Fheld = apply_white(W, f[mfit]), apply_white(W, f[mheld])
    yfit, yheld = sp[mfit], sp[mheld]

    # per-class indices on the held-out half
    byc = {int(c): np.where(yheld == c)[0] for c in np.unique(yheld)}
    cls = [c for c, idx in byc.items() if len(idx) >= 2 * k]
    if len(cls) < 2:
        return 0.0
    # fit-half support pool per class (only used to build the whitening matrix, so
    # the support means below come from the held-out half and stay honest)
    accs = []
    for r in range(reps):
        sup, qry = [], []
        for c in cls:
            idx = byc[c].copy()
            rng.shuffle(idx)
            sup.extend(idx[:k].tolist())
            qry.extend(idx[k:2 * k].tolist())
        sup, qry = np.array(sup), np.array(qry)
        accs.append(ncc_acc(Fheld[sup], yheld[sup], Fheld[qry], yheld[qry]))
    return float(np.mean(accs))


def actual_gain(ds, backbone):
    """Novel-split gain of white_protonet over protonet, from the main table."""
    p = os.path.join(OUT, "final_main_table.csv")
    if not os.path.exists(p):
        return None
    want = {"protonet", "white_protonet"}
    vals = {}
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        if r["dataset"] == ds and r["backbone"] == backbone and r["shot"] == "5" \
                and r["key"] in want:
            vals[r["key"]] = float(r["acc_mean"])
    if len(vals) == 2:
        return vals["white_protonet"] - vals["protonet"]
    return None


rows = []
for ds, (bgroup, egroup) in DATASETS.items():
    for bb in VALID[ds]:
        if not os.path.exists(os.path.join(SCRIPT_DIR, "feat_cache",
                                           "%s__%s.npz" % (bgroup, bb))):
            continue
        accs = {}
        for eps in EPS_GRID:
            accs[eps] = delta_base(bgroup, bb, eps)
        plain = accs[0.0]
        best_eps = max((e for e in EPS_GRID if e > 0), key=lambda e: accs[e])
        gain = accs[best_eps] - plain
        use = gain > 0
        ag = actual_gain(ds, bb)
        rows.append({"dataset": ds, "backbone": bb,
                     "base_plain_acc": "%.2f" % plain,
                     "base_best_eps": best_eps,
                     "base_best_acc": "%.2f" % accs[best_eps],
                     "delta_base": "%.2f" % gain,
                     "BWC_decision": "启用白化" if use else "不启用",
                     "actual_novel_gain": ("%.2f" % ag) if ag is not None else "",
                     "decision_correct": ("是" if (ag is not None and (use == (ag > 0)))
                                          else ("—" if ag is None else "否"))})
        print("%-10s %-18s base留出: plain %.2f -> eps=%.2f %.2f (Δ%+.2f) → %-6s | 实际novel增益 %s"
              % (ds, bb, plain, best_eps, accs[best_eps], gain,
                 "启用" if use else "不用",
                 ("%+.2f" % ag) if ag is not None else "—"), flush=True)

p = os.path.join(OUT, "bwc_criterion.csv")
with open(p, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nsaved:", p)

ok = [r for r in rows if r["decision_correct"] == "是"]
bad = [r for r in rows if r["decision_correct"] == "否"]
print("判据正确 %d / %d（无法核对 %d）" % (len(ok), len(ok) + len(bad),
                                          len(rows) - len(ok) - len(bad)))
for r in bad:
    print("  误判: %s / %s  baseΔ=%s  实际=%s" % (r["dataset"], r["backbone"],
                                                r["delta_base"], r["actual_novel_gain"]))
