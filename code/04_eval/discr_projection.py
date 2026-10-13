# -*- coding: utf-8 -*-
"""
Discriminative projection vs unsupervised whitening.

Motivation
    The paper's BWC uses PCA whitening, i.e. W = Sigma_total^{-1/2}, which ignores the
    Base class labels.  Class-means nearest-neighbour classification is exact when the
    classes share a covariance Sigma_w, in which case the optimal transform is
        W = Sigma_within^{-1/2}
    where Sigma_within is estimated from the labelled Base set.  This script compares:

        none    identity
        pca     Sigma_total^{-1/2}          (current BWC, unsupervised)
        within  Sigma_within^{-1/2}         (supervised, theoretically matched to NCC)
        lda     whiten by Sigma_w then PCA on the class means (supervised, dim <= C-1)

    and tests whether the BWC criterion — held-out Base accuracy under a simulated
    k-shot regime — correctly selects among them.

Run
    python discr_projection.py            # full comparison
    python discr_projection.py --quick    # one backbone, sanity check
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from improve_methods import build_episodes, load, EVAL_SEEDS  # noqa: E402

OUT = os.path.join(SCRIPT_DIR, "final_out")
EPS_GRID = [0.05, 0.1, 0.2, 0.4]

# dataset -> (novel group, base group, shot list)
SETS = [
    ("cub", "cub_novel", "cub_base", [5]),
    ("nab", "nab_novel", "nab_base", [5]),
    ("endb", "endb", "cub_base", [5]),
]
BACKBONES = ["clean_multitask", "clean_species", "clip_vitb16", "dinov2_vitb14"]


def l2(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _inv_sqrt(M, eps):
    d = M.shape[0]
    M = (1 - eps) * M + eps * np.trace(M) / d * np.eye(d, dtype=M.dtype)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 1e-10)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def cov_total(f):
    f = f - f.mean(0, keepdims=True)
    return (f.T @ f) / max(len(f) - 1, 1)


def cov_within(f, y):
    d = f.shape[1]
    Sw = np.zeros((d, d), dtype=np.float64)
    n = 0
    for c in np.unique(y):
        X = f[y == c]
        X = X - X.mean(0, keepdims=True)
        Sw += X.T @ X
        n += len(X)
    return Sw / max(n - 1, 1)


def fit(kind, f, y, eps):
    f = f.astype(np.float64)
    if kind == "none":
        return None
    if kind == "zca":
        return _inv_sqrt(cov_total(f), eps)
    if kind == "within":
        return _inv_sqrt(cov_within(f, y), eps)
    if kind == "lda":
        W = _inv_sqrt(cov_within(f, y), eps)
        Z = l2(f @ W)
        C = np.unique(y)
        M = np.stack([Z[y == c].mean(0) for c in C])
        M = M - M.mean(0, keepdims=True)
        # PCA on class means, keep the top (C-1) directions, then project
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        k = max(2, len(C) - 1)
        return W @ Vt[:k].T
    raise ValueError(kind)


def apply_W(W, f):
    return l2(f if W is None else f.astype(np.float64) @ W)


def ncc_acc(fit_f, fit_y, ev_f, ev_y):
    C = np.unique(fit_y)
    P = l2(np.stack([fit_f[fit_y == c].mean(0) for c in C]))
    pred = C[np.argmax(l2(ev_f) @ P.T, axis=1)]
    return 100.0 * float((pred == ev_y).mean())


def heldout_score(fb, yb, kind, eps, k=5, reps=5, seed=0, split="class"):
    """BWC-style criterion.

    split="class" (default) : fit the projection on HALF THE CLASSES and evaluate
        k-shot NCC on the other half.  This measures what actually matters —
        generalisation to unseen classes — and it is what stops a high-capacity
        discriminant (LDA, C-1 dims) from being chosen purely because it fits the
        classes it was fitted on.
    split="sample"          : the original scheme; fit and evaluate on the same
        classes, which saturates and cannot discriminate.
    """
    rng = np.random.RandomState(seed)
    if split == "class":
        cls = np.unique(yb)
        perm = rng.permutation(len(cls))
        half = len(cls) // 2
        fit_m = np.isin(yb, cls[perm[:half]])
        ev_m = np.isin(yb, cls[perm[half:]])
    else:
        perm = rng.permutation(len(yb))
        cut = len(yb) // 2
        fit_m = np.zeros(len(yb), bool)
        fit_m[perm[:cut]] = True
        ev_m = ~fit_m
    W = fit(kind, fb[fit_m], yb[fit_m], eps)
    F = apply_W(W, fb[ev_m])
    yh = yb[ev_m]
    byc = {int(c): np.where(yh == c)[0] for c in np.unique(yh)}
    cls = [c for c, idx in byc.items() if len(idx) >= 2 * k]
    if len(cls) < 2:
        return 0.0
    accs = []
    for _ in range(reps):
        sup, qry = [], []
        for c in cls:
            idx = byc[c].copy(); rng.shuffle(idx)
            sup += idx[:k].tolist(); qry += idx[k:2 * k].tolist()
        sup, qry = np.array(sup), np.array(qry)
        accs.append(ncc_acc(F[sup], yh[sup], F[qry], yh[qry]))
    return float(np.mean(accs))


def novel_acc(fn, yn, fb, yb, kind, eps, shot):
    W = fit(kind, fb, yb, eps)
    F = apply_W(W, fn)
    eps_eps = build_episodes(yn, shot, EVAL_SEEDS)
    accs = []
    for sup, qry in eps_eps:
        accs.append(ncc_acc(F[sup], yn[sup], F[qry], yn[qry]))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="只跑指定数据集，逗号分隔")
    args = ap.parse_args()

    sets = SETS[:1] if args.quick else SETS
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        sets = [t for t in sets if t[0] in want]
    backs = ["clean_multitask"] if args.quick else BACKBONES
    KINDS = ["none", "zca", "within", "lda"]
    rows = []
    for ds, ngroup, bgroup, shots in sets:
        for bb in backs:
            if not os.path.exists(os.path.join(SCRIPT_DIR, "feat_cache",
                                               "%s__%s.npz" % (bgroup, bb))):
                print("skip %s/%s (no base cache)" % (ds, bb)); continue
            fn, _, yn, _ = load(ngroup, bb)
            fb, _, yb, _ = load(bgroup, bb)
            for shot in shots:
                print("=" * 74)
                print("%s / %s / %d-shot   (novel %d 张, base %d 张)"
                      % (ds, bb, shot, len(fn), len(fb)), flush=True)
                # --- criterion: pick method + eps on held-out Base
                crit = {}
                t0 = time.time()
                for k_ in KINDS:
                    if k_ == "none":
                        crit[("none", 0.0)] = heldout_score(fb, yb, "none", 0.0, k=shot)
                    else:
                        for e in EPS_GRID:
                            crit[(k_, e)] = heldout_score(fb, yb, k_, e, k=shot)
                # break ties toward the SMALLER eps (less aggressive whitening)
                (bk, be), bscore = max(crit.items(), key=lambda x: (x[1], -x[0][1]))
                base_none = crit[("none", 0.0)]
                print("  判据选出: %s eps=%.2f (留出 %.2f)  |  不白化 %.2f  |  %.0fs"
                      % (bk, be, bscore, base_none, time.time() - t0), flush=True)
                # --- unconditional results for every method
                for k_ in KINDS:
                    cand = [(crit[(k_, e)], e) for e in (EPS_GRID if k_ != "none" else [0.0])]
                    _, be2 = max(cand, key=lambda x: (x[0], -x[1]))
                    m, s = novel_acc(fn, yn, fb, yb, k_, be2, shot)
                    rows.append({"dataset": ds, "backbone": bb, "shot": shot,
                                 "method": k_, "eps": be2,
                                 "novel_acc": "%.3f" % m, "novel_std": "%.3f" % s,
                                 "heldout_base": "%.3f" % crit[(k_, be2)]})
                    print("    %-7s eps=%.2f  novel %.2f ± %.2f  (留出 %.2f)"
                          % (k_, be2, m, s, crit[(k_, be2)]), flush=True)
                # --- criterion-selected combination
                m, s = novel_acc(fn, yn, fb, yb, bk, be, shot)
                rows.append({"dataset": ds, "backbone": bb, "shot": shot,
                             "method": "BWC-select(%s)" % bk, "eps": be,
                             "novel_acc": "%.3f" % m, "novel_std": "%.3f" % s,
                             "heldout_base": "%.3f" % bscore})
                print("    >>> 判据选择: %s eps=%.2f → novel %.2f" % (bk, be, m), flush=True)

    p = os.path.join(OUT, "discr_projection.csv")
    # 用 --only 只重算部分数据集时，其余数据集的旧结果必须保留，否则会被整文件覆盖。
    # 这正是濒危集刷新后表 3 变陈旧的原因：discr_projection 没被重跑，
    # 而它一旦被重跑又会抹掉别的数据集。
    done = {t[0] for t in sets}
    keep = []
    if args.only and os.path.exists(p):
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            if r["dataset"] not in done:
                keep.append(r)
        print("保留其它数据集的 %d 行" % len(keep))
    fields = list(rows[0].keys())
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in keep] + rows)
    print("\nsaved: %s（%d 行）" % (p, len(keep) + len(rows)))

    # summary
    print("\n" + "=" * 74)
    print("汇总（novel 5-shot 准确率）")
    print("=" * 74)
    bym = defaultdict(list)
    for r in rows:
        bym[r["method"]].append(float(r["novel_acc"]))
    for k_ in sorted(bym, key=lambda x: -np.mean(bym[x])):
        print("  %-22s 平均 %.2f  (n=%d)" % (k_, np.mean(bym[k_]), len(bym[k_])))


if __name__ == "__main__":
    main()
