# -*- coding: utf-8 -*-
"""
Method-level accuracy improvements, evaluated on any cached feature set.

These are the cheap levers that need no retraining:
  * centering + L2 re-normalisation (the step SimpleShot's paper shows matters most)
  * whitening the base feature space (PCA + shrinkage)
  * flipping TTA (already cached)
  * a support-trained linear probe
  * transductive label propagation (LaplacianShot-style) over the query set

Protocol matches the paper: 50 evaluation episodes with fresh seeds
(1000, 1010, ...), disjoint from any tuning block.

Usage:
    python improve_methods.py clip_vitb16
    python improve_methods.py genus_center clip_vitb16
"""
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from _paths import find_root, FEAT_CACHE  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = FEAT_CACHE
OUT = os.path.join(SCRIPT_DIR, "improve_out")
os.makedirs(OUT, exist_ok=True)

EVAL_SEEDS = [1000 + i * 10 for i in range(50)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(group, backbone):
    z = np.load(os.path.join(CACHE, "%s__%s.npz" % (group, backbone)), allow_pickle=True)
    return z["feats"], z["feats_flip"], z["species"], z["genus"]


def build_episodes(species, n_shot, seeds):
    by_sp = defaultdict(list)
    for idx, s in enumerate(species):
        by_sp[int(s)].append(idx)
    eps = []
    for sd in seeds:
        np.random.seed(sd)
        sup, qry = [], []
        for s, idxs in by_sp.items():
            perm = np.random.permutation(idxs)
            sup.extend(perm[:n_shot].tolist())
            qry.extend(perm[n_shot:].tolist())
        eps.append((np.array(sup), np.array(qry)))
    return eps


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


# ---------------------------------------------------------------------------
# whitening
# ---------------------------------------------------------------------------
def fit_whitening(base_feats, eps=0.1):
    """PCA whitening with shrinkage on the base feature matrix."""
    X = base_feats - base_feats.mean(axis=0, keepdims=True)
    n, d = X.shape
    cov = (X.T @ X) / max(n - 1, 1)
    cov = cov * (1 - eps) + np.eye(d) * (eps * np.trace(cov) / d)
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, 1e-8, None)
    # whitening matrix, then re-scale by sqrt(top eigenvalue) to keep norms sane
    return v @ np.diag(1.0 / np.sqrt(w)) @ v.T


def apply_white(W, X):
    return l2(X @ W.T).astype(np.float32)


# ---------------------------------------------------------------------------
# per-episode methods
# ---------------------------------------------------------------------------
def m_nn(S, yS, Q, **kw):
    C = np.unique(yS)
    pred = yS[np.argmax(Q @ S.T, axis=1)]
    return C, pred


def _protos(S, yS):
    C = np.unique(yS)
    P = np.stack([S[yS == c].mean(axis=0) for c in C])
    return C, P


def m_protonet(S, yS, Q, **kw):
    C, P = _protos(S, yS)
    return C, C[np.argmax(Q @ l2(P).T, axis=1)]


def m_simpleshot(S, yS, Q, **kw):
    C, P = _protos(S, yS)
    d = np.linalg.norm(P[None, :, :] - Q[:, None, :], axis=2)
    return C, C[np.argmin(d, axis=1)]


def _center(S, Q, mu):
    return l2(S - mu[None, :]), l2(Q - mu[None, :])


def m_nn_center(S, yS, Q, **kw):
    """Subtract the support-set mean, L2-normalise, cosine kNN (SimpleShot-CL2 spirit)."""
    mu = S.mean(axis=0)
    S2, Q2 = _center(S, Q, mu)
    return np.unique(yS), yS[np.argmax(Q2 @ S2.T, axis=1)]


def m_protonet_center(S, yS, Q, **kw):
    mu = S.mean(axis=0)
    S2, Q2 = _center(S, Q, mu)
    C, P = _protos(S2, yS)
    return C, C[np.argmax(Q2 @ l2(P).T, axis=1)]


def m_protonet_basecenter(S, yS, Q, base_mu=None, **kw):
    if base_mu is None:
        return m_protonet(S, yS, Q)
    S2, Q2 = _center(S, Q, base_mu)
    C, P = _protos(S2, yS)
    return C, C[np.argmax(Q2 @ l2(P).T, axis=1)]


def _softmax(x, axis=-1, temp=1.0):
    x = x / temp
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def m_laplacian(S, yS, Q, alpha=0.75, k=20, temp=0.1, **kw):
    """Transductive label propagation over the query set (LaplacianShot-style)."""
    C, P = _protos(S, yS)
    Pn = l2(P)
    logits = Q @ Pn.T
    Y0 = _softmax(logits, axis=1, temp=temp)
    # kNN affinity on query features
    nq = Q.shape[0]
    sim = Q @ Q.T
    np.fill_diagonal(sim, -1.0)
    kk = min(k, nq - 1)
    idx = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
    W = np.zeros((nq, nq))
    rows = np.repeat(np.arange(nq), kk)
    W[rows, idx.ravel()] = np.clip(sim[rows, idx.ravel()], 0, None)
    W = np.maximum(W, W.T)
    d = W.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.clip(d, 1e-12, None))
    Ssym = W * dinv[:, None] * dinv[None, :]
    Y = np.linalg.solve(np.eye(nq) - alpha * Ssym, Y0)
    return C, C[np.argmax(Y, axis=1)]


def m_lr(S, yS, Q, **kw):
    """Linear probe on L2-normalised support features (200 Adam steps).

    Seeded so the probe is reproducible: it is the only evaluated rule that runs
    gradient descent.  Two sources of variation had to be removed:
      * random initialisation      -> torch.manual_seed(0)
      * cuDNN autotuning selecting different kernels across processes
        -> cudnn.deterministic / benchmark flags below
    Before both fixes the same run differed by up to 0.054 pp in acc_mean.
    """
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    C = np.unique(yS)
    c2i = {int(c): i for i, c in enumerate(C)}
    St = torch.tensor(l2(S), device=DEVICE, dtype=torch.float32)
    Qt = torch.tensor(l2(Q), device=DEVICE, dtype=torch.float32)
    yt = torch.tensor([c2i[int(l)] for l in yS], device=DEVICE)
    clf = nn.Linear(St.shape[1], len(C)).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=1e-4)
    clf.train()
    for _ in range(200):
        opt.zero_grad()
        F.cross_entropy(clf(St), yt).backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        return C, C[clf(Qt).argmax(dim=1).cpu().numpy()]


RULES = {
    "nn": m_nn,
    "protonet": m_protonet,
    "simpleshot": m_simpleshot,
    "nn_center": m_nn_center,
    "protonet_center": m_protonet_center,
    "protonet_basecenter": m_protonet_basecenter,
    "laplacian": m_laplacian,
    "lr": m_lr,
}


def evaluate(group, backbone, shots=(5,), use_base=False):
    feats, feats_flip, species, genus = load(group, backbone)
    base_mu = None
    whiten_W = None
    # whitening is always fitted on the BASE-class features of the same backbone
    # (never on the evaluation split); for CUB that is also the centering mean.
    try:
        bf, _, _, _ = load("cub_base", backbone)
        if group == "cub_novel":
            base_mu = bf.mean(axis=0)
        t0 = time.time()
        whiten_W = fit_whitening(bf)
        print("  whitening fitted on cub_base (%d images, %d-d), %.1fs"
              % (bf.shape[0], bf.shape[1], time.time() - t0))
    except FileNotFoundError:
        print("  !! no cub_base features for %s, skipping whitening" % backbone)

    sp2g = {}
    for s, g in zip(species, genus):
        sp2g[int(s)] = int(g)

    out = {}
    for shot in shots:
        eps = build_episodes(species, shot, EVAL_SEEDS)
        for kind, tta in (("plain", False), ("tta", True)):
            Fq = feats if not tta else l2(feats + feats_flip)
            for name, rule in RULES.items():
                if name == "protonet_basecenter" and base_mu is None:
                    continue
                accs = []
                t0 = time.time()
                for sup, qry in eps:
                    S, yS, Q, yQ = Fq[sup], species[sup], Fq[qry], species[qry]
                    kw = {}
                    if name == "protonet_basecenter":
                        kw["base_mu"] = base_mu
                    C, pred = rule(S, yS, Q, **kw)
                    accs.append(100.0 * (pred == yQ).mean())
                out[(shot, kind, name)] = np.array(accs)
                print("    %d-shot %-6s %-20s %.2f%%  (%.1fs)"
                      % (shot, kind, name, np.mean(accs), time.time() - t0), flush=True)
        # whitening variants (post-hoc, applied to features)
        if whiten_W is not None:
            Ft = apply_white(whiten_W, feats)
            Ft_tta = apply_white(whiten_W, l2(feats + feats_flip))
            for kind, Fq in (("plain", Ft), ("tta", Ft_tta)):
                for name in ("nn", "protonet", "simpleshot", "lr"):
                    rule = RULES[name]
                    accs = []
                    for sup, qry in eps:
                        C, pred = rule(Fq[sup], species[sup], Fq[qry])
                        accs.append(100.0 * (pred == species[qry]).mean())
                    out[(shot, kind, "white_" + name)] = np.array(accs)
                    print("    %d-shot %-6s %-20s %.2f%%"
                          % (shot, kind, "white_" + name, np.mean(accs)), flush=True)
    return out


def main(backbones, groups=(("cub", "cub_novel"), ("endb", "endb"))):
    summary = []
    for backbone in backbones:
        for ds, group in groups:
            if not os.path.exists(os.path.join(CACHE, "%s__%s.npz" % (group, backbone))):
                print("skip %s/%s (no cache)" % (ds, backbone))
                continue
            print("=" * 92)
            print("dataset=%s backbone=%s" % (ds, backbone), flush=True)
            print("=" * 92)
            shots = (5, 1) if ds == "cub" else (5,)
            res = evaluate(group, backbone, shots=shots, use_base=(group == "cub_novel"))
            for (shot, kind, name), arr in sorted(res.items()):
                summary.append([ds, backbone, shot, kind, name,
                                "%.3f" % arr.mean(), "%.3f" % arr.std()])
    with open(os.path.join(OUT, "method_improvements.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "backbone", "shot", "kind", "method", "acc_mean", "acc_std"])
        w.writerows(summary)

    # side-by-side table: plain vs TTA, per dataset/shot
    print()
    print("=" * 92)
    print("汇总（%）")
    print("=" * 92)
    for ds, group in groups:
        for backbone in backbones:
            keys = [k for k in summary if k[0] == ds and k[1] == backbone]
            if not keys:
                continue
            shots = sorted({k[2] for k in keys})
            methods = sorted({k[4] for k in keys})
            for shot in shots:
                print("\n%s | %s | %d-shot" % (ds, backbone, shot))
                print("%-24s %10s %10s" % ("method", "plain", "tta"))
                for m in methods:
                    p = [k for k in keys if k[2] == shot and k[3] == "plain" and k[4] == m]
                    t = [k for k in keys if k[2] == shot and k[3] == "tta" and k[4] == m]
                    if not p:
                        continue
                    print("%-24s %10s %10s" % (m, p[0][5], t[0][5] if t else "-"))
    print("\nsaved:", os.path.join(OUT, "method_improvements.csv"))


if __name__ == "__main__":
    bbs = sys.argv[1:] or ["clip_vitb16"]
    main(bbs)
