# -*- coding: utf-8 -*-
"""
Calibration family + rule-matched criterion  (paper chapter 4).

The candidate set is
    M = {none} ∪ {pca, within, lda} × eps
and the member is chosen by a criterion that scores k-shot accuracy on held-out
Base CLASSES using the SAME rule that will consume the calibration.  Using a
nearest-class-mean criterion to calibrate a linear probe is a methodological error:
verified to cost up to 10.17 pp (see _tools/criterion_matching_test.py).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 本次实现的是对称（ZCA）白化：W = Σ^{-1/2}，见 identify_whitening.py 的四项判别式。
# 标识符用 zca 而非 pca —— 通常所称的 PCA 白化取 W = Λ^{-1/2}U^T，二者相差一个旋转 U。
MEMBERS = ["zca", "within", "lda"]
# 兼容旧调用：接受 "pca" 作为 zca 的别名
ALIASES = {"pca": "zca"}
EPS_GRID = [0.05, 0.1, 0.2, 0.4]


def l2(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _inv_sqrt(M, eps):
    d = M.shape[0]
    M = (1 - eps) * M + eps * np.trace(M) / d * np.eye(d, dtype=M.dtype)
    w, V = np.linalg.eigh(M)
    return V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-10))) @ V.T


def fit(kind, f, y, eps):
    """Return W for one family member (None = identity)."""
    if kind == "none":
        return None
    f = f.astype(np.float64)
    kind = ALIASES.get(kind, kind)          # 旧名 pca 视为 zca
    if kind == "zca":
        fc = f - f.mean(0, keepdims=True)
        return _inv_sqrt((fc.T @ fc) / max(len(f) - 1, 1), eps)
    if kind == "within":
        d = f.shape[1]
        Sw = np.zeros((d, d))
        n = 0
        for c in np.unique(y):
            X = f[y == c]
            X = X - X.mean(0, keepdims=True)
            Sw += X.T @ X
            n += len(X)
        return _inv_sqrt(Sw / max(n - 1, 1), eps)
    if kind == "lda":
        W = fit("within", f, y, eps)
        Z = l2(f @ W)
        C = np.unique(y)
        M = np.stack([Z[y == c].mean(0) for c in C])
        M = M - M.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        return W @ Vt[:max(2, len(C) - 1)].T
    raise ValueError(kind)


def apply_W(W, f):
    return l2(f if W is None else f.astype(np.float64) @ W)


def _ncc(fs, ys, fq, yq):
    C = np.unique(ys)
    P = l2(np.stack([fs[ys == c].mean(0) for c in C]))
    return 100.0 * float((C[np.argmax(l2(fq) @ P.T, axis=1)] == yq).mean())


def _lr(fs, ys, fq, yq, steps=200):
    C = np.unique(ys)
    c2i = {int(c): i for i, c in enumerate(C)}
    St = torch.tensor(fs, device=DEVICE, dtype=torch.float32)
    Qt = torch.tensor(fq, device=DEVICE, dtype=torch.float32)
    yt = torch.tensor([c2i[int(v)] for v in ys], device=DEVICE)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    clf = nn.Linear(St.shape[1], len(C)).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(clf(St), yt).backward()
        opt.step()
    with torch.no_grad():
        pred = clf(Qt).argmax(1).cpu().numpy()
    return 100.0 * float((C[pred] == yq).mean())


def select(fb, yb, scorer="ncc", k=5, reps=5, seed=0, verbose=False):
    """Rule-matched criterion: fit the calibration on half the Base CLASSES and score
    k-shot accuracy on the other half using `scorer` ('ncc' or 'lr').

    Returns (kind, eps, W, heldout_score).  Ties are broken toward the SMALLER eps.
    """
    rng = np.random.RandomState(seed)
    cls = np.unique(yb)
    perm = rng.permutation(len(cls))
    half = len(cls) // 2
    fit_m = np.isin(yb, cls[perm[:half]])
    ev_m = np.isin(yb, cls[perm[half:]])
    Fev_cache = {}
    byc = None
    best = None
    for kind in ["none"] + MEMBERS:
        for eps in ([0.0] if kind == "none" else EPS_GRID):
            W = fit(kind, fb[fit_m], yb[fit_m], eps)
            Fev = apply_W(W, fb[ev_m])
            yh = yb[ev_m]
            if byc is None:
                byc = {int(c): np.where(yh == c)[0] for c in np.unique(yh)}
                keep = [c for c, idx in byc.items() if len(idx) >= 2 * k]
                draws = []
                for _ in range(reps):
                    sup, qry = [], []
                    for c in keep:
                        idx = byc[c].copy(); rng.shuffle(idx)
                        sup += idx[:k].tolist(); qry += idx[k:2 * k].tolist()
                    draws.append((np.array(sup), np.array(qry)))
            accs = []
            for sup, qry in draws:
                accs.append(_ncc(Fev[sup], yh[sup], Fev[qry], yh[qry]) if scorer == "ncc"
                            else _lr(Fev[sup], yh[sup], Fev[qry], yh[qry]))
            sc = float(np.mean(accs))
            # tie-break toward the smaller eps
            key = (sc, -eps)
            if best is None or key > best[0]:
                best = (key, kind, eps, W, sc)
    _, kind, eps, W, sc = best
    if verbose:
        print("      判据(%s)选出 %s/eps=%.2f 留出 %.2f" % (scorer, kind, eps, sc))
    return kind, eps, W, sc
