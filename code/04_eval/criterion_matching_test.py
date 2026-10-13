# -*- coding: utf-8 -*-
"""
Does the calibration criterion need to match the downstream rule?

The criterion picks a calibration member by scoring k-shot accuracy on held-out Base
classes.  If that score is computed with nearest-class-mean (the rule the calibration
was designed around) but the calibration is then used with a linear probe, we would be
tuning method Y with method X's validation criterion — a methodological error.

This script computes the selection under two scorers and compares what they pick:
    scorer="ncc" : nearest class mean on the calibrated features
    scorer="lr"  : linear probe trained on the k-shot support, evaluated on the query

If the two scorers pick different members, the criterion must be matched to the rule.
"""
import argparse
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import discr_projection as DP   # noqa: E402
from improve_methods import load  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KINDS = ["none", "zca", "within", "lda"]
EPS = [0.05, 0.1, 0.2, 0.4]


def lr_acc(F_sup, y_sup, F_qry, y_qry, steps=200):
    C = np.unique(y_sup)
    c2i = {int(c): i for i, c in enumerate(C)}
    St = torch.tensor(F_sup, device=DEVICE, dtype=torch.float32)
    Qt = torch.tensor(F_qry, device=DEVICE, dtype=torch.float32)
    yt = torch.tensor([c2i[int(v)] for v in y_sup], device=DEVICE)
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
    return 100.0 * float((C[pred] == y_qry).mean())


def score(fb, yb, kind, eps, k=5, reps=3, seed=0, scorer="ncc"):
    rng = np.random.RandomState(seed)
    cls = np.unique(yb)
    perm = rng.permutation(len(cls))
    half = len(cls) // 2
    fit_m = np.isin(yb, cls[perm[:half]])
    ev_m = np.isin(yb, cls[perm[half:]])
    W = DP.fit(kind, fb[fit_m], yb[fit_m], eps)
    Fev = DP.apply_W(W, fb[ev_m])
    yh = yb[ev_m]
    byc = {int(c): np.where(yh == c)[0] for c in np.unique(yh)}
    keep = [c for c, idx in byc.items() if len(idx) >= 2 * k]
    if len(keep) < 2:
        return 0.0
    accs = []
    for _ in range(reps):
        sup, qry = [], []
        for c in keep:
            idx = byc[c].copy(); rng.shuffle(idx)
            sup += idx[:k].tolist(); qry += idx[k:2 * k].tolist()
        sup, qry = np.array(sup), np.array(qry)
        if scorer == "ncc":
            accs.append(DP.ncc_acc(Fev[sup], yh[sup], Fev[qry], yh[qry]))
        else:
            accs.append(lr_acc(Fev[sup], yh[sup], Fev[qry], yh[qry]))
    return float(np.mean(accs))


SETS = [
    ("cub", "cub_novel", "cub_base", "clean_multitask"),
    ("cub", "cub_novel", "cub_base", "clip_vitb16"),
    ("cub", "cub_novel", "cub_base", "dinov2_vitb14"),
    ("nab", "nab_novel", "nab_base", "clean_multitask"),
    ("nab", "nab_novel", "nab_base", "clip_vitb16"),
    ("endb", "endb", "cub_base", "clip_vitb16"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=int, default=len(SETS))
    args = ap.parse_args()

    print("%-6s %-16s %-22s %-22s %s" % ("数据集", "主干", "NCC 判据选出", "线性探针判据选出", "是否相同"))
    print("-" * 92)
    same = 0
    for ds, ngroup, bgroup, bb in SETS[:args.sets]:
        fn, _, yn, _ = load(ngroup, bb)
        fb, _, yb, _ = load(bgroup, bb)
        res = {}
        t0 = time.time()
        for scorer in ("ncc", "lr"):
            cand = {}
            for k_ in KINDS:
                for e in ([0.0] if k_ == "none" else EPS):
                    cand[(k_, e)] = score(fb, yb, k_, e, scorer=scorer)
            best = max(cand.items(), key=lambda x: (x[1], -x[0][1]))
            res[scorer] = (best[0], best[1])
        (k1, e1), s1 = res["ncc"]
        (k2, e2), s2 = res["lr"]
        ok = (k1 == k2)
        same += ok
        # novel accuracy under each selection, to see whether it matters
        n1 = DP.novel_acc(fn, yn, fb, yb, k1, e1, 5)[0]
        n2 = DP.novel_acc(fn, yn, fb, yb, k2, e2, 5)[0]
        print("%-6s %-16s %-22s %-22s %s   (novel %.2f vs %.2f, 差 %+.2f, %.0fs)"
              % (ds, bb, "%s/eps=%.2f" % (k1, e1), "%s/eps=%.2f" % (k2, e2),
                 "相同" if ok else "不同", n1, n2, n2 - n1, time.time() - t0), flush=True)
    print()
    print("两判据选出相同成员的设定：%d/%d" % (same, min(args.sets, len(SETS))))


if __name__ == "__main__":
    main()
