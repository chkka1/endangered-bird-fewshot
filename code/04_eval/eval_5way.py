# -*- coding: utf-8 -*-
"""
5-way evaluation, matching the convention used throughout the few-shot literature.

Why: the main table reports ALL-class N-way (CUB 55, NABirds 146, endangered 70), which
is much harder than the conventional 5-way.  A reviewer comparing our 60.91% on CUB with
the ~70-80% reported in papers would conclude the method is worse, when in fact the
protocols differ.  Reporting both removes that misreading entirely.

Differences from build_episodes (all-class):
  * each episode samples `n_way` classes WITHOUT replacement from the novel set, so
    different episodes use different class subsets (proper 5-way sampling);
  * support/query split per selected class as before.

Writes _tools/final_out/final_5way_table.csv
"""
import csv
import os
import sys
import time
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn as nn                                       # noqa: E402
import torch.nn.functional as TF                            # noqa: E402
import calibration as CAL                                   # noqa: E402
from improve_methods import build_episodes, load, EVAL_SEEDS, l2  # noqa: E402

FO = os.path.join(SCRIPT_DIR, "final_out")
N_WAY = 5
SHOT = 5
SETS = [("cub", "cub_novel", "cub_base"),
        ("nab", "nab_novel", "nab_base"),
        ("endb", "endb", "cub_base")]
BACKBONES = ["clean_multitask", "clean_species", "clip_vitb16", "dinov2_vitb14"]
LABEL = {"protonet": "ProtoNet（无校准）",
         "uncond_protonet": "固定系数白化 + ProtoNet",
         "cal_protonet": "校准方法族 + ProtoNet",
         "linearprobe": "线性探针（无校准）",
         "uncond_linearprobe": "固定系数白化 + 线性探针",
         "cal_linearprobe": "校准方法族 + 线性探针"}
KINDS = ["protonet", "uncond_protonet", "cal_protonet",
         "linearprobe", "uncond_linearprobe", "cal_linearprobe"]


def build_nway(species, n_shot, seeds, n_way=N_WAY):
    """Fixed n_way classes per episode, sampled without replacement."""
    by_sp = defaultdict(list)
    for idx, s in enumerate(species):
        by_sp[int(s)].append(idx)
    classes = sorted(by_sp)
    eps = []
    for sd in seeds:
        rng = np.random.RandomState(sd)
        chosen = rng.choice(classes, size=min(n_way, len(classes)), replace=False)
        sup, qry = [], []
        for c in chosen:
            idxs = np.array(by_sp[int(c)])
            perm = rng.permutation(idxs)
            sup.extend(perm[:n_shot].tolist())
            qry.extend(perm[n_shot:].tolist())
        eps.append((np.array(sup), np.array(qry)))
    return eps


def acc_protonet(F, y, idx):
    out = []
    for sup, qry in idx:
        C = np.unique(y[sup])
        P = l2(np.stack([F[sup][y[sup] == c].mean(0) for c in C]))
        out.append(100.0 * float((C[np.argmax(l2(F[qry]) @ P.T, axis=1)] == y[qry]).mean()))
    return float(np.mean(out)), float(np.std(out))


def acc_lr(F, y, idx):
    out = []
    for sup, qry in idx:
        C = np.unique(y[sup])
        c2i = {int(c): i for i, c in enumerate(C)}
        St = torch.tensor(F[sup], device=CAL.DEVICE, dtype=torch.float32)
        Qt = torch.tensor(F[qry], device=CAL.DEVICE, dtype=torch.float32)
        yt = torch.tensor([c2i[int(v)] for v in y[sup]], device=CAL.DEVICE)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        clf = nn.Linear(St.shape[1], len(C)).to(CAL.DEVICE)
        opt = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(200):
            opt.zero_grad()
            TF.cross_entropy(clf(St), yt).backward()
            opt.step()
        with torch.no_grad():
            pred = C[clf(Qt).argmax(1).cpu().numpy()]
        out.append(100.0 * float((pred == y[qry]).mean()))
    return float(np.mean(out)), float(np.std(out))


def main():
    rows = []
    for ds, ngroup, bgroup in SETS:
        for bb in BACKBONES:
            fp = os.path.join(SCRIPT_DIR, "feat_cache", "%s__%s.npz" % (ngroup, bb))
            bp = os.path.join(SCRIPT_DIR, "feat_cache", "%s__%s.npz" % (bgroup, bb))
            if not (os.path.exists(fp) and os.path.exists(bp)):
                print("skip %s/%s" % (ds, bb)); continue
            fn, _, yn, _ = load(ngroup, bb)
            fb, _, yb, _ = load(bgroup, bb)
            idx = build_nway(yn, SHOT, EVAL_SEEDS)
            ncls = len(np.unique([yn[s] for s, _ in idx[:1]]))
            feats = {"protonet": fn, "linearprobe": fn}
            # calibration chosen by the rule-matched criterion (same as the main table)
            for tag, scorer in (("cal", "ncc"), ("uncond", "zca")):
                if tag == "cal":
                    kind, eps, _, _ = CAL.select(fb, yb, scorer=scorer, k=SHOT)
                    W = CAL.fit(kind, fb, yb, eps)
                else:
                    W = CAL.fit("zca", fb, yb, 0.1)
                feats[tag + "_protonet"] = CAL.apply_W(W, fn)
                feats[tag + "_linearprobe"] = CAL.apply_W(W, fn)
            print("=" * 68)
            print("%s / %s  (%d-way %d-shot, novel %d 张)" % (ds, bb, ncls, SHOT, len(fn)),
                  flush=True)
            for k in KINDS:
                t0 = time.time()
                F = feats[k]
                m, s = (acc_lr if "linearprobe" in k else acc_protonet)(F, yn, idx)
                print("   %-22s %.2f ± %.2f  (%.0fs)" % (LABEL[k], m, s, time.time() - t0),
                      flush=True)
                rows.append({"dataset": ds, "backbone": bb, "protocol": "%d-way %d-shot"
                             % (N_WAY, SHOT), "key": k, "method": LABEL[k],
                             "acc_mean": "%.3f" % m, "acc_std": "%.3f" % s,
                             "episodes": len(idx)})
    p = os.path.join(FO, "final_5way_table.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nsaved: %s (%d 行)" % (p, len(rows)))
    print("\n=== 5-way 汇总（12 设定均值）===")
    for k in KINDS:
        v = [float(r["acc_mean"]) for r in rows if r["key"] == k]
        if v:
            print("   %-22s %.2f" % (LABEL[k], sum(v) / len(v)))


if __name__ == "__main__":
    main()
