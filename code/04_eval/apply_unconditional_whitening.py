# -*- coding: utf-8 -*-
"""
Unconditional whitening baseline, re-run under the SAME protocol as the calibration
family so the two are directly comparable.

Definition used here (matching standard practice): apply PCA whitening to every
setting with a FIXED shrinkage eps = 0.1 — no criterion, no per-setting selection.
This is the deployable "just whiten" baseline.

The family rows (cal_*) come from apply_calibration_family.py.  Both use
build_episodes(..., EVAL_SEEDS) and the same 200-step linear probe, so the only
difference is how the calibration is chosen.

Output: rows uncond_protonet / uncond_linearprobe appended to
final_main_table_calibrated.csv (or written standalone if that file is absent).
"""
import csv
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn as nn                                       # noqa: E402
import torch.nn.functional as TF                            # noqa: E402
import calibration as CAL                                   # noqa: E402
from improve_methods import build_episodes, load, EVAL_SEEDS  # noqa: E402

FO = os.path.join(SCRIPT_DIR, "final_out")
SHOT = 5
FIXED_EPS = 0.1
SETS = [("cub", "cub_novel", "cub_base"),
        ("nab", "nab_novel", "nab_base"),
        ("endb", "endb", "cub_base")]
BACKBONES = ["clean_multitask", "clean_species", "clip_vitb16", "dinov2_vitb14"]
LABEL = {"uncond_protonet": "固定系数白化 + ProtoNet",
         "uncond_linearprobe": "固定系数白化 + 线性探针"}


def _err_breakdown(pred, y_qry, gy_qry, sp2gen):
    """(总错误数, 跨属错误数, 同属错误数)。跨属＝预测属与真实属不同。"""
    wrong = pred != y_qry
    te = int(wrong.sum())
    if te == 0:
        return 0, 0, 0
    pg = np.array([sp2gen.get(int(v), -1) for v in pred])
    tg = np.asarray(gy_qry)
    cross = int((wrong & (pg != tg)).sum())
    return te, cross, te - cross


def _macro_prf(pred, y_true):
    """macro Precision / Recall / F1（%）。先按类求再对类平均；micro 恒等于准确率。"""
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


def eval_protonet(F, y, idx, gy=None, sp2gen=None):
    acc = []
    stat = [0, 0, 0]
    prf = []
    for sup, qry in idx:
        C = np.unique(y[sup])
        P = CAL.l2(np.stack([F[sup][y[sup] == c].mean(0) for c in C]))
        pred = C[np.argmax(CAL.l2(F[qry]) @ P.T, axis=1)]
        acc.append(100.0 * float((pred == y[qry]).mean()))
        prf.append(_macro_prf(pred, y[qry]))
        if gy is not None and sp2gen is not None:
            _a, _b, _c = _err_breakdown(pred, y[qry], gy[qry], sp2gen)
            stat[0] += _a; stat[1] += _b; stat[2] += _c
    if prf:
        _a = np.array(prf)
        stat += [float(_a[:, 0].mean()), float(_a[:, 1].mean()), float(_a[:, 2].mean()),
                 float(_a[:, 0].std()), float(_a[:, 1].std()), float(_a[:, 2].std())]
    else:
        stat += [0.0] * 6
    return float(np.mean(acc)), float(np.std(acc)), stat


def eval_linearprobe(F, y, idx, gy=None, sp2gen=None):
    acc = []
    stat = [0, 0, 0]
    prf = []
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
        acc.append(100.0 * float((pred == y[qry]).mean()))
        prf.append(_macro_prf(pred, y[qry]))
        if gy is not None and sp2gen is not None:
            _a, _b, _c = _err_breakdown(pred, y[qry], gy[qry], sp2gen)
            stat[0] += _a; stat[1] += _b; stat[2] += _c
    if prf:
        _a = np.array(prf)
        stat += [float(_a[:, 0].mean()), float(_a[:, 1].mean()), float(_a[:, 2].mean()),
                 float(_a[:, 0].std()), float(_a[:, 1].std()), float(_a[:, 2].std())]
    else:
        stat += [0.0] * 6
    return float(np.mean(acc)), float(np.std(acc)), stat


def main():
    rows = []
    for ds, ngroup, bgroup in SETS:
        for bb in BACKBONES:
            fp = os.path.join(SCRIPT_DIR, "feat_cache", "%s__%s.npz" % (ngroup, bb))
            bp = os.path.join(SCRIPT_DIR, "feat_cache", "%s__%s.npz" % (bgroup, bb))
            if not (os.path.exists(fp) and os.path.exists(bp)):
                print("skip %s/%s" % (ds, bb)); continue
            fn, _, yn, gn = load(ngroup, bb)
            fb, _, yb, _ = load(bgroup, bb)
            sp2gen = {int(v): int(g) for v, g in zip(yn, gn)}
            idx = build_episodes(yn, SHOT, EVAL_SEEDS)
            # unconditional: PCA whitening at a fixed eps, fitted on ALL Base classes
            W = CAL.fit("zca", fb, yb, FIXED_EPS)
            Fn = CAL.apply_W(W, fn)
            print("=" * 70)
            print("%s / %s   (eps=%.2f 固定)" % (ds, bb, FIXED_EPS), flush=True)
            for rowname, evaluator in (("uncond_protonet", eval_protonet),
                                       ("uncond_linearprobe", eval_linearprobe)):
                t0 = time.time()
                m, s, _st = evaluator(Fn, yn, idx, gy=gn, sp2gen=sp2gen)
                print("   %-18s novel %.2f ± %.2f  (%.0fs)" % (rowname, m, s, time.time() - t0),
                      flush=True)
                rows.append({"dataset": ds, "backbone": bb, "shot": str(SHOT),
                             "key": rowname, "method": LABEL[rowname],
                             "acc_mean": "%.3f" % m, "acc_std": "%.3f" % s,
                             "total_err": str(_st[0]),
                             "cross_genus_err": str(_st[1]),
                             "same_genus_err": str(_st[2]),
                             "cross_share_pct": ("%.3f" % (100.0 * _st[1] / _st[0])
                                                 if _st[0] else ""),
                             "episodes": "50",
                             "precision": "%.3f" % _st[3], "recall": "%.3f" % _st[4],
                             "f1": "%.3f" % _st[5],
                             "precision_std": "%.3f" % _st[6],
                             "recall_std": "%.3f" % _st[7],
                             "f1_std": "%.3f" % _st[8]})

    p = os.path.join(FO, "final_main_table_calibrated.csv")
    if os.path.exists(p):
        base = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        fields = list(base[0].keys())
        base = [r for r in base if not r["key"].startswith("uncond_")]
        out = base + [{k: r.get(k, "") for k in fields} for r in rows]
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(out)
        print("\n已追加到 %s（现 %d 行）" % (os.path.basename(p), len(out)))
    else:
        q = os.path.join(FO, "unconditional_whitening.csv")
        with open(q, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("\n方法族表尚未生成，先写入 %s" % os.path.basename(q))


if __name__ == "__main__":
    main()
