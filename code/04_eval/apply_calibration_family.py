# -*- coding: utf-8 -*-
"""
Calibration-family rows for the main results table.

Replaces the fixed-eps PCA rows (white_*) with family-selected rows (cal_*):
    cal_protonet     member+eps chosen by a criterion scored with the class-mean rule
    cal_linearprobe  member+eps chosen by a criterion scored with the linear probe

Scope: only these two rules.  They are the two the paper reports as its method and as
the strongest baseline; GWP was shown to be inert and TTA is a small add-on, so
extending the family to them adds risk without adding evidence.
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
SETS = [("cub", "cub_novel", "cub_base"),
        ("nab", "nab_novel", "nab_base"),
        ("endb", "endb", "cub_base")]
BACKBONES = ["clean_multitask", "clean_species", "clip_vitb16", "dinov2_vitb14"]
LABEL = {"cal_protonet": "校准方法族 + ProtoNet",
         "cal_linearprobe": "校准方法族 + 线性探针"}


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


def eval_protonet(F, y, idx, gy=None, sp2gen=None):
    acc = []
    stat = [0, 0, 0]
    prf = []          # 逐轮的 (macroP, macroR, macroF1)
    for sup, qry in idx:
        C = np.unique(y[sup])
        P = CAL.l2(np.stack([F[sup][y[sup] == c].mean(0) for c in C]))
        pred = C[np.argmax(CAL.l2(F[qry]) @ P.T, axis=1)]
        acc.append(100.0 * float((pred == y[qry]).mean()))
        prf.append(_macro_prf(pred, y[qry]))
        if gy is not None and sp2gen is not None:
            a, b, c = _err_breakdown(pred, y[qry], gy[qry], sp2gen)
            stat[0] += a; stat[1] += b; stat[2] += c
    if prf:
        _arr = np.array(prf)
        stat += [float(_arr[:, 0].mean()), float(_arr[:, 1].mean()),
                 float(_arr[:, 2].mean()), float(_arr[:, 0].std()),
                 float(_arr[:, 1].std()), float(_arr[:, 2].std())]
    else:
        stat += [0.0] * 6
    return float(np.mean(acc)), float(np.std(acc)), stat


def eval_linearprobe(F, y, idx, gy=None, sp2gen=None):
    acc = []
    stat = [0, 0, 0]
    prf = []          # 逐轮的 (macroP, macroR, macroF1)
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
            a, b, c = _err_breakdown(pred, y[qry], gy[qry], sp2gen)
            stat[0] += a; stat[1] += b; stat[2] += c
    if prf:
        _arr = np.array(prf)
        stat += [float(_arr[:, 0].mean()), float(_arr[:, 1].mean()),
                 float(_arr[:, 2].mean()), float(_arr[:, 0].std()),
                 float(_arr[:, 1].std()), float(_arr[:, 2].std())]
    else:
        stat += [0.0] * 6
    return float(np.mean(acc)), float(np.std(acc)), stat


def main():
    rows = list(csv.DictReader(open(os.path.join(FO, "final_main_table.csv"),
                                   encoding="utf-8-sig")))
    fields = list(rows[0].keys())
    new_rows = []
    # 可用 FE_DATASETS 限定数据集：NABirds 的 cal_linearprobe 每个设定约 36 分钟，
    # 而它的特征并未变化，为改一个数据集重算全部代价过高
    _want = {x.strip() for x in (os.environ.get("FE_DATASETS") or "").split(",") if x.strip()}
    for ds, ngroup, bgroup in SETS:
        if _want and ds not in _want:
            continue
        for bb in BACKBONES:
            fp = os.path.join(SCRIPT_DIR, "feat_cache", "%s__%s.npz" % (ngroup, bb))
            bp = os.path.join(SCRIPT_DIR, "feat_cache", "%s__%s.npz" % (bgroup, bb))
            if not (os.path.exists(fp) and os.path.exists(bp)):
                print("skip %s/%s" % (ds, bb)); continue
            fn, _, yn, gn = load(ngroup, bb)
            fb, _, yb, _ = load(bgroup, bb)
            # 物种→属 映射（属标签此前被丢弃，故误差分解无法计算）
            sp2gen = {int(v): int(g) for v, g in zip(yn, gn)}
            idx = build_episodes(yn, SHOT, EVAL_SEEDS)
            print("=" * 70)
            print("%s / %s   (novel %d 张, base %d 张)" % (ds, bb, len(fn), len(fb)), flush=True)
            for rowname, scorer, evaluator in (
                    ("cal_protonet", "ncc", eval_protonet),
                    ("cal_linearprobe", "lr", eval_linearprobe)):
                t0 = time.time()
                kind, eps, _, hsc = CAL.select(fb, yb, scorer=scorer, k=SHOT)
                # REFIT on ALL Base classes: select() returns the matrix fitted on the
                # criterion's half-split, which discards half the classes and cost
                # ~8.6 pp on CUB / clean_multitask when used directly
                W = CAL.fit(kind, fb, yb, eps)
                Fn = CAL.apply_W(W, fn)
                m, s, _stat = evaluator(Fn, yn, idx, gy=gn, sp2gen=sp2gen)
                print("   %-16s 判据(%s)→%-7s eps=%.2f 留出 %.2f | novel %.2f ± %.2f"
                      " | 跨属错误 %d/%d=%.1f%% (%.0fs)"
                      % (rowname, scorer, kind, eps, hsc, m, s, _stat[1], _stat[0],
                         100.0 * _stat[1] / _stat[0] if _stat[0] else 0.0,
                         time.time() - t0), flush=True)
                print("       macro P %.2f R %.2f F1 %.2f" %
                      (_stat[3], _stat[4], _stat[5]), flush=True)
                new_rows.append({"dataset": ds, "backbone": bb, "shot": str(SHOT),
                                 "key": rowname, "method": LABEL[rowname],
                                 "acc_mean": "%.3f" % m, "acc_std": "%.3f" % s,
                                 "total_err": str(_stat[0]),
                                 "cross_genus_err": str(_stat[1]),
                                 "same_genus_err": str(_stat[2]),
                                 "cross_share_pct": ("%.3f" % (100.0 * _stat[1] / _stat[0])
                                                     if _stat[0] else ""),
                                 "precision": "%.3f" % _stat[3],
                                 "recall": "%.3f" % _stat[4],
                                 "f1": "%.3f" % _stat[5],
                                 "precision_std": "%.3f" % _stat[6],
                                 "recall_std": "%.3f" % _stat[7],
                                 "f1_std": "%.3f" % _stat[8],
                                 "episodes": "50"})

    p = os.path.join(FO, "final_main_table_calibrated.csv")
    # 基表 final_main_table.csv 里没有 cal_* 与 uncond_* 行——它们分别由本脚本与
    # apply_unconditional_whitening.py 写进**校准后**的文件。因此用 FE_DATASETS
    # 限定重算范围时，必须从已有校准文件里保留"未被重算的数据集"的这两类行，
    # 否则会被整体抹掉。
    # 事故记录：早先只认 "cal_"，导致 uncond_protonet / uncond_linearprobe 共 24 行
    # 在一次单数据集试运行中全部丢失（692 → 668 行）。
    _recomputed = {ds for ds, _g, _b in SETS if not _want or ds in _want}
    kept_cal = []
    if os.path.exists(p):
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            _k = r["key"]
            # uncond_* 由 apply_unconditional_whitening.py 产出，本脚本永不生成，
            # 因此无论该数据集是否被重算都必须沿用，否则会被整体抹掉
            if _k.startswith("uncond_"):
                kept_cal.append(r)
            # cal_* 由本脚本产出：只沿用"未被重算"的数据集的行
            elif _k.startswith("cal_") and r["dataset"] not in _recomputed:
                kept_cal.append(r)
    out = ([r for r in rows if not r["key"].startswith("white_")]
           + [{k: r.get(k, "") for k in fields} for r in kept_cal] + new_rows)
    for _c in ("precision", "recall", "f1", "precision_std", "recall_std", "f1_std"):
        if _c not in fields:
            fields.append(_c)
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in fields})
    print("\nsaved: %s  (%d 行，其中新校准行 %d，沿用其它数据集校准行 %d)"
          % (p, len(out), len(new_rows), len(kept_cal)))


if __name__ == "__main__":
    main()
