# -*- coding: utf-8 -*-
"""
Turn the criterion-matching log into a CSV, so table 3 is read from data rather than
hardcoded in the manuscript builder.

This was the last place in the paper where numbers were typed by hand.  Re-running the
experiment is expensive (the NABirds settings take ~25 min each), so the existing log is
parsed once here; the builder then reads the CSV, and a future re-run only needs this
parser re-invoked.

Source line format (from criterion_matching_test.py):
    cub    clean_multitask  pca/eps=0.05   within/eps=0.10   不同  (novel 71.48 vs 61.31, 差 -10.17, 54s)
"""
import csv
import io
import os
import re
from _paths import find_root, FINAL_OUT  # noqa: E402

FO = FINAL_OUT
LOG = os.path.join(FO, "criterion_match.txt")
OUT = os.path.join(FO, "criterion_match.csv")

DS = {"cub": "CUB-200", "nab": "NABirds", "endb": "濒危"}
BB = {"clean_multitask": "领域微调", "clean_species": "无属监督",
      "clip_vitb16": "CLIP B/16", "dinov2_vitb14": "DINOv2 B/14"}
PAT = re.compile(
    r"^(cub|nab|endb)\s+(\w+)\s+(\S+)\s+(\S+)\s+(相同|不同)\s+"
    r"\(novel\s+([\d.]+)\s+vs\s+([\d.]+),\s*差\s*([+-]?[\d.]+),")


def pick(s):
    """'pca/eps=0.05' -> 'pca/0.05'"""
    m = re.match(r"(\w+)/eps=([\d.]+)", s)
    return "%s/%s" % (m.group(1), m.group(2)) if m else s


def main():
    rows = []
    for line in io.open(LOG, encoding="utf-8"):
        m = PAT.match(line.strip())
        if not m:
            continue
        ds, bb, ncc, lr, same, a1, a2, d = m.groups()
        rows.append({
            "dataset": DS.get(ds, ds), "backbone": BB.get(bb, bb),
            "ncc_pick": pick(ncc), "lr_pick": pick(lr),
            "same": "相同" if same == "相同" else "不同",
            # delta = linear-probe-selected minus NCC-selected
            "delta": ("%+.2f" % float(d)).replace("+", "+").replace("-", "-"),
        })
    if not rows:
        print("!! 未从 %s 解析到任何行" % LOG)
        return 1
    with io.open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("解析 %d 个设定 -> %s" % (len(rows), OUT))
    print("%-8s %-16s %-14s %-14s %-6s %8s" % ("数据集", "主干", "NCC 判据", "线性探针判据",
                                               "是否相同", "Novel 差"))
    for r in rows:
        print("%-8s %-16s %-14s %-14s %-6s %8s"
              % (r["dataset"], r["backbone"], r["ncc_pick"], r["lr_pick"],
                 r["same"], r["delta"]))
    diff = sum(1 for r in rows if r["same"] == "不同")
    print("\n%d/%d 个设定两判据选出不同成员" % (diff, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
