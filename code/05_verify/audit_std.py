# -*- coding: utf-8 -*-
"""v6 P1-1：逐格复核表 2 / 表 4 / 表 8 的 ±s（共 108 个）。

审稿人指出：这些标准差无法从稿件内部其他数字推出，是**全稿唯一一批
只能信任作者、读者无法交叉验证的数据**。本脚本把每个 ±s 与结果 CSV
的 std 列逐一比对。

数据源：
    表 2 / 表 8 → final_main_table_calibrated.csv（acc_std 列）
    表 4       → discr_projection.csv（novel_std 列）
"""
import csv
import io
import os
import re

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from _paths import find_root, FINAL_OUT, find_paper  # noqa: E402

ROOT = find_root()
REL = os.path.join(ROOT, "release")
FO = FINAL_OUT
DOCX = os.environ.get("PAPER_DOCX") or find_paper()


def load(p):
    return list(csv.DictReader(io.open(p, encoding="utf-8-sig"))) if os.path.exists(p) else []


def tbc(doc, prefix):
    prev = None
    for c in doc.element.body.iterchildren():
        if c.tag.endswith("}p"):
            prev = Paragraph(c, doc).text.strip()
        elif c.tag.endswith("}tbl"):
            if prev and prev.startswith(prefix):
                return Table(c, doc)
    return None


# ---------- 数据源 ----------
ACC = {}
for r in load(os.path.join(FO, "final_main_table_calibrated.csv")):
    try:
        ACC[(r["dataset"], r["backbone"], r["shot"], r["key"])] = (
            float(r["acc_mean"]), float(r["acc_std"]) if r.get("acc_std") else None)
    except (ValueError, KeyError):
        pass
DP = {}
for r in load(os.path.join(FO, "discr_projection.csv")):
    try:
        DP[(r["dataset"], r["backbone"], r["method"])] = (
            float(r["novel_acc"]), float(r["novel_std"]) if r.get("novel_std") else None)
    except (ValueError, KeyError):
        pass

DSMAP = {"CUB-200": "cub", "NABirds": "nab", "濒危": "endb"}
BBMAP = {"EfficientNetV2-S（领域微调）": "clean_multitask",
         "EfficientNetV2-S（无属监督）": "clean_species",
         "CLIP ViT-B/16": "clip_vitb16", "DINOv2 ViT-B/14": "dinov2_vitb14",
         "领域微调": "clean_multitask", "无属监督": "clean_species",
         "CLIP B/16": "clip_vitb16", "DINOv2 B/14": "dinov2_vitb14"}
CELL = re.compile(r"([\d.]+)\s*±\s*([\d.]+)")

d = Document(DOCX)
bad, total = [], 0
per_table = {}

for cap, label in (("表 2", "表 2"), ("表 4", "表 4"), ("表 8", "表 8")):
    t = tbc(d, cap)
    if t is None:
        print("  %s 未找到" % label)
        continue
    hdr = [c.text.strip() for c in t.rows[0].cells]
    cnt = 0
    cur = None
    for r in t.rows[1:]:
        cells = [" ".join(c.text.split()) for c in r.cells]
        if cells[0] in DSMAP:
            cur = DSMAP[cells[0]]
        bb = BBMAP.get(cells[1])
        if not bb or cur is None:
            continue
        for ci, cell in enumerate(cells):
            m = CELL.search(cell)
            if not m:
                continue
            got_mean, got_std = float(m.group(1)), float(m.group(2))
            # 定位该列对应的方法键
            h = hdr[ci] if ci < len(hdr) else ""
            key = None
            if "不校准" in h:
                key = "protonet"
            elif "固定系数白化" in h:
                key = "uncond_protonet"
            elif h.startswith("方法族") and "−" not in h and "白化" not in h:
                key = "cal_protonet"
            elif "全类 N 路" in h and "无校准" in h:
                key = "protonet"
            elif "全类 N 路" in h and "校准方法族" in h:
                key = "cal_protonet"
            if key:
                src = ACC.get((cur, bb, "5", key))
            elif label == "表 4":
                mk = None
                for kw, k2 in (("不校准", "none"), ("ZCA", "zca"),
                               ("类内白化", "within"), ("判别性", "lda")):
                    if kw in h:
                        mk = k2
                        break
                src = DP.get((cur, bb, mk)) if mk else None
            else:
                src = None
            if not src:
                continue
            total += 1
            cnt += 1
            if src[1] is None:
                bad.append(("%s %s/%s/col%d 无 std" % (label, cur, bb[:8], ci), got_std, "-"))
            elif abs(got_std - src[1]) > 0.011:
                bad.append(("%s %s/%s/col%d" % (label, cur, bb[:8], ci), got_std, src[1]))
    per_table[label] = cnt

print("=== 表 2 / 表 4 / 表 8 的 ±s 逐格复核 ===")
for k, v in per_table.items():
    print("  %s: 核对 %d 个 ±s" % (k, v))
print("  合计 %d 个（审稿人称 108 个）" % total)
print()
if bad:
    print("发现 %d 处不符：" % len(bad))
    for a, b, c in bad[:20]:
        print("  %-40s 稿件=%.2f  数据=%s" % (a, b, c))
else:
    print("全部一致：0 处不符 ✓")
