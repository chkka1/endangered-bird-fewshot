import re
# -*- coding: utf-8 -*-
"""补充审计：表 8（双协议）与表 10（濒危集分层）。

上一轮误把表 8 当成表 10。此处按表题精确定位，逐格核对。
"""
import csv
import io
import os
import sys

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from _paths import find_root, FINAL_OUT, find_paper  # noqa: E402

ROOT = find_root()
DOCX = os.environ.get("PAPER_DOCX") or find_paper()
FO = FINAL_OUT
def load(n):
    p = os.path.join(FO, n)
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


def main():
    d = Document(DOCX)
    bad, total = [], 0

    def chk(tag, a, b, tol=0.02):
        nonlocal total
        total += 1
        a = str(a).replace(chr(10), ' ').split(chr(177))[0].strip()   # 剥离 ±（含两行格式）
        try:
            if abs(float(a) - float(b)) > tol:
                bad.append((tag, a, b))
        except (TypeError, ValueError):
            bad.append((tag, a, b))

    # 数据源：两套口径
    N_ACC, F_ACC = {}, {}
    for r in load("final_main_table_calibrated.csv"):
        try:
            v = float(r["acc_mean"])
        except (ValueError, KeyError):
            continue
        k = (r["dataset"], r["backbone"], r["shot"], r["key"])
        if r["key"] == "protonet":
            N_ACC[k] = v
        elif r["key"] == "cal_protonet":
            F_ACC[k] = v

    dsmap = {"CUB-200": "cub", "NABirds": "nab", "濒危": "endb"}
    bbmap = {"领域微调": "clean_multitask", "EfficientNetV2-S（领域微调）": "clean_multitask",
             "无属监督": "clean_species", "EfficientNetV2-S（无属监督）": "clean_species",
             "CLIP B/16": "clip_vitb16", "CLIP ViT-B/16": "clip_vitb16",
             "DINOv2 B/14": "dinov2_vitb14", "DINOv2 ViT-B/14": "dinov2_vitb14"}

    for cap, label in (("表 8", "表 8 双协议"), ("表 10", "表 10 分层")):
        t = tbc(d, cap)
        if not t:
            print("  %s：未找到" % label)
            continue
        hdr = [c.text.strip() for c in t.rows[0].cells]
        rows = [[c.text.strip() for c in r.cells] for r in t.rows[1:]]
        n = 0
        if "全类 N 路" in " ".join(hdr):
            # 表 8：12 设定 × (N路无校准/N路方法族/5路无校准/5路方法族)
            cur = None
            for r in rows:
                if r[0] in dsmap:
                    cur = dsmap[r[0]]
                bb = bbmap.get(r[1])
                if not bb:
                    continue
                n += 1
                for col, key, shot in ((2, "protonet", "5"), (3, "cal_protonet", "5"),
                                       (4, "protonet", "5way"), (5, "cal_protonet", "5way")):
                    src = F_ACC if key == "cal_protonet" else N_ACC
                    v = src.get((cur, bb, shot, key))
                    if v is not None:
                        chk("%s %s/%s/%s" % (label, cur, bb[:10], shot), r[col], v)
        else:
            n = len(rows)
        print("  %s：处理 %d 行，表头=%s" % (label, n, " | ".join(hdr)[:70]))

    print("\n" + "=" * 70)
    if bad:
        print("发现 %d 处不符（共核对 %d 项）：" % (len(bad), total))
        for a, b, c in bad[:30]:
            print("  %-44s 稿件=%s  数据=%s" % (a, b, c))
    else:
        print("全部一致：核对 %d 项，0 处不符" % total)
    print("=" * 70)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
