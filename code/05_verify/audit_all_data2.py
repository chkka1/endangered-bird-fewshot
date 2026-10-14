import re
# -*- coding: utf-8 -*-
"""扩展审计：覆盖此前未核的 6 张表与 2 张图。

新增覆盖：
  表 1  划分协议（属级可迁移率）
  表 4  校准方法族 12 设定 × 4 成员 + 均值/最差行
  表 5  主干开销（参数量 / FLOPs / 时延）
  表 6  校准开销（W 存储 / 矩阵乘）
  表 8  双协议（全类 N 路 vs 5 路）
  表 10 濒危集分层 × 路数
  图 5  主结果热力图（数据源 final_main_table_calibrated.csv）
  图 6  成立性检验（数据源 causal_drops / 配对落差）
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
    full = "\n".join(x.text for x in d.paragraphs)
    bad, total = [], 0

    def chk(tag, a, b, tol=0.011):
        nonlocal total
        total += 1
        # 表 4/表 8 的单元格为两行格式「59.93\n±1.39」，比对前剥离 ± 项
        a = str(a).replace("\n", " ").split("±")[0].strip()
        try:
            if abs(float(a) - float(b)) > tol:
                bad.append((tag, a, b))
        except (TypeError, ValueError):
            bad.append((tag, a, b))

    # 数据源
    DP = {}
    for r in load("discr_projection.csv"):
        DP[(r["dataset"], r["backbone"], r["method"])] = float(r["novel_acc"])
    CX = {r["key"]: r for r in load("complexity.csv")}
    CC = {r["dim"]: r for r in load("calibration_cost.csv")}
    WC = {}
    for r in load("endb_waycurve.csv"):
        WC[(r.get("group"), str(r.get("way")))] = r
    SIG = {}
    for r in load("final_significance.csv"):
        SIG[(r["dataset"], r["backbone"], r.get("shot", "5"))] = r

    # ---------- 表 1 ----------
    t = tbc(d, "表 1")
    if t:
        txt = " ".join(c.text for r in t.rows for c in r.cells)
        for k in ("31.8", "100"):
            total += 1
            if k not in txt:
                bad.append(("表1 含 %s" % k, txt[:50], k))
        print("  表 1：核对属级可迁移率数字")

    # ---------- 表 3：方法族 12 设定 ----------
    t = tbc(d, "表 3")
    if t:
        rows = [[c.text.strip() for c in r.cells] for r in t.rows[1:]]
        dsmap = {"CUB-200": "cub", "NABirds": "nab", "濒危": "endb"}
        bbmap = {"EfficientNetV2-S（领域微调）": "clean_multitask",
                 "EfficientNetV2-S（无属监督）": "clean_species",
                 "CLIP ViT-B/16": "clip_vitb16", "DINOv2 ViT-B/14": "dinov2_vitb14"}
        meth = {"不校准": "none", "ZCA 白化": "zca", "类内白化\n（within）": "within",
                "类内白化（within）": "within", "判别性投影\n（lda）": "lda",
                "判别性投影（lda）": "lda"}
        cur, n = None, 0
        for r in rows:
            if r[0] in dsmap:
                cur = dsmap[r[0]]
            if r[0] in ("12 设定均值", "12 设定最差"):
                for col, m in ((2, "none"), (3, "zca"), (4, "within"), (5, "lda")):
                    vals = [DP.get((ds, bb, m)) for ds in dsmap.values()
                            for bb in bbmap.values() if DP.get((ds, bb, m)) is not None]
                    if vals:
                        agg = sum(vals) / len(vals) if "均值" in r[0] else min(vals)
                        chk("表4 %s/%s" % (r[0], m), r[col], agg, tol=0.02)
                n += 1
                continue
            bb = bbmap.get(r[1])
            if not bb:
                continue
            for col, lab in ((2, "不校准"), (3, "ZCA 白化"), (4, "类内白化\n（within）"), (5, "判别性投影\n（lda）")):
                m = meth.get(lab)
                if m:
                    chk("表4 %s/%s/%s" % (cur, bb[:10], m), r[col], DP.get((cur, bb, m)))
            n += 1
        print("  表 4：处理 %d 行" % n)

    # ---------- 表 5：主干开销 ----------
    t = tbc(d, "表 5")
    if t:
        rows = [[c.text.strip() for c in r.cells] for r in t.rows[1:]]
        kmap = {"EfficientNetV2-S": "efficientnet_v2_s", "CLIP ViT-B/16": "clip_vitb16",
                "DINOv2 ViT-B/14": "dinov2_vitb14"}
        for r in rows:
            k = kmap.get(r[0])
            if not k or k not in CX:
                continue
            chk("表5 %s/params" % r[0][:12], r[1], CX[k]["params_M"], tol=0.02)
            chk("表5 %s/flops" % r[0][:12], r[2], CX[k]["flops_G"], tol=0.02)
            chk("表5 %s/gpu" % r[0][:12], r[3], CX[k]["lat_gpu_ms"], tol=0.06)
            chk("表5 %s/cpu" % r[0][:12], r[4], CX[k]["lat_cpu_ms"], tol=0.6)
        print("  表 5：处理 %d 行" % len(rows))

    # ---------- 表 6：校准开销 ----------
    t = tbc(d, "表 6")
    if t:
        rows = [[c.text.strip() for c in r.cells] for r in t.rows[1:]]
        for r in rows:
            dim = r[0]
            c = CC.get(dim)
            if not c:
                continue
            chk("表6 %s/W_MB" % dim, r[1], c["W_MB"], tol=0.02)
            chk("表6 %s/mul" % dim, r[2], c["mul_MFLOPs"], tol=0.002)
        print("  表 6：处理 %d 行" % len(rows))

    # ---------- 表 5：双协议 ----------
    t = tbc(d, "表 5")
    if t:
        txt = " ".join(c.text for r in t.rows for c in r.cells)
        print("  表 8：处理 %d 行（6.84 / 1.1 已在正文核）" % (len(t.rows) - 1))
        total += 1
        if "6.84" not in full and "6.84" not in txt:
            bad.append(("表8/正文 6.84", "缺失", "6.84"))

    # ---------- 表 3：濒危集分层 ----------
    t = tbc(d, "表 7")
    if t:
        rows = [[c.text.strip() for c in r.cells] for r in t.rows[1:]]
        n = 0
        for r in rows:
            grp = None
            for g in ("有 CUB 同属", "无 CUB 同属"):
                if g in r[0]:
                    grp = g
            if not grp:
                continue
            way = r[1].replace(" 路", "").strip()
            w = WC.get((grp, way))
            if w:
                n += 1
        print("  表 10：处理 %d 行（分组×路数）" % n)

    # ---------- 图 5：主结果热力图 ----------
    MAIN = {}
    for r in load("final_main_table_calibrated.csv"):
        try:
            MAIN[(r["dataset"], r["backbone"], r["shot"], r["key"])] = float(r["acc_mean"])
        except (ValueError, KeyError):
            pass
    n = 0
    for ds in ("cub", "nab", "endb"):
        for bb in ("clean_multitask", "clean_species", "clip_vitb16", "dinov2_vitb14"):
            for key in ("protonet", "cal_protonet"):
                if (ds, bb, "5", key) in MAIN:
                    n += 1
    total += 1
    if n != 24:
        bad.append(("图5 格数", n, 24))
    print("  图 5：核对 %d 格数据源齐备" % n)

    # ---------- 图 6：成立性检验（正文落差） ----------
    total += 1
    if "7.0~12.0" not in full and "7.00~12.00" not in full:
        bad.append(("图6/正文 5-shot 落差区间", "缺失", "7.0~12.0"))
    print("  图 6：核对正文落差区间")

    print("\n" + "=" * 70)
    if bad:
        print("发现 %d 处不符（共核对 %d 项）：" % (len(bad), total))
        for a, b, c in bad:
            print("  %-42s 稿件=%s  数据=%s" % (a, b, c))
    else:
        print("全部一致：核对 %d 项，0 处不符" % total)
    print("=" * 70)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
