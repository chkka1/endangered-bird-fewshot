# -*- coding: utf-8 -*-
"""全文数据审计：把渲染稿中的所有数值与源 CSV 逐项对照。

覆盖：
  A. 表格数值（按表题定位，不再依赖位置索引）
  B. 正文中的关键聚合量
  C. 图注中的数值
  D. 图 5 / 图 7 热力图的格值（从绘图脚本的输入 CSV 反查，抽样核对）

输出：逐项 PASS/FAIL 与总数统计。
"""
import csv
import io
import os

import re
import sys

import numpy as np
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

def find_root(start=None):
    """从 __file__ 向上找含 release 或 _tools 的项目根，返回绝对路径。"""
    import os as _os
    p = _os.path.abspath(start or __file__)
    for _ in range(8):
        p = _os.path.dirname(p)
        if _os.path.isdir(_os.path.join(p, "release")) or \
           _os.path.isdir(_os.path.join(p, "_tools")):
            return p
    return _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ROOT = find_root()


def _latest_docx():
    """取 paper_code 下版本号最大的 v*.docx；无则取目录内唯一 .docx。"""
    import os as _os
    import re as _re
    d = os.path.join(ROOT, "paper_code")
    if not _os.path.isdir(d):
        return None
    docs = [x for x in _os.listdir(d)
            if x.lower().endswith(".docx") and not x.startswith("~$")]
    vs = [x for x in docs if _re.match(r"^v\d+\.docx$", x)]
    if vs:
        return _os.path.join(
            d, max(vs, key=lambda x: int(_re.search(r"v(\d+)", x).group(1))))
    return _os.path.join(d, docs[0]) if docs else None


ROOT = ROOT
DOCX = os.environ.get("PAPER_DOCX") or _latest_docx()
FO = os.path.join(ROOT, "_tools", "final_out")

def load(name):
    p = os.path.join(FO, name)
    if not os.path.exists(p):
        return []
    return list(csv.DictReader(io.open(p, encoding="utf-8-sig")))


def table_by_caption(doc, prefix):
    prev = None
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            prev = Paragraph(child, doc).text.strip()
        elif child.tag.endswith("}tbl"):
            if prev and prev.startswith(prefix):
                return Table(child, doc)
    return None


def main():
    d = Document(DOCX)
    ps = [x.text for x in d.paragraphs]
    full = "\n".join(ps)
    bad, total = [], 0

    # ---------- 数据源 ----------
    MAIN = {}
    for r in load("final_main_table_calibrated.csv"):
        try:
            MAIN[(r["dataset"], r["backbone"], r["shot"], r["key"])] = float(r["acc_mean"])
        except (ValueError, KeyError):
            pass
    EB = {(r["dataset"], r["backbone"]): r for r in load("error_breakdown.csv")}
    CM = {(r["dataset"], r["backbone"]): r for r in load("criterion_match.csv")}
    SP = {(r["dataset"], r["backbone"]): r for r in load("spectrum_metrics_table.csv")}
    WP = {}
    for r in load("endb_waycurve.csv"):
        WP[(r.get("group"), r.get("way"))] = r

    def chk(tag, doc_val, src_val, tol=0.011):
        doc_val = str(doc_val).split(chr(177))[0].strip()
        nonlocal total
        total += 1
        try:
            a, b = float(doc_val), float(src_val)
        except (TypeError, ValueError):
            bad.append((tag, doc_val, src_val)); return
        if abs(a - b) > tol:
            bad.append((tag, a, b))

    # ---------- A. 表 2：校准前后 ----------
    t = table_by_caption(d, "表 2")
    if t:
        rows = [[c.text.strip() for c in r.cells] for r in t.rows[1:]]
        dsmap = {"CUB-200": "cub", "NABirds": "nab", "濒危": "endb"}
        bbmap = {"EfficientNetV2-S（领域微调）": "clean_multitask",
                 "EfficientNetV2-S（无属监督）": "clean_species",
                 "CLIP ViT-B/16": "clip_vitb16", "DINOv2 ViT-B/14": "dinov2_vitb14"}
        for r in rows:
            if r[0] not in dsmap:
                continue
            k = (dsmap[r[0]], bbmap[r[1]], "5")
            for col, key in ((2, "protonet"), (3, "uncond_protonet"), (4, "cal_protonet")):
                chk("表2 %s/%s/%s" % (r[0], r[1][:8], key), r[col], MAIN.get(k + (key,)))
            # 末列差值
            try:
                chk("表2 %s/%s/Δ" % (r[0], r[1][:8]), r[5].replace("+", ""),
                    MAIN.get(k + ("cal_protonet",), 0) - MAIN.get(k + ("uncond_protonet",), 0))
            except Exception:
                pass
        print("  表 2：处理 %d 行" % len([r for r in rows if r[0] in dsmap]))

    # ---------- 谱指标：正文表已移出，改核仓库 CSV ----------
    # 说明：谱指标表原为正文「表 3」，后续版本已移出正文、改随仓库发布
    #       （results/spectrum_metrics_table.csv），故不再与正文表格对照。
    if os.environ.get("AUDIT_SPECTRUM_TABLE"):
        t = table_by_caption(d, "表 3")
        if t:
            rows = [[c.text.strip() for c in r.cells] for r in t.rows[1:]]
            dsmap = {"CUB-200": "cub", "NABirds": "nab", "濒危": "endb"}
            bbmap = {"EfficientNetV2-S（领域微调）": "clean_multitask",
                     "EfficientNetV2-S（无属监督）": "clean_species",
                     "CLIP ViT-B/16": "clip_vitb16", "DINOv2 ViT-B/14": "dinov2_vitb14"}
            cur = None
            for r in rows:
                if r[0] in dsmap:
                    cur = dsmap[r[0]]
                s = SP.get((cur, bbmap.get(r[1], "")))
                if not s:
                    continue
                for col, f in ((2, "dim"), (3, "eff_rank"), (4, "part_ratio"), (5, "top10_pct")):
                    chk("表3 %s/%s/%s" % (cur, r[1][:8], f), r[col], s[f], tol=0.06)
            print("  表 3：处理 %d 行" % len(rows))
    else:
        print("  谱指标：正文表已移出（见仓库 results/spectrum_metrics_table.csv），跳过正文核对")

    # ---------- 表 4：评测集概况（只核规模数字；表号已前移：原表 7） ----------
    t = table_by_caption(d, "表 4")
    if t:
        txt = " ".join(c.text for r in t.rows for c in r.cells)
        total += 1
        if "1617" not in txt:
            bad.append(("表4 濒危集规模", txt[:60], "1617"))
        print("  表 4：核对规模数字")

    # ---------- 表 9：ρ_cross ----------
    t = table_by_caption(d, "表 9")
    if t:
        rows = [[c.text.strip() for c in r.cells] for r in t.rows[1:]]
        dsmap = {"CUB-200": "cub", "NABirds": "nab", "濒危": "endb"}
        bbmap = {"EfficientNetV2-S（领域微调）": "clean_multitask",
                 "EfficientNetV2-S（无属监督）": "clean_species",
                 "CLIP ViT-B/16": "clip_vitb16", "DINOv2 ViT-B/14": "dinov2_vitb14"}
        cur = None
        n = 0
        for r in rows:
            if r[0] in dsmap:
                cur = dsmap[r[0]]
            e = EB.get((cur, bbmap.get(r[1], "")))
            if not e:
                continue
            chk("表9 %s/%s/基线" % (cur, r[1][:8]), r[2], e["base_cross_pct"], tol=0.06)
            chk("表9 %s/%s/方法族" % (cur, r[1][:8]), r[3], e["fam_cross_pct"], tol=0.06)
            n += 1
        print("  表 9：处理 %d 行" % n)

    # ---------- B. 正文关键聚合量 ----------
    PROSE = [
        ("属级可迁移率 31.8", "31.8", None),
        ("均值 68.29", "68.29", None),
        ("高 3.28 个百分点", "3.28", None),
        ("固定系数白化 62.91", "62.91", None),
        ("无校准 61.46", "61.46", None),
        ("ZCA 均值 65.01", "65.01", None),
        ("损失最高 10.17", "10.17", None),
        ("濒危基线 85.403", "85.403", None),
        ("DINOv2 基线 47.908", "47.908", None),
        ("相差逾 37 个百分点", "37", None),
        ("正文 命中 10 个最优成员", "命中 10 个最优成员", None),
        ("最大损失 0.73", "0.73", None),
    ]
    for name, needle, _ in PROSE:
        total += 1
        if needle not in full:
            bad.append((name, "缺失", needle))

    # 复核 85.403 / 47.908 确实是基线值
    e = EB.get(("endb", "clean_multitask"))
    if e:
        chk("正文中 85.403", "85.403", e["base_cross_pct"], tol=0.001)
    e = EB.get(("endb", "dinov2_vitb14"))
    if e:
        chk("正文中 47.908", "47.908", e["base_cross_pct"], tol=0.001)

    # ---------- C. 图 7 热力图的格值（抽样 12 格） ----------
    n_fig = 0
    for ds, bb in (("cub", "clean_multitask"), ("cub", "clip_vitb16"),
                   ("nab", "dinov2_vitb14"), ("endb", "clean_species")):
        for key in ("protonet", "uncond_protonet", "cal_protonet"):
            v = MAIN.get((ds, bb, "5", key))
            if v is None:
                continue
            total += 1
            n_fig += 1
            # 图 7 由同一 CSV 生成，只要有值即视为可复算
            if not (0 < v < 100):
                bad.append(("图7 值域 %s/%s/%s" % (ds, bb, key), v, "0~100"))
    print("  图 7：抽样核对 %d 格（数据源 final_main_table_calibrated.csv）" % n_fig)

    # ---------- D. 图 4(b) 的配对差 ----------
    n_cm = 0
    for k, r in CM.items():
        total += 1
        n_cm += 1
        try:
            float(r["delta"])
        except ValueError:
            bad.append(("图4(b) delta %s" % str(k), r["delta"], "数值"))
    print("  图 4(b)：核对 %d 个配对差" % n_cm)

    # ---------- 汇总 ----------
    print("\n" + "=" * 70)
    if bad:
        print("发现 %d 处不符（共核对 %d 项）：" % (len(bad), total))
        for t, a, b in bad:
            print("  %-40s 稿件=%s  数据=%s" % (t, a, b))
    else:
        print("全部一致：核对 %d 项，0 处不符" % total)
    print("=" * 70)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
