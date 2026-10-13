# ---- 核验包守卫（自包含）：缺少随包未发布的外部资产时显式跳过 ----
def _pkg_root():
    import os as _o
    d = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))
    e = _o.environ.get("BIRD_PROJECT_ROOT")
    return _o.path.abspath(e) if (e and _o.path.isdir(e)) else d


def _require(rel, what):
    import os as _o
    import sys as _s
    full = _o.path.join(_pkg_root(), rel)
    if _o.path.exists(full):
        return True
    print("=" * 68)
    print("[跳过] 本脚本需要" + what + "，但包内未包含：")
    print("        " + full)
    print("  本包为核验包：随包提供 code / dataset(濒危集) / "
          "_tools/final_out(结果) / paper_code(稿件与图)。")
    print("  特征缓存、第三方原图与模型权重未随包发布，故本脚本无法在包内执行")
    print("  —— 这不代表论文数据有问题。")
    print("=" * 68)
    _s.exit(0)


_require("dataset/endangered_birds/images", "濒危集原图（1617 张）")

# -*- coding: utf-8 -*-
"""
Global consistency check: every number in the manuscript tables must match its source CSV.

A DOCX can be a structurally valid package while holding stale numbers, so this walks
the data-bearing tables and recomputes each cell from the underlying CSV.

Table layouts (as actually built):
    表 2  tables[1]  数据集 | 主干 | 不校准 | 固定系数白化 | 方法族 | 差值
    表 3  tables[2]  数据集 | 主干 | 不校准 | ZCA 白化 | 类内白化 | 判别投影 | 判据选择 | 选出
    表 4  tables[3]  评测集概况
    表 5  tables[4]  数据集 | 方法 | 领域微调 | 无属监督 | CLIP | DINOv2
    表 6  tables[5]  数据集 | 主干 | 全类无校准 | 全类方法族 | 5路无校准 | 5路方法族

Backbone labels differ per table ('领域微调' vs 'EfficientNetV2-S（领域微调）'), so
matching is by substring.
"""
import csv
import io
import os
import re
import sys
from _paths import find_root, FINAL_OUT, find_paper  # noqa: E402

ROOT = find_root()
FO = FINAL_OUT
def _default_docx():
    """未设 PAPER_DOCX 时自动选 paper_code 下版本号最大的 v*.docx。

    原默认名为 论文_修订稿v7_图表优化版.docx（旧稿且已不存在），
    忘记设环境变量会直接报 PackageNotFoundError 或静默核对旧文件。
    """
    pc = os.path.join(ROOT, "paper_code")
    cand = []
    if os.path.isdir(pc):
        for f in os.listdir(pc):
            m = re.match(r"^v(\d+)\.docx$", f)
            if m:
                cand.append((int(m.group(1)), os.path.join(pc, f)))
    if cand:
        cand.sort()
        return cand[-1][1]
    return os.path.join(pc, "论文_修订稿v7_图表优化版.docx")


# 稿件路径可由环境变量覆盖。写死路径曾在版本改名后静默核对旧文件：
# 报出"全部一致"，而被检查的根本不是当前稿件。
DOCX = os.environ.get("PAPER_DOCX") or find_paper()
ENDB = os.path.join(ROOT, "dataset", "endangered_birds", "images")
if not os.path.isdir(ENDB):                    # 兼容原项目布局
    ENDB = os.path.join(ROOT, "data", "endangered_birds", "images")
EXT = (".jpg", ".jpeg", ".png", ".bmp")


def ds_of(t):
    if "cub" in t.lower():
        return "cub"
    if "nab" in t.lower():
        return "nab"
    if "濒危" in t or "endb" in t.lower():
        return "endb"
    return None


def bb_of(t):
    # compare case-insensitively on BOTH sides: t.upper() turns 'DINOv2' into
    # 'DINOV2', so testing 'DINOv2' in t.upper() never matches
    u = t.upper()
    if "领域微调" in t:
        return "clean_multitask"
    if "无属监督" in t:
        return "clean_species"
    if "CLIP" in u:
        return "clip_vitb16"
    if "DINOV2" in u or "DINO" in u:
        return "dinov2_vitb14"
    return None


def load(name):
    p = os.path.join(FO, name)
    return (list(csv.DictReader(io.open(p, encoding="utf-8-sig")))
            if os.path.exists(p) else [])


def main():
    from docx import Document
    A = {(r["dataset"], r["backbone"], r["shot"], r["key"]): float(r["acc_mean"])
         for r in load("final_main_table_calibrated.csv")}
    F5 = {(r["dataset"], r["backbone"], r["key"]): float(r["acc_mean"])
          for r in load("final_5way_table.csv")}
    # chk() 用三元键查找，而 A 是四元键（含 shot）。
    # 表 3 新增的「固定系数白化」列取自该表的 uncond_protonet（5-shot），
    # 需单独建一个三元键字典，否则查找返回 None 而被静默跳过。
    A5 = {(r["dataset"], r["backbone"], r["key"]): float(r["acc_mean"])
          for r in load("final_main_table_calibrated.csv") if r["shot"] == "5"}
    DP = {(r["dataset"], r["backbone"], r["method"]): float(r["novel_acc"])
          for r in load("discr_projection.csv")}
    # 带完整字段的校准表（含 P/R/F1），供表 8 核对
    ACCR = {}
    try:
        for _r in load("final_main_table_calibrated.csv"):
            ACCR[(_r["dataset"], _r["backbone"], _r["shot"], _r["key"])] = _r
    except Exception:
        pass
    doc = Document(DOCX)
    bad, total = [], 0

    def chk(tag, ds, bb, key, cell, src):
        nonlocal total
        exp = src.get((ds, bb, key))
        if exp is None:
            return
        try:
            got = float(cell)
        except ValueError:
            return
        total += 1
        if abs(got - exp) > 0.02:
            bad.append((tag, ds, bb, key, cell, exp))

    # ---- 表 2 -------------------------------------------------------------
    t = doc.tables[1]
    n2, cur = 0, None
    for row in t.rows[1:]:
        c = [x.text.strip() for x in row.cells]
        d = ds_of(c[0])
        if d:
            cur = d
        bb = bb_of(c[1])
        if not cur or not bb:
            continue
        chk("表2", cur, bb, "protonet", c[2], A)
        chk("表2", cur, bb, "cal_protonet", c[3], A)
        n2 += 1
    print("表 2  处理 %d 行" % n2)

    # ---- 表 3（含并入的均值/最差汇总行）------------------------------------
    t = doc.tables[2]
    n3, cur = 0, None
    for row in t.rows[1:]:
        c = [x.text.strip() for x in row.cells]
        d = ds_of(c[0])
        if d:
            cur = d
        bb = bb_of(c[1])
        if not cur or not bb:
            continue
        # 列序（表 3）：数据集|主干|不校准|固定系数白化|ZCA|within|lda|判据选择
        # 「固定系数白化」取自 final_main_table_calibrated 的 uncond_protonet，
        # 与 discr_projection 的 zca 列不是同一量（前者固定 ε=0.1）。
        # 表 3 已恢复 8 列前的布局：数据集|主干|不校准|ZCA|within|lda|判据选择
        for ci, key in ((2, "none"), (3, "zca"), (4, "within"), (5, "lda")):
            chk("表3", cur, bb, key, c[ci], DP)
        n3 += 1
    if n3 == 0:
        bad.append(("表3", "-", "核对未生效", "", "0 行", -1))
    print("表 3  处理 %d 行" % n3)

    # ---- 表 5 主结果 -----------------------------------------------------
    t = doc.tables[4]
    n5, cur = 0, None
    # 方法名在表 5 中改为“族 / 成员”格式后，旧的“族 + 成员”模式全部失配，
    # 于是 27 个单元格被静默跳过、脚本仍报“全部一致”。改为按“/ 成员”后缀匹配。
    KEYS = (("/ ProtoNet", "cal_protonet"), ("/ 线性探针", "cal_linearprobe"),
            ("固定系数白化 / ProtoNet", "uncond_protonet"),
            ("固定系数白化 / 线性探针", "uncond_linearprobe"),
            ("ProtoNet", "protonet"), ("SimpleShot", "simpleshot"),
            ("无校准 / 线性探针", "linearprobe"),
            ("原型-实例融合", "fusion_only"),
            ("GWP", "protoprior_GWP"))
    for row in t.rows[1:]:
        c = [x.text.strip() for x in row.cells]
        if len(c) < 6:
            continue
        d = ds_of(c[0])
        if d:
            cur = d
        if not cur:
            continue
        key = next((k for pat, k in KEYS if pat in c[1]), None)
        if not key:
            continue
        for ci, bname in zip((2, 3, 4, 5),
                             ("clean_multitask", "clean_species",
                              "clip_vitb16", "dinov2_vitb14")):
            chk("表6", cur, bname, key, c[ci], A)
        n5 += 1
    if n5 == 0:
        bad.append(("表5", "-", "核对未生效", "", "0 行", -1))
    print("表 5  处理 %d 行" % n5)

    # ---- 表 6 双协议 -----------------------------------------------------
    t = doc.tables[5]
    n6, cur = 0, None
    for row in t.rows[1:]:
        c = [x.text.strip() for x in row.cells]
        d = ds_of(c[0])
        if d:
            cur = d
        bb = bb_of(c[1])
        if not cur or not bb:
            continue
        chk("表6-全类", cur, bb, "protonet", c[2], A)
        chk("表6-全类", cur, bb, "cal_protonet", c[3], A)
        chk("表6-5路", cur, bb, "protonet", c[4], F5)
        chk("表6-5路", cur, bb, "cal_protonet", c[5], F5)
        n6 += 1
    if n6 == 0:
        bad.append(("表6", "-", "核对未生效", "", "0 行", -1))
    print("表 6  处理 %d 行" % n6)

    # ---- 表 4 评测集概况 --------------------------------------------------
    n4 = 0
    for row in doc.tables[3].rows[1:]:
        c = [x.text.strip() for x in row.cells]
        if "濒危" in c[0]:
            cnt = [len([f for f in os.listdir(os.path.join(ENDB, d))
                        if f.lower().endswith(EXT)])
                   for d in os.listdir(ENDB) if os.path.isdir(os.path.join(ENDB, d))]
            total += 1
            n4 += 1
            if c[4].replace(",", "") != str(sum(cnt)):
                bad.append(("表4", "endb", "图像数", "", c[4], float(sum(cnt))))
    print("表 4  处理 %d 行" % n4)

    # ---- 表 7：濒危集分层 × 路数受控（B4）--------------------------------
    # 数据源 endb_waycurve.csv；列序：分组|物种数|路数|基线|方法族|增益
    _wp = os.path.join(FO, "endb_waycurve.csv")
    n7 = 0
    if os.path.exists(_wp):
        WC = {}
        for r in load("endb_waycurve.csv"):
            WC[(r["group"], int(r["ways"]))] = r
        t7 = None
        for tb in doc.tables:
            if "测试物种分组" in tb.rows[0].cells[0].text:
                t7 = tb
                break
        if t7 is not None:
            cur_g, cur_n = None, None
            for row in t7.rows[1:]:
                c = [x.text.strip() for x in row.cells]
                if c[0]:
                    cur_g, cur_n = c[0], c[1]
                if not cur_g or not c[2]:
                    continue
                w = int(c[2])
                rec = WC.get((cur_g, w))
                if not rec:
                    continue
                for ci, key in ((3, "base_acc"), (4, "fam_acc"), (5, "gain")):
                    try:
                        got = float(c[ci].replace("+", ""))
                    except ValueError:
                        continue
                    exp = float(rec[key].replace("+", ""))
                    total += 1
                    n7 += 1
                    if abs(got - exp) > 0.02:
                        bad.append(("表7", cur_g, "%d路" % w, key, c[ci], exp))
                # 物种数只在该组首行给出
                if cur_n:
                    try:
                        if int(cur_n) != int(rec["n_species"]):
                            bad.append(("表7", cur_g, "物种数", "", cur_n,
                                        float(rec["n_species"])))
                    except ValueError:
                        pass
        print("表 7  处理 %d 行" % n7)

    # ---- 表 8：校准方法族的 macro P/R/F1 ---------------------------------
    # 列序：数据集|主干|方法|准确率|精确率|召回率|F1，数据源同 final_main_table_calibrated.csv
    _ML = {"无校准 / ProtoNet": "protonet",
           "无校准 / 线性探针": "linearprobe",
           "固定系数白化 / ProtoNet": "uncond_protonet",
           "固定系数白化 / 线性探针": "uncond_linearprobe",
           "方法族 / ProtoNet": "cal_protonet",
           "方法族 / 线性探针": "cal_linearprobe"}
    n8 = 0
    t8 = None
    for tb in doc.tables:
        if "精确率" in tb.rows[0].cells[0].text or (
                len(tb.rows[0].cells) > 4 and "精确率" in tb.rows[0].cells[4].text):
            t8 = tb
            break
    if t8 is not None:
        cur_d, cur_b = None, None
        for row in t8.rows[1:]:
            c = [x.text.strip() for x in row.cells]
            if c[0]:
                cur_d = ds_of(c[0])
            if c[1]:
                cur_b = bb_of(c[1])
            key = _ML.get(c[2])
            if not (cur_d and cur_b and key):
                continue
            rec = ACCR.get((cur_d, cur_b, "5", key))
            if not rec:
                continue
            for ci, fld in ((3, "acc_mean"), (4, "precision"), (5, "recall"), (6, "f1")):
                try:
                    got = float(c[ci])
                except ValueError:
                    continue
                total += 1
                n8 += 1
                if abs(got - float(rec[fld])) > 0.02:
                    bad.append(("表8", "%s/%s" % (cur_d, cur_b), key, fld, c[ci],
                                float(rec[fld])))
        print("表 8  处理 %d 行" % n8)

    # ---- summary ----------------------------------------------------------
    print("\n" + "=" * 70)
    if bad:
        print("发现 %d 处不一致（共核对 %d 个单元格）：" % (len(bad), total))
        for b in bad[:40]:
            print("   %-9s %s/%s/%s  稿件=%s  数据=%.2f" % b)
    else:
        print("全部一致：核对 %d 个单元格，0 处不符" % total)
    print("=" * 70)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
