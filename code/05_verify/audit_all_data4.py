import re
# -*- coding: utf-8 -*-
"""表 10（濒危集分层 × 路数）逐格核对。

表头：测试物种分组 | 物种数 | 路数 | 基线 | 校准方法族 | 增益 Δ
数据源：endb_waycurve.csv（8 行：2 组 × 4 档路数）
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
    W = {}
    for r in csv.DictReader(io.open(os.path.join(FO, "endb_waycurve.csv"),
                                    encoding="utf-8-sig")):
        W[(r["group"], r["ways"])] = r

    d = Document(DOCX)
    t = tbc(d, "表 10")
    assert t is not None, "表 10 未找到"
    bad, total = [], 0
    grp = None
    for r in t.rows[1:]:
        c = [" ".join(x.text.split()) for x in r.cells]
        for g in ("有 CUB 同属", "无 CUB 同属"):
            if g in c[0] or (not c[0] and grp == g):
                grp = g
        # 分组列可能因合并单元格为空，沿用上一行
        if not c[0].strip():
            pass
        w = W.get((grp, c[2].strip()))
        if not w:
            continue
        for col, f in ((1, "n_species"), (3, "base_acc"), (4, "fam_acc"), (5, "gain")):
            # 该列为合并/稀疏标注（只在每组首行有值），空单元格属正常排版，跳过
            if not c[col].strip():
                continue
            total += 1
            try:
                if abs(float(c[col].replace("+", "")) - float(w[f])) > 0.02:
                    bad.append(("表10 %s/%s路/%s" % (grp, c[2], f), c[col], w[f]))
            except (TypeError, ValueError):
                bad.append(("表10 %s/%s路/%s" % (grp, c[2], f), c[col], w[f]))

    print("表 10：核对 %d 格" % total)
    print("=" * 70)
    if bad:
        print("发现 %d 处不符：" % len(bad))
        for a, b, c in bad:
            print("  %-40s 稿件=%s  数据=%s" % (a, b, c))
    else:
        print("全部一致：0 处不符")
    print("=" * 70)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
