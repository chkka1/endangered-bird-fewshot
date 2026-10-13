# -*- coding: utf-8 -*-
"""
核对论文所写的卷期页与已核实元数据是否一致。

已核实的 14 条：[2][3][4][5][6][7][8][9][10][11][12][13][22][23]
来源：Crossref（crossref_audit2.json）、多源（multisource_audit.json）、以及本回合补查的 arXiv/DOI
"""
from __future__ import annotations

import io
import json
import os
import re

from docx import Document
from _paths import find_root, FINAL_OUT, find_paper  # noqa: E402

DOCX = os.environ.get("PAPER_DOCX") or find_paper()
FO = FINAL_OUT
# 已核实条目的权威元数据（volume, issue, page, year）
VERIFIED = {
    2:  ("2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)", "", "", "595-604", 2015),
    3:  ("NeurIPS / arXiv:1606.04080", "", "", "3630-3638", 2016),
    4:  ("NeurIPS / arXiv:1703.05175", "", "", "4077-4087", 2017),
    5:  ("arXiv:1911.03584", "", "", "", 2019),
    6:  ("NeurIPS / arXiv:2008.11297", "", "", "", 2020),
    7:  ("ICML / arXiv:2006.15486", "", "", "11660-11670", 2020),
    8:  ("LNCS", "", "", "487-499", 2021),
    9:  ("Aquacultural Engineering", "107", "", "102460", 2024),
    10: ("LNCS", "", "", "493-510", 2022),
    11: ("IJCV / arXiv:2110.04544", "132", "2", "581-595", 2024),
    12: ("arXiv:2104.00298", "", "", "", 2021),
    13: ("TMLR / arXiv:2304.07193", "", "", "", 2024),
    22: ("Journal of Multivariate Analysis", "88", "2", "365-411", 2004),
    23: ("The American Statistician", "72", "4", "309-314", 2018),
}


def refs():
    d = Document(DOCX)
    out = {}
    for p in d.paragraphs:
        m = re.match(r"^\[(\d+)\]\s*(.+)$", p.text.strip())
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def main():
    R = refs()
    print("%-4s %-16s %-12s %s" % ("#", "论文所写卷(期):页", "核实值", "判定"))
    print("-" * 88)
    problems = []
    for n in sorted(VERIFIED):
        text = R.get(n, "")
        ven, vol, iss, page, yr = VERIFIED[n]
        # 从论文条目里抽出形如 卷(期): 页 的模式
        m = re.search(r"(\d{4})[,\s]+(\d+)(?:\((\d+)\))?\s*[:：]\s*([\d\-–]+)", text)
        paper = m.group(0) if m else "（未解析出卷期页）"
        pv = m.group(2) if m else ""
        pi = m.group(3) if m else ""
        pp = m.group(4) if m else ""
        py = m.group(1) if m else ""
        flags = []
        if vol and pv and pv != vol:
            flags.append("卷 %s≠%s" % (pv, vol))
        if iss and pi and pi != iss:
            flags.append("期 %s≠%s" % (pi, iss))
        if page and pp and pp.replace("–", "-") != page:
            flags.append("页 %s≠%s" % (pp, page))
        if py and str(yr) != py:
            flags.append("年 %s≠%s" % (py, yr))
        ok = "一致" if not flags else "；".join(flags)
        if flags:
            problems.append((n, flags))
        print("%-4d %-16s %-12s %s" % (n, paper[:16], ("%s %s(%s):%s" % (yr, vol, iss, page)).strip(), ok))
    print("-" * 88)
    print("需人工核对：%d 条" % len(problems))
    for n, f in problems:
        print("  [%d] %s" % (n, "；".join(f)))
        print("      论文原文: %s" % R.get(n, "")[:150])


if __name__ == "__main__":
    main()
