# -*- coding: utf-8 -*-
"""公式审计：提取全文公式段落，核对编号连续性与正文引用完整性。

检查项：
  A. 公式段落的实际内容（是否为 OMML 数学对象、是否含编号）
  B. 正文引用的"式(N)"是否都有对应公式
  C. 公式编号是否连续
"""
import io
import re
import sys

from docx import Document
from docx.oxml.ns import qn

DOCX = os.environ.get("PAPER_DOCX") or find_paper()

M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"


def main():
    d = Document(DOCX)
    ps = d.paragraphs

    print("=== 含数学对象（OMML）的段落 ===")
    math_paras = []
    for i, p in enumerate(ps):
        n_math = len(p._p.findall(".//" + M + "oMath")) + len(p._p.findall(".//" + M + "oMathPara"))
        if n_math:
            math_paras.append((i, p.text.strip(), n_math))
    print("  共 %d 段含数学对象" % len(math_paras))
    for i, t, n in math_paras[:26]:
        print("   段%-4d [%d] %s" % (i, n, t[:96]))
    if len(math_paras) > 26:
        print("   … 另有 %d 段" % (len(math_paras) - 26))

    full = "\n".join(x.text for x in ps)

    print("\n=== 正文引用的式号 ===")
    refs = sorted(set(re.findall(r"式\((\d+)\)", full)), key=int)
    print("  引用:", refs)

    print("\n=== 公式段落中出现的编号 ===")
    nums = []
    for i, t, n in math_paras:
        m = re.search(r"\((\d+)\)\s*$", t)
        if m:
            nums.append((i, int(m.group(1))))
    print("  编号:", [x[1] for x in nums])

    print("\n" + "=" * 70)
    bad = []
    if nums:
        seq = [x[1] for x in nums]
        if seq != list(range(1, len(seq) + 1)):
            bad.append(("公式编号不连续", seq, list(range(1, len(seq) + 1))))
    for r in refs:
        if int(r) not in [x[1] for x in nums]:
            bad.append(("正文引用式(%s) 但无对应公式" % r, r, "-"))
    # 反向：有公式但正文从未引用
    for _, num in nums:
        if str(num) not in refs:
            bad.append(("式(%d) 存在但正文未引用" % num, num, "-"))

    if bad:
        print("发现 %d 处问题：" % len(bad))
        for a, b, c in bad:
            print("  %-44s %s / %s" % (a, b, c))
    else:
        print("公式编号连续，正文引用与公式一一对应")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
