# -*- coding: utf-8 -*-
"""公式审计：提取全文公式段落，核对编号连续性与正文引用完整性。

检查项：
  A. 公式段落的实际内容（MathType OLE 对象 / OMML 数学对象、是否含编号）
  B. 正文引用的"式(N)"是否都有对应公式（只报真正找不到编号的）
  C. 公式编号是否连续

修复记录（2026-xx）：
  1. 原脚本只认 OMML（m:oMath）判定公式段落，而本稿件公式全部是
     MathType 嵌入对象（w:object，含 w:OLEObject 或 w:OLEObject/w:Embed），
     导致 math_paras 为空 -> nums 为空 -> 所有"式(N)"引用都被误报为缺公式。
     现同时识别 w:object / w:OLEObject / w:pict(v:imagedata) / m:oMath。
  2. 原脚本用 re.search(r"\\((\\d+)\\)\\s*$", text) 只取段落末尾"唯一"编号，
     漏掉同一段内含多个编号的情况（本稿件段 114 同行含 (15) 与 (16)，
     OLE ×2）。现用 finditer 提取段内全部 (n) 编号并按出现顺序记录。
  3. 编号提取兼容：全角括号（）、(n) 前后空白/制表符、编号被拆分到多个
     run 的情况（python-docx 的 paragraph.text 已拼接，故直接对 text 匹配）。
  4. 引用识别兼容 "式(1)" / "式（1）" / "式 ( 1 )" 等写法。
  5. 引用完整性只在"全文确实找不到该编号"时报错；"公式存在但正文未引用"
     降级为 INFO 提示，不再计入问题数（原实现将其算作问题，属误报）。
"""
import io
import re
import sys

import os
from docx import Document
from docx.oxml.ns import qn

def _latest_docx():
    """取 paper_code 下版本号最大的 v*.docx；无则取 release/paper 下的稿件。"""
    import glob as _g
    import re as _re
    cands = []
    for d in (r"D:\soft\bird_project\paper_code",
              r"D:\soft\bird_project\release\paper"):
        if os.path.isdir(d):
            cands += [x for x in _g.glob(os.path.join(d, "*.docx"))
                      if not os.path.basename(x).startswith("~$")]
    vs = [(int(_re.match(r"^v(\d+)\.docx$", os.path.basename(x)).group(1)), x)
          for x in cands if _re.match(r"^v(\d+)\.docx$", os.path.basename(x))]
    if vs:
        return max(vs)[1]
    # 回退：release/paper 下的中文名稿件
    rel = _g.glob(r"D:\soft\bird_project\release\paper\*.docx")
    rel = [x for x in rel if not os.path.basename(x).startswith("~$")]
    return max(rel, key=os.path.getmtime) if rel else (cands[0] if cands else "")


DOCX = os.environ.get("PAPER_DOCX") or _latest_docx()

M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
V = "{urn:schemas-microsoft-com:vml}"

# 正文引用 "式(1)" / "式（1）" / "式 ( 1 )"
RE_REF = re.compile(r"式\s*[（(]\s*(\d+)\s*[）)]")
# 公式段内编号 (1) / （1）；不锚定段尾，允许编号前后有空白、制表符、其它编号
RE_NUM = re.compile(r"[（(]\s*(\d+)\s*[）)]")


def para_objects(p):
    """返回段落内公式对象的构成：(MathType OLE 数, OMML 数)。

    兼容多种嵌入形态：
      * w:object             —— MathType 公式的标准包装（内含 w:OLEObject 本体
                                与 v:imagedata 显示预览，一个 w:object = 一个公式）
      * w:OLEObject          —— 未包在 w:object 里的裸 OLE 对象
      * w:pict + v:imagedata —— 老式 MathType "图片"形态（仅在无 w:object 时计为公式，
                                否则会与同一公式的 OLE 预览重复计数）
      * m:oMath / m:oMathPara—— 新版 Word 原生公式

    计数去重要点（否则 17 个公式会被数成 34 个）：
      w:OLEObject 与 v:imagedata 通常是 w:object 的**子元素**，不能重复累加。
    """
    el = p._p
    n_obj = len(el.findall(".//" + W + "object"))
    oles_all = el.findall(".//" + W + "OLEObject")
    oles_in_obj = sum(len(o.findall(".//" + W + "OLEObject"))
                      for o in el.findall(".//" + W + "object"))
    n_ole_bare = max(0, len(oles_all) - oles_in_obj)
    n_ole = n_obj + n_ole_bare
    if n_ole == 0:
        # 没有 w:object / 裸 OLE 时，才把 VML 图片形态当作公式
        n_ole = len(el.findall(".//" + V + "imagedata"))
    n_omml = (len(el.findall(".//" + M + "oMath"))
              + len(el.findall(".//" + M + "oMathPara")))
    return n_ole, n_omml


def para_text(p):
    """段落纯文本（w:t 已由 python-docx 拼接，编号被拆 run 也不受影响）。"""
    try:
        return p.text or ""
    except Exception:
        return ""


def collect_paragraphs(doc):
    """正文段落 + 表格单元格段落，统一成 (定位标签, 段落) 列表。"""
    out = [("段%d" % i, p) for i, p in enumerate(doc.paragraphs)]
    for ti, t in enumerate(doc.tables):
        for ri, row in enumerate(t.rows):
            for ci, cell in enumerate(row.cells):
                for pi, p in enumerate(cell.paragraphs):
                    out.append(("表%d/r%d/c%d/p%d" % (ti, ri, ci, pi), p))
    return out


def main():
    d = Document(DOCX)
    ps = d.paragraphs
    items = collect_paragraphs(d)

    print("文档: %s" % DOCX)
    print("段落数: %d（正文 %d + 表格 %d）"
          % (len(items), len(ps), len(items) - len(ps)))

    # ---------- A. 公式段落 ----------
    print("\n=== 含数学对象的段落（MathType OLE / OMML）===")
    math_paras = []   # (loc, text, n_ole, n_omml)
    n_ole_total = n_omml_total = 0
    for loc, p in items:
        n_ole, n_omml = para_objects(p)
        n_ole_total += n_ole
        n_omml_total += n_omml
        if n_ole or n_omml:
            math_paras.append((loc, para_text(p).strip(), n_ole, n_omml))
    print("  全文 MathType OLE 对象 %d 个，OMML 对象 %d 个"
          % (n_ole_total, n_omml_total))
    print("  共 %d 段含数学对象" % len(math_paras))
    for loc, t, n_ole, n_omml in math_paras[:40]:
        print("   %-16s [OLE %d / OMML %d] %s" % (loc, n_ole, n_omml, t[:96]))
    if len(math_paras) > 40:
        print("   … 另有 %d 段" % (len(math_paras) - 40))

    # 含公式对象但提取不到编号的段落（可能是无编号的行内公式）
    no_num = [(loc, t, n_ole, n_omml) for loc, t, n_ole, n_omml in math_paras
              if not RE_NUM.search(t)]
    if no_num:
        print("\n  [提示] 含数学对象但未提取到编号的段落 %d 段：" % len(no_num))
        for loc, t, n_ole, n_omml in no_num:
            print("   %-16s [OLE %d / OMML %d] %s" % (loc, n_ole, n_omml, t[:80]))

    # ---------- 全文纯文本 ----------
    full = "\n".join(para_text(p) for _, p in items)

    # ---------- B. 正文引用 ----------
    print("\n=== 正文引用的式号 ===")
    refs = sorted(set(int(x) for x in RE_REF.findall(full)))
    print("  引用:", refs if refs else "（无）")

    # ---------- C. 公式编号 ----------
    print("\n=== 公式段落中出现的编号 ===")
    nums = []          # (loc, num) 按出现顺序，允许一段多个
    for loc, t, n_ole, n_omml in math_paras:
        for m in RE_NUM.finditer(t):
            nums.append((loc, int(m.group(1))))
    num_vals = [n for _, n in nums]
    print("  编号出现顺序:", num_vals if num_vals else "（无）")
    print("  去重排序:", sorted(set(num_vals)) if num_vals else "（无）")

    # 段内编号数 > 公式对象数 或反之，给出提示（不判错）
    for loc, t, n_ole, n_omml in math_paras:
        k = len(RE_NUM.findall(t))
        if n_ole + n_omml > 0 and k > 1:
            print("  [提示] %s 同一段含 %d 个编号（编号=%s，OLE=%d）"
                  % (loc, k, RE_NUM.findall(t), n_ole))

    # ---------- 结论 ----------
    print("\n" + "=" * 70)
    errs = []    # 真问题
    infos = []   # 仅提示

    if num_vals:
        if len(set(num_vals)) != len(num_vals):
            dup = sorted({v for v in num_vals if num_vals.count(v) > 1})
            errs.append(("公式编号重复", dup, "-"))
        lo, hi = min(num_vals), max(num_vals)
        missing = [n for n in range(lo, hi + 1) if n not in set(num_vals)]
        if missing:
            errs.append(("公式编号不连续（缺 %s）" % missing,
                         "%d~%d" % (lo, hi), missing))
        elif sorted(set(num_vals)) != list(range(1, hi + 1)):
            errs.append(("公式编号非从 1 开始连续",
                         sorted(set(num_vals)), "1~%d" % hi))
    elif refs:
        errs.append(("全文未识别到任何公式编号，但正文引用了式号", refs, "-"))

    # 引用完整性：只报"全文确实找不到该编号"的引用
    num_set = set(num_vals)
    for r in refs:
        if r not in num_set:
            errs.append(("正文引用式(%d) 但无对应公式" % r, r, "-"))

    # 反向：公式存在但正文未引用 -> 仅提示，不计入问题
    for _, num in nums:
        if num not in refs:
            infos.append(("式(%d) 存在但正文未引用" % num, num, "-"))

    if errs:
        print("发现 %d 处问题：" % len(errs))
        for a, b, c in errs:
            print("  %-44s %s / %s" % (a, b, c))
    else:
        print("无问题：公式编号连续，正文引用均有对应公式")
    if infos:
        print("\n提示（非问题）%d 条：" % len(infos))
        for a, b, c in infos:
            print("  %-44s %s / %s" % (a, b, c))
    print("=" * 70)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
