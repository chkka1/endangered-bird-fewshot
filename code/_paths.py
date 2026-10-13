# -*- coding: utf-8 -*-
"""包内路径解析（供各脚本导入）。由 rebuild_release.py 生成。

目录布局兼容两种：
  包内（发布形态）   ROOT/results · ROOT/paper
  原项目（开发形态） ROOT/_tools/final_out · ROOT/paper_code
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def find_root(start=None):
    """包根：优先 BIRD_PROJECT_ROOT，否则向上找含 paper 或 paper_code 的目录。"""
    env = os.environ.get("BIRD_PROJECT_ROOT")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    d = os.path.abspath(start or _HERE)
    while True:
        if any(os.path.isdir(os.path.join(d, x))
               for x in ("paper", "paper_code", "_tools", "results")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return _HERE
        d = parent


def _first(*cands):
    for c in cands:
        if os.path.isdir(c):
            return c
    return cands[0]


ROOT = find_root()
TOOLS = os.path.join(ROOT, "_tools")
RESULTS = _first(os.path.join(ROOT, "results"),
                 os.path.join(TOOLS, "final_out"))
FINAL_OUT = RESULTS
FEAT_CACHE = _first(os.path.join(TOOLS, "feat_cache"),
                    os.path.join(ROOT, "feat_cache"))
PAPER_CODE = _first(os.path.join(ROOT, "paper"),
                    os.path.join(ROOT, "paper_code"))
DATA = _first(os.path.join(ROOT, "dataset"), os.path.join(ROOT, "data"))

def find_paper():
    """稿件路径。

    优先取 paper 目录下版本号最大的 v*.docx；若稿件已按论文名命名
    （非 v*.docx），则取该目录下唯一的 .docx。
    """
    import re as _re
    d = PAPER_CODE
    if not os.path.isdir(d):
        return None
    docs = [x for x in os.listdir(d)
            if x.lower().endswith(".docx") and not x.startswith("~$")]
    vs = [x for x in docs if _re.match(r"^v\d+\.docx$", x)]
    if vs:
        return os.path.join(
            d, max(vs, key=lambda x: int(_re.search(r"v(\d+)", x).group(1))))
    return os.path.join(d, docs[0]) if docs else None

