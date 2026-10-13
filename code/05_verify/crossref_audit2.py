# -*- coding: utf-8 -*-
"""
逐条 Crossref 核验（修正版）。

上一版的缺陷：用 query.bibliographic 且 rows=1，Crossref 必然返回"最接近"的条目，
脚本却未判断相似度，把毫不相关的结果也记为"已找到"。

本版加入标题相似度判定，分为三类：
    匹配        —— 返回标题与查询标题高度相似
    未收录      —— Crossref 无相似条目（会议录/中文期刊/技术报告常见）
    可疑        —— 有部分相似但作者/年份/卷期页不一致，需人工核
"""
from __future__ import annotations

import difflib
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request

from docx import Document
from _paths import find_root, find_paper  # noqa: E402
ROOT = find_root()

DOCX = os.environ.get("PAPER_DOCX") or find_paper()
OUT = os.path.join(ROOT, "_tools", "final_out", "crossref_audit2.json")
UA = "ref-audit/1.0 (mailto:research@example.org)"
SIM_OK, SIM_MAYBE = 0.82, 0.55


def fetch(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if k == tries - 1:
                return {"_error": str(e)}
            time.sleep(3 + 3 * k)
    return {}


def norm(s):
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", (s or "").lower())


def extract_refs():
    d = Document(DOCX)
    out = []
    for p in d.paragraphs:
        m = re.match(r"^\[(\d+)\]\s*(.+)$", p.text.strip())
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return sorted(out)


def guess_title(s):
    m = re.search(r"\[[JCRD]\]", s)
    body = s[:m.start()] if m else s
    parts = body.split(". ")
    t = ". ".join(parts[1:]) if len(parts) >= 2 else body
    return t.strip().strip(".")


def main():
    refs = extract_refs()
    print("提取 %d 条\n" % len(refs))
    res = []
    for num, text in refs:
        title = guess_title(text)
        q = urllib.parse.quote(title[:220])
        url = ("https://api.crossref.org/works?rows=3&select=DOI,title,author,"
               "container-title,volume,issue,page,issued&query.bibliographic=" + q)
        data = fetch(url)
        if "_error" in data:
            res.append({"n": num, "ref": text, "title_guess": title,
                        "status": "查询失败", "err": data["_error"]})
            print("[%2d] 查询失败" % num)
            continue
        items = data.get("message", {}).get("items", [])
        best, score = None, 0.0
        for it in items:
            ct = (it.get("title") or [""])[0]
            s = difflib.SequenceMatcher(None, norm(title), norm(ct)).ratio()
            if s > score:
                best, score = it, s
        if best is None or score < SIM_MAYBE:
            res.append({"n": num, "ref": text, "title_guess": title,
                        "status": "未收录", "best_score": round(score, 3),
                        "best_title": ((best or {}).get("title") or [""])[0]})
            print("[%2d] 未收录（最高相似度 %.2f） %s" % (num, score, title[:52]))
        else:
            ct = (best.get("title") or [""])[0]
            au = best.get("author", [])
            au_s = ", ".join(a.get("family", "") for a in au[:3]) if au else ""
            yr = (best.get("issued", {}).get("date-parts") or [[None]])[0][0]
            ven = (best.get("container-title") or [""])[0]
            st = "匹配" if score >= SIM_OK else "可疑"
            res.append({"n": num, "ref": text, "title_guess": title, "status": st,
                        "score": round(score, 3), "title": ct, "authors": au_s,
                        "year": yr, "venue": ven, "vol": best.get("volume", ""),
                        "issue": best.get("issue", ""), "page": best.get("page", ""),
                        "doi": best.get("DOI", "")})
            print("[%2d] %-4s (%.2f) %s" % (num, st, score, ct[:58]))
            print("       %s | %s | %s %s(%s):%s | %s"
                  % (au_s, yr, ven[:30], best.get("volume", ""), best.get("issue", ""),
                     best.get("page", ""), best.get("DOI", "")))
        time.sleep(1.0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n" + "=" * 70)
    for k in ("匹配", "可疑", "未收录", "查询失败"):
        print("  %-6s %d" % (k, sum(1 for r in res if r["status"] == k)))
    print("  saved: %s" % OUT)


if __name__ == "__main__":
    main()
