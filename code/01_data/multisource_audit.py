# -*- coding: utf-8 -*-
"""
用 DBLP 与 arXiv 核验 Crossref 未收录的条目。

来源选择依据：
    DBLP  —— CS 会议录（NeurIPS/ICML/CVPR）收录完整
    arXiv —— 预印本（SimpleShot、EfficientNetV2、DINOv2 等）
    中文期刊（软件学报/自动化学报/中国图象图形学报）两者均不覆盖，需人工查期刊官网。
"""
from __future__ import annotations

import difflib
import io
import json
import os
import re
import time
import urllib.parse
import ssl
import urllib.request
import xml.etree.ElementTree as ET
from _paths import find_root  # noqa: E402
ROOT = find_root()

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

OUT = os.path.join(ROOT, "_tools", "final_out", "multisource_audit.json")
UA = "ref-audit/1.0 (mailto:research@example.org)"

# 12 条 Crossref 未收录 + 4 条可疑 + 2 条失败 = 要核的条目
TARGETS = [
    (1, "The Caltech-UCSD Birds-200-2011 Dataset", "techreport"),
    (3, "Matching Networks for One Shot Learning", "conf"),
    (4, "Prototypical Networks for Few-shot Learning", "conf"),
    (5, "SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning", "arxiv"),
    (6, "Transductive Information Maximization for Few-Shot Learning", "conf"),
    (7, "LaplacianShot: Laplacian Regularized Few-Shot Learning", "conf"),
    (9, "CLIP-FSSC: A transferable visual model for fish and shrimp species classification", "journal"),
    (11, "CLIP-Adapter: Better Vision-Language Models with Feature Adapters", "journal"),
    (12, "EfficientNetV2: Smaller Models and Faster Training", "arxiv"),
    (13, "DINOv2: Learning Robust Visual Features without Supervision", "journal"),
    (14, "小样本学习研究综述", "cn"),
    (15, "基于深度卷积特征的细粒度图像分类研究综述", "cn"),
    (16, "基于小样本学习的图像分类技术综述", "cn"),
    (17, "基于特征变换和度量网络的小样本学习算法", "cn"),
    (18, "基于TC-YOLO模型的北京珍稀鸟类识别方法", "cn"),
    (19, "注意力集合表示的多尺度度量小样本图像分类", "cn"),
    (20, "网络监督数据下的细粒度图像识别综述", "cn"),
    (21, "融合弱监督目标定位的细粒度小样本学习", "cn"),
]


def get(url, tries=3, timeout=30):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
                return r.read()
        except Exception as e:
            if k == tries - 1:
                return b""
            time.sleep(2 + 2 * k)
    return b""


def norm(s):
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", (s or "").lower())


def dblp(title):
    url = "https://dblp.org/search/publ/api?format=json&h=3&q=" + urllib.parse.quote(title)
    raw = get(url)
    if not raw:
        return []
    try:
        hits = json.loads(raw.decode("utf-8"))["result"]["hits"].get("hit", [])
        if isinstance(hits, dict):      # DBLP 只有 1 条结果时返回 dict
            hits = [hits]
    except Exception:
        return []
    out = []
    for h in hits:
        i = h.get("info", {})
        out.append({"title": i.get("title", ""), "venue": i.get("venue", ""),
                    "year": i.get("year", ""), "authors": i.get("authors", {}),
                    "doi": i.get("doi", ""), "type": i.get("type", "")})
    return out


def arxiv(title):
    q = 'ti:"%s"' % title.replace('"', "")
    url = ("https://export.arxiv.org/api/query?max_results=3&search_query="
           + urllib.parse.quote(q))
    raw = get(url)
    if not raw:
        return []
    try:
        root = ET.fromstring(raw)
    except Exception:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in root.findall("a:entry", ns):
        out.append({"title": " ".join((e.findtext("a:title", "", ns) or "").split()),
                    "id": e.findtext("a:id", "", ns),
                    "published": (e.findtext("a:published", "", ns) or "")[:10]})
    return out


def main():
    res = []
    for n, title, kind in TARGETS:
        row = {"n": n, "title": title, "kind": kind}
        # DBLP 对中文标题无效，跳过
        if kind != "cn":
            d = dblp(title)
            best = max(d, key=lambda x: difflib.SequenceMatcher(
                None, norm(title), norm(x["title"])).ratio()) if d else None
            if best:
                sc = difflib.SequenceMatcher(None, norm(title), norm(best["title"])).ratio()
                row["dblp"] = {**best, "score": round(sc, 3)}
            a = arxiv(title)
            if a:
                sc = difflib.SequenceMatcher(None, norm(title), norm(a[0]["title"])).ratio()
                row["arxiv"] = {**a[0], "score": round(sc, 3)}

        ds = row.get("dblp", {}).get("score", 0)
        as_ = row.get("arxiv", {}).get("score", 0)
        if kind == "cn":
            row["status"] = "需人工查期刊官网"
        elif max(ds, as_) >= 0.85:
            row["status"] = "已核实"
        elif max(ds, as_) >= 0.6:
            row["status"] = "部分匹配"
        else:
            row["status"] = "未找到"
        res.append(row)
        d = row.get("dblp"); a = row.get("arxiv")
        print("[%2d] %-8s %s" % (n, row["status"], title[:56]))
        if d:
            print("      DBLP  %.2f | %s | %s %s | %s"
                  % (d["score"], d["title"][:48], d["venue"][:18], d["year"], d.get("doi", "")))
        if a:
            print("      arXiv %.2f | %s | %s" % (a["score"], a["title"][:48], a["published"]))
        time.sleep(1.0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n" + "=" * 70)
    for k in ("已核实", "部分匹配", "未找到", "需人工查期刊官网"):
        print("  %-16s %d" % (k, sum(1 for r in res if r["status"] == k)))
    print("  saved: %s" % OUT)


if __name__ == "__main__":
    main()
