# -*- coding: utf-8 -*-
"""
Build an UNSEEN-GENUS control split for CUB-200.

Motivation
    The genus-stratified protocol guarantees every Novel genus also occurs in Base
    (100% genus overlap).  To measure what that guarantee is actually worth we need
    the opposite condition: Base and Novel must have DISJOINT genus sets, while
    keeping Base at the same size (145 classes) and Novel at a comparable size.

Construction
    Greedily pick the largest possible set of genera whose total class count is
    <= 145, preferring genera with few classes (so more genera fit inside the same
    class budget).  Those classes form Base; every remaining class forms Novel, and
    by construction Novel genera are disjoint from Base genera.

Output: cub_parsed/dataset_split_unseen_genus.json
"""
import json
import os
from collections import defaultdict
from _paths import find_root  # noqa: E402

ROOT = (os.environ.get("BIRD_PROJECT_ROOT")
        or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARSED = os.path.join(ROOT, "cub_parsed")
OUT = os.path.join(PARSED, "dataset_split_unseen_genus.json")

meta = json.load(open(os.path.join(PARSED, "cub_metadata.json"), encoding="utf-8"))
g2c = defaultdict(set)
c2g = {}
for m in meta:
    sid = int(m["species_id"])
    g2c[m["genus"]].add(sid)
    c2g[sid] = m["genus"]
print("物种 %d，属 %d" % (len(c2g), len(g2c)))

BUDGET = 145


def greedy(order):
    base, n = [], 0
    for g in order:
        c = sorted(g2c[g])
        if n + len(c) <= BUDGET:
            base.extend(c)
            n += len(c)
        if n == BUDGET:
            break
    return sorted(base)


candidates = {
    "few_classes_first": greedy(sorted(g2c, key=lambda g: (len(g2c[g]), g))),
    "many_classes_first": greedy(sorted(g2c, key=lambda g: (-len(g2c[g]), g))),
}
for name, base in candidates.items():
    bg = {c2g[c] for c in base}
    novel = sorted(set(c2g) - set(base))
    ng = {c2g[c] for c in novel}
    print("\n[%s] Base %d 类 / %d 属；Novel %d 类 / %d 属；属交集 = %d"
          % (name, len(base), len(bg), len(novel), len(ng), len(bg & ng)))

base = candidates["few_classes_first"]

# Prefer a Novel set that MATCHES the genus-stratified split's structure:
# 39 genera / 55 classes, made of 16 two-species genera + 23 single-species genera.
# That way the only difference between the two protocols is genus overlap.
by_size = defaultdict(list)
for g, cs in g2c.items():
    by_size[len(cs)].append(g)
for k in by_size:
    by_size[k].sort()

need_two, need_one = 16, 23
if len(by_size[2]) >= need_two and len(by_size[1]) >= need_one:
    import random
    random.seed(42)
    g2 = random.sample(by_size[2], need_two)
    g1 = random.sample(by_size[1], need_one)
    novel_g = set(g2) | set(g1)
    novel = sorted(c for g in novel_g for c in g2c[g])
    base = sorted(set(c2g) - set(novel))
    print("\n[matched] 用 16 个双物种属 + 23 个单物种属构造 Novel："
          "Novel %d 类 / %d 属；Base %d 类 / %d 属"
          % (len(novel), len(novel_g), len(base), len({c2g[c] for c in base})))
else:
    print("\n[matched] 素材不足：双物种属 %d（需 %d），单物种属 %d（需 %d）"
          % (len(by_size[2]), need_two, len(by_size[1]), need_one))

bg = {c2g[c] for c in base}
novel = sorted(set(c2g) - set(base))
ng = {c2g[c] for c in novel}
assert not (bg & ng), "Base 与 Novel 的属必须完全不相交"

# compare with the genus-stratified split
gs = json.load(open(os.path.join(PARSED, "dataset_split_by_genus.json"), encoding="utf-8"))
gs_novel_g = {c2g[c] for c in gs["novel_species_ids"]}
gs_base_g = {c2g[c] for c in gs["base_species_ids"]}
print("\n对照：属分层划分 Base %d 类/%d 属，Novel %d 类/%d 属，属交集 %d/ 重叠率 %.0f%%"
      % (len(gs["base_species_ids"]), len(gs_base_g),
         len(gs["novel_species_ids"]), len(gs_novel_g), len(gs_novel_g & gs_base_g),
         100.0 * len(gs_novel_g & gs_base_g) / len(gs_novel_g)))

out = {
    "base_species_ids": base,
    "novel_species_ids": novel,
    "total_genus": len(g2c),
    "base_genus_count": len(bg),
    "novel_genus_count": len(ng),
    "genus_overlap": 0,
    "split_rule": "未见属对照：贪心取类数最少的属填满 145 类作为 Base，其余为 Novel，"
                  "Base 与 Novel 的属集合完全不相交（属重叠率 0%）",
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nsaved:", OUT)
