# -*- coding: utf-8 -*-
"""
Build a genus-stratified split for NABirds, using exactly the rules applied to
CUB (paper_code/split_dataset_by_genus.py, random.seed(42), novel_ratio=0.35):

  R1  genus with 1 species  -> Base
  R2  genus with 2 species  -> 1 Base + 1 Novel
  R3  genus with >=3 species-> n_novel = max(2, int(n * 0.35)) to Novel, rest to Base
  +   every Novel genus must have a species in Base (100% genus overlap)

Input : data/NABirds/nabirds_taxonomy.csv   (class_name, scientific_name, genus, ...)
Output: data/NABirds/dataset_split_by_genus.json
"""
import csv
import json
import os
import random
from collections import defaultdict
from _paths import find_root  # noqa: E402

ROOT = (os.environ.get("BIRD_PROJECT_ROOT")
        or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DST = os.path.join(ROOT, "data", "NABirds")
TAXO = os.path.join(DST, "nabirds_taxonomy.csv")
OUT = os.path.join(DST, "dataset_split_by_genus.json")

# current taxonomy lumps the Northwestern Crow into the American Crow (Corvus caurinus)
MANUAL_FIX = {"Northwestern Crow": ("Corvus caurinus", "Corvus")}

random.seed(42)

rows = []
with open(TAXO, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        name = r["class_name"]
        sci, gen = r["scientific_name"], r["genus"]
        if not gen and name in MANUAL_FIX:
            sci, gen = MANUAL_FIX[name]
        if not gen:
            print("[warn] 无属信息，跳过: %s" % name)
            continue
        rows.append({"class_name": name, "scientific_name": sci, "genus": gen})

# species id = index in the sorted class-name order (stable, like CUB's species_id)
rows.sort(key=lambda r: r["class_name"])
name2id = {r["class_name"]: i for i, r in enumerate(rows)}
for r in rows:
    r["species_id"] = name2id[r["class_name"]]

print("可用类别: %d" % len(rows))

genus_to_species = defaultdict(list)
for r in rows:
    genus_to_species[r["genus"]].append(r["species_id"])
for g in genus_to_species:
    genus_to_species[g] = sorted(genus_to_species[g])

all_genus = sorted(genus_to_species)
print("总属数: %d  总物种数: %d" % (len(all_genus), sum(len(v) for v in genus_to_species.values())))

base, novel = [], []
novel_ratio = 0.35
for g in all_genus:
    sp = genus_to_species[g]
    n = len(sp)
    if n == 1:                                  # R1
        base.extend(sp); continue
    if n == 2:                                  # R2
        sh = sp.copy(); random.shuffle(sh)
        base.append(sh[0]); novel.append(sh[1]); continue
    n_novel = max(2, int(n * novel_ratio))      # R3
    sh = sp.copy(); random.shuffle(sh)
    base.extend(sh[:-n_novel]); novel.extend(sh[-n_novel:])

base, novel = sorted(base), sorted(novel)
assert not (set(base) & set(novel)), "Base 与 Novel 有重叠"

ng = defaultdict(int)
id2genus = {r["species_id"]: r["genus"] for r in rows}
for s in novel:
    ng[id2genus[s]] += 1
bg = set(id2genus[s] for s in base)

print()
print("=" * 78)
print("NABirds 属分层划分结果")
print("=" * 78)
print("Base  %3d 类 / %3d 属" % (len(base), len(bg)))
print("Novel %3d 类 / %3d 属" % (len(novel), len(ng)))
print("Novel 属中在 Base 出现的: %d / %d = %.1f%%"
      % (sum(1 for g in ng if g in bg), len(ng),
         100.0 * sum(1 for g in ng if g in bg) / max(len(ng), 1)))
print("Novel 侧多物种属: %d 个（各有 2 个物种: %s）"
      % (sum(1 for v in ng.values() if v >= 2),
         sorted({v for v in ng.values() if v >= 2})))
print("单物种属: %d" % sum(1 for v in ng.values() if v == 1))

out = {
    "base_species_ids": base,
    "novel_species_ids": novel,
    "total_genus": len(all_genus),
    "species_id_to_class": {str(r["species_id"]): r["class_name"] for r in rows},
    "species_id_to_genus": {str(r["species_id"]): r["genus"] for r in rows},
    "species_id_to_scientific": {str(r["species_id"]): r["scientific_name"] for r in rows},
    "split_rule": "属分层：单物种属全进Base；2物种属1/1；≥3物种属取 max(2, 0.35n) 进Novel；random.seed(42)",
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nsaved:", OUT)
