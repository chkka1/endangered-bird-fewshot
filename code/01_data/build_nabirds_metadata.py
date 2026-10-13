# -*- coding: utf-8 -*-
"""
Build CUB-compatible metadata for NABirds plus a species table.

Outputs (all under data/NABirds/)
    nabirds_metadata.json          [{image_id, image_path, class_name, species_id,
                                     genus, scientific_name, source_split}]
    nabirds_train_test_split.csv   image_id, is_train      (mirrors CUB's csv)
    nabirds_species_table.csv/.md  species table with split membership and counts
"""
import csv
import json
import os
from collections import defaultdict
from _paths import find_root  # noqa: E402

ROOT = (os.environ.get("BIRD_PROJECT_ROOT")
        or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DST = os.path.join(ROOT, "data", "NABirds")
IMAGES = os.path.join(DST, "images")
TAXO = os.path.join(DST, "nabirds_taxonomy.csv")
SPLIT = os.path.join(DST, "dataset_split_by_genus.json")

# taxonomy keyed by a normalised class name
taxo = {}
with open(TAXO, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        key = r["class_name"].strip().lower()
        taxo[key] = {"class_name": r["class_name"].strip(),
                     "scientific_name": r["scientific_name"],
                     "genus": r["genus"]}
print("分类表条目 = %d" % len(taxo))

split = json.load(open(SPLIT, encoding="utf-8"))
sid2cls = split["species_id_to_class"]          # "0" -> class_name
cls2sid = {v.strip().lower(): int(k) for k, v in sid2cls.items()}
base_ids = set(split["base_species_ids"])
novel_ids = set(split["novel_species_ids"])

# scan extracted folders
found = defaultdict(lambda: {"train": [], "test": []})
for sub in ("train", "test"):
    root = os.path.join(IMAGES, sub)
    if not os.path.isdir(root):
        continue
    for cls in sorted(os.listdir(root)):
        cp = os.path.join(root, cls)
        if not os.path.isdir(cp):
            continue
        for fn in sorted(os.listdir(cp)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                found[cls.strip()][sub].append("%s/%s/%s" % (sub, cls, fn))

print("发现类目录 = %d" % len(found))

missing_taxo = [c for c in found if c.lower() not in taxo]
missing_split = [c for c in found if c.lower() not in cls2sid]
print("分类表缺失: %d %s" % (len(missing_taxo), missing_taxo[:5]))
print("划分表缺失: %d %s" % (len(missing_split), missing_split[:5]))

meta, rows, table = [], [], []
iid = 0
for cls in sorted(found, key=lambda c: cls2sid.get(c.lower(), 10 ** 9)):
    key = cls.lower()
    if key not in cls2sid or key not in taxo:
        continue
    sid = cls2sid[key]
    t = taxo[key]
    tr, te = found[cls]["train"], found[cls]["test"]
    for sub, lst in (("train", tr), ("test", te)):
        for rel in lst:
            iid += 1
            meta.append({"image_id": iid, "image_path": rel, "class_name": t["class_name"],
                         "species_id": sid, "genus": t["genus"],
                         "scientific_name": t["scientific_name"], "source_split": sub})
            rows.append({"image_id": iid, "is_train": "1" if sub == "train" else "0"})
    table.append({"species_id": sid, "class_name": t["class_name"],
                  "scientific_name": t["scientific_name"], "genus": t["genus"],
                  "split": "Base" if sid in base_ids else "Novel",
                  "n_train": len(tr), "n_test": len(te), "n_total": len(tr) + len(te)})

with open(os.path.join(DST, "nabirds_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False)
with open(os.path.join(DST, "nabirds_train_test_split.csv"), "w", newline="",
          encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["image_id", "is_train"])
    w.writeheader()
    w.writerows(rows)
table.sort(key=lambda r: (r["split"], r["species_id"]))
with open(os.path.join(DST, "nabirds_species_table.csv"), "w", newline="",
          encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["species_id", "class_name", "scientific_name",
                                      "genus", "split", "n_train", "n_test", "n_total"])
    w.writeheader()
    w.writerows(table)

nbase = [r for r in table if r["split"] == "Base"]
nnovel = [r for r in table if r["split"] == "Novel"]
print()
print("图像总数 = %d（train %d / test %d）"
      % (len(meta), sum(1 for m in meta if m["source_split"] == "train"),
         sum(1 for m in meta if m["source_split"] == "test")))
print("物种表: Base %d 类 / Novel %d 类，共 %d 类" % (len(nbase), len(nnovel), len(table)))
print("Base  图像 train=%d test=%d" % (sum(r["n_train"] for r in nbase),
                                       sum(r["n_test"] for r in nbase)))
print("Novel 图像 train=%d test=%d" % (sum(r["n_train"] for r in nnovel),
                                       sum(r["n_test"] for r in nnovel)))
print("Novel 每类 test 图像: min=%d max=%d 均值=%.1f"
      % (min(r["n_test"] for r in nnovel), max(r["n_test"] for r in nnovel),
         sum(r["n_test"] for r in nnovel) / len(nnovel)))
print("Novel 每类 train 图像: min=%d max=%d"
      % (min(r["n_train"] for r in nnovel), max(r["n_train"] for r in nnovel)))

with open(os.path.join(DST, "nabirds_species_table.md"), "w", encoding="utf-8") as f:
    f.write("# NABirds 物种表（本次使用的 405 类版本）\n\n")
    f.write("> 数据来源：HuggingFace 镜像 `gjuggler/bird-data` 的 NABirds 配置"
            "（train/test 两个 zip）。**该镜像只含 405 个物种，而官方 NABirds v1 为 555 个物种**，"
            "论文中应写作 “NABirds-405（来自 …镜像）” 或改取官方全量版本。\n\n")
    f.write("> 属名经 iNaturalist 分类 API 还原子学名后取得；划分规则同 CUB："
            "单物种属→Base，2 物种属 1/1，≥3 物种属取 max(2, 0.35n) 进 Novel，random.seed(42)。\n\n")
    f.write("| 汇总 | 值 |\n|---|---|\n")
    f.write("| 物种总数 | %d |\n" % len(table))
    f.write("| Base / Novel | %d / %d |\n" % (len(nbase), len(nnovel)))
    f.write("| 图像总数 | %d |\n" % len(meta))
    f.write("| Base 图像 | %d |\n" % sum(r["n_total"] for r in nbase))
    f.write("| Novel 图像 | %d |\n\n" % sum(r["n_total"] for r in nnovel))
    f.write("| 物种ID | 英文名 | 学名 | 属 | 划分 | train | test | 合计 |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    for r in table:
        f.write("| %d | %s | *%s* | %s | %s | %d | %d | %d |\n"
                % (r["species_id"], r["class_name"], r["scientific_name"], r["genus"],
                   r["split"], r["n_train"], r["n_test"], r["n_total"]))
print("\nsaved: nabirds_metadata.json / nabirds_train_test_split.csv / "
      "nabirds_species_table.csv / .md")
