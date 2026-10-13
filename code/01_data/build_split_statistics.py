# -*- coding: utf-8 -*-
"""
Generate split statistics for all three datasets and refresh the release package.

Outputs (written both to the source data dirs and into release/dataset/)
    <dataset>_species_table.csv      per species: genus, split membership, counts
    <dataset>_genus_statistics.csv   per genus: size, Base/Novel membership
    <dataset>_split_summary.md       human-readable summary incl. same-genus pairs

'同属对'(same-genus pair) counts how many unordered pairs of Novel classes share a
genus, and how many Novel classes have at least one same-genus Base class — the two
structural quantities the paper uses to characterise the protocol.
"""
import csv
import io
import json
import os
import shutil
from collections import Counter, defaultdict

ROOT = (os.environ.get("BIRD_PROJECT_ROOT")
        or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL = os.path.join(ROOT, "release", "dataset")
CUB = os.path.join(ROOT, "cub_parsed")
NB = os.path.join(ROOT, "data", "NABirds")
ENDB = os.path.join(ROOT, "data", "endangered_birds")


def pairs(n):
    return n * (n - 1) // 2


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def genus_stats(name, c2g, base, novel, extra_base=None):
    """Per-genus membership table + summary."""
    g2all, g2b, g2n = defaultdict(set), defaultdict(set), defaultdict(set)
    for c, g in c2g.items():
        g2all[g].add(c)
        if c in base:
            g2b[g].add(c)
        if c in novel:
            g2n[g].add(c)
    rows = []
    for g in sorted(g2all):
        rows.append({
            "genus": g,
            "n_species_total": len(g2all[g]),
            "n_in_base": len(g2b[g]),
            "n_in_novel": len(g2n[g]),
            "membership": ("Base+Novel" if g2b[g] and g2n[g]
                           else "Base only" if g2b[g] else "Novel only"),
        })
    nb = {g for g in g2all if g2b[g]}
    nn = {g for g in g2all if g2n[g]}
    ng_novel = len(nn)
    overlap = len(nb & nn)
    novel_cls_with_base_genus = sum(1 for c in novel if c2g[c] in nb)
    same_pairs = sum(pairs(len(v)) for v in g2n.values())
    base_pairs = sum(pairs(len(v)) for v in g2b.values())
    return rows, {
        "dataset": name,
        "n_species": len(c2g),
        "n_genus": len(g2all),
        "n_base_species": len(base),
        "n_novel_species": len(novel),
        "n_base_genus": len(nb),
        "n_novel_genus": ng_novel,
        "n_transferable_novel_genus": overlap,
        "novel_genus_transfer_rate_pct": round(100.0 * overlap / max(ng_novel, 1), 2),
        "n_novel_class_with_base_genus": novel_cls_with_base_genus,
        "novel_class_transfer_rate_pct": round(
            100.0 * novel_cls_with_base_genus / max(len(novel), 1), 2),
        "n_same_genus_pairs_novel": same_pairs,
        "n_same_genus_pairs_base": base_pairs,
        # only genera that actually contain a Novel species: including empty ones
        # would add a "0 species" bucket that reads as a data error
        "genus_size_hist_novel": dict(sorted(
            Counter(len(v) for v in g2n.values() if v).items())),
        "genus_size_hist_all": dict(sorted(Counter(len(v) for v in g2all.values()).items())),
    }


# ------------------------------------------------------------------ CUB
cmeta = json.load(open(os.path.join(CUB, "cub_metadata.json"), encoding="utf-8"))
c2g_cub, name_cub = {}, {}
for m in cmeta:
    c2g_cub[int(m["species_id"])] = m["genus"]
    name_cub[int(m["species_id"])] = m["species_name_en"]
csp = json.load(open(os.path.join(CUB, "dataset_split_by_genus.json"), encoding="utf-8"))
cup = json.load(open(os.path.join(CUB, "dataset_split_unseen_genus.json"), encoding="utf-8"))
cub_base, cub_novel = set(csp["base_species_ids"]), set(csp["novel_species_ids"])
cup_base, cup_novel = set(cup["base_species_ids"]), set(cup["novel_species_ids"])

# ------------------------------------------------------------------ NABirds
nmeta = json.load(open(os.path.join(NB, "nabirds_metadata.json"), encoding="utf-8"))
n_name, n_genus = {}, {}
n_train, n_test = Counter(), Counter()
for m in nmeta:
    s = m["species_id"]
    n_name[s] = m["class_name"]
    n_genus[s] = m["genus"]
    (n_train if m["source_split"] == "train" else n_test)[s] += 1
nsp = json.load(open(os.path.join(NB, "dataset_split_by_genus.json"), encoding="utf-8"))
nup = json.load(open(os.path.join(NB, "dataset_split_unseen_genus.json"), encoding="utf-8"))
nb_base, nb_novel = set(nsp["base_species_ids"]), set(nsp["novel_species_ids"])
nu_base, nu_novel = set(nup["base_species_ids"]), set(nup["novel_species_ids"])

by_name = {n_name[s]: s for s in n_name}
nrows = []
for s in sorted(n_name, key=lambda x: n_name[x]):
    nrows.append({
        "species_id": s,
        "class_name": n_name[s],
        "genus": n_genus[s],
        "split_genus_stratified": "Base" if s in nb_base else "Novel",
        "split_unseen_genus": "Base" if s in nu_base else "Novel",
        "n_train": n_train.get(s, 0),
        "n_test": n_test.get(s, 0),
        "n_total": n_train.get(s, 0) + n_test.get(s, 0),
    })
write_csv(os.path.join(NB, "nabirds_species_table.csv"), nrows)
write_csv(os.path.join(REL, "nabirds", "nabirds_species_table.csv"), nrows)

nrows2 = [{"species_id": r["species_id"], "class_name": r["class_name"],
           "scientific_name": r.get("scientific_name", ""), "genus": r["genus"],
           "split": "Base" if r["species_id"] in nb_base else "Novel",
           "n_train": r["n_train"], "n_test": r["n_test"], "n_total": r["n_total"]}
          for r in nrows]

# ------------------------------------------------------------------ endangered
src = io.open(os.path.join(ROOT, "Endangered", "endangered_birds_dataset.py"),
              encoding="utf-8").read()
# parse the map literal instead of exec'ing the module (it imports torch, which the
# packaging interpreter does not have)
import ast                                                  # noqa: E402
from _paths import find_root  # noqa: E402
sgm = None
for node in ast.parse(src).body:
    if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "SPECIES_GENUS_MAP" for t in node.targets):
        sgm = ast.literal_eval(node.value)
        break
if sgm is None:
    raise SystemExit("未找到 SPECIES_GENUS_MAP")
man = list(csv.DictReader(io.open(os.path.join(ENDB, "image_sources.csv"),
                                  encoding="utf-8-sig")))
e_cnt, e_obs = Counter(), defaultdict(set)
for r in man:
    e_cnt[r["scientific_name"]] += 1
    e_obs[r["scientific_name"]].add(r["observer"])
e2g = {k.replace("_", " "): v[1] for k, v in sgm.items()}
e_sorted = sorted(e2g)                       # stable index mapping
e2i = {name: i for i, name in enumerate(e_sorted)}
e_c2g = {e2i[name]: e2g[name] for name in e_sorted}   # index -> genus
e_novel = set(range(len(e_sorted)))
erows = []
for sci in e_sorted:
    erows.append({"species_id": e2i[sci], "scientific_name": sci,
                  "genus": e2g[sci], "split": "Novel",
                  "n_images": e_cnt.get(sci, 0),
                  "n_observers": len(e_obs.get(sci, []))})
write_csv(os.path.join(REL, "endangered_birds", "species_table.csv"), erows)
write_csv(os.path.join(ENDB, "species_table.csv"), erows)

# ------------------------------------------------------------------ genus stats
summaries = []
stats = [
    ("CUB-200 属分层版", c2g_cub, cub_base, cub_novel),
    ("CUB-200 未见属对照", c2g_cub, cup_base, cup_novel),
    ("NABirds-405 属分层版", n_genus, nb_base, nb_novel),
    ("NABirds-405 未见属对照", n_genus, nu_base, nu_novel),
    ("濒危鸟类（自建）", e_c2g, set(), e_novel),
]
outputs = {
    "CUB-200 属分层版": ("cub", "cub_species_table.csv", "cub_genus_statistics.csv"),
    "CUB-200 未见属对照": (None, None, None),
    "NABirds-405 属分层版": ("nabirds", None, "nabirds_genus_statistics.csv"),
    "NABirds-405 未见属对照": (None, None, None),
    "濒危鸟类（自建）": ("endangered_birds", None, "endangered_birds_genus_statistics.csv"),
}
for label, c2g, base, novel in stats:
    rows, summ = genus_stats(label, c2g, base, novel)
    sub, spec, gen = outputs[label]
    if gen:
        write_csv(os.path.join(REL, sub, gen), rows)
    summ["label"] = label
    summaries.append(summ)

# ------------------------------------------------------------------ CUB species table
g2name = {}
mrows = []
for s in sorted(c2g_cub):
    mrows.append({
        "species_id": s,
        "species_name": name_cub[s],
        "genus": c2g_cub[s],
        "split_genus_stratified": "Base" if s in cub_base else "Novel",
        "split_unseen_genus": "Base" if s in cup_base else "Novel",
    })
write_csv(os.path.join(REL, "cub", "cub_species_table.csv"), mrows)

# ------------------------------------------------------------------ console
print("=" * 78)
print("各划分的属级结构统计")
print("=" * 78)
hdr = ["划分", "类数", "属数", "Base类", "Novel类", "Test属可迁移率", "Novel同属对"]
print("%-24s %5s %5s %6s %6s %12s %10s" % tuple(hdr))
print("-" * 78)
for s in summaries:
    print("%-24s %5d %5d %6d %6d %11.1f%% %10d" % (
        s["label"], s["n_species"], s["n_genus"], s["n_base_species"], s["n_novel_species"],
        s["novel_genus_transfer_rate_pct"], s["n_same_genus_pairs_novel"]))
print()
for s in summaries:
    print("[%s]" % s["label"])
    print("   Novel 属内物种数分布（属内物种数:属数）: %s" % s["genus_size_hist_novel"])
    print("   Novel 类中有同属 Base 类的比例: %.1f%% (%d/%d)"
          % (s["novel_class_transfer_rate_pct"], s["n_novel_class_with_base_genus"],
             s["n_novel_species"]))
    print("   Novel 侧同属对 %d 组；Base 侧同属对 %d 组"
          % (s["n_same_genus_pairs_novel"], s["n_same_genus_pairs_base"]))

with io.open(os.path.join(REL, "SPLIT_STATISTICS.md"), "w", encoding="utf-8") as f:
    f.write("# 三个数据集的划分与属级结构统计\n\n")
    f.write("| 划分 | 类数 | 属数 | Base 类 | Novel 类 | 测试属可迁移率 | Novel 同属对 |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for s in summaries:
        f.write("| %s | %d | %d | %d | %d | %.1f%% | %d |\n" % (
            s["label"], s["n_species"], s["n_genus"], s["n_base_species"],
            s["n_novel_species"], s["novel_genus_transfer_rate_pct"],
            s["n_same_genus_pairs_novel"]))
    f.write("\n## 各划分细节\n\n")
    for s in summaries:
        f.write("### %s\n\n" % s["label"])
        f.write("- Novel 属内物种数分布（`属内物种数:属数`，仅统计含 Novel 物种的属）：`%s`\n"
                % s["genus_size_hist_novel"])
        f.write("- 全部属的属内物种数分布：`%s`\n" % s["genus_size_hist_all"])
        f.write("- Novel 类中有同属 Base 类的比例：**%.1f%%**（%d/%d）\n" % (
            s["novel_class_transfer_rate_pct"], s["n_novel_class_with_base_genus"],
            s["n_novel_species"]))
        f.write("- Novel 侧同属对：**%d 组**；Base 侧同属对：%d 组\n\n" % (
            s["n_same_genus_pairs_novel"], s["n_same_genus_pairs_base"]))
    f.write("> 同属对 = 同一属内两个不同物种构成的无序对，用于衡量同属近缘混淆的压力。\n")
print("\n已生成各数据集的 species_table / genus_statistics 与 SPLIT_STATISTICS.md")
