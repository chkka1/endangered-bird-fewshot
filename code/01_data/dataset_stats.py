# -*- coding: utf-8 -*-
"""
Compute every statistic the dataset section needs, and write it to a JSON the
manuscript builder reads.  Keeps the expanded dataset section tied to the data, in line
with the rest of the paper (no hand-typed numbers).

Outputs _tools/final_out/dataset_stats.json
"""
import csv
import io
import json
import os
from collections import Counter, defaultdict
from _paths import find_root  # noqa: E402

ROOT = find_root()
ENDB = os.path.join(ROOT, "data", "endangered_birds")
MAN = os.path.join(ENDB, "image_sources.csv")
SPEC = os.path.join(ENDB, "species_table.csv")
IM = os.path.join(ENDB, "images")
OUT = os.path.join(ROOT, "_tools", "final_out", "dataset_stats.json")
EXT = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    man = list(csv.DictReader(io.open(MAN, encoding="utf-8-sig")))
    spp = list(csv.DictReader(io.open(SPEC, encoding="utf-8-sig")))

    # ---- per-class image counts from disk ---------------------------------
    counts = {}
    for d in sorted(os.listdir(IM)):
        fp = os.path.join(IM, d)
        if os.path.isdir(fp):
            n = len([f for f in os.listdir(fp) if f.lower().endswith(EXT)])
            if n:
                counts[d.split(".", 1)[1]] = n
    vals = sorted(counts.values())

    # ---- IUCN by species (propagate from any filled row) ------------------
    iucn_by_sp = {}
    for r in man:
        v = (r.get("iucn") or "").strip()
        if v:
            iucn_by_sp[r["scientific_name"]] = v
    # fall back to the species table if it carries the field
    for r in spp:
        v = (r.get("iucn") or r.get("IUCN") or "").strip()
        if v and r["scientific_name"] not in iucn_by_sp:
            iucn_by_sp[r["scientific_name"]] = v

    # ---- genus structure ---------------------------------------------------
    genus2sp = defaultdict(set)
    for s in counts:
        # folder names use underscores (Aquila_chrysaetos): split on "_" not on
        # whitespace, otherwise every species becomes its own genus
        genus2sp[s.split("_")[0]].add(s)
    multi = {g: sorted(v) for g, v in genus2sp.items() if len(v) >= 2}
    # same-genus pairs: sum over genera of C(n,2)
    pairs = sum(len(v) * (len(v) - 1) // 2 for v in genus2sp.values())

    # ---- observers / dates / places ---------------------------------------
    obs = {r.get("observer", "").strip() for r in man if r.get("observer", "").strip()}
    yrs = Counter((r.get("observed_on") or "")[:4] for r in man
                  if (r.get("observed_on") or "")[:4].isdigit())

    stats = {
        "n_species": len(counts),
        "n_genus": len(genus2sp),
        "n_multi_genus": len(multi),
        "n_same_genus_pairs": pairs,
        "n_images": sum(vals),
        "per_class_min": min(vals), "per_class_max": max(vals),
        "per_class_median": vals[len(vals) // 2],
        "per_class_mean": round(sum(vals) / len(vals), 1),
        "n_class_at_max": sum(1 for v in vals if v == max(vals)),
        "n_class_below_max": sum(1 for v in vals if v < max(vals)),
        "n_class_at_min": sum(1 for v in vals if v == min(vals)),
        "license": dict(Counter((r.get("license") or "?").upper() for r in man)),
        "license_verified": dict(Counter(r.get("license_verified", "") for r in man)),
        "n_observers": len(obs),
        "iucn_by_species": dict(Counter(iucn_by_sp.values())),
        "iucn_covered_species": len(iucn_by_sp),
        "year_min": min(yrs) if yrs else "",
        "year_max": max(yrs) if yrs else "",
        "year_n_dated": sum(yrs.values()),
        "year_n_2000plus": sum(v for k, v in yrs.items() if int(k) >= 2000),
        "year_n_pre1950": sum(v for k, v in yrs.items() if int(k) < 1950),
        "n_countries": len({(r.get("place") or "").split(",")[-1].strip()
                            for r in man if r.get("place")}),
    }
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    print("数据集统计 -> %s" % OUT)
    for k, v in stats.items():
        if k != "iucn_by_species":
            print("  %-22s %s" % (k, v))
    print("  iucn_by_species        %s" % stats["iucn_by_species"])
    print("  （IUCN 有记录的物种 %d / %d）" % (stats["iucn_covered_species"],
                                              stats["n_species"]))
    print("\n多物种属（前 10）:")
    for g, v in sorted(multi.items(), key=lambda x: -len(x[1]))[:10]:
        print("   %-16s %d 种: %s" % (g, len(v), ", ".join(x.split()[-1] for x in v)))


if __name__ == "__main__":
    main()
