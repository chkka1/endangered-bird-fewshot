# -*- coding: utf-8 -*-
"""
Build a common-name -> scientific-name -> genus mapping for the 555 NABirds
species, using the iNaturalist taxonomy API.

Why not the original hierarchy.txt: the NABirds archives available on the HF
mirror ship images + class folder names only.  NABirds follows the
eBird/Clements taxonomy; iNaturalist exposes the same binomials, so the genus is
recoverable from the scientific name.

Outputs:
  data/NABirds/nabirds_class_names.txt   one common name per line (folder names)
  data/NABirds/nabirds_taxonomy.csv      class_name, scientific_name, genus, inat_id, matched_name
"""
import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request
from _paths import find_root  # noqa: E402

ROOT = (os.environ.get("BIRD_PROJECT_ROOT")
        or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DST = os.path.join(ROOT, "data", "NABirds")
os.makedirs(DST, exist_ok=True)
NAMES_TXT = os.path.join(DST, "nabirds_class_names.txt")
OUT_CSV = os.path.join(DST, "nabirds_taxonomy.csv")
INA = "https://api.inaturalist.org/v1/taxa"

# fallback: if the source script cannot be fetched, the names come from the zip listing
HF_SCRIPT = ("https://hf-mirror.com/datasets/gjuggler/bird-data/raw/main/bird-data.py")


def fetch_class_list():
    if os.path.exists(NAMES_TXT):
        with open(NAMES_TXT, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
    req = urllib.request.Request(HF_SCRIPT, headers={"User-Agent": "dsh/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        src = r.read().decode("utf-8", "replace")
    m = re.search(r'_CLASSES = """(.*?)"""', src, re.S)
    if not m:
        raise SystemExit("cannot parse _CLASSES from bird-data.py")
    names = [l.strip() for l in m.group(1).splitlines() if l.strip()]
    with open(NAMES_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")
    return names


def lookup(name, tries=3):
    q = urllib.parse.quote(name)
    url = "%s?q=%s&rank=species&per_page=5" % (INA, q)
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dsh/1.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                data = json.load(r)
            return data.get("results", [])
        except Exception:
            time.sleep(1.5 * (t + 1))
    return []


def norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def main():
    names = fetch_class_list()
    print("NABirds 类名: %d 个" % len(names))
    existing = {}
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                existing[row["class_name"]] = row
        print("已有映射: %d 条，将复用" % len(existing))

    rows = []
    for i, name in enumerate(names):
        if name in existing and existing[name].get("scientific_name"):
            rows.append(existing[name])
            continue
        res = lookup(name)
        sci, inat_id, matched = "", "", ""
        for r in res:
            rn = r.get("name", "")
            # scientific name check: two words, first capitalised
            if r.get("rank") == "species" and len(rn.split()) == 2:
                sci = rn
                inat_id = r.get("id", "")
                matched = r.get("preferred_common_name", "")
                # prefer an exact common-name match
                if norm(matched) == norm(name):
                    break
        genus = sci.split()[0] if sci else ""
        rows.append({"class_name": name, "scientific_name": sci, "genus": genus,
                     "inat_id": inat_id, "matched_name": matched})
        if (i + 1) % 25 == 0 or i == len(names) - 1:
            got = sum(1 for r in rows if r["genus"])
            print("  %3d/%d  (已解析属: %d)" % (i + 1, len(names), got), flush=True)
            with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=["class_name", "scientific_name",
                                                  "genus", "inat_id", "matched_name"])
                w.writeheader()
                w.writerows(rows)
        time.sleep(0.6)

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["class_name", "scientific_name", "genus",
                                          "inat_id", "matched_name"])
        w.writeheader()
        w.writerows(rows)

    miss = [r["class_name"] for r in rows if not r["genus"]]
    genera = {}
    for r in rows:
        if r["genus"]:
            genera.setdefault(r["genus"], []).append(r["class_name"])
    print()
    print("解析成功 %d / %d" % (len(rows) - len(miss), len(rows)))
    if miss:
        print("未解析: %s" % miss[:20])
    print("覆盖属数: %d" % len(genera))
    sizes = sorted((len(v) for v in genera.values()), reverse=True)
    print("属内物种数分布(前15): %s" % sizes[:15])
    print("单物种属: %d" % sum(1 for v in genera.values() if len(v) == 1))
    print("多物种属: %d" % sum(1 for v in genera.values() if len(v) > 1))
    print("saved:", OUT_CSV)


if __name__ == "__main__":
    main()
