# -*- coding: utf-8 -*-
"""
Endangered-bird image collection (v2).

Modes
    expand     add the expansion-plan species as new class folders (031, 032, ...)
    recollect  re-download the ORIGINAL 30 classes from CC-licensed sources so the
               whole dataset has uniform, recorded provenance.  The old images are
               MOVED to a timestamped backup folder first, never deleted.

Sources
    naming / taxonomy : eBird (Clements) taxonomy, no API key required
    images            : iNaturalist open API, restricted to CC0 / CC-BY / CC-BY-NC
                        every image records licence, observer, date, place, observation URL

Target images per class: 25 when the genus ends up with >=2 species in the final
set (so same-genus confusion is measurable), else 20.  With 5-shot support that
leaves 15-20 query images per class, roughly halving the per-class variance
compared with the original 10-image classes.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from _paths import find_root  # noqa: E402

ROOT = (os.environ.get("BIRD_PROJECT_ROOT")
        or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IUCN = os.path.join(ROOT, "data", "iucn")
EBIRD_DIR = os.path.join(ROOT, "data", "ebird")
EBIRD_TAXO = os.path.join(EBIRD_DIR, "ebird_taxonomy.csv")
ENDB = os.path.join(ROOT, "data", "endangered_birds")
IMAGES = os.path.join(ENDB, "images")
BACKUP = os.path.join(ENDB, "backup_original30")
MANIFEST = os.path.join(ENDB, "image_sources.csv")
PLAN = os.path.join(IUCN, "endangered_expansion_plan.csv")

LICENSES = "cc0,cc-by,cc-by-nc"
UA = {"User-Agent": "dsh-endangered-benchmark/1.0 (research dataset construction)"}
sys.path.insert(0, os.path.join(ROOT, "Endangered"))

# current taxonomy differs from some names hard-coded in the old code
SYNONYM = {
    "Grus vipio": "Antigone vipio",
    "Gorsachius magnificus": "Oroanassa magnifica",
    "Oroanassa magnifica": "Oroanassa magnifica",
    "Calamornis heudei": "Paradoxornis heudei",
    "Schoeniclus aureolus": "Emberiza aureola",
    "Calidris pygmaea": "Calidris pygmaea",
}


def http_json(url, tries=3, pause=1.5):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except Exception:
            time.sleep(pause * (t + 1))
    return None


def download(url, dest, tries=3):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            if os.path.getsize(dest) > 2000:
                return True
            os.remove(dest)
        except Exception:
            time.sleep(1.5 * (t + 1))
    return False


def load_ebird():
    if not os.path.exists(EBIRD_TAXO):
        os.makedirs(EBIRD_DIR, exist_ok=True)
        url = "https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=csv&locale=en"
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=180) as r, open(EBIRD_TAXO, "wb") as f:
            shutil.copyfileobj(r, f)
    idx = {}
    with open(EBIRD_TAXO, encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            sci = (row.get("scientific name") or row.get("SCIENTIFIC_NAME") or "").strip()
            if not sci:
                continue
            idx[sci.lower()] = {
                "sci": sci,
                "com": (row.get("common name") or row.get("COMMON_NAME") or "").strip(),
                "code": (row.get("species code") or row.get("SPECIES_CODE") or "").strip(),
                "family": (row.get("family") or row.get("FAMILY_SCI_NAME") or "").strip()}
    return idx


def resolve_name(sci, idx):
    """Map a possibly outdated scientific name onto the eBird/Clements name."""
    if sci in SYNONYM:
        return SYNONYM[sci], idx.get(SYNONYM[sci].lower(), {})
    hit = idx.get(sci.lower())
    if hit:
        return hit["sci"], hit
    # fall back: same species epithet, genus sharing a prefix
    parts = sci.split()
    if len(parts) == 2:
        g, e = parts[0], parts[1]
        for k, v in idx.items():
            vp = v["sci"].split()
            if len(vp) == 2 and vp[1].lower() == e.lower() and \
                    vp[0][:3].lower() == g[:3].lower():
                return v["sci"], v
    return sci, {}


def inat_photos(sci, need):
    out, page = [], 1
    while len(out) < need and page <= 8:
        url = ("https://api.inaturalist.org/v1/observations?taxon_name=%s"
               "&quality_grade=research&photos=true&photo_license=%s"
               "&per_page=30&page=%d&order_by=votes"
               % (urllib.parse.quote(sci), LICENSES, page))
        data = http_json(url)
        if not data or not data.get("results"):
            break
        for ob in data["results"]:
            ph = ob.get("photos") or []
            if not ph:
                continue
            p = ph[0]
            if not p.get("url"):
                continue
            out.append({"photo_url": p["url"],
                        "license": (p.get("license_code") or "").upper(),
                        "observer": (ob.get("user") or {}).get("login", ""),
                        "observed_on": ob.get("observed_on", ""),
                        "place": ob.get("place_guess", ""),
                        "observation_id": ob.get("id"),
                        "observation_url": "https://www.inaturalist.org/observations/%s" % ob.get("id")})
            if len(out) >= need:
                break
        page += 1
        time.sleep(0.6)
    return out


def load_original_species():
    src = open(os.path.join(ROOT, "Endangered", "endangered_birds_dataset.py"),
               encoding="utf-8").read()
    ns = {}
    exec(compile(src.split("class EndangeredBirdsDataset")[0], "m", "exec"), ns)
    m = ns["SPECIES_GENUS_MAP"]
    out = []
    for name, (gid, genus, cn) in m.items():
        out.append({"sci": name.replace("_", " "), "genus": genus, "cn": cn})
    out.sort(key=lambda d: d["sci"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["expand", "recollect"], required=True)
    ap.add_argument("--tiers", nargs="*", default=None)
    ap.add_argument("--phase-a", action="store_true",
                    help="expand: take the T1+T2 minimal-viable subset (40 species)")
    ap.add_argument("--per-species", type=int, default=0)
    ap.add_argument("--species", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ebird = load_ebird()
    print("eBird 分类表: %d 个分类单元" % len(ebird))

    # ---------------- build the work list ----------------
    work = []          # dicts: sci, genus, common, iucn, tier, folder, target
    if args.mode == "recollect":
        orig = load_original_species()
        print("原有物种: %d 个" % len(orig))
    else:
        orig = []

    plan = []
    if os.path.exists(PLAN):
        plan = [r for r in csv.DictReader(open(PLAN, encoding="utf-8-sig"))
                if not r["recommendation"].startswith("不建议")]
        order = {"T1": 0, "T2": 1, "T3": 2, "T4": 3}
        plan.sort(key=lambda r: order[r["tier"]])
        if args.phase_a:
            plan = [r for r in plan if r["tier"] in ("T1", "T2")][:40]
        if args.tiers:
            plan = [r for r in plan if r["tier"] in args.tiers]

    # genus census over the FINAL set (original + all plan species) -> target images
    genus_total = Counter()
    for d in orig:
        genus_total[d["genus"]] += 1
    for r in csv.DictReader(open(PLAN, encoding="utf-8-sig")):
        genus_total[r["genus"]] += 1

    def target_for(genus):
        return 25 if genus_total.get(genus, 1) >= 2 else 20

    existing = sorted(d for d in os.listdir(IMAGES)
                      if os.path.isdir(os.path.join(IMAGES, d))) if os.path.isdir(IMAGES) else []
    used_nums = [int(d.split(".")[0]) for d in existing if d.split(".")[0].isdigit()]
    next_idx = (max(used_nums) + 1) if used_nums else 1
    present = {d.split(".", 1)[1].lower() for d in existing if "." in d}

    if args.mode == "recollect":
        for i, d in enumerate(orig, 1):
            sci, eb = resolve_name(d["sci"], ebird)
            folder = "%03d.%s" % (i, sci.replace(" ", "_"))
            # keep the folder that currently holds this species if the name changed
            cur = [f for f in existing if f.lower().endswith(d["sci"].replace(" ", "_").lower())]
            if cur:
                folder = cur[0]
            work.append({"sci": sci, "raw": d["sci"], "genus": d["genus"],
                         "common": eb.get("com", d["cn"]), "iucn": "", "tier": "原有",
                         "folder": folder, "target": args.per_species or target_for(d["genus"])})
    else:
        skipped = 0
        for r in plan:
            sci, eb = resolve_name(r["scientificName"], ebird)
            if sci.replace(" ", "_").lower() in present:
                skipped += 1
                continue
            work.append({"sci": sci, "raw": r["scientificName"], "genus": r["genus"],
                         "common": eb.get("com", r["commonName"]), "iucn": r["redlistCategory"],
                         "tier": r["tier"], "folder": None,
                         "target": args.per_species or int(r["target_images"])})
        if skipped:
            print("已在数据集中、跳过 %d 个物种" % skipped)
        for w in work:
            w["folder"] = "%03d.%s" % (next_idx, w["sci"].replace(" ", "_"))
            next_idx += 1

    if args.species:
        work = work[:args.species]

    print("模式=%s  待处理物种=%d%s\n" % (args.mode, len(work), "  (dry-run)" if args.dry_run else ""))

    # ---------------- back up the original images once ----------------
    if args.mode == "recollect" and not args.dry_run:
        os.makedirs(BACKUP, exist_ok=True)
        moved = 0
        for w in work:
            d = os.path.join(IMAGES, w["folder"])
            if not os.path.isdir(d):
                continue
            dst = os.path.join(BACKUP, w["folder"])
            if os.path.isdir(dst) and len(os.listdir(dst)) > 0:
                continue          # already backed up in an earlier run
            os.makedirs(dst, exist_ok=True)
            for fn in os.listdir(d):
                shutil.move(os.path.join(d, fn), os.path.join(dst, fn))
                moved += 1
        print("原有图像已备份到 %s（%d 张），原目录已清空待重采\n" % (BACKUP, moved))

    manifest_rows = []
    if os.path.exists(MANIFEST) and not args.dry_run:
        manifest_rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))

    log_rows, short = [], []
    for i, w in enumerate(work):
        dst = os.path.join(IMAGES, w["folder"])
        have = len([f for f in os.listdir(dst)]) if os.path.isdir(dst) else 0
        print("[%d/%d] %-34s 属=%-14s 目标=%2d 已有=%2d -> %s"
              % (i + 1, len(work), w["sci"], w["genus"], w["target"], have, w["folder"]), flush=True)

        photos = inat_photos(w["sci"], w["target"])
        if len(photos) < 5:
            print("        可用 CC 照片仅 %d 张，跳过" % len(photos))
            short.append((w["sci"], len(photos)))
            # recollect mode emptied this folder up front: restore the backup so a
            # class can never be left with zero images
            if args.mode == "recollect" and not args.dry_run:
                bdir = os.path.join(BACKUP, w["folder"])
                if os.path.isdir(bdir):
                    os.makedirs(dst, exist_ok=True)
                    for fn in os.listdir(bdir):
                        shutil.copy2(os.path.join(bdir, fn), os.path.join(dst, fn))
                    print("        已恢复原有 %d 张，避免该类变空" % len(os.listdir(bdir)))
            continue
        if args.dry_run:
            print("        可用照片 %d 张（dry-run）" % len(photos))
            time.sleep(0.4)
            continue

        os.makedirs(dst, exist_ok=True)
        ok = 0
        for j, p in enumerate(photos, 1):
            # 'medium' (~500 px) is plenty for 224-300 px training input and is
            # 2-3x smaller than 'large'; the open-data CDN is slow from this network
            big = (p["photo_url"].replace("/square.", "/medium.")
                   .replace("/large.", "/medium."))
            ext = os.path.splitext(urllib.parse.urlparse(big).path)[1] or ".jpg"
            fn = "%s_%03d%s" % (w["sci"].replace(" ", "_"), j, ext)
            fp = os.path.join(dst, fn)
            if os.path.exists(fp):
                ok += 1
                continue
            if download(big, fp):
                ok += 1
                manifest_rows.append({
                    "folder": w["folder"], "file": fn,
                    "species_code": ebird.get(w["sci"].lower(), {}).get("code", ""),
                    "scientific_name": w["sci"], "common_name": w["common"],
                    "genus": w["genus"], "iucn": w["iucn"], "tier": w["tier"],
                    "license": p["license"], "observer": p["observer"],
                    "observed_on": p["observed_on"], "place": p["place"],
                    "observation_url": p["observation_url"], "photo_url": big})
            time.sleep(0.2)
        print("        下载 %d/%d" % (ok, len(photos)))
        log_rows.append((w["folder"], w["sci"], w["genus"], ok, w["target"]))
        if not args.dry_run:
            with open(MANIFEST, "w", newline="", encoding="utf-8-sig") as f:
                wr = csv.DictWriter(f, fieldnames=["folder", "file", "species_code",
                                                   "scientific_name", "common_name", "genus",
                                                   "iucn", "tier", "license", "observer",
                                                   "observed_on", "place", "observation_url",
                                                   "photo_url"])
                wr.writeheader()
                wr.writerows(manifest_rows)

    if not args.dry_run:
        print("\n来源清单: %s（%d 条）" % (MANIFEST, len(manifest_rows)))
        print("完成 %d 个类" % len(log_rows))
        print("图像不足(<5)而跳过的: %s" % (short if short else "无"))
        total = sum(r[3] for r in log_rows)
        print("本次下载 %d 张" % total)


if __name__ == "__main__":
    main()
