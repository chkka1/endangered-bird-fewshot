# -*- coding: utf-8 -*-
"""
Cache features for the UNSEEN-GENUS control condition.

Groups (mirroring the genus-stratified caches)
    cubunseen_base   train-split images of the control Base (145 classes / 101 genera)
    cubunseen_novel  test-split images of the control Novel (55 classes / 39 genera,
                     genera disjoint from Base)
"""
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import cache_features as cf   # noqa: E402
from _paths import find_root  # noqa: E402

ROOT = (os.environ.get("BIRD_PROJECT_ROOT")
        or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARSED = os.path.join(ROOT, "cub_parsed")
CACHE = cf.CACHE
BATCH = 32

meta = json.load(open(os.path.join(PARSED, "cub_metadata.json"), encoding="utf-8"))
with open(os.path.join(PARSED, "cub_train_test_split.csv"), encoding="utf-8-sig") as f:
    is_train = {r["image_id"].strip(): r["is_train"].strip() for r in csv.DictReader(f)}
split = json.load(open(os.path.join(PARSED, "dataset_split_unseen_genus.json"),
                       encoding="utf-8"))
base_ids = set(split["base_species_ids"])
novel_ids = set(split["novel_species_ids"])
genera = sorted({m["genus"] for m in meta})
g2id = {g: i for i, g in enumerate(genera)}

groups = {}
for gname, ids, want_train in (("cubunseen_base", base_ids, True),
                               ("cubunseen_novel", novel_ids, False)):
    items = []
    for m in meta:
        sid = int(m["species_id"])
        if sid not in ids:
            continue
        if (is_train.get(str(m["image_id"]).strip()) == "1") != want_train:
            continue
        items.append((m["image_path"], sid, g2id[m["genus"]]))
    groups[gname] = items
    print("%-16s %d 张" % (gname, len(items)))

# sanity: genus overlap between the two groups must be zero
bg = {g for _, _, g in groups["cubunseen_base"]}
ng = {g for _, _, g in groups["cubunseen_novel"]}
print("Base 属 %d / Novel 属 %d / 交集 %d  <- 必须为 0" % (len(bg), len(ng), len(bg & ng)))
assert not (bg & ng)


def resolve(rel):
    p = os.path.join(cf.CUB_ROOT, "images", "base", rel)
    return p if os.path.exists(p) else os.path.join(cf.CUB_ROOT, "images", "novel", rel)


def extract(model, items, desc):
    feats, flips = [], []
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        x = torch.stack([cf.TRANSFORM(Image.open(resolve(p)).convert("RGB"))
                         for p, _, _ in chunk]).to(cf.DEVICE)
        with torch.no_grad():
            f = F.normalize(model.avgpool(model.features(x)).flatten(1), dim=1)
            g = F.normalize(model.avgpool(model.features(torch.flip(x, dims=[3]))).flatten(1), dim=1)
        feats.append(f.cpu()); flips.append(g.cpu())
        if (i // BATCH) % 20 == 0:
            print("    %s %d/%d" % (desc, min(i + BATCH, len(items)), len(items)), flush=True)
    return torch.cat(feats).numpy(), torch.cat(flips).numpy()


def main(tags):
    for tag in tags:
        print("=" * 80)
        print("MODEL %s" % tag, flush=True)
        model = cf.build_model(cf.MODELS[tag])
        for gname, items in groups.items():
            out = os.path.join(CACHE, "%s__%s.npz" % (gname, tag))
            if os.path.exists(out):
                print("  skip %s" % os.path.basename(out))
                continue
            f, g = extract(model, items, gname)
            np.savez_compressed(out, feats=f, feats_flip=g,
                                species=np.array([s for _, s, _ in items], dtype=np.int64),
                                genus=np.array([q for _, _, q in items], dtype=np.int64),
                                path=np.array([p for p, _, _ in items]))
            print("  saved %s %s" % (os.path.basename(out), f.shape), flush=True)
        del model
        torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main(sys.argv[1:] or ["unseen_multitask"])
