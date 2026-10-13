# -*- coding: utf-8 -*-
"""
Cache frozen features for NABirds, in the same npz layout the evaluation uses.

Groups
    nab_base   train-split images of the 259 genus-stratified Base classes
               (only used to fit whitening / genus prototypes — never evaluated)
    nab_novel  test-split images of the 146 Novel classes (the evaluation pool)

Usage:
    python cache_nabirds_features.py clean_multitask clip_vitb16 dinov2_vitb14
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import cache_features as cf            # noqa: E402  (EfficientNet checkpoints)
import cache_timm_features as ctf      # noqa: E402  (timm backbones)
from _paths import find_root  # noqa: E402

ROOT = (os.environ.get("BIRD_PROJECT_ROOT")
        or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DST = os.path.join(ROOT, "data", "NABirds")
IMAGES = os.path.join(DST, "images")
CACHE = cf.CACHE
BATCH = 32

meta = json.load(open(os.path.join(DST, "nabirds_metadata.json"), encoding="utf-8"))
split = json.load(open(os.path.join(DST, "dataset_split_by_genus.json"), encoding="utf-8"))
base_ids = set(split["base_species_ids"])
novel_ids = set(split["novel_species_ids"])

genera = sorted({m["genus"] for m in meta})
g2id = {g: i for i, g in enumerate(genera)}
print("NABirds: %d 图像, %d 属; Base %d 类, Novel %d 类"
      % (len(meta), len(genera), len(base_ids), len(novel_ids)))

groups = {
    "nab_base": [("/".join([m["source_split"], m["class_name"], os.path.basename(m["image_path"])]),
                  m["species_id"], g2id[m["genus"]])
                 for m in meta if m["species_id"] in base_ids and m["source_split"] == "train"],
    "nab_novel": [("/".join([m["source_split"], m["class_name"], os.path.basename(m["image_path"])]),
                   m["species_id"], g2id[m["genus"]])
                  for m in meta if m["species_id"] in novel_ids and m["source_split"] == "test"],
}
for k, v in groups.items():
    print("  %-10s %d 张" % (k, len(v)))


def extract(model, items, transform, desc, is_timm):
    """is_timm=False: EfficientNet checkpoint with a classifier head -> pooled features.
       is_timm=True : timm model built with num_classes=0 -> model(x) is already pooled."""
    def fwd(x):
        if is_timm:
            return model(x)
        return model.avgpool(model.features(x)).flatten(1)

    feats, flips = [], []
    t0 = time.time()
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        x = torch.stack([transform(Image.open(os.path.join(IMAGES, p)).convert("RGB"))
                         for p, _, _ in chunk]).to(cf.DEVICE)
        with torch.no_grad():
            f = F.normalize(fwd(x).float(), dim=1)
            g = F.normalize(fwd(torch.flip(x, dims=[3])).float(), dim=1)
        feats.append(f.cpu())
        flips.append(g.cpu())
        if (i // BATCH) % 25 == 0:
            d = min(i + BATCH, len(items))
            print("    %s %d/%d (%.0fs, %.0f img/s)"
                  % (desc, d, len(items), time.time() - t0, d / max(time.time() - t0, 1e-6)),
                  flush=True)
    return torch.cat(feats).numpy(), torch.cat(flips).numpy()


def main(names):
    for tag in names:
        print("=" * 80)
        print("MODEL %s" % tag, flush=True)
        if tag in cf.MODELS:                       # EfficientNet checkpoint
            model = cf.build_model(cf.MODELS[tag])
            tf = cf.TRANSFORM
            is_timm = False
        else:                                       # timm backbone
            model, cfg = ctf.build(ctf.TIMM_MODELS[tag])
            tf = ctf.timm.data.create_transform(**cfg, is_training=False)
            is_timm = True
        for gname, items in groups.items():
            out = os.path.join(CACHE, "%s__%s.npz" % (gname, tag))
            if os.path.exists(out):
                print("  skip existing %s" % os.path.basename(out))
                continue
            f, g = extract(model, items, tf, gname, is_timm)
            np.savez_compressed(
                out, feats=f, feats_flip=g,
                species=np.array([s for _, s, _ in items], dtype=np.int64),
                genus=np.array([q for _, _, q in items], dtype=np.int64),
                path=np.array([p for p, _, _ in items]))
            print("  saved %s shape=%s" % (os.path.basename(out), f.shape), flush=True)
        del model
        torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main(sys.argv[1:] or ["clean_multitask", "clip_vitb16", "dinov2_vitb14"])
