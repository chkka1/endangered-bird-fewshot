# -*- coding: utf-8 -*-
"""
Cache frozen features from strong timm backbones (CLIP / DINOv2 / ImageNet-21k ViT).

Why: the paper's current backbone is EfficientNetV2-S fine-tuned on 145 base
classes, and its frozen features only reach ProtoNet 78.3% on CUB 5-shot.
Modern vision backbones give far better frozen features; switching backbone is
the single largest accuracy lever available here.

Usage:
    python cache_timm_features.py clip_vitb16
    python cache_timm_features.py clip_vitb16 dinov2_vitb14
"""
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import cache_features as cf  # noqa: E402  (reuses the dataset item builders)
from _paths import find_root  # noqa: E402
ROOT = find_root()

CACHE = cf.CACHE
BATCH = 32

TIMM_MODELS = {
    "clip_vitb16": "vit_base_patch16_clip_224.openai",
    "clip_vitb16_12k": "vit_base_patch16_clip_224.openai_ft_in12k_in1k",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "in21k_vitb16": "vit_base_patch16_224_in21k",
    "swin_base": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
}


def build(name):
    """Build with the model's own default input size (DINOv2 uses 518, not 224)."""
    m = timm.create_model(name, pretrained=True, num_classes=0)
    m = m.to(cf.DEVICE).eval()
    cfg = timm.data.resolve_data_config({}, model=m)
    print("  %s | params %.1fM | input=%s | mean=%s std=%s"
          % (name, sum(p.numel() for p in m.parameters()) / 1e6,
             tuple(cfg["input_size"]), [round(v, 3) for v in cfg["mean"]],
             [round(v, 3) for v in cfg["std"]]))
    return m, cfg


@torch.no_grad()
def extract(model, items, transform, desc):
    feats, flips = [], []
    t0 = time.time()
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        x = torch.stack([transform(Image.open(p).convert("RGB")) for p, _, _ in chunk])
        x = x.to(cf.DEVICE)
        f = F.normalize(model(x), dim=1).float()
        g = F.normalize(model(torch.flip(x, dims=[3])), dim=1).float()
        feats.append(f.cpu())
        flips.append(g.cpu())
        if (i // BATCH) % 20 == 0:
            done = min(i + BATCH, len(items))
            print("    %s %d/%d (%.1fs, %.0f img/s)"
                  % (desc, done, len(items), time.time() - t0, done / max(time.time() - t0, 1e-6)),
                  flush=True)
    return torch.cat(feats).numpy(), torch.cat(flips).numpy()


def main(names):
    meta, is_train, genus_split, genus2id = cf.load_meta()
    base_ids = set(genus_split["base_species_ids"])
    novel_ids = set(genus_split["novel_species_ids"])
    groups = {
        "cub_novel": cf.cub_items(meta, is_train, genus2id, novel_ids, want_train=False),
        "cub_base": cf.cub_items(meta, is_train, genus2id, base_ids, want_train=True),
        "endb": cf.endb_items(),
    }
    for gname, items in groups.items():
        print("group %-10s n=%d" % (gname, len(items)))

    for tag in names:
        arch = TIMM_MODELS[tag]
        print("=" * 80)
        print("MODEL %s (%s)" % (tag, arch), flush=True)
        model, cfg = build(arch)
        transform = timm.data.create_transform(**cfg, is_training=False)
        for gname, items in groups.items():
            out = os.path.join(CACHE, "%s__%s.npz" % (gname, tag))
            if os.path.exists(out):
                print("  skip existing %s" % os.path.basename(out))
                continue
            f, g = extract(model, items, transform, gname)
            np.savez_compressed(
                out, feats=f, feats_flip=g,
                species=np.array([s for _, s, _ in items], dtype=np.int64),
                genus=np.array([q for _, _, q in items], dtype=np.int64),
                path=np.array([os.path.relpath(p, cf.ROOT) for p, _, _ in items]))
            print("  saved %s shape=%s" % (os.path.basename(out), f.shape), flush=True)
        del model
        torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    names = sys.argv[1:] or ["clip_vitb16"]
    main(names)
