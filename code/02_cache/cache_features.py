# ---- 核验包守卫（自包含）：缺少随包未发布的外部资产时显式跳过 ----
def _pkg_root():
    import os as _o
    d = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))
    e = _o.environ.get("BIRD_PROJECT_ROOT")
    return _o.path.abspath(e) if (e and _o.path.isdir(e)) else d


def _require(rel, what):
    import os as _o
    import sys as _s
    full = _o.path.join(_pkg_root(), rel)
    if _o.path.exists(full):
        return True
    print("=" * 68)
    print("[跳过] 本脚本需要" + what + "，但包内未包含：")
    print("        " + full)
    print("  本包为核验包：随包提供 code / dataset(濒危集) / "
          "_tools/final_out(结果) / paper_code(稿件与图)。")
    print("  特征缓存、第三方原图与模型权重未随包发布，故本脚本无法在包内执行")
    print("  —— 这不代表论文数据有问题。")
    print("=" * 68)
    _s.exit(0)


_require("_tools/feat_cache", "冻结特征缓存（feat_cache）")

# -*- coding: utf-8 -*-
"""
One-time feature cache for the bird few-shot evaluation.

Why: in the original scripts the frozen backbone is re-run for every method,
every episode and every hyper-parameter combination.  The backbone output does
not depend on any of those, so we extract each image exactly once (plus its
horizontal flip, needed by TTA) and cache the L2-normalised 1280-d vectors.
Every later ablation then runs as cheap CPU linear algebra.

Outputs (in _tools/feat_cache/):
  cub_novel__<model>.npz   feats, feats_flip, species, genus, image_path
  cub_base__<model>.npz    feats, feats_flip, species, genus
  endb__<model>.npz        feats, feats_flip, species, genus, image_path
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.utils.parametrizations as param
from PIL import Image

ROOT = (os.environ.get("BIRD_PROJECT_ROOT")
        or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = FEAT_CACHE
os.makedirs(CACHE, exist_ok=True)

CUB_ROOT = os.path.join(ROOT, "data", "CUB_200_2011")
PARSED = os.path.join(ROOT, "cub_parsed")
ENDB_ROOT = os.path.join(ROOT, "data", "endangered_birds")
CKPT = os.path.join(ROOT, "checkpoints")

IMG_SIZE = 300
BATCH = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    # Obsolete exploratory backbones. Their weights were moved to
    # _checkpoints_stale/ because the numbers computed from them were voided by the
    # image-level leakage (see README section 1). They are kept here only so that an
    # existing cache can still be read; re-extracting requires restoring the weights.
    # "genus_center": "cub_cosine_multitask_genus_center.pth",
    # "genus_split": "cub_cosine_multitask_genus_split.pth",
    # "clean" 已于审计后移除：其权重属被图像级泄漏作废的一批（见 README 节 1），
    # 缓存已在 feat_cache_quarantine/。若需复现旧结果，先恢复权重再取消注释。
    # "clean": "cub_pretrained_clean_baseline.pth",
    # retrained on the corrected 145-class Base tree, ImageNet-1k init (no CUB lineage)
    "clean_species": "cub_base_species_only.pth",
    "clean_multitask": "cub_base_multitask_clean.pth",
    # trained on the UNSEEN-GENUS control Base (145 classes / 101 genera)
    "unseen_species": "cub_unseen_species.pth",
    "unseen_multitask": "cub_unseen_multitask.pth",
    # leakage-free reference: pure ImageNet weights, never trained on CUB.
    # This is also the initialisation the paper's backbone started from.
    "imagenet": None,
}

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- endangered species -> genus map, copied verbatim from
#     Endangered/endangered_birds_dataset.py so ids stay identical ---
sys.path.insert(0, os.path.join(ROOT, "Endangered"))
from endangered_birds_dataset import SPECIES_GENUS_MAP  # noqa: E402
from _paths import find_root, FEAT_CACHE  # noqa: E402


# --------------------------------------------------------------------------
def load_meta():
    with open(os.path.join(PARSED, "cub_metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(PARSED, "cub_train_test_split.csv"), encoding="utf-8-sig") as f:
        split_rows = list(csv.DictReader(f))
    is_train = {r["image_id"].strip(): r["is_train"].strip() for r in split_rows}
    with open(os.path.join(PARSED, "dataset_split_by_genus.json"), encoding="utf-8") as f:
        genus_split = json.load(f)

    genera = sorted({m["genus"] for m in meta})
    genus2id = {g: i for i, g in enumerate(genera)}
    return meta, is_train, genus_split, genus2id


def cub_items(meta, is_train, genus2id, species_filter, want_train):
    # resolve by actual directory content, not by species_id range: the
    # base/novel trees now follow the genus-stratified split exactly.
    base_folders = set(os.listdir(os.path.join(CUB_ROOT, "images", "base")))
    out = []
    for m in meta:
        sid = int(m["species_id"])
        if sid not in species_filter:
            continue
        split = is_train.get(str(m["image_id"]).strip())
        if (split == "1") != want_train:
            continue
        cls_folder = m["image_path"].split("/")[0]
        sub = "base" if cls_folder in base_folders else "novel"
        path = os.path.join(CUB_ROOT, "images", sub, m["image_path"])
        out.append((path, sid, genus2id[m["genus"]]))
    return out


def endb_items():
    species_list = sorted(SPECIES_GENUS_MAP.keys())
    sp2id = {n: i for i, n in enumerate(species_list)}
    # SPECIES_GENUS_MAP[name] = (species_id, genus_name, chinese_name)
    # 属编号必须由属名映射得到——此前误用 [0]（物种 id），
    # 使缓存的 genus 列实际为 70 个物种编号而非 45 个属。
    genera = sorted({v[1] for v in SPECIES_GENUS_MAP.values()})
    gen2id = {g: i for i, g in enumerate(genera)}
    img_root = os.path.join(ENDB_ROOT, "images")
    out = []
    for folder in sorted(os.listdir(img_root)):
        folder_path = os.path.join(img_root, folder)
        if not os.path.isdir(folder_path):
            continue
        name = folder.split(".", 1)[1] if "." in folder else folder
        if name not in SPECIES_GENUS_MAP:
            print("[warn] unmapped folder, skipped: %s" % folder)
            continue
        gid = gen2id[SPECIES_GENUS_MAP[name][1]]
        sid = sp2id[name]
        for fn in sorted(os.listdir(folder_path)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                out.append((os.path.join(folder_path, fn), sid, gid))
    return out


def build_model(weight_name):
    """Same architecture as the evaluation scripts; only backbone weights matter."""
    if weight_name is None:
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        print("  ImageNet-only backbone (reference, never trained on CUB)")
        return m.to(DEVICE).eval()
    m = models.efficientnet_v2_s(weights=None)
    m.classifier[1] = param.weight_norm(nn.Linear(1280, 145, bias=False))
    m.genus_classifier = nn.Sequential(
        nn.Dropout(p=0.2), param.weight_norm(nn.Linear(1280, 140, bias=False)))
    path = os.path.join(CKPT, weight_name)
    sd = torch.load(path, map_location=DEVICE, weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    backbone_missing = [k for k in missing if k.startswith(("features", "avgpool"))]
    print("  load %s | missing_in_backbone=%d | unexpected=%d"
          % (weight_name, len(backbone_missing), len(unexpected)))
    if backbone_missing:
        print("  !! backbone weights missing: %s" % backbone_missing[:5])
    m = m.to(DEVICE).eval()
    return m


@torch.no_grad()
def extract(model, items, desc):
    feats, flips = [], []
    t0 = time.time()
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        imgs = torch.stack([TRANSFORM(Image.open(p).convert("RGB")) for p, _, _ in chunk]).to(DEVICE)
        f = model.avgpool(model.features(imgs)).flatten(1)
        f = F.normalize(f, dim=1)
        g = model.avgpool(model.features(torch.flip(imgs, dims=[3]))).flatten(1)
        g = F.normalize(g, dim=1)
        feats.append(f.cpu())
        flips.append(g.cpu())
        if (i // BATCH) % 10 == 0:
            print("    %s %d/%d  (%.1fs)" % (desc, min(i + BATCH, len(items)), len(items),
                                             time.time() - t0), flush=True)
    return torch.cat(feats).numpy(), torch.cat(flips).numpy()


def main():
    meta, is_train, genus_split, genus2id = load_meta()
    base_ids = set(genus_split["base_species_ids"])
    novel_ids = set(genus_split["novel_species_ids"])
    groups = {
        "cub_novel": cub_items(meta, is_train, genus2id, novel_ids, want_train=False),
        "cub_base": cub_items(meta, is_train, genus2id, base_ids, want_train=True),
        "endb": endb_items(),
    }
    for gname, items in groups.items():
        print("group %-10s n=%d" % (gname, len(items)))

    for mname, weight in MODELS.items():
        if len(sys.argv) > 1 and mname not in sys.argv[1:]:
            continue
        print("=" * 80)
        print("MODEL %s (%s)" % (mname, weight), flush=True)
        model = build_model(weight)
        for gname, items in groups.items():
            out = os.path.join(CACHE, "%s__%s.npz" % (gname, mname))
            if os.path.exists(out):
                print("  skip existing %s" % os.path.basename(out))
                continue
            print("  extracting %s ..." % gname, flush=True)
            f, g = extract(model, items, gname)
            np.savez_compressed(
                out, feats=f, feats_flip=g,
                species=np.array([s for _, s, _ in items], dtype=np.int64),
                genus=np.array([q for _, _, q in items], dtype=np.int64),
                path=np.array([os.path.relpath(p, ROOT) for p, _, _ in items]))
            print("  saved %s  shape=%s" % (os.path.basename(out), f.shape), flush=True)
        del model
        torch.cuda.empty_cache()
    print("DONE. cache dir = %s" % CACHE)


if __name__ == "__main__":
    main()
