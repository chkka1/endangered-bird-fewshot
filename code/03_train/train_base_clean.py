# -*- coding: utf-8 -*-
"""
Clean retraining on the (now physically correct) 145-class genus-stratified Base.

Why: the existing checkpoints descend from `cub_pretrained_clean_baseline.pth`,
which has a 150-way species head and was produced while 35 evaluation classes
still lived inside images/base.  The image tree has since been reorganised, but
those checkpoints cannot be un-contaminated — they must be retrained.

Two stages, mirroring the project's own pipeline:
  stage species    : ImageNet-1k init -> 145 Base classes, species loss only
                     ==> this is the TRUE CONTROL backbone (no genus supervision)
  stage multitask  : from the species checkpoint -> + genus cls + genus centre
                     ==> this is the CLEAN MAIN model

Both use the same augmentations, cosine classifier and schedule as
paper_code/train_cosine_multitask_genus_split.py, so results stay comparable.

Usage:
    python train_base_clean.py --stage species   --epochs 60 --lr 2e-4
    python train_base_clean.py --stage multitask --epochs 30 --lr 2e-5
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.utils.parametrizations as param
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from _paths import find_root  # noqa: E402

ROOT = (os.environ.get("BIRD_PROJECT_ROOT")
        or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUB = os.path.join(ROOT, "data", "CUB_200_2011")
IMAGES = os.path.join(CUB, "images")
PARSED = os.path.join(ROOT, "cub_parsed")
CKPT = os.path.join(ROOT, "checkpoints")
LOGDIR = os.path.join(ROOT, "_tools", "train_out")
os.makedirs(LOGDIR, exist_ok=True)

IMG_SIZE = 300
FEAT_DIM = 1280
TEMP = 5.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve(image_path):
    """Layout-agnostic image lookup (base tree first, then novel)."""
    p = os.path.join(IMAGES, "base", image_path)
    return p if os.path.exists(p) else os.path.join(IMAGES, "novel", image_path)


class CubBase(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, species_id, genus_id, label = self.items[i]
        img = Image.open(resolve(rel)).convert("RGB")
        return {"image": self.transform(img),
                "species_id": species_id,
                "genus_label": genus_id,
                "label": label}


def build_items(split_name, genus2id, base_ids, want_train):
    meta = json.load(open(os.path.join(PARSED, "cub_metadata.json"), encoding="utf-8"))
    import csv
    with open(os.path.join(PARSED, "cub_train_test_split.csv"), encoding="utf-8-sig") as f:
        is_train = {r["image_id"].strip(): r["is_train"].strip() for r in csv.DictReader(f)}
    label_of = {sid: i for i, sid in enumerate(sorted(base_ids))}
    out = []
    for m in meta:
        sid = int(m["species_id"])
        if sid not in base_ids:
            continue
        if (is_train.get(str(m["image_id"]).strip()) == "1") != want_train:
            continue
        out.append((m["image_path"], sid, genus2id[m["genus"]], label_of[sid]))
    return out


def build_model(num_species, num_genus):
    m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    m.classifier[1] = param.weight_norm(nn.Linear(FEAT_DIM, num_species, bias=False))
    nn.init.kaiming_normal_(m.classifier[1].weight)
    m.genus_classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        param.weight_norm(nn.Linear(FEAT_DIM, num_genus, bias=False)))
    nn.init.kaiming_normal_(m.genus_classifier[1].weight)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["species", "multitask"], required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--init", default=None,
                    help="checkpoint to initialise the backbone from "
                         "(default: ImageNet-1k for stage species, "
                         "cub_base_species_only.pth for stage multitask)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", default="dataset_split_by_genus.json",
                    help="split json under cub_parsed/ that defines the Base set")
    ap.add_argument("--limit-batches", type=int, default=0,
                    help="debug: stop each epoch after N batches")
    args = ap.parse_args()

    epochs = args.epochs if args.epochs is not None else (60 if args.stage == "species" else 30)
    lr = args.lr if args.lr is not None else (2e-4 if args.stage == "species" else 2e-5)
    out = args.out or os.path.join(
        CKPT, "cub_base_species_only.pth" if args.stage == "species"
        else "cub_base_multitask_clean.pth")

    split = json.load(open(os.path.join(PARSED, args.split), encoding="utf-8"))
    base_ids = set(split["base_species_ids"])
    meta = json.load(open(os.path.join(PARSED, "cub_metadata.json"), encoding="utf-8"))
    genera = sorted({m["genus"] for m in meta})
    genus2id = {g: i for i, g in enumerate(genera)}
    num_genus = len(genera)

    tr_items = build_items("train", genus2id, base_ids, True)
    va_items = build_items("test", genus2id, base_ids, False)
    num_species = len(base_ids)
    print("stage=%s  epochs=%d  lr=%g  batch=%d" % (args.stage, epochs, lr, args.batch))
    print("Base 类数=%d  训练图=%d  验证图=%d  属数=%d"
          % (num_species, len(tr_items), len(va_items), num_genus))
    print("输出: %s" % out)

    tr_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    va_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    tr_loader = DataLoader(CubBase(tr_items, tr_tf), batch_size=args.batch,
                           shuffle=True, num_workers=0, drop_last=True)
    va_loader = DataLoader(CubBase(va_items, va_tf), batch_size=args.batch,
                           shuffle=False, num_workers=0)

    model = build_model(num_species, num_genus)

    init_path = args.init
    if init_path is None and args.stage == "multitask":
        init_path = os.path.join(CKPT, "cub_base_species_only.pth")
    if init_path and os.path.exists(init_path):
        sd = torch.load(init_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("初始化自 %s | missing=%d unexpected=%d"
              % (os.path.basename(init_path), len(missing), len(unexpected)))
    elif args.stage == "species":
        print("初始化自 torchvision ImageNet-1k 权重（无任何 CUB 血统）")
    else:
        print("!! 未找到 %s，将使用 ImageNet-1k 初始化" % init_path)

    model = model.to(DEVICE)

    lambda_genus = 0.25 if args.stage == "multitask" else 0.0
    lambda_center = 0.008 if args.stage == "multitask" else 0.0

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    patience = 8
    best_acc, no_improve = 0.0, 0
    logpath = os.path.join(LOGDIR, "train_%s.log" % args.stage)
    lf = open(logpath, "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        lf.write(msg + "\n")
        lf.flush()

    log("=" * 90)
    log("START %s | %s | epochs=%d lr=%g" % (args.stage, time.ctime(), epochs, lr))

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        tot_loss, correct, total = 0.0, 0, 0
        nb = 0
        for bi, batch in enumerate(tqdm(tr_loader, desc="ep%d" % (epoch + 1), leave=False)):
            if args.limit_batches and bi >= args.limit_batches:
                break
            imgs = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            glabels = batch["genus_label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                feats = model.avgpool(model.features(imgs)).flatten(1)
                fn = F.normalize(feats, dim=1)
                slog = model.classifier[1](model.classifier[0](fn)) * TEMP
                loss = crit(slog, labels)
                if lambda_genus > 0:
                    glog = model.genus_classifier(fn) * TEMP
                    loss = loss + lambda_genus * crit(glog, glabels)
                if lambda_center > 0:
                    lc = torch.tensor(0.0, device=DEVICE)
                    ng = 0
                    for g in torch.unique(glabels):
                        mask = glabels == g
                        if mask.sum() > 1:
                            center = fn[mask].mean(dim=0)
                            lc = lc + (1 - fn[mask] @ center).mean()
                            ng += 1
                    if ng:
                        loss = loss + lambda_center * (lc / ng)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item()
            nb += 1
            correct += (slog.argmax(1) == labels).sum().item()
            total += labels.size(0)
        sched.step()

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for batch in va_loader:
                imgs = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    fn = F.normalize(model.avgpool(model.features(imgs)).flatten(1), dim=1)
                    slog = model.classifier[1](model.classifier[0](fn)) * TEMP
                vc += (slog.argmax(1) == labels).sum().item()
                vt += labels.size(0)
        acc = 100.0 * vc / max(vt, 1)
        log("epoch %2d/%d | loss %.4f | train %.2f%% | val %.2f%% | %.0fs"
            % (epoch + 1, epochs, tot_loss / max(nb, 1), 100.0 * correct / max(total, 1),
               acc, time.time() - t0))
        if acc > best_acc:
            best_acc, no_improve = acc, 0
            torch.save(model.state_dict(), out)
            log("  saved best -> %s (%.2f%%)" % (out, best_acc))
        else:
            no_improve += 1
            if no_improve >= patience:
                log("  early stop at epoch %d" % (epoch + 1))
                break

    log("DONE %s | best val = %.2f%% | %s" % (args.stage, best_acc, out))
    lf.close()


if __name__ == "__main__":
    main()
