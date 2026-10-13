# -*- coding: utf-8 -*-
"""
Train a feature extractor on the NABirds genus-stratified Base.

Mirrors `train_base_clean.py` (CUB) so the two causal tests use the same recipe:
ImageNet-1k init -> species-only cosine head -> +genus classification + genus centre.

This model is then evaluated on BOTH NABirds Novel sets:
    Novel-stratified : 146 classes whose genera DO occur in Base   (genus transferred)
    Novel-unseen     : 146 classes whose genera do NOT occur in Base
Since neither Novel set participated in training, the only difference is whether the
genus was seen — a within-model control, cleaner than the CUB cross-model one.

Usage
    python train_nabirds_base.py --stage species   --epochs 60 --lr 2e-4
    python train_nabirds_base.py --stage multitask --epochs 30 --lr 2e-5 --init <ckpt>
"""
import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.utils.parametrizations as param
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from _paths import find_root  # noqa: E402

ROOT = (os.environ.get("BIRD_PROJECT_ROOT")
        or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NB = os.path.join(ROOT, "data", "NABirds")
CKPT = os.path.join(ROOT, "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_TF = T.Compose([
    T.RandomResizedCrop(300, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
VAL_TF = T.Compose([
    T.Resize(340), T.CenterCrop(300), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class NabirdsDS(Dataset):
    def __init__(self, items, tf, c2i, g2i):
        self.items, self.tf, self.c2i, self.g2i = items, tf, c2i, g2i

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        # A single truncated JPEG in 15 k images must not kill a 20 h run: walk
        # forward to the next readable sample instead of raising in the worker.
        for off in range(8):
            rel, sid, gen = self.items[(i + off) % len(self.items)]
            try:
                img = Image.open(os.path.join(NB, "images", rel)).convert("RGB")
                return self.tf(img), self.c2i[sid], self.g2i[gen]
            except Exception:
                continue
        # fall back to a blank image so the batch still forms
        return self.tf(Image.new("RGB", (300, 300))), self.c2i[sid], self.g2i[gen]


def load_items(split_name="by_genus"):
    meta = json.load(open(os.path.join(NB, "nabirds_metadata.json"), encoding="utf-8"))
    # --split selects the genus-stratified split (100% genus transferability) or the
    # unseen-genus control (0%), which is what the causal test in 5.6 compares
    split = json.load(open(os.path.join(NB, "dataset_split_%s.json" % split_name),
                           encoding="utf-8"))
    base_ids = set(split["base_species_ids"])
    genera = sorted({m["genus"] for m in meta})
    g2i = {g: i for i, g in enumerate(genera)}
    train, val = [], []
    for m in meta:
        if m["species_id"] not in base_ids:
            continue
        it = (m["image_path"], m["species_id"], m["genus"])
        (train if m["source_split"] == "train" else val).append(it)
    classes = sorted({s for _, s, _ in train})
    c2i = {c: i for i, c in enumerate(classes)}
    return train, val, c2i, g2i


def build(num_species, num_genus, init):
    m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    if init:
        sd = torch.load(init, map_location="cpu", weights_only=True)
        m.load_state_dict(sd, strict=False)
        print("  初始化自 %s" % os.path.basename(init))
    m.classifier[1] = param.weight_norm(nn.Linear(1280, num_species, bias=False))
    m.genus_classifier = nn.Sequential(
        nn.Dropout(p=0.2), param.weight_norm(nn.Linear(1280, num_genus, bias=False)))
    return m.to(DEVICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["species", "multitask"], required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=32)  # 48 + other jobs -> OOM
    ap.add_argument("--init", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)   # 8 exhausted the Windows commit charge
    ap.add_argument("--resume", default=None)
    ap.add_argument("--val-every", type=int, default=5,
                    help="run validation every N epochs (val set is as large as train)")
    ap.add_argument("--start-epoch", type=int, default=1)
    ap.add_argument("--split", default="by_genus",
                    choices=["by_genus", "unseen_genus"],
                    help="by_genus = stratified (100%% genus transfer); "
                         "unseen_genus = control (0%%)")
    args = ap.parse_args()

    epochs = args.epochs if args.epochs is not None else (60 if args.stage == "species" else 30)
    lr = args.lr if args.lr is not None else (2e-4 if args.stage == "species" else 2e-5)
    out = args.out or os.path.join(
        CKPT, "nabirds_base_%s.pth" % ("species_only" if args.stage == "species" else "multitask"))

    train, val, c2i, g2i = load_items(getattr(args, "split", "by_genus"))
    print("stage=%s  Base 类=%d  训练图=%d  验证图=%d  属=%d"
          % (args.stage, len(c2i), len(train), len(val), len(g2i)))
    print("输出: %s" % out)

    model = build(len(c2i), len(g2i), args.init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr = DataLoader(NabirdsDS(train, TRAIN_TF, c2i, g2i), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0, pin_memory=True)
    va = DataLoader(NabirdsDS(val, VAL_TF, c2i, g2i), batch_size=args.batch,
                    shuffle=False, num_workers=args.workers,
                    persistent_workers=args.workers > 0)

    best = 0.0
    vacc = None
    _resumed_ep = None
    if getattr(args, "resume", None) and os.path.exists(args.resume):
        _ck = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        if isinstance(_ck, dict) and "model" in _ck:
            model.load_state_dict(_ck["model"])
            try:
                opt.load_state_dict(_ck["optim"])
                sched.load_state_dict(_ck["sched"])
            except Exception as e:
                print("  (优化器/调度器状态未能恢复: %s)" % str(e)[:60], flush=True)
            _resumed_ep = _ck.get("epoch")
            best = float(_ck.get("best", 0.0))
            print("resumed from %s | 已完成 %s 轮 | best %.2f%%"
                  % (args.resume, _resumed_ep, best), flush=True)
        else:
            # backward compatible with the old bare state_dict
            model.load_state_dict(_ck)
            print("resumed weights only from %s（旧格式，优化器状态不可恢复）"
                  % args.resume, flush=True)
    # if the checkpoint knows its epoch, continue from there unless overridden
    _start = args.start_epoch
    if _resumed_ep and args.start_epoch <= 1:
        _start = _resumed_ep + 1
        print("自动从第 %d 轮继续（检查点记录了轮次）" % _start, flush=True)
    for ep in range(_start, epochs + 1):
        model.train()
        t0, tot, corr = time.time(), 0, 0
        for x, y, g in tr:
            x, y, g = x.to(DEVICE), y.to(DEVICE), g.to(DEVICE)
            f = model.avgpool(model.features(x)).flatten(1)
            logits = model.classifier[1](f)
            loss = F.cross_entropy(logits * 20.0, y)
            if args.stage == "multitask":
                gl = model.genus_classifier(f)
                loss = loss + 0.3 * F.cross_entropy(gl * 20.0, g)
                # genus centre: pull each class mean toward its genus mean
                cn = F.normalize(f, dim=1)
                gl2 = torch.zeros(len(g2i), 1280, device=DEVICE).index_add_(
                    0, g, cn)
                cnt = torch.zeros(len(g2i), device=DEVICE).index_add_(
                    0, g, torch.ones_like(g, dtype=torch.float))
                gm = F.normalize(gl2 / cnt.clamp(min=1)[:, None], dim=1)
                loss = loss + 0.1 * (1 - (cn * gm[g]).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += y.numel(); corr += (logits.argmax(1) == y).sum().item()
        sched.step()

        # validate every args.val_every epochs (and on the final one): the val set is
        # as large as the train set, so validating each epoch doubled the cost
        do_val = (ep % args.val_every == 0) or (ep == epochs) or (vacc is None)
        if do_val:
            model.eval(); vc = vt = 0
            with torch.no_grad():
                for x, y, g in va:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    f = model.avgpool(model.features(x)).flatten(1)
                    vc += (model.classifier[1](f).argmax(1) == y).sum().item()
                    vt += y.numel()
            vacc = 100.0 * vc / max(vt, 1)
        else:
            model.train()
        flag = ""
        # always keep a resumable copy: the previous run lost 3 h because only a
        # new-best checkpoint was written and the process was killed by an OOM
        lastp = out.replace(".pth", "_last.pth")
        # full state so a resume continues the SAME optimisation trajectory
        torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best,
                    "stage": args.stage}, lastp)
        if vacc > best:
            best = vacc
            torch.save(model.state_dict(), out)
            flag = "  saved"
        print("epoch %2d/%d | loss %.4f | train %.2f%% | val %s | %.0fs%s"
              % (ep, epochs, loss.item(), 100.0 * corr / max(tot, 1),
                 ("%.2f%%" % vacc) if do_val else "skipped",
                 time.time() - t0, flag), flush=True)
    print("DONE %s | best val = %.2f%% | %s" % (args.stage, best, out))


if __name__ == "__main__":
    main()
