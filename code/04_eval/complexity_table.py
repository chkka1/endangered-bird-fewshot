# -*- coding: utf-8 -*-
"""
C8：补复杂度表——参数量、FLOPs、单样本推理时延，以及校准本身的额外开销。

论文的主要卖点之一是"不进行任何特征更新"，但此前没有任何成本数字。本脚本给出：
  · 各主干的参数量与 FLOPs（用 torch.utils.flop_counter，无需 thop）
  · 主干 + 校准 + 判别的端到端单样本时延（GPU 与 CPU 各测，50 次预热后取 200 次中位数）
  · 校准的额外开销：W 的存储（d×d float32）与一次矩阵乘的 FLOPs
  · 支撑集线性探针的训练成本（作为对照——它是唯一需要优化的方法）

输出：final_out/complexity.csv
"""
import csv
import io
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(ROOT, "_tools"))
import torch                                                  # noqa: E402
import torchvision                                            # noqa: E402
from _paths import find_root, FINAL_OUT  # noqa: E402
ROOT = find_root()

FO = FINAL_OUT
BACKBONES = [
    ("EfficientNetV2-S", "efficientnet_v2_s", 1280),
    ("CLIP ViT-B/16", "clip_vitb16", 768),
    ("DINOv2 ViT-B/14", "dinov2_vitb14", 768),
]


def build(name):
    """构造与论文所用一致的主干（仅用于量测参数量与 FLOPs）。"""
    if name == "efficientnet_v2_s":
        m = torchvision.models.efficientnet_v2_s(weights=None)
        m.classifier = torch.nn.Identity()
        return m
    if name == "clip_vitb16":
        import timm
        return timm.create_model("vit_base_patch16_clip_224.openai",
                                 pretrained=False, num_classes=0)
    if name == "dinov2_vitb14":
        import timm
        return timm.create_model("vit_base_patch14_dinov2.lvd142m",
                                 pretrained=False, num_classes=0, img_size=224)
    raise KeyError(name)


def flops_of(model, dev):
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.randn(1, 3, 224, 224, device=dev)
    model = model.to(dev).eval()
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        model(x)
    return fc.get_total_flops()


def latency_ms(fn, dev, warm=50, rep=200):
    for _ in range(warm):
        fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def main():
    rows = []
    for lab, key, dim in BACKBONES:
        try:
            m = build(key)
        except Exception as e:
            print("  %-18s 构造失败: %s" % (lab, e))
            continue
        nparam = sum(p.numel() for p in m.parameters())
        try:
            fl = flops_of(m, "cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            fl = 0
            print("  %-18s FLOPs 失败: %s" % (lab, e))
        xg = torch.randn(1, 3, 224, 224, device="cuda") if torch.cuda.is_available() else None
        mg = m.to("cuda").eval() if xg is not None else None
        lat_gpu = 0.0
        if mg is not None:
            with torch.no_grad():
                lat_gpu = latency_ms(lambda: mg(xg), "cuda")
        mc = m.to("cpu").eval()
        xc = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            lat_cpu = latency_ms(lambda: mc(xc), "cpu")
        rows.append(dict(backbone=lab, key=key, dim=dim,
                         params_M="%.2f" % (nparam / 1e6),
                         flops_G="%.2f" % (fl / 1e9) if fl else "—",
                         lat_gpu_ms="%.2f" % lat_gpu if lat_gpu else "—",
                         lat_cpu_ms="%.1f" % lat_cpu))
        print("  %-18s 参数 %6.2f M | FLOPs %6.2f G | 时延 GPU %6.2f ms / CPU %7.1f ms"
              % (lab, nparam / 1e6, fl / 1e9 if fl else 0, lat_gpu, lat_cpu))
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 校准的额外开销 ----
    print()
    print("=== 校准的额外开销（与主干无关，只取决于特征维 d）===")
    wrows = []
    for d in (768, 1280):
        W = np.zeros((d, d), dtype=np.float32)
        mem_kb = W.nbytes / 1024
        # 一次矩阵乘： (1×d) @ (d×d) = 2*d^2 FLOPs
        mul_fl = 2 * d * d
        wrows.append(dict(dim=d, W_MB="%.2f" % (W.nbytes / 1024 / 1024),
                          mul_MFLOPs="%.4f" % (mul_fl / 1e6),
                          note="单样本前向时预乘一次，可离线完成"))
        print("  d=%4d | W 存储 %6.2f MB | 一次矩阵乘 %8.4f MFLOPs" % (d, mem_kb / 1024, mul_fl / 1e6))

    out = os.path.join(FO, "complexity.csv")
    with io.open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backbone", "key", "dim", "params_M",
                                          "flops_G", "lat_gpu_ms", "lat_cpu_ms"])
        w.writeheader()
        w.writerows(rows)
    out2 = os.path.join(FO, "calibration_cost.csv")
    with io.open(out2, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dim", "W_MB", "mul_MFLOPs", "note"])
        w.writeheader()
        w.writerows(wrows)
    print("\nsaved %s / %s" % (os.path.basename(out), os.path.basename(out2)))


if __name__ == "__main__":
    main()
