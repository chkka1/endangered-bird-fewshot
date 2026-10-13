# -*- coding: utf-8 -*-
"""
抽取 CUB Base 145 类的**测试**划分特征，用于与濒危集做可比的距离度量。

背景：现有的 cub_base 缓存是训练划分（want_train=True），正是微调所用的图像，
以它度量"类内距离"会因记忆化而偏小，与濒危集（全新图像）不可比。
本脚本按同一套绘制逻辑取 Base 类的测试图像，使两侧都是"模型未见过的图像"。
"""
from __future__ import annotations

import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
os.chdir(TOOLS)

import cache_features as cf                                    # noqa: E402
from _paths import find_root  # noqa: E402
ROOT = find_root()


def main():
    meta, is_train, genus_split, genus2id = cf.load_meta()
    base_ids = set(genus_split["base_species_ids"])
    items = cf.cub_items(meta, is_train, genus2id, base_ids, want_train=False)
    print("Base 类测试划分：%d 张 / %d 类" % (len(items), len({s for _, s, _ in items})))

    import sys as _s
    _bb = _s.argv[1] if len(_s.argv)>1 else "imagenet"
    out = os.path.join(cf.CACHE, "cub_basetest__%s.npz" % _bb)
    if os.path.exists(out):
        print("已存在，跳过：%s" % out)
        return
    model = cf.build_model(cf.MODELS[_bb])
    f, g = cf.extract(model, items, "cub_basetest")
    np.savez_compressed(
        out, feats=f, feats_flip=g,
        species=np.array([s for _, s, _ in items], dtype=np.int64),
        genus=np.array([q for _, _, q in items], dtype=np.int64),
        path=np.array([os.path.relpath(p, cf.ROOT) for p, _, _ in items]))
    print("saved %s  shape=%s" % (out, f.shape))


if __name__ == "__main__":
    main()
