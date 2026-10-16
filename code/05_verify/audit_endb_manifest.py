# -*- coding: utf-8 -*-
"""濒危集清单内部一致性审计（新增，补 run_all.cmd 的检查缺口）

背景：读者按 README 的「一键核验」运行时会检查 6 个脚本，但**没有一个**检查
      species_table.csv 与 image_sources.csv 的逐物种 n_images 是否一致。
      2026-10 曾发现两文件的 n_images 合计为 1674（旧值）而权威清单只有 1617 行，
      导致「读者一核就发现对不上」。本脚本即为补上该缺口。

检查项：
  A. species_table.csv 行数 = 70，且物种名与 image_sources.csv 完全一致
  B. 逐物种 n_images == image_sources.csv 中该物种的行数
  C. 逐物种 n_observers == image_sources.csv 中该物种的唯一 observer 数
  D. species_table.csv 的 n_images 合计 == image_sources.csv 行数（1617）
  E. endangered_birds_genus_statistics.csv 的 n_images 合计 == 1617，
     且逐属 n_images == species_table 中该属的合计
  F. 论文 4.1 节口径：170 物种 / 45 属 / 1617 张 / 每类 10~25 张（中位 24，均值 23.1）
  G. 45 个属的 n_in_base / membership 与 CUB 属分层划分一致（若 cub_parsed 可用）

退出码：0 = 全部一致；1 = 有不一致。
"""
import csv
import io
import os
import sys
from collections import Counter, defaultdict

import numpy as np


def find_root(start=None):
    p = os.path.abspath(start or __file__)
    for _ in range(8):
        p = os.path.dirname(p)
        if os.path.isdir(os.path.join(p, "dataset")) and \
           os.path.isdir(os.path.join(p, "results")):
            return p
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT = os.environ.get("ENDB_ROOT") or find_root()
D = os.path.join(ROOT, "dataset", "endangered_birds")
if not os.path.isdir(D):
    D = os.path.join(ROOT, "release", "dataset", "endangered_birds")

SP = os.path.join(D, "species_table.csv")
SRC = os.path.join(D, "image_sources.csv")
GEN = os.path.join(D, "endangered_birds_genus_statistics.csv")

bad = []
ok = []


def chk(tag, cond, detail=""):
    if cond:
        ok.append(tag)
    else:
        bad.append((tag, detail))


def load(p):
    if not os.path.exists(p):
        return None
    return list(csv.DictReader(io.open(p, encoding="utf-8-sig")))


def main():
    print("=" * 72)
    print("濒危集清单内部一致性审计")
    print("=" * 72)
    print("  目录: %s" % D)

    sp = load(SP)
    src = load(SRC)
    gen = load(GEN)

    if sp is None or src is None:
        print("  [跳过] 缺少 species_table.csv 或 image_sources.csv")
        return 0

    # A
    chk("A species_table 行数 = 70", len(sp) == 70, "实际 %d" % len(sp))
    S = {r["scientific_name"].strip() for r in sp}
    I = {r["scientific_name"].strip() for r in src}
    chk("A 物种名集合一致", S == I,
        "只在一侧: %s" % sorted(S ^ I)[:5])

    # B / C
    cnt = Counter()
    obs = defaultdict(set)
    for r in src:
        n = r["scientific_name"].strip()
        cnt[n] += 1
        if r.get("observer"):
            obs[n].add(r["observer"].strip())

    mismatch_n = [(r["scientific_name"].strip(), int(r["n_images"]), cnt.get(r["scientific_name"].strip(), 0))
                  for r in sp if int(r["n_images"]) != cnt.get(r["scientific_name"].strip(), 0)]
    chk("B 逐物种 n_images 与 image_sources 行数一致", not mismatch_n,
        "%d 个不符，例: %s" % (len(mismatch_n), mismatch_n[:3]))

    mismatch_o = [(r["scientific_name"].strip(), int(r.get("n_observers") or 0),
                   len(obs.get(r["scientific_name"].strip(), ())))
                  for r in sp if int(r.get("n_observers") or 0) != len(obs.get(r["scientific_name"].strip(), ()))]
    chk("C 逐物种 n_observers 与唯一 observer 数一致", not mismatch_o,
        "%d 个不符，例: %s" % (len(mismatch_o), mismatch_o[:3]))

    # D
    tot_sp = sum(int(r["n_images"]) for r in sp)
    chk("D n_images 合计 == image_sources 行数", tot_sp == len(src),
        "species_table 合计 %d vs image_sources %d" % (tot_sp, len(src)))

    # E
    if gen is not None:
        tot_gen = sum(int(r["n_images"]) for r in gen)
        chk("E genus_statistics 合计 == image_sources 行数", tot_gen == len(src),
            "genus 合计 %d vs %d" % (tot_gen, len(src)))
        by_g = Counter()
        for r in sp:
            by_g[r["genus"].strip()] += int(r["n_images"])
        bad_g = [(r["genus"], int(r["n_images"]), by_g.get(r["genus"].strip(), 0))
                 for r in gen if int(r["n_images"]) != by_g.get(r["genus"].strip(), 0)]
        chk("E 逐属 n_images 与 species_table 合计一致", not bad_g,
            "%d 个不符" % len(bad_g))

    # F
    v = np.array([int(r["n_images"]) for r in sp])
    chk("F 论文口径：总数 1617", int(v.sum()) == 1617, "实际 %d" % v.sum())
    chk("F 论文口径：每类 10~25 张", v.min() == 10 and v.max() == 25,
        "实际 min %d / max %d" % (v.min(), v.max()))
    chk("F 论文口径：中位 24", int(np.median(v)) == 24, "实际 %d" % int(np.median(v)))
    chk("F 论文口径：均值 23.1", abs(v.mean() - 23.1) < 0.05, "实际 %.2f" % v.mean())
    n_genera = len({r["genus"].strip() for r in sp})
    chk("F 论文口径：45 个属", n_genera == 45, "实际 %d" % n_genera)

    # G（可选）
    cub = None
    for cand in (os.path.join(ROOT, "cub_parsed", "dataset_split_by_genus.json"),
                 os.path.join(ROOT, "..", "cub_parsed", "dataset_split_by_genus.json")):
        if os.path.exists(cand):
            cub = cand
            break
    if cub and gen is not None:
        import json
        j = json.load(io.open(cub, encoding="utf-8"))
        meta = os.path.join(os.path.dirname(cub), "cub_metadata.json")
        c2g = {}
        if os.path.exists(meta):
            m = json.load(io.open(meta, encoding="utf-8"))
            for rec in (m if isinstance(m, list) else m.get("species", [])):
                if isinstance(rec, dict) and rec.get("genus"):
                    cid = rec.get("id") or rec.get("species_id") or rec.get("class_id")
                    if cid is not None:
                        c2g[int(cid)] = rec["genus"]
        if c2g:
            base_g = {c2g[c] for c in j["base_species_ids"] if c in c2g}
            bad_m = [(r["genus"], r["membership"]) for r in gen
                     if (r["genus"] in base_g) != (r["n_in_base"] == "1")]
            chk("G 属级归属与 CUB 划分一致", not bad_m, "%d 个不符" % len(bad_m))
            nbase = sum(1 for r in gen if r["n_in_base"] == "1")
            chk("G 30/45 属在 CUB Base 中", nbase == 30, "实际 %d" % nbase)
    else:
        print("  [跳过] G 项：cub_parsed 不可用（核验包未随包发布）")

    print()
    print("  通过 %d 项" % len(ok))
    for t in ok:
        print("    [OK] %s" % t)
    if bad:
        print()
        print("  不一致 %d 项：" % len(bad))
        for t, d in bad:
            print("    [!!] %s  -- %s" % (t, d))
    print()
    print("=" * 72)
    print("总判定：%s" % ("全部一致" if not bad else "存在不一致，请核对"))
    print("=" * 72)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
