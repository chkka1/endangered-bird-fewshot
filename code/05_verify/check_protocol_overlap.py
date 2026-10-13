# -*- coding: utf-8 -*-
"""Genus overlap of the split protocols the paper compares against.

Facts established first:
  * cub_train_test_split.csv is byte-identical to the official CUB-200-2011
    train_test_split.txt: a per-IMAGE ~30/29 split covering ALL 200 classes.
  * images/base(150) and images/novel(50) are a custom class-level reorganisation,
    NOT part of the official CUB protocol.
So "CUB-200 标准划分" must be described precisely. This script measures how much
genus structure each candidate protocol already preserves.
"""
import csv
import json
import os
from collections import defaultdict

ROOT = (os.environ.get("BIRD_PROJECT_ROOT")
        or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARSED = os.path.join(ROOT, "cub_parsed")
with open(os.path.join(PARSED, "cub_metadata.json"), encoding="utf-8") as f:
    meta = json.load(f)
with open(os.path.join(PARSED, "dataset_split_by_genus.json"), encoding="utf-8") as f:
    gsplit = json.load(f)

sp2gen = {int(m["species_id"]): m["genus"] for m in meta}
all_cls = sorted(sp2gen)


def report(train_ids, test_ids, tag):
    tr_g = set(sp2gen[i] for i in train_ids)
    te_g = set(sp2gen[i] for i in test_ids)
    inter = tr_g & te_g
    gn = defaultdict(list)
    for i in test_ids:
        gn[sp2gen[i]].append(i)
    multi = {g: v for g, v in gn.items() if len(v) >= 2}
    # 每个测试类在训练集中是否有同属物种
    mate = sum(1 for i in test_ids if sp2gen[i] in tr_g)
    print("%-34s 训练%3d类 测试%3d类 | 测试属%2d 中可迁移%2d (%.0f%%) | 测试类有同属训练类 %3d/%3d (%.0f%%) | 测试侧多类属 %d" % (
        tag, len(train_ids), len(test_ids), len(te_g), len(inter),
        100.0 * len(inter) / len(te_g), mate, len(test_ids),
        100.0 * mate / len(test_ids), len(multi)))


print("=" * 130)
print("不同划分协议下的属级重叠")
print("=" * 130)
report(all_cls, all_cls, "官方 per-image 划分（同类两侧都有）")
report(all_cls[:100], all_cls[100:], "社区惯例 100/100 类切分 (1-100/101-200)")
report(all_cls[:150], all_cls[150:], "150/50 类切分 (1-150/151-200)")
report(gsplit["base_species_ids"], gsplit["novel_species_ids"], "本文属分层 145/55")

# 随机 100/50/50 的期望值（按类随机抽）
import random
random.seed(0)
rs = []
# 同时累计「测试类中有同属训练类的比例」（R_same）——表 1 该栏此前缺数据
rs_cls = []
for _ in range(200):
    pool = all_cls[:]
    random.shuffle(pool)
    tr, te = pool[:100], pool[100:150]
    tr_g = set(sp2gen[i] for i in tr)
    te_g = set(sp2gen[i] for i in te)
    rs.append(100.0 * len(tr_g & te_g) / len(te_g))
    rs_cls.append(100.0 * sum(1 for i in te if sp2gen[i] in tr_g) / len(te))
_rm = sum(rs) / len(rs)
_rsd = (sum((x - _rm) ** 2 for x in rs) / len(rs)) ** 0.5
_cm = sum(rs_cls) / len(rs_cls)
_csd = (sum((x - _cm) ** 2 for x in rs_cls) / len(rs_cls)) ** 0.5
print("%-34s 测试属可迁移率 = %.1f%% ± %.1f%%" % ("随机 100/50 类切分(200 次)", _rm, _rsd))
print("%-34s 测试类有同属训练类 = %.1f%% ± %.1f%%" % ("", _cm, _csd))

# 写入 split_stats.json，供表 1 与稿件动态引用
import json as _json
import os as _os
from _paths import find_root  # noqa: E402
_SS = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "final_out", "split_stats.json")
if _os.path.exists(_SS):
    _d = _json.load(open(_SS, encoding="utf-8"))
    _d["cub_random_cls_mean"] = round(_cm, 2)
    _d["cub_random_cls_std"] = round(_csd, 2)
    _json.dump(_d, open(_SS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n已写入 split_stats.json：cub_random_cls_mean=%.2f, cub_random_cls_std=%.2f" % (_cm, _csd))

print()
print("结论：")
print("  - 官方 per-image 划分下训练与测试共享全部 200 个类，属重叠 100%（平凡）。")
print("  - 按类切分并不保证测试侧是新属：社区惯例 100/100 切分的属级可迁移率仅 2%，")
print("    150/50 顺序切分为 13%，按类随机 100/50 的 200 次均值为 31.8%±5.7%。")
print("  - 本文协议把该比例显式设定为 100%，并另设 0% 的未见属对照以支持因果归因。")
