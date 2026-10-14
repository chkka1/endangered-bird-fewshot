# 训练划分来源记录（PROVENANCE）

**目的**：回应「训练日志未记录 `--split` 取值」这一取证缺口，使「未见属对照主干是在
`dataset_split_unseen_genus.json` 的 Base 上训练」这一结论**可被独立追溯**。

---

## 1. 脚本已加固

`_tools/train_base_clean.py` 已加入 `PROVENANCE` 日志行（补丁：`_tools/_archive_patches/add_split_provenance.py`）：

```
PROVENANCE split_file=<划分文件名> sha256=<该 JSON 的 SHA-256>
PROVENANCE base_classes=<Base 类数> base_genera=<Base 属数> novel_classes=<Novel 类数>
PROVENANCE checkpoint=<输出路径> init=<初始化来源>
PROVENANCE argv=<完整命令行>
```

→ **此后任何一次训练，其 Base 集合都可从日志直证。**

## 2. 已有 checkpoint 的来源记录

训练已完成，日志无法追溯补记。以下为**从多重独立证据交叉推断**的结果：

| checkpoint | 推断的 `--split` | 依据 |
|---|---|---|
| `checkpoints\cub_unseen_species.pth` | `dataset_split_unseen_genus.json` | ① checkpoint 命名含 `unseen`；② `cache_unseen_features.py` L6-L8 docstring 直书 `cubunseen_base … (145 classes / 101 genera)`；③ `final_eval.py` L148-L150 注释 `Base 145 classes / 101 genera, … 0% genus overlap`；④ `train_out\train_species.log` L34/L115 记录 `epochs=60 lr=0.0002` 与输出路径 |
| `checkpoints\cub_unseen_multitask.pth` | 同上 | ① 命名含 `unseen`；② `train_out\train_multitask.log` L36/L38 记录 `epochs=30 lr=2e-05` 与输出路径；③ `final_eval.py` 的 `("cubunseen", "cubunseen_novel", "cubunseen_base", [1, 5])` 映射 |

## 3. 划分文件自身的可核验字段

| 文件 | `base_genus_count` | `novel_genus_count` | `genus_overlap` | Base 类数 | Novel 类数 |
|---|---|---|---|---|---|
| `cub_parsed\dataset_split_by_genus.json`（属分层） | 140（全库属数） | 39 | 39（100%） | 145 | 55 |
| `cub_parsed\dataset_split_unseen_genus.json`（未见属对照） | **101** | **39** | **0** | **145** | **55** |

**闭合性**：101 + 39 = 140 属 ✓；145 + 55 = 200 类 ✓

**权威汇总**：`release\results\split_stats.txt` L6-L7、L13、L17；`release\dataset\SPLIT_STATISTICS.md` L5-L6、L15-L16、L22-L23

## 4. 未能直证的部分（如实声明）

- **`train_out\train_species.log` 与 `train_multitask.log` 中无 `--split` 字样**
  （该脚本此前不打印该参数；现已修补）
- **`--split` 的默认值即为 `dataset_split_by_genus.json`**（属分层版）；
  `cub_unseen_*` 系列 checkpoint 的命名与缓存/评测脚本的映射共同指向未见属对照，
  但**原始命令行未留档**
- 若需完全直证，可用修补后的脚本重跑一次（`--split dataset_split_unseen_genus.json`），
  或在投稿信的数据可用性声明中说明该推断链

---

**生成时间**：2026-10-14
**相关补丁**：`_tools/_archive_patches/add_split_provenance.py`
**相关取证报告**：`release\paper\_content_logic_audit.md`（4.5 节属集合调查）
