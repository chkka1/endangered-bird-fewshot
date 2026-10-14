# 面向濒危鸟类小样本识别的属分层基准 —— 数据集

本包提供论文《面向濒危鸟类小样本识别的属分层基准与基类白化校准方法》所用的三个评测集
的**划分文件、元数据与（仅限自建集的）图像**。

---

## 一、包内有什么

```
dataset/
├── README.md                      本文件
├── LICENSE                        各数据集的许可条款（务必先读）
├── SPLIT_STATISTICS.md            各划分的属级结构统计（属可迁移率、同属对）
├── CHECKSUMS.sha256               全部文件的 SHA256
│
├── endangered_birds/              ★ 自建集：含图像，可自由再分发
│   ├── images/<NNN.Species_name>/  70 个类目录，1617 张图像
│   ├── image_sources.csv           逐张来源：许可/观察者/日期/地点/观察链接
│   ├── species_table.csv           70 个物种：学名/属/张数/独立观察者数
│   ├── endangered_birds_genus_statistics.csv  45 个属的规模与成员构成
│   └── species_zh.json             中文名对照（部分待补）
│
├── cub/                           CUB-200-2011：只含划分，图像请自行下载
│   ├── splits/dataset_split_by_genus.json        属分层划分（Base 145 / Novel 55）
│   ├── splits/dataset_split_unseen_genus.json    未见属对照划分（属重叠 0%）
│   ├── splits/cub_train_test_split.csv           与官方 train_test_split.txt 逐字节一致
│   ├── cub_metadata.json                         物种/属/学名/图像相对路径
│   ├── cub_species_table.csv                     200 个物种 × 两种划分的归属
│   └── cub_genus_statistics.csv                  140 个属的规模与 Base/Novel 成员
│
└── nabirds/                       NABirds-405：只含划分，图像请自行下载
    ├── splits/dataset_split_by_genus.json        属分层划分（Base 259 / Novel 146）
    ├── splits/dataset_split_unseen_genus.json    未见属对照划分（属重叠 0%）
    ├── splits/nabirds_train_test_split.csv       逐图 train/test 标记
    ├── nabirds_metadata.json                     与 CUB 同构的元数据
    ├── nabirds_species_table.csv                 405 个类 × 两种划分的归属 + 张数
    ├── nabirds_genus_statistics.csv              210 个属的规模与 Base/Novel 成员
    └── nabirds_taxonomy.csv                      类名 → 学名/属
```

---

## 二、许可条款（重要）

| 数据集 | 图像是否随包分发 | 条款 |
|---|---|---|
| **濒危鸟类（自建）** | ✅ **是** | 全部为 **CC0 / CC-BY / CC-BY-NC**；需保留署名与非商业使用限制 |
| **CUB-200-2011** | ❌ 否 | 版权属加州理工学院，请从[官方页面](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)获取 |
| **NABirds** | ❌ 否 | 版权属原发布方，请从[官方页面](https://dl.allaboutbirds.org/nabirds)获取 |

**为什么只分发自建集的图像**：CUB 与 NABirds 的图像版权不属于本文作者，重新分发会违反其
原始条款。本包只提供**由我们构建的划分与元数据**，这些是我们自己的贡献，可自由使用。

**濒危集的许可构成**（1617 张）：

| 许可 | 张数 | 要求 |
|---|---|---|
| CC-BY-NC | 1437 | 署名 + **非商业使用** |
| CC-BY | 181 | 署名 |
| CC0 | 59 | 无限制 |

使用时须**保留 `image_sources.csv` 中的署名信息**（观察者、观察链接与许可），并遵守
非商业使用限制。逐张来源记录已随包提供，便于使用者自行核验。

> 打包时执行了自动许可校验：1678 条来源记录中，剔除 3 条重复记录与 **1 张照片级许可为空
> （即未授权共享）的图像**，最终 1617 张全部为 CC 授权。

---

## 三、三个数据集的划分

| 数据集 | Base（训练特征提取器） | Novel（小样本评测） | Novel 测试图像 | 属级可迁移率 |
|---|---|---|---|---|
| CUB-200 属分层版 | 145 类 / 140 属 | 55 类 / 39 属 | 1,627 | **100%** |
| NABirds-405 属分层版 | 259 类 / 210 属 | 146 类 / 89 属 | 8,741 | **100%** |
| 濒危鸟类 | —（无 Base 划分） | 70 类 / 45 属 | 1,673 | — |

**属分层划分规则**（CUB 与 NABirds 使用同一套规则）：

- **R1**　单物种属：该物种全部划入 Base
- **R2**　多物种属：属内 3 个及以上物种时，至少 2 个划入 Novel，其余划入 Base
- **R3**　属级重叠约束：**Novel 中每个属必须在 Base 中存在同属物种**

R3 是本基准的核心。常规划分下该比例极低（按类顺序切分 150/50 为 13%，随机 100/50 为
31.8%±5.7%），而本文协议将其提升至 100%。

**未见属对照划分**（`dataset_split_unseen_genus.json`）：在 CUB 上取 16 个双物种属与
23 个单物种属（55 类 / 39 属）作为 Novel，其余 101 个属的 145 个类作为 Base，使
**Base 与 Novel 的属集合完全不相交**。其 Novel 的类数、属数与属内物种数分布与属分层版
**完全配对**，两者唯一的差别就是属是否可迁移——用于量化该约束本身的价值。

---

## 四、濒危鸟类评测集的构建规则

数据集的**筛选规则完全可复现**，不含主观挑选：

1. **物种筛选**：从 IUCN 受威胁名录中选取其**属**在 CUB 属分层 Base（140 属）中出现过的
   物种。这样做的目的是让评测物种与训练侧存在属级关联，与 R3 约束一致。
2. **命名体系**：以 **eBird/Clements** 分类为准（通过 eBird Taxonomy API 获取，无需密钥），
   统一学名与科属信息。
3. **图像来源**：**iNaturalist** 上质量等级为 `research`、且照片许可为
   CC0 / CC-BY / CC-BY-NC 的观察记录。
4. **每类张数**：目标 25 张；若该属在数据集中只有 1 个物种则取 20 张（避免单物种属因样本
   过多而主导统计）。
5. **去重**：按来源照片 URL 去重，确保同一观察不会同时出现在支撑集与查询集中。
6. **可追溯**：逐张记录许可、观察者、拍摄日期、地点与观察链接，并执行许可校验。

最终得到 **70 个物种 / 45 个属 / 1617 张图像**，其中 **24 个属含 2 个及以上物种（共 26 组同属对）**，
可支撑"同属近缘混淆"的统计分析。IUCN 等级分布：VU 288、EN 291、CR 121 张。

---

## 五、如何使用

### 5.1 濒危集

```python
import csv, os
root = "dataset/endangered_birds"
images = os.path.join(root, "images")
classes = sorted(os.listdir(images))            # "001.Aquila_chrysaetos", ...
species = [c.split(".", 1)[1] for c in classes] # "Aquila_chrysaetos"
# 属与 IUCN 等级见 species_table.csv；逐张来源见 image_sources.csv
```

### 5.2 CUB / NABirds

下载原数据集后，按 `*_metadata.json` 中的 `image_path` 字段定位图像；划分由
`splits/dataset_split_by_genus.json` 给出：

```python
import json
sp = json.load(open("dataset/cub/splits/dataset_split_by_genus.json"))
base_ids, novel_ids = sp["base_species_ids"], sp["novel_species_ids"]
```

NABirds 的 `image_path` 形如 `train/<class_name>/<file>.jpg`，相对于解压后的 `images/` 根目录。

### 5.3 评测协议要求

复现本文数字时须遵守：

- **每个 episode**：从 Novel 类的**测试图像**中采样 k 个支撑样本与其余作为查询
- **种子**：调参块 `[0, 10, …, 240]`（25 轮）与评测块 `[1000, 1010, …, 1490]`（50 轮）
  **必须分离**，不得在评测块上选择超参数
- **白化矩阵**：只能用 Base 类特征拟合，绝不可使用 Novel 特征
- **指标**：报告准确率的同时应给出跨属/同属错误分解（同属混淆是本任务的核心难点）

---

## 六、已知限制

1. **NABirds 版本**：本包所用为 **405 个物种**，而官方 NABirds v1 为 555 个物种。
   引用时**必须写作 "NABirds-405"** 并注明来源，不得自称标准 NABirds。
2. **濒危集规模**：70 类 / 1617 张，相对 CUB 与 NABirds 规模较小；但其价值在于
   同属对结构（24 组）与完全可追溯的 CC 授权来源。
3. **类间张数不均**：多数类 25 张，少数（如珍稀种）因 iNaturalist 可用观察有限而更少。
   具体见 `species_table.csv` 的 `n_images` 字段。
4. **单次观察可能贡献多张图**：少数类别中同一观察者的多次观察被采入，`species_table.csv`
   给出了每类的独立观察者数，可用于评估拍摄批次偏差。
5. **中文名不完整**：`species_zh.json` 中部分物种中文名为空。

---

## 七、引用

若使用本数据集或其划分，请引用本文（待补充正式出版信息）：

> 面向濒危鸟类小样本识别的属分层基准与基类白化校准方法

并同时引用原始数据集：

- CUB-200-2011：Wah C, Branson S, Welinder P, et al. The Caltech-UCSD Birds-200-2011
  Dataset. CNS-TR-2011-001, California Institute of Technology, 2011.
- NABirds：Van Horn G, Branson S, Farrell R, et al. Building a Bird Recognition App and
  Large Scale Dataset with Citizen Scientists. CVPR, 2015: 595-604.
