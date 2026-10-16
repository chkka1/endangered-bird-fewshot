# 面向濒危鸟类小样本识别的属分层基准与基类校准方法族

本仓库为论文《面向濒危鸟类小样本识别的属分层基准与基类校准方法族》的
可复现材料，包含代码、评测集清单、全部结果文件与稿件图源。

---

## 目录结构

```
.
├── code/          评测与训练代码（62 个 .py 脚本，按流水线分 6 阶段）
│   ├── 01_data/       数据构建：采集、过滤、去重、许可核验、划分生成
│   ├── 02_cache/      冻结特征缓存抽取
│   ├── 03_train/      主干微调（EfficientNetV2-S）
│   ├── 04_eval/       小样本评测、校准方法族与判据
│   ├── 05_verify/     数值一致性核验脚本
│   ├── 06_paper/      表与图的生成脚本
│   ├── _paths.py      共享路径模块
│   ├── README.md      代码说明与核验脚本清单
│   ├── requirements.txt
│   └── run_all.cmd    一键核验
├── dataset/       评测集（1617 张濒危鸟类图像 + 逐张来源清单；CUB/NABirds 仅划分与元数据）
│   ├── README.md          构建流程、规模、许可说明
│   ├── LICENSE            图像许可分条款说明
│   └── CHECKSUMS.sha256   逐文件校验和
├── results/       全部结果文件（122 个 CSV / JSON / TXT）
│   ├── final_main_table_calibrated.csv   主结果（含 P/R/F1）
│   ├── final_significance.csv            配对 t 检验
│   ├── error_breakdown.csv               同属/跨属错误分解
│   ├── criterion_match.csv               判据评分器一致性
│   ├── criterion_design_points.json      三个设计点消融原始记录
│   ├── spectrum_metrics.csv              三数据集 × 四主干的谱指标
│   ├── complexity.csv / calibration_cost.csv   计算开销实测
│   └── endb_stratified.csv / endb_waycurve.csv 濒危集分层与路数曲线
├── LICENSE        代码与结果文件的许可（MIT，含适用范围说明）
├── CITATION.cff   引用信息
└── .gitignore
```

---

## 核心结论速查

| 项 | 数值 |
|---|---|
| 属级可迁移率（按类切分 → 属分层） | 10%~31.8% → 100% |
| 该约束本身的价值（配对对照） | 5-shot 下降 7.0~12.0 个百分点 |
| 校准方法族 12 设定平均准确率 | 68.29%（较最优单一校准方法 +3.28） |
| 判据选中最优成员 | 10/12（11 个差距 ≤0.5 个百分点） |
| 固定系数白化的增益方向 | 领域微调 +7~10；CLIP 最高 −15.6 |
| 主干造成的跨属错误占比差 | 濒危集 47.9%（DINOv2）对 85.4%（领域微调） |

---

## 复现方式

### 环境

```bash
pip install -r code/requirements.txt
```

主要依赖：PyTorch 2.5.1+cu121、torchvision 0.20.1、timm 1.0.30、numpy 2.0.2、
open_clip、python-docx、matplotlib。评测在单张 RTX 4060 Laptop 上完成。

### 路径配置

脚本以环境变量 `BIRD_PROJECT_ROOT` 指定仓库根；未设置时自动查找含
`paper` / `results` / `dataset` 的目录。

```bash
# Linux / macOS
export BIRD_PROJECT_ROOT=/path/to/repo
# Windows
set BIRD_PROJECT_ROOT=D:\path\to\repo
```

### 一键核验

```bash
code/run_all.cmd          # Windows
```

依次运行 6 个核验脚本，**全部退出码 0 即表示稿件与结果一致**：

| 脚本 | 覆盖 | 项数 |
|---|---|---|
| `verify_prose_aggregates.py` | 正文聚合量 | 25 |
| `audit_endb_manifest.py` | 濒危集清单内部一致性（species_table vs image_sources） | 14 |
| `check_causal_drops.py` | 因果检验落差 | — |
| `audit_all_data.py` | 表 2/3/7/9 + 正文 + 图 7 + 图 4(b) | 153 |
| `audit_all_data2.py` | 表 1/4/5/6/8/10 + 图 5/6 | 73 |
| `audit_all_data3.py` | 表 8 双协议 | 24 |
| `audit_all_data4.py` | 表 10 分层 | 逐格 |

### 关于未随仓库发布的资产

以下资产体积大或受第三方许可限制，**未随仓库发布**：

| 资产 | 说明 |
|---|---|
| `_tools/feat_cache/` | 冻结特征缓存（数百 MB ~ GB） |
| `data/` | CUB / NABirds 原始图像 |
| `cub_parsed/` | 划分与解析中间件 |
| `checkpoints/` | 主干权重 |

依赖这些资产的脚本（如 `audit_cache_freshness.py`、`verify_all_methods.py`）
在缺该目录时会**打印说明并以退出码 0 跳过**——**这不代表论文数据有问题**。

因此本仓库的性质是**核验包**：可对照 `results/` 的结果文件与论文稿件
逐格核验论文全部数值；从第三方原图重训重评需自行补齐上表资产。

---

---

## 本仓库包含什么、不包含什么

**本仓库（GitHub）包含**：代码、评测集划分与清单、全部结果文件、许可与引用信息。

**本仓库不包含论文稿件。** 原因：稿件在投稿与修回期间会持续修改，
放进仓库会造成版本混乱；期刊要求的可复现材料是代码、数据与结果，
稿件由投稿系统单独提交。

**因此 `code/05_verify/` 下的核验脚本在克隆本仓库后无法直接运行** ——
它们需要一份稿件做对照。要运行，把稿件放入 `paper/` 目录，
或用环境变量 `PAPER_DOCX` 指定路径。详见 `code/README.md`。

---

## 数据许可

- **代码与结果文件**：**MIT**，见根目录 `LICENSE`。
- **濒危鸟类集图像**：各自遵循其原始 CC 许可（**逐张**见
  `dataset/endangered_birds/image_sources.csv`），整体条款见 `dataset/LICENSE`。
- **第三方数据集**：CUB-200-2011 与 NABirds 的**图像版权不属作者**，
  本仓库**不包含亦不授权**其图像，仅提供划分与元数据；使用者须从官方渠道获取。

濒危集图像**全部为 CC0 / CC-BY / CC-BY-NC 授权**，已逐张向 iNaturalist 与
GBIF 的公开接口核验，并剔除不可再分发者。

物种命名体系采用 **eBird 分类**（其 taxonomy API 无需密钥）；
受威胁等级参照 **IUCN 红色名录**。使用时请按各自条款引用。

---

## 引用

见 `CITATION.cff`。
