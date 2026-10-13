# code/ —— 评测与训练代码

按流水线阶段分层，共 **67 个脚本**（含 `_paths.py`）。

| 阶段 | 脚本数 | 职责 |
|---|---:|---|
| `01_data/` | 10 | 数据构建：采集、过滤、去重、许可核验、划分生成 |
| `02_cache/` | 5 | 冻结特征缓存抽取 |
| `03_train/` | 3 | 主干微调（EfficientNetV2-S） |
| `04_eval/` | 21 | 小样本评测、校准方法族与判据 |
| `05_verify/` | 20 | 数值一致性核验脚本 |
| `06_paper/` | 7 | 表与图的生成脚本 |

`_paths.py` 为共享路径模块：解析仓库根与各数据目录，兼容发布形态
（`results/`、`paper/`）与开发形态（`_tools/final_out/`、`paper_code/`）。

## 环境

```bash
pip install -r requirements.txt
```

## 路径配置

脚本以环境变量 `BIRD_PROJECT_ROOT` 指定仓库根；未设置时自动向上查找
含 `paper` / `results` / `dataset` 的目录。

```bash
# Linux / macOS
export BIRD_PROJECT_ROOT=/path/to/repo
# Windows
set BIRD_PROJECT_ROOT=D:\path\to\repo
```

## 一键核验

```bash
run_all.cmd        # Windows
```

依次运行 6 个核验脚本，**全部退出码 0 即表示稿件与结果一致**：

| 脚本 | 覆盖 | 项数 |
|---|---|---|
| `verify_prose_aggregates.py` | 正文聚合量 | 25 |
| `check_causal_drops.py` | 因果检验落差 | — |
| `audit_all_data.py` | 表 2/3/7/9 + 正文 + 图 7 + 图 4(b) | 153 |
| `audit_all_data2.py` | 表 1/4/5/6/8/10 + 图 5/6 | 73 |
| `audit_all_data3.py` | 表 8 双协议 | 24 |
| `audit_all_data4.py` | 表 10 分层 | 逐格 |

## 说明

- 依赖 `_tools/feat_cache/` 的脚本在缺该目录时**打印说明并以退出码 0 跳过**，
  这不代表论文数据有问题。
- `05_verify/consistency_check.py` 为历史实现（表结构变更后其表 3 区块会误报），
  未被 `run_all.cmd` 调用；其作用已由 `audit_all_data*.py` 覆盖，
  详见 `05_verify/README_consistency_check.md`。

## 关于稿件

**本仓库不包含论文稿件。** 原因：稿件在投稿与修回期间会持续修改，
放进仓库会造成版本混乱；期刊要求的可复现材料是代码、数据与结果。

因此 `05_verify/` 下的核验脚本**在克隆本仓库后无法直接运行** ——
它们需要一份稿件做对照。要运行，二选一：

```bash
# 方式一：把稿件放到 paper/ 目录
mkdir -p paper && cp <稿件>.docx paper/

# 方式二：用环境变量指定稿件路径
export PAPER_DOCX=/path/to/稿件.docx      # Linux / macOS
set PAPER_DOCX=D:\path\to\稿件.docx     # Windows
```

稿件可从论文投稿系统或通信作者处获取。

**不需要稿件也能跑的部分**：`01_data/` ~ `04_eval/` 的流水线脚本，
以及 `results/` 下的结果文件本身（可直接打开查看）。
