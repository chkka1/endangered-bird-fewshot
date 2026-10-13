# -*- coding: utf-8 -*-
"""生成投稿《计算机工程与应用》的四份材料（占位符版）。

  ① 投稿信（cover letter）
  ② 作者贡献声明
  ③ 利益冲突声明
  ④ 数据可用性声明

占位符统一写作【待填：…】，便于投稿前一次性替换。
内容取自稿件 v192 的实际数据与结论，不虚构任何数字。
"""
import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt

OUT = r"D:\Desktop\投稿材料_计算机工程与应用"
os.makedirs(OUT, exist_ok=True)

CN = "宋体"
EN = "Times New Roman"


def new_doc():
    d = Document()
    s = d.sections[0]
    # 版心 16.6 × 22.9 cm（与稿件一致）
    s.page_width, s.page_height = Pt(595), Pt(842)
    s.left_margin = s.right_margin = Pt(70)
    s.top_margin = s.bottom_margin = Pt(72)
    st = d.styles["Normal"]
    st.font.name = EN
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), CN)
    return d


def P(d, text, size=10.5, bold=False, align="left", indent=False, after=6):
    p = d.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.25
    if indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "just":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    elif align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = bold
    r.font.name = EN
    r._element.rPr.rFonts.set(qn("w:eastAsia"), CN)
    return p


def H(d, text, size=12):
    return P(d, text, size=size, bold=True, after=8)


TITLE = "【待填：论文中文标题】"
AUTHORS = "【待填：全部作者姓名，按署名顺序】"
AFFIL = "【待填：单位全称，城市 邮编】"
CORR = "【待填：通信作者姓名】"
EMAIL = "【待填：通信作者邮箱】"
PHONE = "【待填：联系电话】"
FUND = "【待填：基金项目名称与编号；若无则填“无”】"

# ============================================================ ① 投稿信
d = new_doc()
P(d, "投  稿  信", size=15, bold=True, align="center", after=14)

P(d, "尊敬的《计算机工程与应用》编辑部：", after=8)
P(d, "", after=2)

P(d, "　　您好！现向贵刊投寄研究论文《" + TITLE + "》，恳请审阅。",
  align="just", after=8)

H(d, "一、论文的增量")
P(d, "　　本文面向濒危鸟类监测中小样本识别的实际需求，主要贡献有三：",
  align="just", after=6)
for s in [
    "（1）提出属分层划分协议。现有小样本评测多按物种随机切分基类与新颖类，"
    "同一属的物种常被分到两侧，导致评测高估模型对未见属的判别力。本文改为"
    "**按属分层**，并量化该约束本身的影响：在读的三个评测集上，5-shot 准确率"
    "较按类切分下降 7.0~12.0 个百分点（均值 9.03）。",
    "（2）提出基类校准方法族与类别不相交判据。方法族在 12 个设定上的平均"
    "准确率为 68.29%，较最优单一校准方法高 3.28 个百分点；判据在其中 10 个"
    "设定上选中最优成员，11 个设定的差距不超过 0.5 个百分点，最大损失 0.73。",
    "（3）发现并解释了一个反直觉现象：固定系数白化的增益方向随主干而反转 —— "
    "在领域微调主干上为正（+7~10 个百分点），在 CLIP 上最高为 −15.6。本文用"
    "特征协方差谱的陡峭程度统一解释了这一分歧，并据此给出方法族的适用边界。",
]:
    P(d, "　　" + s.replace("**", ""), align="just", after=5)

H(d, "二、与同类工作的区别")
P(d, "　　本文不与某一具体少样本方法竞争，而是提供**评测协议与校准方法族**"
     "两层贡献，并给出该协议下方法选择的判据。为避免评价口径混淆，全文的"
     "所有数值均可在随稿提供的可复现材料中逐格核验：核验脚本对照结果文件与"
     "稿件正文，共覆盖 153 + 73 + 24 + 25 项数值及表 2/表 4/表 8 的全部 108 "
     "个标准差，全部通过。", align="just", after=8)

H(d, "三、可复现材料")
P(d, "　　随稿提供可复现材料包，含：评测与训练代码（Python，按流水线分 6 阶段）、"
     "三个评测集的划分清单、自建濒危鸟类集的逐张图像来源与许可清单"
     "（1617 张，全部为 CC 授权）、全部结果文件，以及一键核验脚本。",
     align="just", after=6)
P(d, "　　材料获取方式：【待填：仓库地址 / 或“随稿以附件形式提供，亦可向通信"
     "作者索取”】", align="just", after=8)

H(d, "四、声明")
for s in [
    "1. 本文未曾公开发表，亦未一稿多投；全体作者均已阅读并同意投稿。",
    "2. 所有作者对本研究均有实质性贡献，署名顺序无争议。",
    "3. 不存在利益冲突。",
    "4. 研究涉及的图像数据均来自公开渠道，许可状况已逐张核验（见材料包）。",
]:
    P(d, "　　" + s, align="just", after=4)

P(d, "", after=6)
P(d, "　　恳请贵刊安排审稿。如需补充材料，请随时与我联系。", align="just", after=10)

for x in ("此致", "敬礼"):
    P(d, "　　" + x, after=6)
P(d, "", after=8)
P(d, "通信作者：" + CORR, align="right", after=4)
P(d, "单位：" + AFFIL, align="right", after=4)
P(d, "邮箱：" + EMAIL, align="right", after=4)
P(d, "电话：" + PHONE, align="right", after=4)
P(d, "日期：【待填：年 月 日】", align="right", after=4)

d.save(os.path.join(OUT, "01_投稿信.docx"))

# ============================================================ ② 作者贡献声明
d = new_doc()
P(d, "作者贡献声明", size=15, bold=True, align="center", after=12)
P(d, "论文题目：" + TITLE, after=10)
P(d, "全体作者依据国际医学期刊编辑委员会（ICMJE）及本刊要求，对本研究的"
     "贡献声明如下：", align="just", after=10)

P(d, "【待填：逐位作者填写，格式如下。请按署名顺序列出，并删去不适用项】",
  bold=True, align="just", after=8)
for i, role in enumerate([
    "研究构思与设计、结果解释、论文撰写与修改",
    "数据采集与整理、许可核验、代码实现",
    "实验实施、结果分析、图表制作",
    "研究指导、经费支持、论文审阅",
], start=1):
    P(d, "作者 %d（姓名：%s）：%s。" % (i, "【待填】", role), align="just", after=5)

P(d, "【待填：增删行数以匹配实际作者人数】", bold=True, align="just", after=10)

H(d, "确认事项")
for s in [
    "1. 全体作者均已阅读并同意本文的最终版本投稿。",
    "2. 全体作者同意对研究内容的真实性负责。",
    "3. 署名顺序经全体作者协商一致，无争议。",
    "4. 不存在代写、代投情形。",
]:
    P(d, "　　" + s, align="just", after=4)

P(d, "", after=10)
P(d, "全体作者签名：", after=6)
P(d, "【待填：签名或电子签名】", after=6)
P(d, "日期：【待填：年 月 日】", after=6)
d.save(os.path.join(OUT, "02_作者贡献声明.docx"))

# ============================================================ ③ 利益冲突声明
d = new_doc()
P(d, "利益冲突声明", size=15, bold=True, align="center", after=12)
P(d, "论文题目：" + TITLE, after=10)

P(d, "　　本文全体作者郑重声明：", align="just", after=8)
for s in [
    "1. 本研究不存在任何可能影响研究客观性的经济利益冲突，包括但不限于："
    "与本研究相关的商业资助、股权、专利、顾问费或酬金。",
    "2. 不存在任何非经济利益冲突，包括但不限于：学术竞争关系、个人关系或"
    "机构利益。",
    "3. 本研究未接受任何可能影响结果解释的资助。基金资助情况为：" + FUND,
    "4. 作者与所使用数据集（CUB-200-2011、NABirds 及自建濒危鸟类集）的"
    "提供方之间不存在利益关联。",
    "5. 若在审稿或出版过程中出现新的利益冲突，作者将及时向编辑部披露。",
]:
    P(d, "　　" + s, align="just", after=6)

P(d, "", after=10)
P(d, "全体作者签名：", after=6)
P(d, "【待填：签名或电子签名】", after=6)
P(d, "日期：【待填：年 月 日】", after=6)
d.save(os.path.join(OUT, "03_利益冲突声明.docx"))

# ============================================================ ④ 数据可用性声明
d = new_doc()
P(d, "数据与代码可用性声明", size=15, bold=True, align="center", after=12)
P(d, "论文题目：" + TITLE, after=10)

H(d, "一、自建数据集（濒危鸟类集）")
P(d, "　　本文自建的濒危鸟类数据集包含 70 个物种、45 个属、共 1617 张图像，"
     "全部来自 iNaturalist 与 GBIF 的公开观测记录。每张图像的来源页面、"
     "观察者、许可类型与核验状态已逐张记录（1617 行），随材料包提供。"
     "许可构成为：CC-BY-NC 1377 张、CC-BY 181 张、CC0 59 张，"
     "均已剔除不可再分发的图像。", align="just", after=8)

H(d, "二、第三方数据集")
P(d, "　　CUB-200-2011 与 NABirds 的图像版权不属于本文作者，本文不包含"
     "亦不授权其图像，仅提供划分与元数据。使用者须从官方渠道获取原图"
     "并遵守各自的许可条款。", align="just", after=8)

H(d, "三、代码")
P(d, "　　全部实验代码以 Python 实现，按流水线分为 6 个阶段（数据构建、"
     "特征缓存、主干微调、小样本评测、数值核验、图表生成），随材料包提供。",
  align="just", after=6)
P(d, "　　运行环境：PyTorch 2.5.1+cu121、torchvision 0.20.1、timm 1.0.30、"
     "numpy 2.0.2、open_clip、python-docx，评测在单张 NVIDIA RTX 4060 "
     "Laptop GPU 上完成。", align="just", after=8)

H(d, "四、结果文件与核验")
P(d, "　　全部结果文件（主结果表、显著性检验、错误分解、判据一致性、"
     "谱指标、计算开销、濒危集分层等）随材料包提供。材料包内含核验脚本，"
     "可对照结果文件与稿件正文逐格复核论文全部数值。", align="just", after=8)

H(d, "五、获取方式")
P(d, "　　【待填：仓库地址】", align="just", after=4)
P(d, "　　或：随稿以附件形式提供，亦可向通信作者（" + EMAIL + "）索取。",
  align="just", after=8)

P(d, "", after=8)
P(d, "通信作者：" + CORR, align="right", after=4)
P(d, "日期：【待填：年 月 日】", align="right", after=4)
d.save(os.path.join(OUT, "04_数据可用性声明.docx"))

print("已生成 4 份材料 →", OUT)
for f in sorted(os.listdir(OUT)):
    print("  %-26s %6.1f KB" % (f, os.path.getsize(os.path.join(OUT, f)) / 1024))
