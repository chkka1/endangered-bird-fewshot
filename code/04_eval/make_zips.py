# -*- coding: utf-8 -*-
"""把材料打包为 ZIP。

两个包：
  ① 可复现材料包  ← D:\\soft\\bird_project\\release
  ② 投稿材料      ← D:\\Desktop\\投稿材料_计算机工程与应用

压缩策略（按已压缩与否分流，兼顾速度与体积）：
  · .jpg / .png / .docx / .zip 等**已压缩**格式 → ZIP_STORED（再压几乎无收益，反而慢）
  · 文本类（.py/.md/.csv/.json/.svg/.cff/.txt/.cmd/LICENSE/.sha256）→ ZIP_DEFLATED

排除：__pycache__、*.pyc、.git、临时 .bak
"""
import os
import sys
import time
import zipfile
from _paths import find_root  # noqa: E402
ROOT = find_root()

JOBS = [
    (os.path.join(ROOT, "release"),
     r"D:\Desktop\bird_project_release.zip",
     "可复现材料包"),
    (r"D:\Desktop\投稿材料_计算机工程与应用",
     r"D:\Desktop\submission_materials.zip",
     "投稿材料"),
]

STORED_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".docx", ".xlsx",
              ".pptx", ".zip", ".gz", ".7z", ".pdf", ".mp4", ".npz"}
SKIP_DIRS = {"__pycache__", ".git", ".idea", ".vscode"}
SKIP_EXT = {".pyc", ".pyo", ".bak", ".tmp"}


def classify(p):
    e = os.path.splitext(p)[1].lower()
    return zipfile.ZIP_STORED if e in STORED_EXT else zipfile.ZIP_DEFLATED


def pack(src, dst, label):
    if not os.path.isdir(src):
        print("  [跳过] 目录不存在：%s" % src)
        return None
    if os.path.exists(dst):
        os.remove(dst)

    files = []
    for root, dirs, fs in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in fs:
            if os.path.splitext(f)[1].lower() in SKIP_EXT:
                continue
            p = os.path.join(root, f)
            files.append((p, os.path.relpath(p, src).replace("\\", "/")))

    total_raw = sum(os.path.getsize(p) for p, _ in files)
    print("\n%s" % ("=" * 66))
    print("%s：%d 个文件 / %.1f MB" % (label, len(files), total_raw / 1024 / 1024))
    print("=" * 66)

    t0 = time.time()
    n = 0
    with zipfile.ZipFile(dst, "w", allowZip64=True) as z:
        for p, rel in files:
            # 顶层加一个目录名，解压后不会散落一地
            z.write(p, arcname=os.path.basename(src) + "/" + rel,
                    compress_type=classify(p))
            n += 1
            if n % 300 == 0:
                print("  已写入 %d/%d …" % (n, len(files)))

    sz = os.path.getsize(dst)
    print("  完成：%s" % dst)
    print("  %d 个文件 | 原始 %.1f MB → 压缩后 %.1f MB（%.0f%%）| 耗时 %.0f 秒"
          % (len(files), total_raw / 1024 / 1024, sz / 1024 / 1024,
             sz / total_raw * 100, time.time() - t0))
    return dst


if __name__ == "__main__":
    outs = []
    for src, dst, label in JOBS:
        r = pack(src, dst, label)
        if r:
            outs.append(r)
    print("\n" + "=" * 66)
    for o in outs:
        print("  %-46s %8.1f MB" % (os.path.basename(o),
                                    os.path.getsize(o) / 1024 / 1024))
