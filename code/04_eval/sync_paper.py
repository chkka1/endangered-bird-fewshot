import re
# -*- coding: utf-8 -*-
"""一键同步：把 paper_code/ 下的最新稿件同步到 release/paper/，并做完整收尾。

执行内容（按序）：
  1) 找 paper_code/ 下版本号最大的 v*.docx
  2) 清空 release/paper/ 下所有 .docx（只应留一个）
  3) 以**论文名**复制过去
  4) 校验 SHA256 一致
  5) 清理 __pycache__ / *.pyc
  6) 重生成 dataset/CHECKSUMS.sha256
  7) 跑仓库自检
  8) 重新打包 ZIP 到桌面

用法：
    python sync_paper.py            # 全流程
    python sync_paper.py --no-zip   # 跳过打包（快）
"""
import hashlib
import io
import os
import shutil
import subprocess
import sys
import zipfile
from _paths import find_root  # noqa: E402

ROOT = find_root()
SRC_DIR = os.path.join(ROOT, "paper_code")
REL = os.path.join(ROOT, "release")
PAPER_DIR = os.path.join(REL, "paper")
TITLE = "面向濒危鸟类小样本识别的属分层基准与基类校准方法族"
DOCX_NAME = TITLE + ".docx"
ZIP_DST = r"D:\Desktop\bird_project_release.zip"
PY = sys.executable

W = 68


def hr(t=""):
    print("\n" + "=" * W)
    if t:
        print(t)
        print("=" * W)


def sha(p):
    return hashlib.sha256(io.open(p, "rb").read()).hexdigest()


def main():
    do_zip = "--no-zip" not in sys.argv
    hr("一键同步：稿件 → 发布仓库")

    # ---- 1) 找最新稿件 ----
    import re
    cands = [x for x in os.listdir(SRC_DIR)
             if re.match(r"^v\d+\.docx$", x) and not x.startswith("~$")]
    if not cands:
        print("  ✗ paper_code/ 下没有 v*.docx")
        return 1
    latest = max(cands, key=lambda x: int(re.search(r"v(\d+)", x).group(1)))
    src = os.path.join(SRC_DIR, latest)
    print("  源稿件   : %s（%.2f MB）" % (latest, os.path.getsize(src) / 1048576))

    # ---- 2) 清空旧 docx ----
    removed = []
    for f in os.listdir(PAPER_DIR):
        if f.lower().endswith(".docx") and not f.startswith("~$"):
            os.remove(os.path.join(PAPER_DIR, f))
            removed.append(f)
    if removed:
        print("  已删除   : %s" % ", ".join(removed))

    # ---- 3) 以论文名复制 ----
    dst = os.path.join(PAPER_DIR, DOCX_NAME)
    shutil.copy2(src, dst)
    print("  已复制为 : %s" % DOCX_NAME)

    # ---- 4) 校验 ----
    h1, h2 = sha(src), sha(dst)
    ok = (h1 == h2)
    print("  SHA256   : %s  %s" % (h1[:16], "✓ 一致" if ok else "✗ 不一致"))
    if not ok:
        return 1

    # ---- 5) 清缓存 ----
    n_pc = 0
    for root, dirs, _ in os.walk(REL):
        for d in list(dirs):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d))
                dirs.remove(d)
                n_pc += 1
    for root, _, fs in os.walk(REL):
        for f in fs:
            if f.endswith(".pyc"):
                os.remove(os.path.join(root, f))
    print("  清理缓存 : 删除 %d 个 __pycache__" % n_pc)

    # ---- 6) 重生成 CHECKSUMS ----
    DS = os.path.join(REL, "dataset")
    lines = []
    for root, dirs, fs in os.walk(DS):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in sorted(fs):
            if f == "CHECKSUMS.sha256":
                continue
            p = os.path.join(root, f)
            lines.append("%s  %s" % (sha(p),
                                     os.path.relpath(p, DS).replace("\\", "/")))
    lines.sort()
    io.open(os.path.join(DS, "CHECKSUMS.sha256"), "w", encoding="utf-8",
            newline="\n").write("\n".join(lines) + "\n")
    print("  校验和   : %d 条" % len(lines))

    # ---- 7) 自检 ----
    hr("仓库自检")
    r = subprocess.run([PY, os.path.join(ROOT, "_tools", "release_selfcheck.py")],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    tail = [l for l in (r.stdout or "").split("\n") if l.strip()][-3:]
    for l in tail:
        print("  " + l.strip())
    self_ok = (r.returncode == 0)

    # ---- 8) 打包 ----
    if do_zip:
        hr("打包 ZIP")
        SKIP = {"__pycache__", ".git"}
        ST = {".jpg", ".png", ".docx", ".svg"}
        for root, dirs, _ in os.walk(REL):
            for d in list(dirs):
                if d == "__pycache__":
                    shutil.rmtree(os.path.join(root, d))
                    dirs.remove(d)
        if os.path.exists(ZIP_DST):
            os.remove(ZIP_DST)
        files = []
        for root, dirs, fs in os.walk(REL):
            dirs[:] = [d for d in dirs if d not in SKIP]
            for f in fs:
                if f.endswith((".pyc", ".bak")):
                    continue
                p = os.path.join(root, f)
                files.append((p, os.path.relpath(p, REL).replace("\\", "/")))
        with zipfile.ZipFile(ZIP_DST, "w", allowZip64=True) as z:
            for p, rel in files:
                ext = os.path.splitext(p)[1].lower()
                z.write(p, arcname="release/" + rel,
                        compress_type=zipfile.ZIP_STORED if ext in ST
                        else zipfile.ZIP_DEFLATED)
        print("  %s" % ZIP_DST)
        print("  %d 个文件 | %.1f MB" % (len(files),
                                        os.path.getsize(ZIP_DST) / 1048576))

    hr("完成")
    print("  稿件     : %s" % DOCX_NAME)
    print("  自检     : %s" % ("✓ 通过" if self_ok else "✗ 见上方输出"))
    if do_zip:
        print("  ZIP      : %s" % ZIP_DST)
    return 0 if self_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
