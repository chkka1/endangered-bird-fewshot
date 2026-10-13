# -*- coding: utf-8 -*-
"""
Renumber endangered-bird image files to a contiguous 001..N sequence and update
image_sources.csv to match.

Why: quarantine_quality moved duplicates/person images out, leaving index gaps
(one folder ran 004..027 with only 25 files).  The re-collection script then used
the file COUNT as the next index, so new files collided with existing ones
(15 folders ended up with two files sharing an index, e.g. _024.jpeg and _024.jpg).

Renaming is done in two passes through a temporary prefix so that no rename ever
overwrites a file that has not been moved yet.
"""
import csv
import io
import os
import re
from _paths import find_root  # noqa: E402

ROOT = find_root()
ENDB = os.path.join(ROOT, "data", "endangered_birds")
IM = os.path.join(ENDB, "images")
MAN = os.path.join(ENDB, "image_sources.csv")
EXT = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    rows = list(csv.DictReader(io.open(MAN, encoding="utf-8-sig")))
    by_key = {(r["folder"], r["file"]): r for r in rows}
    changed = 0
    for d in sorted(x for x in os.listdir(IM) if os.path.isdir(os.path.join(IM, x))):
        fp = os.path.join(IM, d)
        files = [f for f in os.listdir(fp) if f.lower().endswith(EXT)]
        if not files:
            continue

        def key(f):
            m = re.search(r"_(\d+)\.", f) if False else re.search(r"_(\d+)\.", f)
            return (int(m.group(1)) if m else 10 ** 9, f)

        files.sort(key=key)
        sci = d.split(".", 1)[1]
        plan = []
        for i, f in enumerate(files, 1):
            ext = os.path.splitext(f)[1]
            # .jpeg -> .jpg is a pure rename (identical format); done so a class
            # does not mix extensions.  .png is kept as-is to avoid re-encoding.
            if ext.lower() == ".jpeg":
                ext = ".jpg"
            new = "%s_%03d%s" % (sci, i, ext)
            if new != f:
                plan.append((f, new))
        if not plan:
            continue
        # pass 1: move everything that changes to a unique temp name
        tmp = []
        for i, (old, new) in enumerate(plan):
            t = "__tmp%04d__" % i
            os.rename(os.path.join(fp, old), os.path.join(fp, t))
            tmp.append((t, new, old))
        # pass 2: temp -> final
        for t, new, old in tmp:
            os.rename(os.path.join(fp, t), os.path.join(fp, new))
            r = by_key.get((d, old))
            if r is not None:
                r["file"] = new
            changed += 1
        print("  %-34s %d 张重编号" % (d, len(plan)))

    print("\n共重编号 %d 个文件" % changed)
    fields = list(rows[0].keys())
    with io.open(MAN, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("清单已同步：%s（%d 条）" % (MAN, len(rows)))

    # verify
    bad = 0
    for d in sorted(x for x in os.listdir(IM) if os.path.isdir(os.path.join(IM, x))):
        files = [f for f in os.listdir(os.path.join(IM, d)) if f.lower().endswith(EXT)]
        idx = sorted(int(re.search(r"_(\d+)\.", f).group(1)) for f in files
                     if re.search(r"_(\d+)\.", f))
        if idx != list(range(1, len(files) + 1)):
            print("  !! %s 仍异常: %s" % (d, idx[:6]))
            bad += 1
    print("验证：%s" % ("全部类编号连续 001..N" if bad == 0 else "%d 个类仍异常" % bad))


if __name__ == "__main__":
    main()
