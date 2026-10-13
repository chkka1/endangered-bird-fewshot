# -*- coding: utf-8 -*-
"""
Training queue for the NABirds causal test.

Runs four stages in order, and RETRIES each one automatically until it completes:

    1  stratified  species     -> nabirds_base_species_only.pth
    2  stratified  multitask   -> nabirds_base_multitask.pth
    3  unseen      species     -> nabirds_unseen_species.pth
    4  unseen      multitask   -> nabirds_unseen_multitask.pth

Why a queue: the causal test in 5.6 compares a genus-stratified model against an
unseen-genus model trained with the SAME recipe.  If one is interrupted and the other
is not, the pair is no longer comparable — so resuming must be exact.  The trainer now
writes a full checkpoint (model + optimiser + scheduler + epoch) and, when resumed
without --start-epoch, continues from the recorded epoch by itself.

Each stage is retried up to MAX_TRIES times; a stage that finishes (exit 0) moves on.
"""
import os
import subprocess
import sys
import time
from _paths import find_root  # noqa: E402

ROOT = find_root()
T = os.path.join(ROOT, "_tools")
CK = os.path.join(ROOT, "checkpoints")
LOG = os.path.join(ROOT, "_tools", "final_out")
PY = sys.executable
MAX_TRIES = 5

# (label, split, stage, epochs, lr, out, init-from)
QUEUE = [
    ("属分层 species", "by_genus", "species", 60, "2e-4",
     "nabirds_base_species_only.pth", None),
    ("属分层 multitask", "by_genus", "multitask", 30, "2e-5",
     "nabirds_base_multitask.pth", "nabirds_base_species_only.pth"),
    ("未见属 species", "unseen_genus", "species", 60, "2e-4",
     "nabirds_unseen_species.pth", None),
    ("未见属 multitask", "unseen_genus", "multitask", 30, "2e-5",
     "nabirds_unseen_multitask.pth", "nabirds_unseen_species.pth"),
]


def done(out):
    """A stage is finished only when its marker exists.

    Checking the .pth alone is wrong: the trainer writes a best checkpoint after the
    FIRST epoch, so an interrupted run looked complete.
    """
    return os.path.exists(os.path.join(CK, out + ".done"))


def run_stage(label, split, stage, epochs, lr, out, init):
    outp = os.path.join(CK, out)
    lastp = outp.replace(".pth", "_last.pth")
    logf = os.path.join(LOG, "train_%s_%s.log" % (split, stage))
    for attempt in range(1, MAX_TRIES + 1):
        cmd = [PY, os.path.join(T, "train_nabirds_base.py"),
               "--split", split, "--stage", stage,
               "--epochs", str(epochs), "--lr", lr,
               "--workers", "4", "--val-every", "5",
               "--out", outp]
        if os.path.exists(lastp):
            cmd += ["--resume", lastp]
            state = "续训"
        elif init and os.path.exists(os.path.join(CK, init)):
            cmd += ["--init", os.path.join(CK, init)]
            state = "初始化自 %s" % init
        else:
            state = "从头训练"
        print("\n" + "=" * 74)
        print("[%s] 第 %d/%d 次尝试 | %s | %s" % (label, attempt, MAX_TRIES, state,
                                                  time.strftime("%H:%M:%S")))
        print("=" * 74, flush=True)
        with open(logf, "a", encoding="utf-8") as f:
            f.write("\n===== attempt %d %s =====\n" % (attempt, time.strftime("%F %T")))
            f.flush()
            r = subprocess.run(cmd, cwd=T, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode == 0 and os.path.exists(outp):
            with open(outp + ".done", "w") as fh:      # completion marker
                fh.write(time.strftime("%F %T\n"))
            print("[%s] 完成 | %s" % (label, outp), flush=True)
            return True
        print("[%s] 失败 (exit %d)，30 秒后续训 ..." % (label, r.returncode), flush=True)
        time.sleep(30)
    print("[%s] 连续 %d 次失败，跳过。" % (label, MAX_TRIES), flush=True)
    return False


def main():
    print("NABirds 因果检验训练队列 | 开始 %s" % time.strftime("%F %T"))
    print("已完成:", [l for l, _, _, _, _, o, _ in QUEUE if done(o)] or "无")
    for label, split, stage, epochs, lr, out, init in QUEUE:
        if done(out):
            print("\n[%s] 已存在，跳过" % label)
            continue
        run_stage(label, split, stage, epochs, lr, out, init)
    print("\n队列结束 %s" % time.strftime("%F %T"))
    for label, _, _, _, _, out, _ in QUEUE:
        print("  %-18s %s" % (label, "OK" if done(out) else "缺失"))


if __name__ == "__main__":
    main()
