# -*- coding: utf-8 -*-
"""
Unified re-run of the bird few-shot evaluation, on cached frozen-backbone features.

Fixes / additions relative to the original scripts
-------------------------------------------------
A1  every result is decomposed into total / cross-genus / same-genus errors, so
    "the genus prior pushes mistakes towards same-genus" can be checked directly.
A2  hyper-parameters are selected ONLY on a held-out episode block (TUNE) and
    every reported number comes from a disjoint block (EVAL).
A4  the inert "local candidate mask" gets its own ablation.
A5  CUB and the endangered set share ONE hyper-parameter set.
B2  a clean-backbone control (no genus-aware training) is included.
B3  linear probe is reported.
B4  paired t-tests for CUB and for the endangered set, on EVAL episodes.

Episode rule is identical to the original scripts:
    np.random.seed(ep * 10); one permutation per species, first-appearance order.
"""
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from _paths import find_root, FEAT_CACHE  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = FEAT_CACHE
OUT = os.path.join(SCRIPT_DIR, "rerun_out")
os.makedirs(OUT, exist_ok=True)
ROOT = (os.environ.get("BIRD_PROJECT_ROOT")
        or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARSED = os.path.join(ROOT, "cub_parsed")

EPISODES = 50
# tuning block: original seeds, first 25 episodes
TUNE_SEEDS = [i * 10 for i in range(25)]
# reported block: 50 FRESH episodes whose seeds never appear during tuning
EVAL_SEEDS = [1000 + i * 10 for i in range(EPISODES)]
# original episode seeds, used only to validate the pipeline against the old CSV
ORIG_SEEDS = [i * 10 for i in range(EPISODES)]
EVAL_EP = list(range(EPISODES))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = ["genus_center", "genus_split", "clean"]
DATASETS = {"cub": "cub_novel", "endb": "endb"}

ORIG_DEFAULT = dict(fw=0.6, p1=1.025, top_k=10, cand=15, calib=0.02, base_w=0.5)

log_lines = []


def log(msg=""):
    print(msg, flush=True)
    log_lines.append(str(msg))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_cache(group, model):
    z = np.load(os.path.join(CACHE, "%s__%s.npz" % (group, model)), allow_pickle=True)
    return z["feats"], z["feats_flip"], z["species"], z["genus"]


def build_base_library(model):
    """Genus prototypes / intra-genus similarity statistics from base-class images."""
    feats, _, species, genus = load_cache("cub_base", model)
    proto, mu, sig = {}, {}, {}
    for g in np.unique(genus):
        fg = feats[genus == g]
        p = fg.mean(axis=0)
        proto[int(g)] = p / max(np.linalg.norm(p), 1e-12)
        n = fg.shape[0]
        if n >= 2:
            inner = fg @ fg.T
            triu = inner[np.triu_indices(n, k=1)]
            mu[int(g)] = float(triu.mean())
            sig[int(g)] = float(triu.std()) + 1e-6
        else:
            mu[int(g)], sig[int(g)] = 0.0, 1.0
    return proto, mu, sig


def build_episodes(species, n_shot, seeds):
    by_sp = defaultdict(list)
    for idx, s in enumerate(species):
        by_sp[int(s)].append(idx)
    eps = []
    for sd in seeds:
        np.random.seed(sd)
        sup, qry = [], []
        for s, idxs in by_sp.items():
            perm = np.random.permutation(idxs)
            sup.extend(perm[:n_shot].tolist())
            qry.extend(perm[n_shot:].tolist())
        eps.append((np.array(sup), np.array(qry)))
    return eps


# ---------------------------------------------------------------------------
# episode container
# ---------------------------------------------------------------------------
class Ep:
    def __init__(self, feats, feats_flip, species, genus, sup, qry, use_tta=False):
        self.S = feats[sup]
        self.yS = species[sup]
        self.gS = genus[sup]
        q = feats[qry]
        if use_tta:
            q = q + feats_flip[qry]
            q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
        self.Q = q
        self.yQ = species[qry]
        self.gQ = genus[qry]

        self.C = np.unique(self.yS)
        c2i = {int(c): i for i, c in enumerate(self.C)}
        Pu = np.stack([self.S[self.yS == c].mean(axis=0) for c in self.C])
        self.Pu = Pu
        self.P = Pu / np.clip(np.linalg.norm(Pu, axis=1, keepdims=True), 1e-12, None)
        self.gP = np.array([int(self.gS[self.yS == c][0]) for c in self.C])
        self.sup_cls = np.array([c2i[int(l)] for l in self.yS])
        # support species -> genus (for turning a prediction into a genus)
        self.sp2g = {int(s): int(g) for s, g in zip(self.yS, self.gS)}

        self.simS = self.Q @ self.S.T
        self.simP = self.Q @ self.P.T
        self.sim_sc = np.full((self.Q.shape[0], len(self.C)), -1.0)
        for i, c in enumerate(self.C):
            self.sim_sc[:, i] = self.simS[:, self.yS == c].max(axis=1)

        # compact genus index over support genera
        sup_g = np.unique(self.gS)
        self.gmap = {int(g): i for i, g in enumerate(sup_g)}
        self.gcompact = np.array([self.gmap[int(g)] for g in self.gS])
        self.nG = len(sup_g)
        self.glist = [int(g) for g in sup_g]

        self.g_mean, self.g_std = {}, {}
        for g in sup_g:
            fg = self.S[self.gS == g]
            if fg.shape[0] >= 2:
                inner = fg @ fg.T
                triu = inner[np.triu_indices(fg.shape[0], k=1)]
                self.g_mean[int(g)] = float(triu.mean())
                self.g_std[int(g)] = float(triu.std()) + 1e-6
            else:
                self.g_mean[int(g)], self.g_std[int(g)] = 0.0, 1.0

    # --- genus ranking --------------------------------------------------
    def genus_scores(self, sim, top_k, base_proto=None, base_w=0.5):
        nq = sim.shape[0]
        k = min(top_k, sim.shape[1])
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        score = np.zeros((nq, self.nG))
        rows = np.repeat(np.arange(nq), k)
        cols = self.gcompact[idx].ravel()
        np.add.at(score, (rows, cols), sim[np.arange(nq)[:, None], idx].ravel())
        if base_proto:
            for g in self.glist:
                if g in base_proto:
                    score[:, self.gmap[g]] += base_w * (self.Q @ base_proto[g])
        return score

    @staticmethod
    def top2(score, glist):
        order = np.argsort(-score, axis=1)
        t1 = np.array([glist[i] for i in order[:, 0]])
        t2 = np.array([glist[i] for i in order[:, 1]]) if score.shape[1] >= 2 else None
        return t1, t2

    def cand_mask(self, sim, cand):
        m = np.zeros_like(sim, dtype=bool)
        c = min(cand, sim.shape[1])
        idx = np.argpartition(-sim, c - 1, axis=1)[:, :c]
        np.put_along_axis(m, idx, True, axis=1)
        return m

    def fused(self, fw):
        return fw * self.simP + (1.0 - fw) * self.sim_sc


# ---------------------------------------------------------------------------
# scoring rules:  rule(ep, params) -> predicted species ids (Nq,)
# ---------------------------------------------------------------------------
def r_nn(ep, p):
    return ep.yS[np.argmax(ep.simS, axis=1)]


def r_protonet(ep, p):
    return ep.C[np.argmax(ep.simP, axis=1)]


def r_simpleshot(ep, p):
    # squared Euclidean distance expanded as ||q||^2 + ||c||^2 - 2 q.c, which avoids
    # materialising the (Nq, C, D) tensor (that needed 5.6 GiB on NABirds)
    q2 = (ep.Q ** 2).sum(axis=1)[:, None]
    p2 = (ep.Pu ** 2).sum(axis=1)[None, :]
    d2 = q2 + p2 - 2.0 * (ep.Q @ ep.Pu.T)
    return ep.C[np.argmin(d2, axis=1)]


def r_matchingnet(ep, p):
    tau = p.get("tau", 1.0)
    a = np.exp((ep.simS - ep.simS.max(axis=1, keepdims=True)) / tau)
    a /= a.sum(axis=1, keepdims=True)
    sc = np.zeros((ep.Q.shape[0], len(ep.C)))
    for i in range(len(ep.C)):
        sc[:, i] = a[:, ep.sup_cls == i].sum(axis=1)
    return ep.C[np.argmax(sc, axis=1)]


def _genus_boost(sim, ep, t1, t2, b1, b2, mask=None):
    w = np.ones_like(sim)
    w = np.where(ep.gS[None, :] == t1[:, None], b1, w)
    if t2 is not None:
        w = np.where(ep.gS[None, :] == t2[:, None], b2, w)
    if mask is not None:
        w = np.where(mask, w, 1.0)
    return sim * w


def _genus_of_pred(ep, pred):
    return np.array([ep.sp2g[int(v)] for v in pred])


def r_gsingle(ep, p):
    t1, t2 = ep.top2(ep.genus_scores(ep.simS, p["top_k"]), ep.glist)
    return ep.yS[np.argmax(_genus_boost(ep.simS, ep, t1, None, p["b1"], 1.0), axis=1)]


def r_ablationA(ep, p):
    m = ep.cand_mask(ep.simS, p["cand"])
    t1, t2 = ep.top2(ep.genus_scores(ep.simS, p["top_k"]), ep.glist)
    return ep.yS[np.argmax(_genus_boost(ep.simS, ep, t1, None, p["b1"], 1.0, m), axis=1)]


def r_ablationB(ep, p):
    t1, t2 = ep.top2(ep.genus_scores(ep.simS, p["top_k"]), ep.glist)
    return ep.yS[np.argmax(_genus_boost(ep.simS, ep, t1, t2, p["b1"], p["b2"]), axis=1)]


def r_full(ep, p):
    m = ep.cand_mask(ep.simS, p["cand"])
    t1, t2 = ep.top2(ep.genus_scores(ep.simS, p["top_k"]), ep.glist)
    return ep.yS[np.argmax(_genus_boost(ep.simS, ep, t1, t2, p["b1"], p["b2"], m), axis=1)]


def r_full_nomask(ep, p):
    return r_ablationB(ep, p)


def r_full_old(ep, p):
    """Sample-level family with the hyper-parameters hard-coded in the original
    scripts; used only to validate the new pipeline against the frozen CSV."""
    return r_full(ep, dict(p, b1=ORIG_DEFAULT["p1"], b2=ORIG_DEFAULT["p2"]
                           if "p2" in ORIG_DEFAULT else 1.02,
                           top_k=ORIG_DEFAULT["top_k"], cand=ORIG_DEFAULT["cand"]))


def r_calibrated(ep, p):
    m = ep.cand_mask(ep.simS, p["cand"])
    t1, t2 = ep.top2(ep.genus_scores(ep.simS, p["top_k"]), ep.glist)
    gm = np.array([ep.g_mean[int(g)] for g in ep.gS])
    gs = np.array([ep.g_std[int(g)] for g in ep.gS])
    z = np.clip((ep.simS - gm[None, :]) / gs[None, :], 0.0, 3.0)
    w = 1.0 + p["calib"] * z
    w = np.where(ep.gS[None, :] == t1[:, None], w * p["b1"], w)
    if t2 is not None:
        w = np.where(ep.gS[None, :] == t2[:, None], w * p["b2"], w)
    w = np.where(m, w, 1.0)
    return ep.yS[np.argmax(ep.simS * w, axis=1)]


def r_baseaug(ep, p):
    m = ep.cand_mask(ep.simS, p["cand"])
    t1, t2 = ep.top2(ep.genus_scores(ep.simS, p["top_k"], p.get("base_proto"), p["base_w"]),
                     ep.glist)
    return ep.yS[np.argmax(_genus_boost(ep.simS, ep, t1, t2, p["b1"], p["b2"], m), axis=1)]


# ---- proto-instance fusion family -----------------------------------------
def _proto_z(ep, p):
    """z-score weight from the fused score, using base-genus stats when available
    (CUB variant) else the support-genus stats (endangered variant)."""
    sim = ep.fused(p["fw"])
    bp_mu = p.get("base_mu")
    if bp_mu:
        mu = np.array([bp_mu.get(int(g), ep.g_mean[int(g)]) for g in ep.gP])
        sg = np.array([p["base_sigma"].get(int(g), ep.g_std[int(g)]) for g in ep.gP])
    else:
        mu = np.array([ep.g_mean[int(g)] for g in ep.gP])
        sg = np.array([ep.g_std[int(g)] for g in ep.gP])
    z = np.clip((sim - mu[None, :]) / sg[None, :], 0.0, 3.0)
    return sim, 1.0 + p["calib"] * z


def r_fusion_only(ep, p):
    """Prototype + nearest-instance fusion, no genus prior, no calibration."""
    return ep.C[np.argmax(ep.fused(p["fw"]), axis=1)]


def r_fusion_calib(ep, p):
    """Fusion + z calibration, still no genus prior."""
    sim, w = _proto_z(ep, p)
    return ep.C[np.argmax(sim * w, axis=1)]


def r_protoprior(ep, p):
    """The paper's GWP: fusion + z calibration + graded genus boost on the
    prototype-level fused score.

    Faithful to test_ablation_genus_prior.py / test_endangered_ablation.py:
      * the z-score calibration multiplies EVERY class score,
      * only the graded genus boost is gated by the local candidate mask.
    """
    sim, w = _proto_z(ep, p)
    if p.get("base_proto"):
        sc = ep.genus_scores(ep.simS, p["top_k"], p["base_proto"], p["base_w"])
    else:
        sc = ep.genus_scores(ep.simS, p["top_k"])
    t1, t2 = ep.top2(sc, ep.glist)

    boost = np.ones_like(sim)
    boost = np.where(ep.gP[None, :] == t1[:, None], p["p1"], boost)
    if t2 is not None:
        boost = np.where(ep.gP[None, :] == t2[:, None], p["p2"], boost)
    if p.get("use_mask", True):
        m = ep.cand_mask(sim, p["cand"])
        boost = np.where(m, boost, 1.0)
    return ep.C[np.argmax(sim * w * boost, axis=1)]


def r_protoprior_old(ep, p):
    """GWP evaluated with the hyper-parameters hard-coded in the original scripts,
    kept only so the new pipeline can be validated against the frozen CSV."""
    q = dict(p)
    q.update(fw=ORIG_DEFAULT["fw"], p1=ORIG_DEFAULT["p1"],
             p2=1.0 + (ORIG_DEFAULT["p1"] - 1.0) * 0.8,
             top_k=ORIG_DEFAULT["top_k"], cand=ORIG_DEFAULT["cand"])
    return r_protoprior(ep, q)


def r_linearprobe(ep, p):
    classes = ep.C
    c2i = {int(c): i for i, c in enumerate(classes)}
    S = torch.tensor(ep.S, device=DEVICE)
    Q = torch.tensor(ep.Q, device=DEVICE)
    y = torch.tensor([c2i[int(l)] for l in ep.yS], device=DEVICE)
    clf = nn.Linear(S.shape[1], len(classes)).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=1e-4)
    clf.train()
    for _ in range(200):
        opt.zero_grad()
        F.cross_entropy(clf(S), y).backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        return classes[clf(Q).argmax(dim=1).cpu().numpy()]


RULES = {
    "nn": r_nn,
    "protonet": r_protonet,
    "simpleshot": r_simpleshot,
    "matchingnet": r_matchingnet,
    "gsingle": r_gsingle,
    "ablationA_single_local": r_ablationA,
    "ablationB_dual_global": r_ablationB,
    "full_masked": r_full,
    "full_nomask": r_full_nomask,
    "full_oldparam": r_full_old,
    "calibrated": r_calibrated,
    "baseaug": r_baseaug,
    "fusion_only": r_fusion_only,
    "fusion_calib": r_fusion_calib,
    "protoprior_GWP": r_protoprior,
    "protoprior_oldparam": r_protoprior_old,
    "protoprior_nomask": lambda ep, p: r_protoprior(ep, dict(p, use_mask=False)),
    "linearprobe": r_linearprobe,
}
TTA_RULES = {
    "simpleshot_tta": r_simpleshot,
    "protonet_tta": r_protonet,
    "fusion_only_tta": r_fusion_only,
    "protoprior_tta": r_protoprior,
    "protoprior_oldparam_tta": r_protoprior_old,
}


def make_params(p, tta=False):
    """Translate the compact hyper-parameter record into rule parameters."""
    q = dict(p)
    q["b1"] = p["p1"]
    q["b2"] = p["p2"]
    return q


def score_one(pred, ep):
    """Accuracy / total / cross-genus / same-genus errors for one episode."""
    ok = pred == ep.yQ
    wrong = ~ok
    gpred = np.array([ep.sp2g[int(v)] for v in pred])
    return (100.0 * ok.mean(), int(wrong.sum()),
            int((wrong & (gpred != ep.gQ)).sum()),
            int((wrong & (gpred == ep.gQ)).sum()))


def evaluate_rule(rule, eps, params):
    acc, tot, cro, sam = [], [], [], []
    for ep in eps:
        a, t, c, s = score_one(rule(ep, params), ep)
        acc.append(a); tot.append(t); cro.append(c); sam.append(s)
    return tuple(np.array(x) for x in (acc, tot, cro, sam))


# ---------------------------------------------------------------------------
# hyper-parameter selection on TUNE episodes only
# ---------------------------------------------------------------------------
def tune(cub_eps5, base_lib, tune_idx):
    """Tune the two method families independently, on the TUNE block only.

    Family 'proto'  : prototype-instance fusion family (fusion_only, fusion_calib, GWP)
    Family 'sample' : sample-level genus-prior family (gsingle / A / B / Full / calibrated)
    """
    proto, mu, sig = base_lib
    eps = list(cub_eps5)
    tk_grid = [5, 10, 15, 20]
    cd_grid = [8, 15, 30]
    out = {}
    t0 = time.time()

    # ---- family 'proto' : fusion weight x proto-level genus boost ----
    fw_grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    p1_grid = [1.0, 1.005, 1.01, 1.02, 1.025, 1.04, 1.05]
    rows = []
    for fw in fw_grid:
        for p1 in p1_grid:
            for tk in tk_grid:
                for cd in cd_grid:
                    p = dict(fw=fw, p1=p1, p2=1.0 + (p1 - 1.0) * 0.8, top_k=tk, cand=cd,
                             calib=0.02, base_w=0.5, use_mask=True, b1=p1, b2=1.0,
                             base_proto=proto, base_mu=mu, base_sigma=sig)
                    acc, tot, cro, sam = evaluate_rule(r_protoprior, eps, p)
                    rows.append(dict(fw=fw, p1=p1, top_k=tk, cand=cd,
                                     acc=float(acc.mean()), cross=float(cro.mean())))
    rows.sort(key=lambda r: -r["acc"])
    out["proto_grid"] = rows
    out["best_proto"] = rows[0]
    # how often does tuning keep the genus boost at all?
    no_boost = float(np.mean([r["acc"] for r in rows if r["p1"] == 1.0]))
    with_boost = [r for r in rows if r["p1"] > 1.0]
    best_boost = with_boost[0] if with_boost else None
    log("    [proto family] %d combos, %d TUNE episodes, %.1fs" % (len(rows), len(eps), time.time() - t0))
    log("      best          : fw=%.2f p1=%.3f top_k=%d cand=%d acc=%.2f%%"
        % (rows[0]["fw"], rows[0]["p1"], rows[0]["top_k"], rows[0]["cand"], rows[0]["acc"]))
    log("      best with p1>1: fw=%.2f p1=%.3f top_k=%d cand=%d acc=%.2f%%  (rank %d/%d)"
        % (best_boost["fw"], best_boost["p1"], best_boost["top_k"], best_boost["cand"],
           best_boost["acc"], rows.index(best_boost) + 1, len(rows)) if best_boost else "")
    log("      mean acc over all p1==1.0 combos = %.2f%%" % no_boost)
    old = [r for r in rows if r["fw"] == ORIG_DEFAULT["fw"] and r["p1"] == ORIG_DEFAULT["p1"]
           and r["top_k"] == ORIG_DEFAULT["top_k"] and r["cand"] == ORIG_DEFAULT["cand"]]
    if old:
        log("      original hard-coded params: acc=%.2f%% (rank %d/%d)"
            % (old[0]["acc"], rows.index(old[0]) + 1, len(rows)))
        out["old_default_proto"] = old[0]

    # ---- family 'sample' : sample-level genus boost ----
    b1_grid = [1.0, 1.01, 1.02, 1.025, 1.04, 1.06, 1.10, 1.15]
    rows2 = []
    for b1 in b1_grid:
        for tk in tk_grid:
            for cd in cd_grid:
                p = dict(fw=0.6, p1=b1, p2=1.0 + (b1 - 1.0) * 0.8, top_k=tk, cand=cd,
                         calib=0.02, base_w=0.5, use_mask=True,
                         b1=b1, b2=1.0 + (b1 - 1.0) * 0.8)
                acc, tot, cro, sam = evaluate_rule(r_full, eps, p)
                acc_nm, _, cro_nm, _ = evaluate_rule(r_full_nomask, eps, p)
                rows2.append(dict(b1=b1, top_k=tk, cand=cd, acc=float(acc.mean()),
                                  acc_nomask=float(acc_nm.mean()),
                                  cross=float(cro.mean()), cross_nomask=float(cro_nm.mean())))
    rows2.sort(key=lambda r: -r["acc"])
    out["sample_grid"] = rows2
    out["best_sample"] = rows2[0]
    log("      [sample family] best: b1=%.3f top_k=%d cand=%d acc=%.2f%%  "
        "(nomask variant of same params: %.2f%%)"
        % (rows2[0]["b1"], rows2[0]["top_k"], rows2[0]["cand"], rows2[0]["acc"],
           rows2[0]["acc_nomask"]))
    log("      [sample family] %d combos on %d TUNE episodes" % (len(rows2), len(eps)))
    return out


# ---------------------------------------------------------------------------
# main experiment
# ---------------------------------------------------------------------------
SAMPLE_KEYS = {"gsingle", "ablationA_single_local", "ablationB_dual_global",
               "full_masked", "full_nomask", "calibrated", "baseaug"}
PROTO_KEYS = {"fusion_only", "fusion_calib", "protoprior_GWP", "protoprior_nomask",
              "protoprior_oldparam"}
TTA_PROTO_KEYS = {"fusion_only_tta", "protoprior_tta", "protoprior_oldparam_tta"}


def run():
    base_libs = {m: build_base_library(m) for m in MODELS}
    results = {}
    tuning_report = {}

    # ---- hyper-parameter selection FIRST, on the CUB 5-shot TUNE block only ----
    tune_json = os.path.join(OUT, "tuning_grid.json")
    if os.path.exists(tune_json):
        with open(tune_json, encoding="utf-8") as f:
            cached = json.load(f)
        if "best_proto" in cached and "best_sample" in cached:
            tuning_report = cached
            log("  reused cached tuning grids from %s" % tune_json)
    if not tuning_report:
        feats, feats_flip, species, genus = load_cache("cub_novel", "genus_center")
        tune_idx = build_episodes(species, 5, TUNE_SEEDS)
        tune_eps = [Ep(feats, feats_flip, species, genus, s, q, use_tta=False)
                    for s, q in tune_idx]
        tuning_report = tune(tune_eps, base_libs["genus_center"], None)
        with open(tune_json, "w", encoding="utf-8") as f:
            json.dump(tuning_report, f, ensure_ascii=False, indent=1)
        del tune_eps, feats, feats_flip, species, genus

    bp = tuning_report["best_proto"]
    bs = tuning_report["best_sample"]

    for ds, group in DATASETS.items():
        for model in MODELS:
            feats, feats_flip, species, genus = load_cache(group, model)
            base_proto, base_mu, base_sig = base_libs[model]
            has_base = group.startswith("cub")
            shots = [1, 5] if ds == "cub" else [5]
            for shot in shots:
                eps_idx = build_episodes(species, shot, EVAL_SEEDS)
                eps = [Ep(feats, feats_flip, species, genus, s, q, use_tta=False)
                       for s, q in eps_idx]
                eps_tta = [Ep(feats, feats_flip, species, genus, s, q, use_tta=True)
                           for s, q in eps_idx]

                common = dict(calib=0.02, base_w=0.5, use_mask=True,
                              base_proto=base_proto if has_base else None,
                              base_mu=base_mu if has_base else None,
                              base_sigma=base_sig if has_base else None)
                p_proto = dict(common, fw=bp["fw"], p1=bp["p1"],
                               p2=1.0 + (bp["p1"] - 1.0) * 0.8,
                               top_k=bp["top_k"], cand=bp["cand"], b1=1.0, b2=1.0)
                p_sample = dict(common, fw=0.6, p1=bs["b1"],
                                p2=1.0 + (bs["b1"] - 1.0) * 0.8,
                                top_k=bs["top_k"], cand=bs["cand"],
                                b1=bs["b1"], b2=1.0 + (bs["b1"] - 1.0) * 0.8)

                tag = "%s/%s/%dshot" % (ds, model, shot)
                log("  running %s ..." % tag)
                for name, rule in RULES.items():
                    params = p_sample if name in SAMPLE_KEYS else p_proto
                    results[(ds, model, shot, name)] = evaluate_rule(rule, eps, params)
                for name, rule in TTA_RULES.items():
                    params = p_proto if name in TTA_PROTO_KEYS else p_sample
                    results[(ds, model, shot, name)] = evaluate_rule(rule, eps_tta, params)

    # ---- pipeline fidelity check on the ORIGINAL episode seeds ----
    feats, feats_flip, species, genus = load_cache("cub_novel", "genus_center")
    _bp, _bm, _bs = base_libs["genus_center"]
    p_old = dict(calib=0.02, base_w=0.5, use_mask=True, fw=ORIG_DEFAULT["fw"],
                 p1=ORIG_DEFAULT["p1"], p2=1.0 + (ORIG_DEFAULT["p1"] - 1.0) * 0.8,
                 top_k=ORIG_DEFAULT["top_k"], cand=ORIG_DEFAULT["cand"],
                 b1=ORIG_DEFAULT["p1"], b2=1.0 + (ORIG_DEFAULT["p1"] - 1.0) * 0.8,
                 base_proto=_bp, base_mu=_bm, base_sigma=_bs)
    for shot in (1, 5):
        oidx = build_episodes(species, shot, ORIG_SEEDS)
        oeps = [Ep(feats, feats_flip, species, genus, s, q, use_tta=False) for s, q in oidx]
        oeps_tta = [Ep(feats, feats_flip, species, genus, s, q, use_tta=True) for s, q in oidx]
        for name, rule in [("nn", r_nn), ("protonet", r_protonet),
                           ("simpleshot", r_simpleshot), ("full_oldparam", r_full_old),
                           ("protoprior_oldparam", r_protoprior_old)]:
            results[("repro", "genus_center", shot, name)] = evaluate_rule(rule, oeps, p_old)
        results[("repro", "genus_center", shot, "protoprior_oldparam_tta")] = \
            evaluate_rule(r_protoprior_old, oeps_tta, p_old)
    del feats, feats_flip, species, genus, oeps, oeps_tta

    np.save(os.path.join(OUT, "results.npy"), results, allow_pickle=True)
    with open(tune_json, "w", encoding="utf-8") as f:
        json.dump(tuning_report, f, ensure_ascii=False, indent=1)
    return results, tuning_report


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
METHOD_LABEL = {
    "nn": "NN Baseline (cosine kNN)",
    "protonet": "ProtoNet (class prototype)",
    "simpleshot": "SimpleShot (euclidean NCM)",
    "matchingnet": "MatchingNet (attn kNN tau=1)",
    "gsingle": "global single-genus (no mask)",
    "ablationA_single_local": "A: single-genus + local mask",
    "ablationB_dual_global": "B: dual-genus global (no mask)",
    "full_masked": "Full: dual-genus + local mask",
    "full_nomask": "Full without mask (== B)",
    "full_oldparam": "Full with original hard-coded params",
    "calibrated": "z-calibrated + dual genus",
    "baseaug": "genus prior + base prototypes",
    "fusion_only": "FUSION ONLY (no genus prior)",
    "fusion_calib": "fusion + z-calib (no genus prior)",
    "protoprior_GWP": "GWP/ProtoPrior (paper, tuned)",
    "protoprior_oldparam": "GWP with original hard-coded params",
    "protoprior_nomask": "GWP without local mask",
    "linearprobe": "Linear probe (support-trained)",
    "simpleshot_tta": "SimpleShot + TTA",
    "protonet_tta": "ProtoNet + TTA",
    "fusion_only_tta": "FUSION ONLY + TTA",
    "protoprior_tta": "GWP + TTA",
    "protoprior_oldparam_tta": "GWP + TTA (original params)",
}


def report(results, tuning):
    order = list(METHOD_LABEL.keys())

    def stat(ds, model, shot, m, idx, episodes):
        if (ds, model, shot, m) not in results:
            return None
        acc, tot, cro, sam = results[(ds, model, shot, m)]
        return np.array([acc[i] for i in episodes]), np.array([cro[i] for i in episodes]), \
            np.array([tot[i] for i in episodes]), np.array([sam[i] for i in episodes])

    csv_rows = []
    sig_rows = []

    for ds, group in DATASETS.items():
        for model in MODELS:
            shots = [1, 5] if ds == "cub" else [5]
            for shot in shots:
                log("")
                log("=" * 118)
                rows_txt = []
                for m in order:
                    if (ds, model, shot, m) not in results:
                        continue
                    acc, cro, tot, sam = stat(ds, model, shot, m, None, EVAL_EP)
                    rows_txt.append((m, acc, cro, tot, sam))
                if not rows_txt:
                    continue
                steps = "%d+%d" % (len(tuning.get("proto_grid", [])),
                                   len(tuning.get("sample_grid", []))) if tuning else "-"
                log("%s | backbone=%s | %d-shot | EVAL episodes=%d | uniform hyper-params "
                    "(tuned on TUNE block only)" % (ds, model, shot, len(EVAL_EP)))
                log("-" * 118)
                log("%-38s %8s %6s %10s %11s %11s %8s" %
                    ("method", "Acc%", "Std", "总错误", "跨属错误", "同属错误", "跨属占比"))
                base_mean = None
                for m, acc, cro, tot, sam in rows_txt:
                    if m == "nn":
                        base_mean = acc.mean()
                for m, acc, cro, tot, sam in rows_txt:
                    share = 100.0 * cro.mean() / max(tot.mean(), 1e-9)
                    log("%-38s %8.2f %6.2f %10.1f %11.1f %11.1f %7.1f%%" %
                        (METHOD_LABEL[m], acc.mean(), acc.std(), tot.mean(), cro.mean(),
                         sam.mean(), share))
                    csv_rows.append([ds, model, shot, m, METHOD_LABEL[m],
                                     "%.3f" % acc.mean(), "%.3f" % acc.std(),
                                     "%.2f" % tot.mean(), "%.2f" % cro.mean(),
                                     "%.2f" % sam.mean(), "%.3f" % share,
                                     len(EVAL_EP), steps])

                # paired t-tests on EVAL episodes
                for key, mk in [("acc", 0), ("cross", 1), ("total", 2), ("same", 3)]:
                    ref = "protoprior_GWP" if "protoprior_GWP" in [r[0] for r in rows_txt] else None
                    if ref is None:
                        continue
                    a = results[(ds, model, shot, ref)][mk]
                    for other in ["simpleshot", "protonet", "nn", "fusion_only",
                                  "fusion_calib", "protoprior_oldparam",
                                  "linearprobe", "simpleshot_tta"]:
                        if (ds, model, shot, other) not in results:
                            continue
                        if other == "simpleshot_tta" and key not in ("acc", "cross"):
                            continue
                        b = results[(ds, model, shot, other)][mk]
                        av = np.array([a[i] for i in EVAL_EP])
                        bv = np.array([b[i] for i in EVAL_EP])
                        t, pv = stats.ttest_rel(av, bv)
                        d = av - bv
                        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(),
                                              scale=stats.sem(d)) if d.std() > 0 else (np.nan, np.nan)
                        star = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
                        log("    [%s] GWP vs %-18s dMean=%+8.4f  t=%7.3f  p=%.4f %-3s CI=[%+.3f,%+.3f]"
                            % (key, other, d.mean(), t, pv, star, ci[0], ci[1]))
                        sig_rows.append([ds, model, shot, key, "GWP vs " + other,
                                         "%.4f" % d.mean(), "%.3f" % t, "%.4f" % pv, star,
                                         "[%.3f,%.3f]" % (ci[0], ci[1])])

    with open(os.path.join(OUT, "main_table.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "backbone", "shot", "key", "method", "acc_mean", "acc_std",
                    "total_err", "cross_genus_err", "same_genus_err", "cross_share_pct",
                    "eval_episodes", "tuning_combos"])
        w.writerows(csv_rows)
    with open(os.path.join(OUT, "significance.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "backbone", "shot", "metric", "comparison", "mean_diff",
                    "t_stat", "p_value", "sig", "ci95"])
        w.writerows(sig_rows)

    # per-episode arrays, for re-plotting
    with open(os.path.join(OUT, "per_episode.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "backbone", "shot", "episode", "method", "acc", "total_err",
                    "cross_genus_err", "same_genus_err"])
        for (ds, model, shot, m), (acc, tot, cro, sam) in results.items():
            for i in range(EPISODES):
                w.writerow([ds, model, shot, i + 1, m, "%.3f" % acc[i], tot[i], cro[i], sam[i]])

    # reproduction check against the frozen CSV from the original scripts
    log("")
    log("=" * 118)
    log("复现校验：本次 EVAL 块 vs 原始 50 轮 CSV（注意原始数字为 50 轮全量、且用旧超参）")
    log("=" * 118)
    try:
        orig = {}
        with open(os.path.join(ROOT, "paper_code", "ablation_genus_prior_results.csv"),
                  encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                orig[(r["n_shot"], r["method"])] = (float(r["acc_mean"]),
                                                    float(r["cross_genus_err_mean"]))
        pairs = [("nn", "NN Baseline (Cosine k-NN)"),
                 ("protonet", "ProtoNet (class prototype mean)"),
                 ("simpleshot", "SimpleShot: class-mean prototype (no renorm)"),
                 ("full_oldparam",
                  "Full: dual-genus graded + local candidate (fixed boost)"),
                 ("protoprior_oldparam",
                  "ProtoPrior: class-prototype matching + genus prior + base aug"),
                 ("protoprior_oldparam_tta",
                  "ProtoPrior+TTA: proto-prior with flip-averaged query feat")]
        all_ep = list(range(EPISODES))
        for shot in (1, 5):
            for key, label in pairs:
                if (str(shot), label) not in orig:
                    continue
                oa, oc = orig[(str(shot), label)]
                if ("repro", "genus_center", shot, key) not in results:
                    continue
                acc, tot, cro, sam = results[("repro", "genus_center", shot, key)]
                acc = np.array([acc[i] for i in all_ep])
                cro = np.array([cro[i] for i in all_ep])
                log("  %s-shot %-38s 原始(50轮)=%6.2f%%/%6.1f  本次(50轮,旧超参,原始种子)=%6.2f%%/%6.1f  Δacc=%+.2f"
                    % (str(shot), key, oa, oc, acc.mean(), cro.mean(), acc.mean() - oa))
    except Exception as e:  # noqa: BLE001
        log("  复现校验失败: %r" % (e,))

    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    log("device = %s" % DEVICE)
    log("TUNE episodes = 25 (seeds %s ...)" % TUNE_SEEDS[:3])
    log("EVAL episodes = %d fresh-seed episodes (seeds %s ...)" % (len(EVAL_SEEDS), EVAL_SEEDS[:3]))
    log("ORIG episodes = %d (used only for the fidelity check)" % len(ORIG_SEEDS))
    res, tun = run()
    report(res, tun)
    log("")
    log("产物目录: %s" % OUT)
