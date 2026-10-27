"""Mechanism diagnostics of EviGrid on the fresh test split (from prediction files only).
 voi:  predicted gain V_t(1)-v_t of one more round vs. the realised gain (quality of the answer EviGrid selects at t+1 minus at t)
 post: calibration of v_t = max_a pi_t(a) against the quality of the selected answer (reliability bins, ECE)
usage: python grid_analysis.py p_full,p_full_s1,p_full_s2 out.json"""
import json, sys, numpy as np
from scipy.stats import spearmanr
from common import DATASETS, T, load_traj, split
tags, out = sys.argv[1].split(","), sys.argv[2]
res = {}
for ds in DATASETS:
    te = split(load_traj("qwen25-7b", ds), "test2")
    res[ds] = {}
    for tag in tags:
        P = json.load(open(f"../preds/qwen25-7b/essn_{tag}.json"))
        pg, rg, pv, rv, gp = [], [], [], [], []
        for tr in te:
            p = P[f"{ds}::{tr['qid']}"]
            q = [tr["rounds"][max(p["S"][t], key=lambda x: x[1])[0]]["f1"] for t in range(T + 1)]
            for t in range(T):
                pg.append(p["V"][t][1] - p["V"][t][0]); rg.append(q[t + 1] - q[t])
            for t in range(T + 1):
                pv.append(p["V"][t][0]); rv.append(q[t])
        pg, rg, pv, rv = map(np.array, (pg, rg, pv, rv))
        qs = np.quantile(pg, [0, .2, .4, .6, .8, 1.0]); qs[-1] += 1e-9
        bins = [[float(pg[(pg >= a) & (pg < b)].mean()), float(rg[(pg >= a) & (pg < b)].mean()), int(((pg >= a) & (pg < b)).sum())] for a, b in zip(qs[:-1], qs[1:])]
        eb = np.linspace(0, 1, 11); ece = 0.0; rel = []
        for a, b in zip(eb[:-1], eb[1:]):
            m = (pv >= a) & (pv < b + (b == 1))
            if m.sum():
                ece += m.mean() * abs(pv[m].mean() - rv[m].mean()); rel.append([float(pv[m].mean()), float(rv[m].mean()), int(m.sum())])
        res[ds][tag] = {"voi_spearman": float(spearmanr(pg, rg)[0]), "voi_bins": bins, "voi_pred_mean": float(pg.mean()), "voi_real_mean": float(rg.mean()),
                        "ece": float(ece), "rel": rel, "neg_gain_viol": float((pg < -1e-6).mean())}
json.dump(res, open(out, "w"))
for ds in DATASETS:
    print(ds, {t: (round(r["voi_spearman"], 3), round(r["ece"], 3), round(r["voi_pred_mean"], 3), round(r["voi_real_mean"], 3)) for t, r in res[ds].items()})
