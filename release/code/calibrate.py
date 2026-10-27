"""Calibration experiment for the answer posterior and the predicted value of one more round (CPU only).

Quantities (per question, per round t; q_t = F1 of the answer EviGrid itself would return at round t):
  v_t         predicted value of stopping            vs  q_t
  g_t(k)      predicted gain V_t(k) - v_t            vs  q_{t+k} - q_t
Metrics: expected calibration error (10 equal-mass bins) of v_t and of the gain, Spearman rank correlation,
calibration slope of the gain, all on the fresh test split.
Recalibration: monotone (isotonic) maps fitted on the DEVELOPMENT pool only (calib, test, calib2; never test2/cert):
  v'_t = h_0(v_t),   V'_t(k) = v'_t + h_k(g_t(k))   (made non-decreasing in k, clipped to 1)
The recalibrated file is evaluated by evaluate.py exactly like the raw one (operating point on dev, once on test2).

usage: python calibrate.py <tags comma-separated>      writes preds/<traj>/essn_<tag>_cal.json and results/calibration.json
"""
import hashlib, json, sys, numpy as np
from scipy.stats import spearmanr
import os
from common import ROOT, DATASETS, T, load_traj, split
TRAJ = os.environ.get("TRAJ", "qwen25-7b")

DEV, TEST = "calib,test,calib2", "test2"


def pav(x, y):
    """isotonic (non-decreasing) regression by pool-adjacent-violators; returns a function."""
    o = np.argsort(x, kind="stable")
    xs, ys = np.asarray(x)[o], np.asarray(y, float)[o]
    blocks = []  # [sum, count, x_sum]
    for xi, yi in zip(xs, ys):
        blocks.append([yi, 1, xi])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop(); blocks[-1][0] += b[0]; blocks[-1][1] += b[1]; blocks[-1][2] += b[2]
    bx = np.array([b[2] / b[1] for b in blocks]); by = np.array([b[0] / b[1] for b in blocks])
    return lambda z: np.interp(z, bx, by)


def ece(p, y, nb=10):
    q = np.quantile(p, np.linspace(0, 1, nb + 1)); q[-1] += 1e-9
    tot = 0.0
    for a, b in zip(q[:-1], q[1:]):
        m = (p >= a) & (p < b)
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def states(P, trajs, ds):
    """arrays over all (question, t): v, q ; and over (question, t, k): predicted gain, realised gain, k"""
    v, q, gp, gr, kk = [], [], [], [], []
    for tr in trajs:
        p = P[f"{ds}::{tr['qid']}"]
        qs = [tr["rounds"][max(p["S"][t], key=lambda x: x[1])[0]]["f1"] for t in range(T + 1)]
        for t in range(T + 1):
            v.append(p["V"][t][0]); q.append(qs[t])
            for k in range(1, T - t + 1):
                gp.append(p["V"][t][k] - p["V"][t][0]); gr.append(qs[t + k] - qs[t]); kk.append(k)
    return [np.array(a) for a in (v, q, gp, gr, kk)]


def metrics(v, q, gp, gr):
    sl = np.polyfit(gp, gr, 1)[0] if gp.std() > 0 else float("nan")
    return {"ece_v": ece(v, q), "ece_gain": ece(gp, gr), "rho": float(spearmanr(gp, gr)[0]), "slope": float(sl),
            "pred_gain": float(gp.mean()), "real_gain": float(gr.mean()), "v_mean": float(v.mean()), "q_mean": float(q.mean())}


def fold(qid):
    return int(hashlib.md5(str(qid).encode()).hexdigest(), 16) % 2


def fit(P, trajs, ds):
    v, q, gp, gr, kk = states(P, trajs, ds)
    return pav(v, q), {k: pav(gp[kk == k], gr[kk == k]) for k in range(1, T + 1)}


def apply(p, h0, hk):
    V = np.array(p["V"], float); Vn = V.copy()
    for t in range(T + 1):
        v0 = float(h0(V[t, 0])); Vn[t, 0] = v0; run = v0
        for k in range(1, T - t + 1):
            run = max(run, v0 + float(hk[k](V[t, k] - V[t, 0]))); Vn[t, k] = min(run, 1.0)
    return {**p, "V": np.clip(Vn, 1e-5, 1 - 1e-5).round(4).tolist()}


def crossfit(tags):
    """development questions: maps fitted on the other fold (out-of-fold); all other splits: maps fitted on the whole pool.
    The operating point is then selected on out-of-fold calibrated development predictions."""
    for tag in tags:
        P = json.load(open(f"{ROOT}/preds/{TRAJ}/essn_{tag}.json"))
        Pc = {}
        for ds in DATASETS:
            tr = load_traj(TRAJ, ds)
            dev = split(tr, DEV)
            full = fit(P, dev, ds)
            folds = [fit(P, [x for x in dev if fold(x["qid"]) != f], ds) for f in (0, 1)]
            for x in split(tr, "calib,test,calib2,cert,test2,test3"):
                key = f"{ds}::{x['qid']}"
                if key not in P:
                    continue
                h = folds[fold(x["qid"])] if x["split"] in DEV.split(",") else full
                Pc[key] = apply(P[key], *h)
        json.dump(Pc, open(f"{ROOT}/preds/{TRAJ}/essn_{tag}_cf.json", "w"))
        print("crossfit", tag, len(Pc), flush=True)


def main():
    if sys.argv[1] == "crossfit":
        return crossfit(sys.argv[2].split(","))
    tags = sys.argv[1].split(",")
    out = {}
    for tag in tags:
        P = json.load(open(f"{ROOT}/preds/{TRAJ}/essn_{tag}.json"))
        Pc = {}
        out[tag] = {}
        for ds in DATASETS:
            tr = load_traj(TRAJ, ds)
            dev, te = split(tr, DEV), split(tr, TEST)
            v, q, gp, gr, kk = states(P, dev, ds)
            h0 = pav(v, q)
            hk = {k: pav(gp[kk == k], gr[kk == k]) for k in range(1, T + 1)}
            raw = metrics(*[a for a in states(P, te, ds)[:4]])
            for tr_ in split(tr, "calib,test,calib2,cert,test2,test3"):
                key = f"{ds}::{tr_['qid']}"
                if key not in P:
                    continue
                p = P[key]
                V = np.array(p["V"], float)
                Vn = V.copy()
                for t in range(T + 1):
                    v0 = float(h0(V[t, 0]))
                    Vn[t, 0] = v0
                    run = v0
                    for k in range(1, T - t + 1):
                        run = max(run, v0 + float(hk[k](V[t, k] - V[t, 0])))
                        Vn[t, k] = min(run, 1.0)
                Pc[key] = {**p, "V": np.clip(Vn, 1e-5, 1 - 1e-5).round(4).tolist()}
            cal = metrics(*[a for a in states(Pc, te, ds)[:4]])
            out[tag][ds] = {"raw": raw, "cal": cal}
            print(tag, ds, "raw", {k: round(x, 3) for k, x in raw.items() if k in ("ece_v", "ece_gain", "rho", "slope", "pred_gain", "real_gain")},
                  "cal", {k: round(x, 3) for k, x in cal.items() if k in ("ece_v", "ece_gain", "rho", "slope", "pred_gain", "real_gain")}, flush=True)
            # reliability curves (raw / calibrated) on the test split
            for name, PP in (("raw", P), ("cal", Pc)):
                v_, q_, gp_, gr_, _ = states(PP, te, ds)
                def rel(a, b, nb=8):
                    e = np.quantile(a, np.linspace(0, 1, nb + 1)); e[-1] += 1e-9
                    return [[float(a[(a >= x) & (a < y)].mean()), float(b[(a >= x) & (a < y)].mean())] for x, y in zip(e[:-1], e[1:]) if ((a >= x) & (a < y)).any()]
                out[tag][ds][name + "_rel_v"] = rel(v_, q_); out[tag][ds][name + "_rel_gain"] = rel(gp_, gr_)
        json.dump(Pc, open(f"{ROOT}/preds/{TRAJ}/essn_{tag}_cal.json", "w"))
    json.dump(out, open(f"{ROOT}/results/calibration.json", "w"))


if __name__ == "__main__":
    main()
