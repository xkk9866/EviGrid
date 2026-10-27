"""Value-of-information analysis (test split of the final protocol).

1. decomposition   for every state (q, t) and horizon k, predicted D = delta (s - v), N = gamma (1 - s) against the realised
                   D* = f(argmax_{a in A_t} pi_{t+k}(a)) - f(argmax_{a in A_t} pi_t(a)),
                   N* = f(argmax_{a in A_{t+k}} pi_{t+k}(a)) - f(argmax_{a in A_t} pi_{t+k}(a)).
2. drivers         at the operating point selected on the development pool: which term carries the predicted gain of
                   every decision to continue (disambiguation or discovery).
3. headroom        calibration of the predicted gain V(k) - v against the realised gain f(sel_{t+k}) - f(sel_t), in bins of
                   the headroom 1 - v_t, for the VOI head and for an unconstrained head (Prop. error propagation).
usage: python voi_analysis.py <main_tag> <free_tag> [traj] [dev] [test]
"""
import json, sys, numpy as np
from common import ROOT, DATASETS, T, load_traj, split
from evaluate import ESSNPolicy, select, key

main_tag, free_tag = sys.argv[1], sys.argv[2]
traj = sys.argv[3] if len(sys.argv) > 3 else "qwen25-7b"
DEV = sys.argv[4] if len(sys.argv) > 4 else "calib,test,calib2"
TEST = sys.argv[5] if len(sys.argv) > 5 else "test2"
LAM = list(np.concatenate([[0.0], np.geomspace(0.002, 2.0, 40)]))
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
PM = json.load(open(f"{ROOT}/preds/{traj}/essn_{main_tag}.json"))
PF = json.load(open(f"{ROOT}/preds/{traj}/essn_{free_tag}.json"))
out = {}
for ds in DATASETS:
    trs = load_traj(traj, ds)
    dev, te = split(trs, DEV), split(trs, TEST)
    ref = np.mean([tr["rounds"][T]["cum_tok"] for tr in dev])
    cc = np.mean([[r["cum_tok"] for r in tr["rounds"]] for tr in dev], 0) / ref
    g = select(dev, ESSNPolicy(PM, cc), LAM, ref)
    rows = {k: {"D": [], "N": [], "Ds": [], "Ns": []} for k in range(1, T + 1)}
    drive = {"dis": 0, "disc": 0}
    head = {name: {b: ([], []) for b in range(len(BINS) - 1)} for name in ("voi", "free")}
    for tr in te:
        R = tr["rounds"]
        f = lambda r: R[r]["f1"]
        for name, P in (("voi", PM), ("free", PF)):
            p = P[key(tr)]
            V = np.array(p["V"])
            sel = [max(p["S"][t], key=lambda x: x[1])[0] for t in range(T + 1)]
            for t in range(T):
                b = int(np.digitize(1 - V[t, 0], BINS)) - 1
                for k in range(1, T - t + 1):
                    head[name][b][0].append(V[t, k] - V[t, 0]); head[name][b][1].append(f(sel[t + k]) - f(sel[t]))
        p = PM[key(tr)]
        for t in range(T):
            S_t = p["S"][t]
            v, s = max(x[1] for x in S_t), p["suff"][t]
            sel_t = max(S_t, key=lambda x: x[1])[0]
            for k in range(1, T - t + 1):
                dl, gm = p["rates"][t][0][k - 1], p["rates"][t][1][k - 1]
                S_k = p["S"][t + k]
                a_in = max([x for x in S_k if x[0] <= t] or S_k, key=lambda x: x[1])[0]
                a_all = max(S_k, key=lambda x: x[1])[0]
                rows[k]["D"].append(dl * (s - v)); rows[k]["N"].append(gm * (1 - s))
                rows[k]["Ds"].append(f(a_in) - f(sel_t)); rows[k]["Ns"].append(f(a_all) - f(a_in))
        V = np.array(p["V"])
        for t in range(T):
            cont = [V[t, k] - g * (cc[t + k] - cc[t]) for k in range(1, T - t + 1)]
            kb = int(np.argmax(cont)) + 1
            if V[t, 0] >= cont[kb - 1]:
                break
            S_t = p["S"][t]
            v, s = max(x[1] for x in S_t), p["suff"][t]
            dis, disc = p["rates"][t][0][kb - 1] * (s - v), p["rates"][t][1][kb - 1] * (1 - s)
            drive["dis" if dis >= disc else "disc"] += 1
    res = {k: {m: round(100 * float(np.mean(x)), 2) for m, x in r.items()} for k, r in rows.items() if r["D"]}
    tot = max(1, drive["dis"] + drive["disc"])
    hc = {name: {f"{BINS[b]:.1f}-{min(BINS[b + 1], 1):.1f}": [len(x), round(100 * float(np.mean(x)), 2) if x else None,
                                                                   round(100 * float(np.mean(y)), 2) if y else None]
                 for b, (x, y) in hb.items()} for name, hb in head.items()}
    out[ds] = {"by_k": res, "continue_by_disambiguation_%": round(100 * drive["dis"] / tot, 1), "n_continue": tot,
               "lambda": g, "headroom": hc}
    print(ds, json.dumps(out[ds]), flush=True)
json.dump(out, open(f"{ROOT}/results/voi_{main_tag}_{traj}.json", "w"), indent=1)
