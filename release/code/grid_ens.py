"""Seed ensemble: average of the value profiles and answer posteriors of independently trained EviGrid models."""
import json, os, sys, numpy as np
TRAJ = os.environ.get("TRAJ", "qwen25-7b")
tags, out = sys.argv[1].split(","), sys.argv[2]
Ps = [json.load(open(f"../preds/{TRAJ}/essn_{t}.json")) for t in tags]
res = {}
for k in Ps[0]:
    V = np.mean([P[k]["V"] for P in Ps], 0)
    S = [[[r, round(float(np.mean([P[k]["S"][t][n][1] for P in Ps])), 4)] for n, (r, _) in enumerate(Ps[0][k]["S"][t])] for t in range(len(Ps[0][k]["S"]))]
    suff = np.mean([P[k]["suff"] for P in Ps], 0)
    res[k] = {"V": V.round(4).tolist(), "S": S, "suff": suff.round(4).tolist(), "P": suff.round(4).tolist(), "lik_used": Ps[0][k]["lik_used"], "extra_tok": [0] * 6}
json.dump(res, open(f"../preds/{TRAJ}/essn_{out}.json", "w"))
print("ens", len(res))
