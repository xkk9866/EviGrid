"""Decisions of calibrated EviGrid and Stop-RAG (operating points of the main table) on the case-study questions."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import f1
R = os.path.join(os.path.dirname(__file__), "..")
M = json.load(open(f"{R}/results/v14/f_test2_all.json"))["main"]
P = json.load(open(f"{R}/release/preds/essn_p_full_cf.json"))
PR = json.load(open(f"{R}/release/preds/essn_p_full.json"))
Q = json.load(open(f"{R}/release/preds/stoprag.json"))
C = json.load(open(f"{R}/results/case_candidates.json", encoding="utf8"))
TOK = {"hotpotqa": [0.19, 0.72, 1.63, 2.91, 4.58, 6.64], "2wiki": [0.18, 0.71, 1.58, 2.79, 4.36, 6.28], "musique": [0.19, 0.75, 1.67, 2.97, 4.64, 6.70]}
keys = sys.argv[1:]
for k in keys:
    ds = k.split("::")[0]; c = np.array(TOK[ds]) / TOK[ds][-1]
    lam = M[ds]["ESSN"]["param"]; V = np.array(P[k]["V"])
    stop = next((t for t in range(5) if V[t, 0] >= max(V[t, j] - lam * (c[t + j] - c[t]) for j in range(1, 6 - t))), 5)
    sel = max(PR[k]["S"][stop], key=lambda x: x[1])
    g = M[ds]["Stop-RAG"]["param"]
    tau, mt = (g if isinstance(g, (list, tuple)) else (g, 0))
    qs = Q[k]
    sr = next((t for t in range(6) if t >= mt and qs[t][0] - qs[t][1] >= tau), 5)
    case = next(x for x in C if x["key"] == k)
    print(k, "| lam", lam, "| EviGrid stop", stop, "select round", sel, "| v,V1..3", V[stop, :4].round(3),
          "| raw post", PR[k]["S"][stop], "| Stop-RAG param", g, "stops", sr, "draft", case["drafts"][sr], case["q_f1"][sr])
    print("   drafts", case["drafts"], case["q_f1"])
    print("   titles", case["titles"])
