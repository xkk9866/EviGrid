"""Fast check of ESSN-type predictions: selection F1 at T, calibrated operating point F1/Tok, frontier AUC.
usage: python quick_eval.py name1,name2,...   (preds/<traj>/essn_<name>.json)"""
import json, sys, numpy as np
from common import ROOT, DATASETS, T, load_traj, split
from evaluate import ESSNPolicy, evaluate, select, pareto_auc, key

traj = sys.argv[2] if len(sys.argv) > 2 else "qwen25-7b"
names = sys.argv[1].split(",")
P = {n: json.load(open(f"{ROOT}/preds/{traj}/essn_{n}.json")) for n in names}
lam = list(np.concatenate([[0.0], np.geomspace(0.002, 2.0, 40)]))
for ds in DATASETS:
    trs = load_traj(traj, ds)
    cal, te = split(trs, "calib"), split(trs, "test")
    ref = np.mean([tr["rounds"][T]["cum_tok"] for tr in cal])
    cc = np.mean([[r["cum_tok"] for r in tr["rounds"]] for tr in cal], 0) / ref
    for n in names:
        p = P[n]
        selT = np.mean([tr["rounds"][max(p[key(tr)]["S"][T], key=lambda x: x[1])[0]]["f1"] for tr in te]) * 100
        pol = ESSNPolicy(p, cc)
        g = select(cal, pol, lam, ref)
        m = evaluate(te, pol(g))
        auc, _ = pareto_auc(te, pol, lam, ref)
        print(f"{ds:9s} {n:22s} selT {selT:5.1f} | F1 {m['F1']:5.1f} EM {m['EM']:5.1f} Ret {m['Ret']:.2f} Tok {m['Tok']:6.0f} AUC {auc:5.1f}", flush=True)
