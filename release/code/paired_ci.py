"""Paired bootstrap confidence intervals over test questions (EviGrid vs every baseline).

For every dataset and baseline X, questions are resampled with replacement (B draws, the SAME draw for both
controllers) and the following are recomputed for EviGrid minus X: F1 at the operating point, tokens per question, and the
area under the F1-cost frontier (AUC; the frontier is rebuilt from the per-question outcomes of every grid point,
the operating points themselves stay fixed because they were selected on the development pool).
usage: python paired_ci.py <results_dir> <tag> <out.json>   (tag: v15 -> v15_all.json / v15_ens.json)
"""
import json, sys
import numpy as np

RES, TAG, OUT = sys.argv[1:4]
B = 2000
DS = ["hotpotqa", "2wiki", "musique"]
BASE = ["Fixed-5", "Fixed-3", "Answer-Stability", "Conf-Threshold", "SIM-RAG", "Stop-RAG", "SIM-RAG (+select)", "Stop-RAG (+select)",
        "SIM-RAG + LLM-lik rerank", "Stop-RAG + LLM-lik rerank", "HALT", "S2G-Judge"]
A = json.load(open(f"{RES}/{TAG}_all.json"))
E = json.load(open(f"{RES}/{TAG}_ens.json"))
AA = np.load(f"{RES}/{TAG}_all_arr.npz")
EA = np.load(f"{RES}/{TAG}_ens_arr.npz")


def auc_from(arr, tokref, idx):
    f = arr[:, 0, :][:, idx].mean(1)
    x = arr[:, 1, :][:, idx].mean(1) / tokref
    o = np.argsort(x, kind="stable")
    x, f = x[o], f[o]
    best = np.maximum.accumulate(f)
    xs = np.concatenate([[0.0], np.minimum(x, 1.0), [1.0]])
    ys = np.concatenate([[0.0], best, [best[-1]]])
    return float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(ys, xs))


def ci(a):
    return [float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


rng = np.random.default_rng(0)
res = {}
for ds in DS:
    n = len(A["main"][ds]["ESSN"]["perq"])
    idxs = rng.integers(0, n, size=(B, n))
    res[ds] = {}
    for who, M, AR in (("EviGrid", A["main"][ds], AA), ("EviGrid-Ens", E["main"][ds], EA)):
        f_me = np.array(M["ESSN"]["perq"]) * 100; t_me = np.array(M["ESSN"]["pertok"])
        arr_me = AR[f"{ds}__ESSN"]; tr_me = float(AR[f"tokref_{ds}__ESSN"])
        auc_me = np.array([auc_from(arr_me, tr_me, ix) for ix in idxs]) * 100
        for b in BASE:
            Mb, ARb = A["main"][ds], AA
            if b not in Mb:
                continue
            f_b = np.array(Mb[b]["perq"]) * 100; t_b = np.array(Mb[b]["pertok"])
            d_f = np.array([(f_me[ix] - f_b[ix]).mean() for ix in idxs])
            d_t = np.array([(t_me[ix] - t_b[ix]).mean() for ix in idxs])
            r = {"dF1": ci(d_f), "dTok": ci(d_t), "p_F1_le0": float((d_f <= 0).mean())}
            if f"{ds}__{b}" in ARb.files:
                arr_b = ARb[f"{ds}__{b}"]; tr_b = float(ARb[f"tokref_{ds}__{b}"])
                d_a = auc_me - np.array([auc_from(arr_b, tr_b, ix) for ix in idxs]) * 100
                r["dAUC"] = ci(d_a); r["p_AUC_le0"] = float((d_a <= 0).mean())
            res[ds][f"{who} vs {b}"] = r
        res[ds][f"{who}"] = {"F1": ci(f_me[idxs].mean(1)), "Tok": ci(t_me[idxs].mean(1)), "AUC": ci(auc_me)}
    print(ds, "done", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
for ds in DS:
    print("==", ds)
    for k, v in res[ds].items():
        if " vs " in k and k.startswith("EviGrid vs") or k.startswith("EviGrid-Ens vs Stop-RAG"):
            print(f"{k:45s} dF1 {v['dF1'][0]:+.2f} [{v['dF1'][1]:+.2f},{v['dF1'][2]:+.2f}]  dTok {v['dTok'][0]:+.0f}" +
                  (f"  dAUC {v['dAUC'][0]:+.2f} [{v['dAUC'][1]:+.2f},{v['dAUC'][2]:+.2f}]" if "dAUC" in v else ""))
