"""Generalisation across backbones with one protocol: operating points on the development split of the backbone
(calib), one evaluation on the fresh test split (test2).

  zero-shot  controllers trained on Qwen2.5-7B trajectories, applied unchanged (*_from_qwen25-7b)
  few-shot   InHand initialised from the Qwen2.5-7B model and adapted on 100 training questions per dataset (e12_fs)
  in-domain  controllers trained on the backbone's own 1,200 training questions per dataset (three seeds)
  native     for the search agent: its own stopping decision and answer

usage: python report_backbones.py llama31-8b,qwen3-4b,qwen25-7b-bm25,searchr1
"""
import json, os, sys
import numpy as np
from common import ROOT, DATASETS, T, load_traj, split
from evaluate import ESSNPolicy, run_fixed, run_threshold, evaluate, select, pareto_auc, key

LAM = [0.0] + list(np.geomspace(0.002, 2.0, 40))
tg = lambda taus: [(float(t), m) for t in taus for m in (0, 1)]
out = {}
for bb in sys.argv[1].split(","):
    if not os.path.exists(f"{ROOT}/traj/{bb}/hotpotqa.jsonl.gz"):
        continue
    P = lambda n: json.load(open(f"{ROOT}/preds/{bb}/{n}.json"))
    has = lambda n: os.path.exists(f"{ROOT}/preds/{bb}/{n}.json")
    out[bb] = {}
    for ds in DATASETS:
        trajs = load_traj(bb, ds)
        dev, test = split(trajs, "calib"), split(trajs, "test2")
        if not dev or not test:
            continue
        ref = np.mean([tr["rounds"][T]["cum_tok"] for tr in dev])
        cc = np.mean([[r["cum_tok"] for r in tr["rounds"]] for tr in dev], 0) / ref
        R = {}
        for k in (1, 2, 3, T):
            R[f"Fixed-{k}"] = evaluate(test, lambda tr, k=k: run_fixed(tr, k))
        fams = {"Answer-Stability": [(lambda m: run_threshold(lambda tr, t: tr["rounds"][t]["same_prev"], 0.5, min_t=m), [1, 2, 3])],
                "Confidence": [(lambda g: run_threshold(lambda tr, t: tr["rounds"][t]["lp_mean"], g[0], min_t=g[1]),
                                tg(np.quantile([r["lp_mean"] for tr in dev for r in tr["rounds"]], np.linspace(0, 1, 41))))]}

        def stop_pol(p):
            return (lambda g: run_threshold(lambda tr, t: p[key(tr)][t][0] - p[key(tr)][t][1], g[0], min_t=g[1]), tg(np.linspace(-0.5, 0.3, 41)))

        def sim_pol(p):
            return (lambda g: run_threshold(lambda tr, t: p[key(tr)][t][0], g[0], min_t=g[1]), tg(np.linspace(0.05, 1.0, 39)))

        def ess_pol(p):
            return (ESSNPolicy(p, cc), LAM)
        for label, base, mk in (("Stop-RAG (zero-shot)", "stoprag_from_qwen25-7b", stop_pol), ("SIM-RAG (zero-shot)", "simrag_from_qwen25-7b", sim_pol),
                                ("InHand (zero-shot)", "essn_e12_m_from_qwen25-7b", ess_pol), ("InHand (few-shot, 100/ds)", "essn_e12_fs", ess_pol),
                                ("Stop-RAG (in-domain)", "stoprag", stop_pol), ("SIM-RAG (in-domain)", "simrag", sim_pol),
                                ("InHand (in-domain)", "essn_e12_m", ess_pol)):
            runs = [mk(P(base + sfx)) for sfx in ("", "_s1", "_s2") if has(base + sfx)]
            if runs:
                fams[label] = runs
        for label, runs in fams.items():
            ms = []
            for pol, grid in runs:
                g = select(dev, pol, grid, ref)
                m = evaluate(test, pol(g))
                m["AUC"] = pareto_auc(test, pol, grid, ref)[0]
                ms.append(m)
            R[label] = {c: float(np.mean([m[c] for m in ms])) for c in ("EM", "F1", "Tok", "Ret", "AUC")}
            R[label]["F1_std"] = float(np.std([m["F1"] for m in ms], ddof=1)) if len(ms) > 1 else 0.0
            R[label]["n"] = len(ms)
        if test[0].get("native_answer") is not None:
            R["Agent (native stop)"] = evaluate(test, lambda tr: (tr["native_stop"], "native", -tr["rounds"][tr["native_stop"]]["cum_probe"]))
            R["Agent (native stop) + InHand select"] = None
        R["Oracle"] = evaluate(test, lambda tr: (lambda t: (t, t, 0))(int(np.argmax([r["f1"] for r in tr["rounds"]]))))
        if R.get("Agent (native stop) + InHand select", 1) is None:
            # the agent keeps its own stopping time; InHand only selects among the drafts in hand at that time
            best = "essn_e12_m" if has("essn_e12_m") else "essn_e12_m_from_qwen25-7b"
            pe = P(best)
            lik = lambda tr: sum(tr["rounds"][u].get("lik_tok", 0) for u in range(tr["native_stop"] + 1))
            R["Agent (native stop) + InHand select"] = evaluate(test, lambda tr: (tr["native_stop"], max(pe[key(tr)]["S"][tr["native_stop"]], key=lambda x: x[1])[0], lik(tr)))
        out[bb][ds] = {k: {c: v for c, v in m.items() if c != "perq"} for k, m in R.items()}
        print(bb, ds, {k: round(m["F1"], 1) for k, m in R.items()}, flush=True)
json.dump(out, open(f"{ROOT}/results/backbones_v12.json", "w"), indent=1)
print("saved")
