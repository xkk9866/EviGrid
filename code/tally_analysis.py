"""Mechanism diagnostics for TALLY (and, for contrast, the previous controller ESSN), on the fresh test split.

  coherence   martingale residual of the posterior: for every candidate a in A_t, pi_{t+1}(a) - pi_t(a), binned by
              pi_t(a); a coherent posterior has mean residual 0 in every bin (reported: weighted |mean residual|)
  voi         calibration of the predicted value of one more round, V_t(1) - v_t, against the realised gain of the
              selected answer f(a_hat_{t+1}) - f(a_hat_t); binned, plus Spearman correlation
  monotone    fraction of states with V_t(1) < v_t - 1e-3 (0 by construction for TALLY up to Monte-Carlo error)
  split       gain over Fixed-5 / best fixed budget, separately for questions whose drafts change across rounds and
              questions whose drafts never change (the answers-in-hand mechanism can only act on the former)
  terms       share of the predicted value of information carried by disambiguation and by discovery

usage: python tally_analysis.py --main tally_m [--ref e12_m] [--traj qwen25-7b]
"""
import argparse, json, os
import numpy as np
from scipy.stats import spearmanr
from common import ROOT, DATASETS, T, load_traj, split, normalize
from evaluate import ESSNPolicy, select, evaluate, key

LAM = [0.0] + list(np.geomspace(0.002, 2.0, 40))


def probs_by_key(tr, S_t):
    return {normalize(tr["rounds"][r]["answer"]): p for r, p in S_t}


def coherence(test, P):
    res = []
    for tr in test:
        pr = P[key(tr)]
        for t in range(T):
            a, b = probs_by_key(tr, pr["S"][t]), probs_by_key(tr, pr["S"][t + 1])
            for k, p in a.items():
                if k in b:
                    res.append((p, b[k] - p))
    x = np.array(res)
    bins = np.minimum((x[:, 0] * 10).astype(int), 9)
    w = [(bins == i).mean() * abs(x[bins == i, 1].mean()) for i in range(10) if (bins == i).any()]
    return {"wabs_resid": float(sum(w)), "mean_resid": float(x[:, 1].mean()), "n": int(len(x))}


def sel_f1(tr, S_t):
    return tr["rounds"][max(S_t, key=lambda x: x[1])[0]]["f1"]


def voi(test, P):
    pred, real = [], []
    viol = 0
    for tr in test:
        pr = P[key(tr)]
        V = np.array(pr["V"])
        for t in range(T):
            pred.append(V[t, 1] - V[t, 0])
            real.append(sel_f1(tr, pr["S"][t + 1]) - sel_f1(tr, pr["S"][t]))
            viol += V[t, 1] < V[t, 0] - 1e-3
    pred, real = np.array(pred), np.array(real)
    q = np.quantile(pred, np.linspace(0, 1, 6))
    b = np.clip(np.searchsorted(q, pred, side="right") - 1, 0, 4)
    rel = [[float(pred[b == i].mean()), float(real[b == i].mean()), int((b == i).sum())] for i in range(5) if (b == i).any()]
    return {"spearman": float(spearmanr(pred, real).correlation), "bins": rel, "monotone_viol": float(viol / len(pred)),
            "pred_mean": float(pred.mean()), "real_mean": float(real.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="tally_m")
    ap.add_argument("--ref", default="e12_m")
    ap.add_argument("--traj", default="qwen25-7b")
    ap.add_argument("--dev", default="calib,test,calib2")
    ap.add_argument("--test", default="test2")
    a = ap.parse_args()
    pre = a.main.rsplit("_", 1)[0]
    load = lambda n: json.load(open(f"{ROOT}/preds/{a.traj}/essn_{n}.json")) if os.path.exists(f"{ROOT}/preds/{a.traj}/essn_{n}.json") else None
    out = {}
    for ds in DATASETS:
        trajs = load_traj(a.traj, ds)
        calib, test = split(trajs, a.dev), split(trajs, a.test)
        ref = np.mean([tr["rounds"][T]["cum_tok"] for tr in calib])
        cc = np.mean([[r["cum_tok"] for r in tr["rounds"]] for tr in calib], 0) / ref
        o = {}
        for name, fn in (("TALLY", a.main), ("ESSN", a.ref)):
            P = load(fn)
            if P is None:
                continue
            o[name] = {"coherence": coherence(test, P), "voi": voi(test, P)}
        # mechanism split: questions whose drafts change vs never change
        P = load(a.main)
        if P is not None:
            pol = lambda g: ESSNPolicy(P, cc, charge_lik=False)(g)
            g = select(calib, pol, LAM, ref)
            m = evaluate(test, pol(g))
            changes = np.array([len({normalize(r["answer"]) for r in tr["rounds"][1:]}) > 1 for tr in test])
            f5 = np.array([tr["rounds"][T]["f1"] for tr in test])
            fb = {k: np.array([tr["rounds"][k]["f1"] for tr in test]) for k in range(1, T + 1)}
            kbest = max(fb, key=lambda k: np.mean([tr["rounds"][k]["f1"] for tr in calib]))
            pq = np.array(m["perq"])
            o["split"] = {grp: {"n": int(mask.sum()), "TALLY": 100 * float(pq[mask].mean()),
                                "Fixed-5": 100 * float(f5[mask].mean()), f"Fixed-{kbest}": 100 * float(fb[kbest][mask].mean())}
                          for grp, mask in (("drafts change", changes), ("drafts constant", ~changes)) if mask.any()}
            # share of the predicted value of information by term (same model, planning variants)
            Pd, Pn = load(f"{pre}_noDisc"), load(f"{pre}_noDisamb")
            if Pd and Pn:
                tot = dis = dsc = 0.0
                for tr in test:
                    k_ = key(tr)
                    for t in range(T):
                        v0 = P[k_]["V"][t][0]
                        tot += P[k_]["V"][t][1] - v0
                        dis += Pd[k_]["V"][t][1] - v0  # disambiguation only
                        dsc += Pn[k_]["V"][t][1] - v0  # discovery only
                o["terms"] = {"disamb_share": dis / max(tot, 1e-9), "disc_share": dsc / max(tot, 1e-9)}
        out[ds] = o
        print(ds, json.dumps(o, default=float)[:600], flush=True)
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    json.dump(out, open(f"{ROOT}/results/tally_analysis_{a.main}_{a.traj}.json", "w"), indent=1, default=float)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
