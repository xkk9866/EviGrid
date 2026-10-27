"""Reviewer-requested analyses on the shared trajectories.

  1. multi-seed mean/std for the learned controllers (ESSN, Stop-RAG, SIM-RAG)
  2. paired bootstrap 95% CI of the F1 difference ESSN - baseline (same test questions)
  3. selection-only comparison: given the SAME stopping time, which answer to return
  4. Learn-then-Test certification of the stop-and-select policy family indexed by the cost multiplier lambda
     (fixed-sequence testing), with p-values, calibration UCB and test CI
  5. calibration (ECE) of P_suff vs "all gold evidence retrieved" and of Q vs answer correctness
  6. gold-evidence recall of the shared retriever per round (corpus transparency)

usage: python analysis.py --traj qwen25-7b
"""
import argparse, json, math, os
from collections import Counter
import numpy as np
from scipy.stats import beta
from common import ROOT, DATASETS, T, load_traj, split, normalize
from evaluate import (ESSNPolicy, run_threshold, run_threshold_rerank, evaluate, select, hb_pvalue, premature_risk, key, pareto_auc)

KAPPA_GRID = None
LAM = [0.0] + list(np.geomspace(0.002, 2.0, 40))


def per_q_f1(trajs, pol):
    out = []
    for tr in trajs:
        tau, r, _ = pol(tr)
        out.append(tr["rounds"][r]["f1"])
    return np.array(out)


def boot_ci(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return 100 * d.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5), float((m <= 0).mean())


def cp_interval(k, n, alpha=0.05):
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y, dtype=float)
    e = 0.0
    for i in range(bins):
        m = (p >= i / bins) & (p < (i + 1) / bins + (1e-9 if i == bins - 1 else 0))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="qwen25-7b")
    a = ap.parse_args()
    P = lambda n: json.load(open(f"{ROOT}/preds/{a.traj}/{n}.json"))
    has = lambda n: os.path.exists(f"{ROOT}/preds/{a.traj}/{n}.json")
    out = {"seeds": {}, "boot": {}, "select": {}, "ltt": {}, "calib": {}, "recall": {}}
    for ds in DATASETS:
        trajs = load_traj(a.traj, ds)
        calib, test = split(trajs, "calib"), split(trajs, "test")
        tok_ref = np.mean([tr["rounds"][T]["cum_tok"] for tr in calib])
        cc = np.mean([[r["cum_tok"] for r in tr["rounds"]] for tr in calib], 0) / tok_ref
        tg = lambda taus: [(float(t), m) for t in taus for m in (0, 1)]

        def essn_mk(pe):
            pol = ESSNPolicy(pe, cc, select=True)
            return pol, [float(l) for l in LAM]

        def stop_mk(pq):
            return (lambda g: run_threshold(lambda tr, t: pq[key(tr)][t][0] - pq[key(tr)][t][1], g[0], min_t=g[1]),
                    tg(np.linspace(-0.5, 0.3, 41)))

        def sim_mk(ps):
            return (lambda g: run_threshold(lambda tr, t: ps[key(tr)][t][0], g[0], min_t=g[1]), tg(np.linspace(0.05, 1.0, 39)))

        def nosel_mk(pe):
            return ESSNPolicy(pe, cc, select=False), [float(l) for l in LAM]

        # ---------------- 1. seeds: learned baselines + the ESSN decomposition variants
        fams = {"ESSN": ("essn_full", essn_mk), "Stop-RAG": ("stoprag", stop_mk), "SIM-RAG": ("simrag", sim_mk),
                "ESSN w/o select": ("essn_full", nosel_mk),
                "ESSN-noReverify": ("essn_noReverify", essn_mk), "ESSN-noContrast": ("essn_noContrast", essn_mk),
                "ESSN-latestTarget": ("essn_latestTarget", essn_mk), "ESSN-twoHead": ("essn_twoHead", essn_mk),
                "ESSN-simple": ("essn_simple", essn_mk), "ESSN-decision": ("essn_decision", essn_mk),
                "ESSN-dsimple": ("essn_dsimple", essn_mk), "ESSN-noLik": ("essn_noLik", essn_mk),
                "ESSN-mono": ("essn_mono", essn_mk)}
        seeds, chosen, perq = {}, {}, {}
        for name, (base, mkf) in fams.items():
            vals, vecs = [], []
            for sfx in ["", "_s1", "_s2"]:
                fn = base + sfx
                if not has(fn):
                    continue
                mk, grid = mkf(P(fn))
                g = select(calib, mk, grid, tok_ref)
                m = evaluate(test, mk(g))
                auc = pareto_auc(test, mk, grid, tok_ref)[0]
                vals.append((m["F1"], m["EM"], m["Tok"], auc, m["Ret"]))
                vecs.append(np.array(m["perq"]))
                if sfx == "":
                    chosen[name] = (mk, g)
            if not vals:
                continue
            v = np.array(vals)
            sd = lambda c: v[:, c].std(ddof=1) if len(v) > 1 else 0.0
            seeds[name] = {"n": len(v), "F1_mean": v[:, 0].mean(), "F1_std": sd(0), "EM_mean": v[:, 1].mean(), "EM_std": sd(1),
                           "Tok_mean": v[:, 2].mean(), "AUC_mean": v[:, 3].mean(), "AUC_std": sd(3), "Ret_mean": v[:, 4].mean()}
            perq[name] = np.mean(vecs, 0)  # per-question F1 averaged over seeds
        out["seeds"][ds] = seeds

        # ---------------- 2. paired bootstrap. ONE protocol for every number in the paper: methods with seeds use
        # their seed-averaged per-question F1; deterministic methods use the per-question F1 that evaluate.py
        # recorded at the operating point it selected (identical to Table 1). Hence Delta = difference of table means.
        MAIN = json.load(open(f"{ROOT}/results/{a.traj}.json"))["main"][ds]
        f_es = perq["ESSN"]
        comps = {k: perq[k] for k in ("Stop-RAG", "SIM-RAG") if k in perq}
        for k in ("HALT", "HALT (all-match)", "S2G-Judge", "Stop-RAG + rerank", "SIM-RAG + rerank",
                  "Stop-RAG + LLM-lik rerank", "SIM-RAG + LLM-lik rerank", "Answer-Stability", "Fixed-2", "Fixed-3", "Fixed-5"):
            if k in MAIN and "perq" in MAIN[k]:
                comps[k] = np.array(MAIN[k]["perq"])
        out["boot"][ds] = {k: boot_ci(f_es, v) for k, v in comps.items()}
        out["perq_mean"] = out.get("perq_mean", {})
        out["perq_mean"][ds] = {k: 100 * float(np.mean(v)) for k, v in comps.items()}

        # ---------------- 3. selection-only at a common stopping time
        pe, ps, pq = P("essn_full"), P("simrag"), P("stoprag")
        prr = P("rerank") if has("rerank") else None

        # ---------------- 3b. empirical submartingale check on real test trajectories (seed-0 ESSN):
        # predicted value v_t = max_a pi_t(a), realised quality of the selected answer, latest draft's quality,
        # and how often each of them DROPS from one round to the next
        for _tag in ("essn_full", "essn_mono"):
            if not has(_tag):
                continue
            pm = P(_tag)
            mono = {"pred_value": [], "selected_f1": [], "latest_f1": [], "drop_selected": [], "drop_latest": [], "drop_pred": []}
            sel = lambda tr, u: tr["rounds"][max(pm[key(tr)]["S"][u], key=lambda x: x[1])[0]]["f1"]
            for t in range(T + 1):
                mono["pred_value"].append(100 * float(np.mean([pm[key(tr)]["V"][t][0] for tr in test])))
                mono["selected_f1"].append(100 * float(np.mean([sel(tr, t) for tr in test])))
                mono["latest_f1"].append(100 * float(np.mean([tr["rounds"][t]["f1"] for tr in test])))
                if t < T:
                    mono["drop_selected"].append(100 * float(np.mean([sel(tr, t + 1) < sel(tr, t) for tr in test])))
                    mono["drop_latest"].append(100 * float(np.mean([tr["rounds"][t + 1]["f1"] < tr["rounds"][t]["f1"] for tr in test])))
                    mono["drop_pred"].append(100 * float(np.mean([pm[key(tr)]["V"][t + 1][0] < pm[key(tr)]["V"][t][0] - 1e-6 for tr in test])))
            out.setdefault("mono", {}).setdefault(ds, {})[_tag] = mono
        pe = P("essn_full")
        sel = {}
        for stop_name, stop_fn in [("full budget (t=T)", lambda tr: T), ("ESSN stop time", lambda tr: chosen["ESSN"][0](chosen["ESSN"][1])(tr)[0])]:
            res = Counter()
            for tr in test:
                R = tr["rounds"]
                t = stop_fn(tr)
                k = key(tr)
                pool = list(range(t + 1))
                votes = Counter(normalize(R[i]["answer"]) for i in pool if normalize(R[i]["answer"]) not in ("", "unknown"))
                maj = max(pool, key=lambda i: (votes.get(normalize(R[i]["answer"]), 0), i))
                choices = {
                    "Last draft": t,
                    "Max confidence (log-prob)": max(pool, key=lambda i: R[i]["lp_mean"]),
                    "Majority vote over drafts": maj,
                    "SIM-RAG critic argmax": max(pool, key=lambda i: ps[k][i][0]),
                    "Stop-RAG Q(stop) argmax": max(pool, key=lambda i: pq[k][i][0]),
                    "ESSN posterior, cached at production round": max(pool, key=lambda i: pe[k]["S"][i][0][1]),
                    "ESSN posterior, re-verified under current evidence": max(pe[k]["S"][t], key=lambda x: x[1])[0],
                    "Oracle among drafts": max(pool, key=lambda i: R[i]["f1"]),
                    **({"Generator likelihood under current evidence": max(R[t]["lik"].items(), key=lambda x: x[1][0])[0]} if R[t].get("lik") else {}),
                    **({"Re-verification reranker (cross-encoder)": max(prr[k]["S"][t], key=lambda x: x[1])[0]} if has("rerank") else {}),
                }
                for n, i in choices.items():
                    res[n] += R[i]["f1"]
            sel[stop_name] = {n: 100 * v / len(test) for n, v in res.items()}
        out["select"][ds] = sel

        # ---------------- 4. LTT for the full stop-and-select policy family
        lt = {}
        # The certified family is fixed before any calibration data is seen. It contains every parameter value
        # of every variant of a controller: all min-retrieval variants of a threshold baseline and, for ESSN,
        # BOTH value heads (free and submartingale-constrained). Each chain is tested by fixed-sequence testing at
        # level 0.1 / #chains (Bonferroni), so the family-wise error over the whole family is <= 0.1. Choices made
        # on the same calibration questions (value head, min-retrieval variant, operating point) are choices
        # among members of this family and are therefore covered by the guarantee.
        for name in ["ESSN", "Stop-RAG", "SIM-RAG"]:
            if name == "ESSN":
                groups = {"free": (chosen["ESSN"][0], sorted(float(l) for l in LAM))}
                if "ESSN-mono" in chosen:
                    groups["mono"] = (chosen["ESSN-mono"][0], sorted(float(l) for l in LAM))
            else:
                mk0, grid = stop_mk(pq) if name == "Stop-RAG" else sim_mk(ps)
                groups = {v: (mk0, sorted([g for g in grid if g[1] == v], key=lambda g: -g[0]))
                          for v in sorted({g[1] for g in grid})}
            for eps in (0.05, 0.10):
                level = 0.1 / len(groups)
                best = None
                for d, (mk, seq) in groups.items():
                    cert = None
                    for g in seq:  # fixed-sequence: conservative -> aggressive
                        L = premature_risk(calib, mk(g))
                        p = hb_pvalue(L.mean(), len(L), eps)
                        if p <= level:
                            cert = (g, p, L.mean())
                        else:
                            break
                    if cert is not None:
                        m = evaluate(calib, mk(cert[0]))
                        if best is None or m["Tok"] < best[3]:
                            best = (cert[0], cert[1], cert[2], m["Tok"], mk, d)
                if best is None:
                    lt[f"{name}@{eps}"] = {"certified": False, "n_chains": len(groups)}
                    continue
                g, p, rcal, _, mk, member = best
                Lt = premature_risk(test, mk(g))
                lo, hi = cp_interval(int(Lt.sum()), len(Lt))
                m = evaluate(test, mk(g))
                ucb = cp_interval(int(round(rcal * len(calib))), len(calib), 0.2)[1]
                lt[f"{name}@{eps}"] = {"certified": True, "member": str(member), "n_chains": len(groups), "level": level, "param": list(g) if isinstance(g, tuple) else g, "p": p,
                                       "calib_risk": 100 * rcal, "calib_ucb90": 100 * ucb, "test_risk": 100 * Lt.mean(),
                                       "test_ci": [100 * lo, 100 * hi], "F1": m["F1"], "EM": m["EM"], "Tok": m["Tok"], "Ret": m["Ret"], "overall_err": 100 * float(np.mean(np.array(m["perq"]) < 0.5))}
        out["ltt"][ds] = lt

        # ---------------- 5. calibration
        ps_, gs_, q_, y_, s_, h_, u_, w_ = [], [], [], [], [], [], [], []
        for tr in test:
            k = key(tr)
            for t, r in enumerate(tr["rounds"]):
                if r["gold_all"] >= 0:
                    ps_.append(pe[k]["P"][t]); gs_.append(r["gold_all"])
                for ri, qv in pe[k]["S"][t]:
                    q_.append(qv); y_.append(tr["rounds"][ri]["f1"] >= 0.5)
                # selected-answer value max_a pi_t(a) vs. correctness of the answer actually selected
                ri_sel = max(pe[k]["S"][t], key=lambda x: x[1])[0]
                s_.append(pe[k]["V"][t][0]); h_.append(tr["rounds"][ri_sel]["f1"] >= 0.5)
                # answer-in-hand sufficiency 1 - pi_t(not found) vs. "some candidate is correct"
                if "suff" in pe[k]:
                    u_.append(pe[k]["suff"][t]); w_.append(max(tr["rounds"][ri]["f1"] for ri, _ in pe[k]["S"][t]) >= 0.5)
        from sklearn.metrics import roc_auc_score
        out["calib"][ds] = {"Psuff_ECE": ece(ps_, gs_), "Psuff_AUROC": roc_auc_score(gs_, ps_) if len(set(gs_)) > 1 else None,
                            "Q_ECE": ece(q_, y_), "Q_AUROC": roc_auc_score(y_, q_),
                            "Val_ECE": ece(s_, h_), "Val_AUROC": roc_auc_score(h_, s_) if len(set(h_)) > 1 else None,
                            "Suff_ECE": ece(u_, w_) if u_ else None, "Suff_AUROC": roc_auc_score(w_, u_) if len(set(w_)) > 1 else None}

        # ---------------- 6. gold-evidence recall per round (test)
        out["recall"][ds] = [100 * np.mean([max(r["gold_cov"], 0) for r in [tr["rounds"][t] for tr in test]]) for t in range(T + 1)]
        print(ds, json.dumps({k: out[k][ds] for k in out}, default=float)[:3000], flush=True)
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    json.dump(out, open(f"{ROOT}/results/analysis_{a.traj}.json", "w"), indent=1, default=float)
    print("saved")


if __name__ == "__main__":
    main()
