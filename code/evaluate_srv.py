"""Offline evaluation of all stopping policies on shared trajectories.

Every policy maps a trajectory to (stop round tau, returned answer round, extra cost).
Free parameters (thresholds / lambda) are selected on the calibration split with the SAME rule for
every method: maximise  U = F1 - KAPPA * Tok / Tok(Fixed-T).  We also report the Pareto
area (F1 vs tokens) and Learn-then-Test risk control.

usage: python evaluate.py --traj qwen25-7b [--essn full,noCNN,noReverify,latestTarget,...]
"""
import argparse, json, math, os
import numpy as np
from scipy.stats import binom
from common import ROOT, DATASETS, T, load_traj, split, normalize

KAPPA = 0.1


# ----------------------------------------------------------------------------- policies
def run_fixed(tr, k):
    return k, k, 0


def run_threshold(score_fn, tau, min_t=0, extra_fn=None):
    def f(tr):
        extra = 0
        for t in range(T + 1):
            if extra_fn:
                extra += extra_fn(tr, t)
            if t >= min_t and score_fn(tr, t) >= tau:
                return t, t, extra
        return T, T, extra
    return f


def run_threshold_select(score_fn, tau, min_t=0):
    """threshold stopping + return the draft with the highest score among drafts seen so far"""
    def f(tr):
        for t in range(T + 1):
            if t >= min_t and score_fn(tr, t) >= tau:
                break
        return t, max(range(t + 1), key=lambda i: score_fn(tr, i)), 0
    return f


def agreement(tr, t):
    r = tr["rounds"][t]
    g = normalize(r["answer"])
    s = [normalize(x) for x in r.get("samples", [])]
    return (1 + sum(x == g for x in s)) / (1 + len(s))


def run_threshold_rerank(score_fn, tau, S, min_t=0, extra_fn=None):
    """existing stopper + re-verification reranker: stop by the baseline rule, return the candidate that the
    reranker scores highest under the evidence available at the stopping round"""
    base = run_threshold(score_fn, tau, min_t=min_t, extra_fn=extra_fn)

    def f(tr):
        t, _, extra = base(tr)
        return t, max(S[key(tr)]["S"][t], key=lambda x: x[1])[0], extra
    return f


def judge_cost(tr, t):
    return tr["rounds"][t].get("judge_ptoks", 0) + 1


def sc_cost(tr, t):
    return tr["rounds"][t].get("sample_ctoks", 0)


class ESSNPolicy:
    def __init__(self, preds, cost_curve, select=True, profile=True, charge_lik=True):
        self.p, self.c, self.select, self.profile, self.charge = preds, cost_curve, select, profile, charge_lik

    def __call__(self, param):
        lam = param
        def f(tr):
            pr = self.p[key(tr)]
            V = np.array(pr["V"])
            for t in range(T + 1):
                if t == T:
                    break
                if self.profile:
                    cont = max(V[t, k] - lam * (self.c[t + k] - self.c[t]) for k in range(1, T - t + 1))
                    if V[t, 0] >= cont:
                        break
                elif V[t, 0] >= lam:
                    break
            # extra cost: generator-side candidate scoring of every round up to the stop (prefix KV reused)
            extra = sum(tr["rounds"][u].get("lik_tok", 0) for u in range(t + 1)) if (self.charge and pr.get("lik_used")) else 0
            extra += sum(pr.get("extra_tok", [0] * (T + 1))[:t + 1])  # full-field scoring of newborn drafts
            if not self.select:
                return t, t, extra
            # stop-and-select: candidate with the highest value under the CURRENT evidence
            r, q = max(pr["S"][t], key=lambda x: x[1])
            return t, r, extra
        return f


def key(tr):
    return f"{tr['dataset']}::{tr['qid']}"


# ----------------------------------------------------------------------------- metrics
def evaluate(trajs, policy):
    em = f1 = acc = rounds = tok = 0.0
    early = over = 0
    perq = []
    for tr in trajs:
        tau, ans_t, extra = policy(tr)
        r = tr["rounds"][ans_t]
        em += r["em"]; f1 += r["f1"]; acc += r["acc"]
        perq.append(round(r["f1"], 4))
        rounds += tau
        tok += tr["rounds"][tau]["cum_tok"] + extra
        later = max(x["f1"] for x in tr["rounds"][tau:])
        early += r["f1"] < 0.5 <= later
        over += any(x["f1"] >= 0.5 for x in tr["rounds"][:tau]) and r["f1"] < 0.5
    n = len(trajs)
    return {"perq": perq, "EM": 100 * em / n, "F1": 100 * f1 / n, "Acc": 100 * acc / n, "Ret": rounds / n, "Tok": tok / n,
            "early": 100 * early / n, "over": 100 * over / n}


def premature_risk(trajs, policy):
    losses = []
    for tr in trajs:
        tau, ans_t, _ = policy(tr)
        r = tr["rounds"][ans_t]
        losses.append(float(r["f1"] < 0.5 and any(x["f1"] >= 0.5 for x in tr["rounds"][tau + 1:])))
    return np.array(losses)


def hb_pvalue(rhat, n, eps):
    """Hoeffding-Bentkus p-value for H0: R > eps (Bates et al. / Angelopoulos et al. LTT)."""
    def h(a, b):
        a = min(max(a, 1e-12), 1 - 1e-12)
        return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))
    p_h = math.exp(-n * h(min(rhat, eps), eps)) if rhat < eps else 1.0
    p_b = math.e * binom.cdf(math.ceil(n * rhat), n, eps)
    return min(p_h, p_b)


def ltt_select(calib, make_policy, grid_safe_to_risky, eps, delta):
    """Fixed-sequence testing from the most conservative parameter to the cheapest one."""
    chosen = None
    for g in grid_safe_to_risky:
        loss = premature_risk(calib, make_policy(g))
        if hb_pvalue(loss.mean(), len(loss), eps) <= delta:
            chosen = g
        else:
            break
    return chosen


# ----------------------------------------------------------------------------- main
def select(calib, make_policy, grid, tok_ref):
    best, best_u = None, -1e9
    for g in grid:
        m = evaluate(calib, make_policy(g))
        u = m["F1"] - KAPPA * 100 * m["Tok"] / tok_ref
        if u > best_u:
            best, best_u = g, u
    return best


def pareto_auc(test, make_policy, grid, tok_ref):
    pts = []
    for g in grid:
        m = evaluate(test, make_policy(g))
        pts.append((m["Tok"] / tok_ref, m["F1"]))
    pts.sort()
    xs, ys, best = [0.0], [0.0], 0.0
    for x, y in pts:
        best = max(best, y)
        xs.append(min(x, 1.0)); ys.append(best)
    xs.append(1.0); ys.append(best)
    return float(np.trapezoid(ys, xs)), pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="qwen25-7b")
    ap.add_argument("--essn", default="full")
    ap.add_argument("--essn_preds_from", default="")  # transfer: ESSN trained on other traj
    ap.add_argument("--out", default="")
    ap.add_argument("--main", default="full")  # which ESSN prediction file is the main model
    a = ap.parse_args()
    sfx = f"_from_{a.essn_preds_from}" if a.essn_preds_from else ""
    # trained controllers transferred from another generator carry a suffix; zero-shot LLM controllers do not
    fn = lambda name: f"{ROOT}/preds/{a.traj}/{name}{'' if name in ('halt', 's2g') else sfx}.json"
    P = lambda name: json.load(open(fn(name)))
    have = lambda name: os.path.exists(fn(name))
    res, risk, curves, costmatch = {}, {}, {}, {}
    for ds in DATASETS:
        try:
            trajs = load_traj(a.traj, ds)
        except FileNotFoundError:
            continue
        calib, test = split(trajs, "calib"), split(trajs, "test")
        tok_ref = np.mean([tr["rounds"][T]["cum_tok"] for tr in calib])
        cost_curve = np.mean([[r["cum_tok"] for r in tr["rounds"]] for tr in calib], 0) / tok_ref
        R = {}
        for k in range(T + 1):
            R[f"Fixed-{k}"] = evaluate(test, lambda tr, k=k: run_fixed(tr, k))
        qgrid = lambda vals: sorted(set(np.quantile(vals, np.linspace(0, 1, 41)).tolist()))
        tg = lambda taus: [(float(t), m) for t in taus for m in (0, 1)]  # (threshold, min rounds)
        # each entry: (make_policy(param), grid, riskiness(param)) ; riskiness increases as the policy stops earlier
        methods = {
            "Conf-Threshold": (lambda g: run_threshold(lambda tr, t: tr["rounds"][t]["lp_mean"], g[0], min_t=g[1]),
                               tg(qgrid([r["lp_mean"] for tr in calib for r in tr["rounds"]])), lambda g: -g[0]),
            "Answer-Stability": (lambda m: run_threshold(lambda tr, t: tr["rounds"][t]["same_prev"], 0.5, min_t=m), [1, 2, 3],
                                 lambda m: -m),
            "Self-Consistency": (lambda g: run_threshold(agreement, g[0], min_t=g[1], extra_fn=sc_cost),
                                 tg([0.4, 0.6, 0.8, 1.0, 1.01]), lambda g: -g[0]),
            "LLM-Judge": (lambda g: run_threshold(lambda tr, t: tr["rounds"][t]["judge_yes"], g[0], min_t=g[1], extra_fn=judge_cost),
                          tg(qgrid([r["judge_yes"] for tr in calib for r in tr["rounds"]])), lambda g: -g[0]),
        }
        if have("adaptive"):
            pa = P("adaptive")
            mp = {0: 0, 1: 1, 2: T}
            R["Adaptive-RAG"] = evaluate(test, lambda tr: (mp[int(np.argmax(pa[key(tr)]))],) * 2 + (0,))
        if have("simrag"):
            ps = P("simrag")
            methods["SIM-RAG"] = (lambda g: run_threshold(lambda tr, t: ps[key(tr)][t][0], g[0], min_t=g[1]),
                                  tg(np.linspace(0.05, 1.0, 39)), lambda g: -g[0])
        if have("simrag"):
            methods["SIM-RAG (+select)"] = (lambda g: run_threshold_select(lambda tr, t: ps[key(tr)][t][0], g[0], min_t=g[1]),
                                            tg(np.linspace(0.05, 1.0, 39)), lambda g: -g[0])
        if have("stoprag"):
            pq = P("stoprag")
            methods["Stop-RAG"] = (lambda g: run_threshold(lambda tr, t: pq[key(tr)][t][0] - pq[key(tr)][t][1], g[0], min_t=g[1]),
                                   tg(np.linspace(-0.5, 0.3, 41)), lambda g: -g[0])
        if any("lik" in r for tr in calib[:5] for r in tr["rounds"]):
            LR = {key(tr): {"S": [[[ci, sv] for ci, (sv, n) in tr["rounds"][t].get("lik", {}).items()] or [[t, 0.0]]
                                  for t in range(T + 1)]} for tr in calib + test}
            lc = lambda tr, t: tr["rounds"][t].get("lik_tok", 0)
            if have("stoprag"):
                methods["Stop-RAG + LLM-lik rerank"] = (lambda g: run_threshold_rerank(lambda tr, t: pq[key(tr)][t][0] - pq[key(tr)][t][1], g[0], LR, min_t=g[1], extra_fn=lc),
                                                        tg(np.linspace(-0.5, 0.3, 41)), lambda g: -g[0])
            if have("simrag"):
                methods["SIM-RAG + LLM-lik rerank"] = (lambda g: run_threshold_rerank(lambda tr, t: ps[key(tr)][t][0], g[0], LR, min_t=g[1], extra_fn=lc),
                                                       tg(np.linspace(0.05, 1.0, 39)), lambda g: -g[0])
        if have("stoprag"):
            methods["Stop-RAG (+select)"] = (lambda g: run_threshold_select(lambda tr, t: pq[key(tr)][t][0] - pq[key(tr)][t][1], g[0], min_t=g[1]),
                                             tg(np.linspace(-0.5, 0.3, 41)), lambda g: -g[0])
        if have("halt"):
            ph = P("halt")
            hc = lambda tr, t: ph[key(tr)]["tok"][t]
            # HALT default = all_match (threshold 1.0); the sweep over the matched-claim fraction gives its frontier
            methods["HALT"] = (lambda g: run_threshold(lambda tr, t: ph[key(tr)]["match_frac"][t], g[0], min_t=g[1], extra_fn=hc),
                               tg([1.0, 0.75, 0.67, 0.5, 0.34, 0.25, 0.01]), lambda g: -g[0])
            R["HALT (all-match)"] = evaluate(test, run_threshold(lambda tr, t: ph[key(tr)]["match_frac"][t], 1.0, extra_fn=hc))
        if have("s2g"):
            pg = P("s2g")
            gc = lambda tr, t: pg[key(tr)]["tok"][t]
            methods["S2G-Judge"] = (lambda g: run_threshold(lambda tr, t: pg[key(tr)]["p"][t], g[0], min_t=g[1], extra_fn=gc),
                                    tg(qgrid([v for x in pg.values() for v in x["p"]])), lambda g: -g[0])
        if have("rerank"):
            prr = P("rerank")
            if have("simrag"):
                methods["SIM-RAG + rerank"] = (lambda g: run_threshold_rerank(lambda tr, t: ps[key(tr)][t][0], g[0], prr, min_t=g[1]),
                                               tg(np.linspace(0.05, 1.0, 39)), lambda g: -g[0])
            if have("stoprag"):
                methods["Stop-RAG + rerank"] = (lambda g: run_threshold_rerank(lambda tr, t: pq[key(tr)][t][0] - pq[key(tr)][t][1], g[0], prr, min_t=g[1]),
                                                tg(np.linspace(-0.5, 0.3, 41)), lambda g: -g[0])
            if have("halt"):
                methods["HALT + rerank"] = (lambda g: run_threshold_rerank(lambda tr, t: ph[key(tr)]["match_frac"][t], g[0], prr, min_t=g[1], extra_fn=hc),
                                            tg([1.0, 0.75, 0.67, 0.5, 0.34, 0.25, 0.01]), lambda g: -g[0])
        lam_grid = list(np.concatenate([[0.0], np.geomspace(0.002, 2.0, 40)]))
        for name in a.essn.split(","):
            if not have(f"essn_{name}"):
                continue
            pe = P(f"essn_{name}")
            label = "ESSN" if name == a.main else f"ESSN-{name}"
            ch = "noLik" not in name
            pol = ESSNPolicy(pe, cost_curve, select=name != "noSelect", profile="noProfile" not in name, charge_lik=ch)
            if pol.profile:
                methods[label] = (pol, lam_grid, lambda g: g)
            else:
                methods[label] = (pol, list(np.linspace(0.0, 1.0, 41)), lambda g: -g)
            if name == a.main:
                methods["ESSN w/o select"] = (ESSNPolicy(pe, cost_curve, select=False, charge_lik=ch), lam_grid, lambda g: g)
        chosen = {}
        for name, (mk, grid, _) in methods.items():
            g = select(calib, mk, grid, tok_ref)
            chosen[name] = g
            R[name] = evaluate(test, mk(g))
            R[name]["param"] = g
            auc, curves[f"{ds}::{name}"] = pareto_auc(test, mk, grid, tok_ref)
            R[name]["AUC"] = auc
        cm = {}
        for name in methods:
            pts = curves[f"{ds}::{name}"]
            ok = [x for x, y in pts if y >= R[f"Fixed-{T}"]["F1"]]
            cm[name] = min(ok) if ok else 9.0
        costmatch[ds] = cm
        R["Oracle"] = evaluate(test, lambda tr: (lambda t: (t, t, 0))(int(np.argmax([r["f1"] for r in tr["rounds"]]))))
        # upper bound of evidence-coverage stopping: stop at the first round at which ALL annotated supporting
        # documents have been retrieved (T if never) and return the draft of that round; not deployable
        onset = lambda tr: next((t for t, r in enumerate(tr["rounds"]) if r["gold_all"] == 1), T)
        R["Oracle-Sufficiency"] = evaluate(test, lambda tr: (onset(tr), onset(tr), 0))
        res[ds] = R
        # ---------------- risk control (LTT) for methods with a monotone knob
        # The certified family of a controller is fixed BEFORE looking at calibration data: every value of its
        # parameter, every min-retrieval variant and, for ESSN, both value heads (free and constrained). Each
        # chain (one variant / one head) is tested by fixed-sequence testing at level delta / #chains
        # (Bonferroni), so the family-wise error is <= delta over the whole family. Any data-dependent choice
        # among the certified members (here: calibration utility) is then covered by the guarantee, including
        # the choice of the value head and of the min-retrieval variant made on the same calibration questions.
        rk = {}
        DELTA = 0.1
        # certification uses the fresh calibration split calib2, which no development decision has ever touched
        # (models, heads, ablations and operating points were chosen on calib); calib if calib2 is unavailable
        cert_set = split(trajs, "calib2") or calib
        fam = {n: [n] for n in methods if n not in ("Answer-Stability",) and not n.startswith("ESSN-")}
        for name, members in fam.items():
            chains = []
            for mname in members:
                mk, grid, riskiness = methods[mname]
                variants = sorted({g[1] for g in grid if isinstance(g, tuple)}) or [None]
                for v in variants:
                    sub = [g for g in grid if v is None or g[1] == v]
                    chains.append((mname, mk, sorted(sub, key=riskiness)))
            for eps in (0.05, 0.10):
                cert = []
                for mname, mk, g_sorted in chains:
                    for g in g_sorted:
                        loss = premature_risk(cert_set, mk(g))
                        if hb_pvalue(loss.mean(), len(loss), eps) <= DELTA / len(chains):
                            cert.append((mname, mk, g, hb_pvalue(loss.mean(), len(loss), eps), loss.mean()))
                        else:
                            break
                if not cert:
                    rk[f"{name}@{eps}"] = {"feasible": False, "n_chains": len(chains)}
                    continue
                def util(c):
                    m = evaluate(cert_set, c[1](c[2]))
                    return m["F1"] - KAPPA * 100 * m["Tok"] / tok_ref
                mname, mk, g, pv, rcal = max(cert, key=util)
                m = evaluate(test, mk(g))
                rk[f"{name}@{eps}"] = {"feasible": True, "member": mname, "n_chains": len(chains), "p": pv, "param": g,
                                       "n_cert": len(cert_set), "n_certified": len(cert),
                                       "risk_cal": 100 * rcal, "risk": 100 * premature_risk(test, mk(g)).mean(),
                                       "overall_err": 100 - m["EM"], "EM": m["EM"],
                                       "F1": m["F1"], "Tok": m["Tok"], "Ret": m["Ret"]}
        risk[ds] = rk
        print(f"\n=== {ds} (n_test={len(test)}, Tok(Fixed-{T})={tok_ref:.0f})")
        for name, m in R.items():
            print(f"{name:22s} EM {m['EM']:5.1f} F1 {m['F1']:5.1f} Acc {m['Acc']:5.1f} Ret {m['Ret']:.2f} "
                  f"Tok {m['Tok']:7.0f} early {m['early']:4.1f} over {m['over']:4.1f} AUC {m.get('AUC', float('nan')):5.1f}")
    out = a.out or f"{ROOT}/results/{a.traj}{'_from_' + a.essn_preds_from if a.essn_preds_from else ''}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"main": res, "risk": risk, "curves": curves, "costmatch": costmatch}, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
