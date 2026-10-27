"""Shared utilities: trajectory loading, answer metrics, per-state labels and costs."""
import gzip, json, os, re, string
from collections import Counter

ROOT = os.environ.get("ESRAG_ROOT", os.path.expanduser("~/research/esrag"))
DATASETS = ["hotpotqa", "2wiki", "musique"]
T = 5


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc)


def scores(pred, golds):
    em = max(float(normalize(pred) == normalize(g)) for g in golds)
    fs = max(f1(pred, g) for g in golds)
    acc = max(float(normalize(g) in normalize(pred)) for g in golds if normalize(g)) if any(normalize(g) for g in golds) else 0.0
    return em, fs, acc


def load_traj(tag, ds):
    with gzip.open(f"{ROOT}/traj/{tag}/{ds}.jsonl.gz", "rt", encoding="utf-8") as f:
        trajs = [json.loads(l) for l in f]
    for tr in trajs:
        gold = set(tr.get("gold_titles", []))
        seen, cum, cprobe = set(), 0, 0
        prev = None
        if tr.get("native_answer") is not None:  # agent trajectories: the agent's own stop and answer
            tr["native_em"], tr["native_f1"], tr["native_acc"] = scores(tr["native_answer"], tr["answers"])
        for r in tr["rounds"]:
            r["em"], r["f1"], r["acc"] = scores(r["answer"], tr["answers"])
            seen |= set(r["new_titles"])
            r["gold_cov"] = (len(gold & seen) / len(gold)) if gold else -1.0
            r["gold_all"] = float(bool(gold) and gold <= seen) if gold else -1.0
            # answer probes of agent trajectories are part of every draft-based controller's cost
            cum += r["ptoks"] + r["ctoks"] + r.get("probe_toks", 0)
            cprobe += r.get("probe_toks", 0)
            r["cum_tok"], r["cum_probe"] = cum, cprobe
            r["same_prev"] = float(prev is not None and normalize(prev) == normalize(r["answer"]))
            prev = r["answer"]
    # generator-side likelihood field (lik_score.py), if available: row t scores every candidate under E_t
    lp = f"{ROOT}/data/lik/{tag}/{ds}.json.gz"
    if os.path.exists(lp):
        with gzip.open(lp, "rt", encoding="utf-8") as f:
            lik = json.load(f)
        for tr in trajs:
            L = lik.get(tr["qid"])
            for t, r in enumerate(tr["rounds"]):
                r["lik"] = {int(ci): (s, n) for ci, s, n in L["rows"][t]} if L else {}
                r["lik_tok"] = L["tok"][t] if L else 0
    return trajs


def split(trajs, name):
    """name may list several splits, e.g. "calib,test,calib2" (the development pool of the final protocol)"""
    names = set(name.split(","))
    return [t for t in trajs if t["split"] in names]
