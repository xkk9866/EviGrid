"""Find real trajectories for the case study / Fig. 1: a correct draft is overwritten later,
and the calibrated EviGrid posterior still puts its mass on the correct earlier draft."""
import gzip, json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import f1, normalize
R = os.path.join(os.path.dirname(__file__), "..", "release")
P = json.load(open(f"{R}/preds/essn_p_full_cf.json"))
out = []
for ds in ["hotpotqa", "2wiki", "musique"]:
    for line in gzip.open(f"{R}/data/traj/{ds}.jsonl.gz", "rt", encoding="utf8"):
        tr = json.loads(line)
        k = f"{ds}::{tr['qid']}"
        if k not in P: continue
        rs = tr["rounds"]; gold = tr["answers"]
        q = [max(f1(r["answer"], g) for g in gold) for r in rs]
        S = P[k]["S"]; V = P[k]["V"]
        if len(rs) < 6 or q[-1] > 0 or max(q) < 1: continue
        first_ok = next(t for t in range(6) if q[t] >= 1)
        if first_ok > 3: continue
        cand = S[-1]; best = max(cand, key=lambda x: x[1])
        if q[best[0]] < 1 or len(cand) > 3: continue
        if len(tr["question"]) > 95: continue
        out.append({"key": k, "split": tr.get("split"), "q": tr["question"], "gold": gold, "drafts": [r["answer"] for r in rs], "q_f1": q,
                    "titles": [r["new_titles"] for r in rs], "reason": [r["reasoning"] for r in rs], "S": S, "V": V})
print(len(out))
json.dump(out[:400], open(f"{R}/../results/case_candidates.json", "w"), indent=1)
for o in out[:25]:
    print("----", o["key"], o["split"], "|", o["q"], "| gold", o["gold"])
    for t in range(6): print("  r", t, repr(o["drafts"][t]), o["q_f1"][t], o["titles"][t], "| S:", o["S"][t], "| v/V:", o["V"][t][:3])
