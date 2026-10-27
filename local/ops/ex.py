"""examples where InHand (e12_m) at its selected operating point returns a correct earlier draft while Fixed-5 is wrong"""
import json, sys, gzip
import numpy as np
sys.path.insert(0, ".")
from common import load_traj, split, normalize, T, ROOT
from evaluate import ESSNPolicy, select, key
LAM = [0.0] + list(np.geomspace(0.002, 2.0, 40))
P = json.load(open("../preds/qwen25-7b/essn_e12_m.json"))
for ds in ["hotpotqa", "2wiki"]:
    trajs = load_traj("qwen25-7b", ds)
    dev, test = split(trajs, "calib,test,calib2"), split(trajs, "test2")
    ref = np.mean([tr["rounds"][T]["cum_tok"] for tr in dev])
    cc = np.mean([[r["cum_tok"] for r in tr["rounds"]] for tr in dev], 0) / ref
    pol = ESSNPolicy(P, cc)
    lam = select(dev, pol, LAM, ref)
    f = pol(lam)
    print(ds, "lambda", lam)
    n = 0
    corpus = None
    for tr in test:
        R = tr["rounds"]
        t, r, extra = f(tr)
        if R[T]["f1"] > 0 or R[r]["f1"] < 1 or r == t or len(tr["question"]) > 100:
            continue
        nd = len({normalize(x["answer"]) for x in R[1:]})
        if nd < 3:
            continue
        if corpus is None:
            corpus = [json.loads(l) for l in gzip.open(f"{ROOT}/data/proc/{ds}/corpus.jsonl.gz", "rt", encoding="utf-8")]
        pr = P[key(tr)]
        print("=" * 70)
        print(tr["qid"], tr["question"], "| gold", tr.get("answers") or tr.get("answer"), "| stop", t, "ret", r, "tok", R[t]["cum_tok"] + extra)
        for u, x in enumerate(R):
            print(f" t={u} new={[corpus[p]['title'] for p in x['new_pids']]} draft={x['answer'][:40]!r} f1={x['f1']:.2f} tok={x['cum_tok']}")
            print("    S=", pr["S"][u], "suff=", pr["suff"][u], "V=", [round(v, 3) for v in pr["V"][u]], "rates=", pr.get("rates", [[None]])[u] if "rates" in pr else "")
        n += 1
        if n >= 4:
            break
