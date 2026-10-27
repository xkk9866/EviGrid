import json, sys, os
sys.path.insert(0, ".")
from common import load_traj, split
from collections import Counter
for ds in ["hotpotqa", "2wiki", "musique"]:
    tr = load_traj("qwen25-7b", ds)
    print(ds, Counter(t["split"] for t in tr))
    for n in ["essn_tally_m", "essn_e12_m", "simrag", "stoprag", "simrag_s2", "stoprag_s2", "halt", "s2g"]:
        f = "../preds/qwen25-7b/" + n + ".json"
        if not os.path.exists(f):
            continue
        P = json.load(open(f))
        print("  ", n, len(P), {s: sum((ds + "::" + t["qid"]) in P for t in split(tr, s)) for s in ["calib", "test", "calib2", "cert", "test2"]})
