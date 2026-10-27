"""Merge partial rollouts (and their likelihood fields) into one trajectory set per generator.

usage: python merge_traj.py <target_tag> <part_tag> [<part_tag> ...]
Trajectories are keyed by (split, qid); a part never overwrites an existing entry of another split.
"""
import gzip, json, os, sys
from common import ROOT, DATASETS


def rd(path):
    if not os.path.exists(path):
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main(target, parts):
    for ds in DATASETS:
        base = rd(f"{ROOT}/traj/{target}/{ds}.jsonl.gz")
        have = {(t["split"], t["qid"]) for t in base}
        lik_p = f"{ROOT}/data/lik/{target}/{ds}.json.gz"
        lik = json.load(gzip.open(lik_p, "rt", encoding="utf-8")) if os.path.exists(lik_p) else {}
        added = 0
        for p in parts:
            for t in rd(f"{ROOT}/traj/{p}/{ds}.jsonl.gz"):
                if (t["split"], t["qid"]) not in have:
                    base.append(t); have.add((t["split"], t["qid"])); added += 1
            pl = f"{ROOT}/data/lik/{p}/{ds}.json.gz"
            if os.path.exists(pl):
                for k, v in json.load(gzip.open(pl, "rt", encoding="utf-8")).items():
                    lik.setdefault(k, v)
        os.makedirs(f"{ROOT}/traj/{target}", exist_ok=True)
        with gzip.open(f"{ROOT}/traj/{target}/{ds}.jsonl.gz", "wt", encoding="utf-8") as f:
            for t in base:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        if lik:
            os.makedirs(f"{ROOT}/data/lik/{target}", exist_ok=True)
            with gzip.open(lik_p, "wt", encoding="utf-8") as f:
                json.dump(lik, f)
        print(target, ds, "added", added, "total", len(base), "lik", len(lik), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
