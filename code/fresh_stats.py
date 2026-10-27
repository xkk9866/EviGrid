"""Controller-free statistics of the fresh test split: fixed budgets, over-retrieval, oracle selection, perfect
document-level sufficiency. usage: python fresh_stats.py <traj> [split]"""
import sys, numpy as np
from common import load_traj, split, T, DATASETS

traj = sys.argv[1]
sp = sys.argv[2] if len(sys.argv) > 2 else "test2"
for ds in DATASETS:
    try:
        te = split(load_traj(traj, ds), sp)
    except FileNotFoundError:
        continue
    if not te:
        continue
    F = np.array([[r["f1"] for r in tr["rounds"]] for tr in te])
    fixed = 100 * F.mean(0)
    over = 100 * np.mean([(F[i, :T].max() >= 0.5) and F[i, T] < 0.5 for i in range(len(te))])
    oracle = 100 * F.max(1).mean()
    on = [next((t for t, r in enumerate(tr["rounds"]) if r["gold_all"] == 1), T) for tr in te]
    suff = 100 * np.mean([F[i, on[i]] for i in range(len(te))])
    print(ds, len(te), "fixed", np.round(fixed, 1), "best fixed", round(fixed[1:].max(), 1), f"over-retrieval@T {over:.1f}",
          f"oracle {oracle:.1f}", f"perfect-sufficiency {suff:.1f} (ret {np.mean(on):.2f})", flush=True)
