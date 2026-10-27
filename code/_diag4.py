"""Where is the oracle gap? Selection errors at T of the best existing posterior (v9), categorised."""
import json, numpy as np
from collections import Counter
from common import load_traj, split, normalize, T
from essn import candidates

ess = json.load(open('../preds/qwen25-7b/essn_v9.json'))
for ds in ['hotpotqa', '2wiki', 'musique']:
    trs = split(load_traj('qwen25-7b', ds), 'test')
    c = Counter()
    gap = Counter()
    for tr in trs:
        R = tr['rounds']
        C = candidates(R, T)
        f = {ci: R[ci]['f1'] for ci in C}
        S = ess[f"{ds}::{tr['qid']}"]['S'][T]
        sel = max(S, key=lambda x: x[1])[0]
        best = max(f.values())
        c['n'] += 1
        c['ncand'] += len(C)
        if best >= 0.5:
            c['inhand'] += 1
            if f[sel] < 0.5:
                c['selerr'] += 1
                bi = max(C, key=lambda ci: f[ci])
                cnt = sum(normalize(r['answer']) == normalize(R[bi]['answer']) for r in R)
                cs = sum(normalize(r['answer']) == normalize(R[sel]['answer']) for r in R)
                c['selerr_bestIsLast'] += int(normalize(R[bi]['answer']) == normalize(R[T]['answer']))
                c['selerr_selIsLast'] += int(normalize(R[sel]['answer']) == normalize(R[T]['answer']))
                c['selerr_bestCntGt'] += int(cnt > cs)
                c['selerr_bestCntLt'] += int(cnt < cs)
                c['selerr_selUnknown'] += int(normalize(R[sel]['answer']) in ('unknown', ''))
                c['selerr_bestBorn0'] += int(min(t for t, r in enumerate(R) if normalize(r['answer']) == normalize(R[bi]['answer'])) == 0)
        gap['oracle'] += best
        gap['sel'] += f[sel]
        gap['partial'] += (best - f[sel]) * (best < 0.5 or f[sel] >= 0.5)
    n = c['n']
    print(ds, {k: (round(v / n, 3) if k == 'ncand' else v) for k, v in c.items()},
          {k: round(100 * v / n, 1) for k, v in gap.items()})
