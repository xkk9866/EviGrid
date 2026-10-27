"""Is document-level evidence sufficiency a good *selection/stop* signal if known exactly?
Policy 'onset': return the draft of the first round at which all gold supporting documents are retrieved (T if never)."""
import numpy as np
from common import load_traj, split, normalize, T

for ds in ['hotpotqa', '2wiki', 'musique']:
    trs = split(load_traj('qwen25-7b', ds), 'test')
    res = {k: [] for k in ['last', 'onset', 'onset+1', 'onset_ret', 'never', 'f_onset_given_suff', 'f_last_given_suff']}
    for tr in trs:
        R = tr['rounds']
        on = next((t for t, r in enumerate(R) if r['gold_all'] == 1), None)
        res['last'].append(R[T]['f1'])
        res['never'].append(on is None)
        s = T if on is None else on
        res['onset'].append(R[s]['f1'])
        res['onset+1'].append(R[min(s + 1, T)]['f1'])
        res['onset_ret'].append(s)
        if on is not None:
            res['f_onset_given_suff'].append(R[on]['f1'])
            res['f_last_given_suff'].append(R[T]['f1'])
    print(ds, {k: round(float(np.mean(v)) * (1 if k in ('onset_ret',) else 100), 1) for k, v in res.items()})
