"""Which candidate-level generator signals carry the selection power on each dataset? (no encoder, seconds)
Softmax-linear and small-MLP selectors over candidate features at every round t (trained on train, all t),
selection F1 on test at T and averaged over t."""
import math, random, sys, numpy as np, torch, torch.nn as nn
from common import load_traj, split, normalize, T
from essn import candidates

torch.manual_seed(0); random.seed(0)


def rowp(R, t):
    L = R[t].get("lik", {})
    if not L:
        return {}
    m = max(s for s, _ in L.values())
    w = {ci: math.exp(s - m) for ci, (s, _) in L.items()}
    Z = sum(w.values())
    return {ci: v / Z for ci, v in w.items()}


def feats(tr, t, groups):
    R = tr["rounds"]
    C = candidates(R, t)
    rp = [rowp(R, u) for u in range(t + 1)]
    X, y = [], []
    for ci in C:
        L = R[t].get("lik", {})
        sv, n = L.get(ci, (-30.0, 1))
        sv = max(sv, -30.0)
        birth = min(u for u in range(t + 1) if normalize(R[u]["answer"]) == normalize(R[ci]["answer"]))
        hist = [rp[u].get(ci2, 0.0) for u in range(birth, t + 1)
                for ci2 in [max([c for c in rp[u]] + [-1], key=lambda c: c if c >= 0 and normalize(R[c]["answer"]) == normalize(R[ci]["answer"]) else -1)]]
        hist = [h for h in hist] or [0.0]
        f = {"lik": [sv / 5, sv / max(n, 1) / 2],
             "rho": [rp[t].get(ci, 0.0)],
             "hist": [float(np.mean(hist)), hist[-1] - hist[0]],
             "cur": [float(ci == t)], "age": [(t - birth) / T],
             "cnt": [sum(normalize(R[u]["answer"]) == normalize(R[ci]["answer"]) for u in range(t + 1)) / (t + 1)],
             "lp": [R[birth]["lp_mean"] / 5, R[birth]["lp_min"] / 10]}
        X.append(sum((f[g] for g in groups), []))
        y.append(R[ci]["f1"])
    return torch.tensor(X), torch.tensor(y)


GROUPS = [["lik"], ["lik", "rho"], ["lik", "rho", "hist"], ["lik", "cur", "age"], ["lik", "cnt"],
          ["lik", "lp"], ["lik", "rho", "hist", "cur", "age", "cnt", "lp"]]
for ds in sys.argv[1].split(","):
    trs = load_traj("qwen25-7b", ds)
    tr_, te = split(trs, "train"), split(trs, "test")
    for groups in GROUPS:
        D = [feats(tr, t, groups) for tr in tr_ for t in range(T + 1)]
        E = [[feats(tr, t, groups) for t in range(T + 1)] for tr in te]
        for kind in ("lin", "mlp"):
            dim = D[0][0].shape[1]
            m = nn.Linear(dim, 1) if kind == "lin" else nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 1))
            opt = torch.optim.Adam(m.parameters(), 0.01)
            for ep in range(3):
                random.shuffle(D)
                for i in range(0, len(D), 64):
                    loss = 0
                    for X, y in D[i:i + 64]:
                        z = y.max()
                        tg = z * y / y.sum().clamp(min=1e-6) if z > 0 else torch.ones_like(y) / len(y)
                        loss = loss - (tg * torch.log_softmax(m(X).squeeze(-1), 0)).sum()
                    opt.zero_grad(); (loss / 64).backward(); opt.step()
            with torch.no_grad():
                sel = np.array([[y[int(m(X).squeeze(-1).argmax())].item() for X, y in q] for q in E])
            print(ds, kind, "+".join(groups), "selT", round(100 * sel[:, T].mean(), 1), "sel_avg_t", round(100 * sel.mean(), 1), flush=True)
