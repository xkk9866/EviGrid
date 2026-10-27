"""How much does the generator's full likelihood field (every candidate under every nested evidence state) carry?
A tiny row-wise sequence model over the field only, trained on train, selection F1 on test at every t."""
import sys, math, random, numpy as np, torch, torch.nn as nn
from common import load_traj, split, normalize, T
from essn_bf import hypotheses, field

ds = sys.argv[1]
torch.manual_seed(0); random.seed(0)


def rows(tr, causal=True):
    R = tr["rounds"]
    hs, _ = hypotheses(R)
    Fd = field(tr, hs, "qwen25-7b")
    if Fd is None:
        return None
    out = []
    for t in range(T + 1):
        al = [i for i, h in enumerate(hs) if h["round"] <= t]
        X = torch.full((len(al) + 1, t + 1, 3), 0.0)
        for s in range(t + 1):
            lps = []
            for j, i in enumerate(al):
                sv, n = Fd[s].get(i, (-30.0, 1))
                sv = max(sv, -30.0); lps.append(sv)
                X[j, s] = torch.tensor([sv / 5, sv / max(n, 1) / 2, 0.0])
            mass = min(sum(math.exp(x) for x in lps), 1 - 1e-4)
            X[-1, s] = torch.tensor([math.log1p(-mass) / 5, 0.0, 1.0])
        y = torch.tensor([hs[i]["f1"] for i in al])
        out.append((X, y))
    return out


class M(nn.Module):
    def __init__(self, use_seq=True):
        super().__init__()
        self.use_seq = use_seq
        self.inp = nn.Linear(3, 32)
        self.gru = nn.GRU(32, 32, batch_first=True)
        self.out = nn.Linear(32, 1)

    def forward(self, X):
        if not self.use_seq:
            X = X[:, -1:]
        h, _ = self.gru(torch.tanh(self.inp(X)))
        return self.out(h[:, -1]).squeeze(-1)


trs = load_traj("qwen25-7b", ds)
D = {sp: [r for r in (rows(tr) for tr in split(trs, sp)) if r] for sp in ("train", "test")}
for use_seq in (False, True):
    m = M(use_seq)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for ep in range(4):
        random.shuffle(D["train"])
        for q in D["train"]:
            loss = 0
            for X, y in q:
                z = y.max(); tg = torch.cat([z * y / y.sum().clamp(min=1e-6), (1 - z).view(1)])
                loss = loss - (tg * torch.log_softmax(m(X), 0)).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        sel = np.zeros(T + 1)
        for q in D["test"]:
            for t, (X, y) in enumerate(q):
                sel[t] += y[int(m(X)[:-1].argmax())].item()
    print(ds, "seq" if use_seq else "last-row", "sel F1 by t", np.round(100 * sel / len(D["test"]), 1), flush=True)
