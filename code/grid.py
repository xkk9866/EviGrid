"""EviGrid: an answer-evidence grid transformer for stop-and-select in iterative RAG.

The state of an iterative RAG loop at round t is ONE object, the answer-evidence grid G_t:
  rows    = the answers in hand A_t (every distinct draft so far) plus a null row (the bare question),
  columns = the evidence batches E^(0..t) in order of arrival (column 0 = no evidence, closed book),
  cell    = a cross-encoder reading of (question + "Answer: a", passages of batch u), plus the generator's own
            reading of a under E_u (relative likelihood share, mean token log-likelihood) and whether the
            generator drafted a in round u.
A single axial transformer reads the grid: column attention lets the answers compete for the same evidence
(and the null row says what the evidence supports when no candidate is right), causal row attention
accumulates each answer's support over the evidence in order of arrival. Every candidate is read against every
batch, including batches retrieved after it was drafted (re-verification is a property of the grid).
Readout at the last column:
  posterior  pi_t over A_t + {not found}         -> v_t = max_a pi_t(a), s_t = 1 - pi_t(bot), selection argmax
  value      two monotone rates (disambiguation / discovery) of the exact value-of-information decomposition
             V_t(k) = v_t + delta_t(k) (s_t - v_t) + gamma_t(k) (1 - s_t)
Training: cross-entropy of pi_t to the graded truth + binary cross-entropy of V_t(k) to the realised quality of
the answer EviGrid itself would select k rounds later. Nothing else.

Ablations: single (only the latest draft is in hand), noCompete (no column attention: answers scored
independently), cached (a candidate only sees batches up to its birth), noLik (no generator reading),
freeHead (unconstrained value head), noNull (null row = learned constant instead of the question reading).

usage: python grid.py train --tag g_full [--ablate single] [--seed 0]
"""
import argparse, gzip, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from common import ROOT, DATASETS, T, load_traj, split, normalize

ENC = "BAAI/bge-base-en-v1.5"
DEVT = os.environ.get("DEV", "cuda")
ADT = torch.float16 if DEVT == "cuda" else torch.bfloat16
GRAN = os.environ.get("GRAN", "passage")
MAXLEN = int(os.environ.get("MAXLEN", 192 if GRAN == "passage" else 320))
EVAL_SPLITS = ("calib", "test", "calib2", "cert", "test2", "test3")
D = 256


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


# ------------------------------------------------------------------------------------------------ grid construction
def prep(tr, corpus, ab):
    """rows: distinct answers in order of first appearance (+ null); for each round t the visible rows; cells as text pairs."""
    R = tr["rounds"]
    keys = [normalize(r["answer"]) for r in R]
    rows, birth, text = [], {}, {}
    for t, k in enumerate(keys):
        if k not in birth:
            birth[k] = t; rows.append(k); text[k] = R[t]["answer"]
    # columns: the evidence in order of arrival, one column per passage (GRAN=batch: one per retrieval round);
    # column 0 is the empty evidence of the closed-book round
    if GRAN == "batch":
        cols = [("", 0)] + [("\n".join(f"{corpus[p]['title']}: {corpus[p]['text']}" for p in r["new_pids"]), u)
                            for u, r in enumerate(R) if u > 0]
    else:
        cols = [("", 0)] + [(f"{corpus[p]['title']}: {corpus[p]['text']}", u) for u, r in enumerate(R) if u > 0
                            for p in r["new_pids"]]
    col_round = [u for _, u in cols]
    cut = [sum(1 for u in col_round if u <= t) for t in range(T + 1)]  # visible columns at round t
    # generator reading: row u of the likelihood field scores candidate round-indices ci <= u
    gen = {}
    for u, r in enumerate(R):
        L = r.get("lik", {}) if "noLik" not in ab else {}
        if not L:
            continue
        m = max(s for s, _ in L.values())
        Z = sum(math.exp(s - m) for s, _ in L.values())
        for ci, (s, n) in L.items():
            gen[(keys[ci], u)] = (math.exp(s - m) / Z, max(s / max(n, 1), -10.0) / 5)
    q = tr["question"]
    pairs_a, pairs_b, cell_feat = [], [], []
    for k in rows + [None]:
        claim = f"{q} Answer: {text[k] or 'unknown'}" if k is not None else f"{q} Answer:"
        for text_b, u in cols:
            pairs_a.append(claim); pairs_b.append(text_b)
            g = gen.get((k, u)) if k is not None else None
            drafted = float(k is not None and keys[u] == k)
            cell_feat.append([g[0], g[1], 1.0, drafted] if g else [0.0, 0.0, 0.0, drafted])
    f1 = {k: max(R[t]["f1"] for t in range(T + 1) if keys[t] == k) for k in rows}
    last = {k: [max([u for u in range(t + 1) if keys[u] == k], default=-1) for t in range(T + 1)] for k in rows}
    return {"rows": rows, "birth": birth, "pairs": (pairs_a, pairs_b), "cf": cell_feat, "f1": f1, "last": last,
            "col_round": col_round, "cut": cut,
            "keys": keys, "lik_used": any(r.get("lik") for r in R) and "noLik" not in ab}


class AxialLayer(nn.Module):
    def __init__(self, d, h=4, compete=True):
        super().__init__()
        self.compete = compete
        self.col = nn.MultiheadAttention(d, h, batch_first=True)
        self.row = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff = mlp(d, 2 * d, d)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, X):  # X: (rows, cols, d)
        Rn, C, d = X.shape
        if self.compete:  # answers (and null) compete for the evidence of the same batch
            Y = self.n1(X).transpose(0, 1)                                   # (cols, rows, d)
            X = X + self.col(Y, Y, Y, need_weights=False)[0].transpose(0, 1)
        Y = self.n2(X)                                                       # causal accumulation along the evidence
        mask = torch.triu(torch.ones(C, C, dtype=torch.bool, device=X.device), 1)
        X = X + self.row(Y, Y, Y, attn_mask=mask, need_weights=False)[0]
        return X + self.ff(self.n3(X))


class EviGrid(nn.Module):
    def __init__(self, ablate=()):
        super().__init__()
        self.ab = set(ablate)
        self.enc = AutoModel.from_pretrained(ENC)
        self.enc.gradient_checkpointing_enable()  # memory bounded by the chunk size, not the grid size
        h =self.enc.config.hidden_size
        self.cell = nn.Linear(h + 4, D)
        self.col_pos = nn.Embedding(T + 1, D)
        self.null_tag = nn.Parameter(torch.zeros(D))
        self.null_const = nn.Parameter(torch.zeros(T + 1, D))  # noNull ablation
        self.layers = nn.ModuleList([AxialLayer(D, compete="noCompete" not in self.ab) for _ in range(2)])
        self.norm = nn.LayerNorm(D)
        self.score = mlp(D, D, 1)       # candidate logit
        self.score_null = mlp(D, D, 1)  # not-found logit
        self.rate = mlp(3 * D, D, 2 * T)  # disambiguation / discovery increments
        self.free = mlp(3 * D, D, T)      # freeHead ablation

    def encode(self, a, b, tok, dev, chunk=16):
        out = []
        for i in range(0, len(a), chunk):
            e = tok(a[i:i + chunk], b[i:i + chunk], truncation="only_second", max_length=MAXLEN, padding=True,
                    return_tensors="pt").to(dev)
            out.append(self.enc(**e).last_hidden_state[:, 0])
        return torch.cat(out).float()

    def forward(self, g, tok, dev):
        nr = len(g["rows"]) + 1
        H = self.encode(*g["pairs"], tok, dev)
        cf = torch.tensor(g["cf"], device=dev)
        nc = len(g["col_round"])
        X = self.cell(torch.cat([H, cf], -1)).view(nr, nc, D) + self.col_pos(torch.tensor(g["col_round"], device=dev))[None]
        null = self.null_const[torch.tensor(g["col_round"], device=dev)] if "noNull" in self.ab else X[-1]
        X = torch.cat([X[:-1], null.unsqueeze(0) + self.null_tag], 0)
        out = []
        for t in range(T + 1):
            vis = [i for i, k in enumerate(g["rows"]) if g["birth"][k] <= t]
            if "single" in self.ab:
                vis = [g["rows"].index(g["keys"][t])]
            cols = g["cut"][t]
            Xt = X[vis + [nr - 1], :cols]
            if "cached" in self.ab:  # a candidate only sees evidence up to its last drafting round
                lim = torch.tensor([g["cut"][g["last"][g["rows"][i]][t]] - 1 for i in vis] + [cols - 1], device=dev)
                keep = (torch.arange(cols, device=dev)[None] <= lim[:, None]).float()[..., None]
                Xt = Xt * keep
            for L in self.layers:
                Xt = L(Xt)
            Z = self.norm(Xt[:, -1])  # state of every row after the latest evidence
            if "cached" in self.ab:
                Z = self.norm(torch.stack([Xt[j, int(lim[j])] for j in range(len(vis) + 1)]))
            logits = torch.cat([self.score(Z[:-1]).squeeze(-1), self.score_null(Z[-1:]).squeeze(-1)])
            pi = torch.softmax(logits.float(), -1)
            j = int(pi[:-1].argmax())
            ctx = torch.cat([Z[-1], Z[j], Z[:-1].mean(0)])
            v, s = pi[:-1].max(), 1 - pi[-1]
            if "freeHead" in self.ab:
                Vk = torch.sigmoid(self.free(ctx).float())
            else:
                rt = 1 - torch.exp(-torch.cumsum(F.softplus(self.rate(ctx).float().view(2, T)), -1))
                Vk = v + rt[0] * (s - v) + rt[1] * (1 - s)
            out.append({"logits": logits, "pi": pi, "vis": vis, "V": torch.cat([v.view(1), Vk]).clamp(1e-5, 1 - 1e-5)})
        return out


def targets(g, out):
    """graded posterior targets and the realised quality of the answer EviGrid would select at every round"""
    sel = []
    for t, o in enumerate(out):
        y = torch.tensor([g["f1"][g["rows"][i]] for i in o["vis"]], device=o["pi"].device)
        o["y"] = y
        sel.append(float(y[int(o["pi"][:-1].detach().argmax())]))
    return sel


def loss_fn(g, out):
    sel = targets(g, out)
    lq, lv, nv = 0.0, 0.0, 0
    for t, o in enumerate(out):
        y = o["y"]; z = y.max()
        tgt = torch.cat([z * y / y.sum().clamp(min=1e-6), (1 - z).view(1)])
        lq = lq - (tgt * F.log_softmax(o["logits"].float(), -1)).sum()
        for k in range(1, T + 1 - t):
            p = o["V"][k]
            lv = lv - (sel[t + k] * torch.log(p) + (1 - sel[t + k]) * torch.log(1 - p)); nv += 1
    return lq / len(out) + lv / max(nv, 1)


def predict(model, items, tok, dev):
    model.eval()
    res = {}
    with torch.no_grad(), torch.autocast(DEVT, dtype=ADT):
        for tr, g in items:
            out = model(g, tok, dev)
            V, S, suff = [], [], []
            for t, o in enumerate(out):
                v = o["V"].float().cpu().numpy().tolist()
                V.append([round(x, 4) for x in v])
                S.append([[g["last"][g["rows"][i]][t], round(float(o["pi"][n]), 4)] for n, i in enumerate(o["vis"])])
                suff.append(round(1 - float(o["pi"][-1]), 4))
            res[f"{tr['dataset']}::{tr['qid']}"] = {"V": V, "S": S, "suff": suff, "P": suff, "lik_used": g["lik_used"]}
    model.train()
    return res


class _Corpus:
    def __init__(self, ds):
        with gzip.open(f"{ROOT}/data/proc/{ds}/corpus.jsonl.gz", "rt", encoding="utf-8") as f:
            self.c = [json.loads(l) for l in f]

    def __getitem__(self, p):
        return self.c[p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--tag", default="g_full")
    ap.add_argument("--traj", default="qwen25-7b")
    ap.add_argument("--src_traj", default="qwen25-7b")
    ap.add_argument("--ablate", default="")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    ab = [x for x in a.ablate.split(",") if x]
    dev = DEVT
    tok = AutoTokenizer.from_pretrained(ENC)
    model = EviGrid(ab).to(dev)
    items = {}
    for ds in DATASETS:
        cp = _Corpus(ds)
        items[ds] = [(tr, prep(tr, cp, ab)) for tr in load_traj(a.traj, ds)]
    if os.environ.get("SMOKE"):
        n = int(os.environ["SMOKE"])
        items = {ds: v[:n] + [x for x in v if x[0]["split"] != "train"][:n] for ds, v in items.items()}
    evals = lambda: {k: v for ds in DATASETS for k, v in
                     predict(model, [x for x in items[ds] if x[0]["split"] in EVAL_SPLITS], tok, dev).items()}
    if a.mode == "predict":
        model.load_state_dict(torch.load(f"{ROOT}/models/grid_{a.src_traj}_{a.tag}.pt", map_location=dev))
        os.makedirs(f"{ROOT}/preds/{a.traj}", exist_ok=True)
        json.dump(evals(), open(f"{ROOT}/preds/{a.traj}/essn_{a.tag}_from_{a.src_traj}.json", "w"))
        return
    train = [x for ds in DATASETS for x in items[ds] if x[0]["split"] == "train"]
    enc_p = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    oth_p = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": 2e-5}, {"params": oth_p, "lr": 3e-4}], weight_decay=0.01)
    steps = a.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler(DEVT, enabled=DEVT == "cuda")
    step, t0, run, skipped = 0, time.time(), 0.0, 0
    for ep in range(a.epochs):
        random.shuffle(train)
        for tr, g in train:
            opt.zero_grad()
            try:  # the GPU is shared with other jobs: a transient OOM skips one question instead of the run
                with torch.autocast(DEVT, dtype=ADT):
                    out = model(g, tok, dev)
                loss = loss_fn(g, out)
                scaler.scale(loss).backward()
            except torch.OutOfMemoryError:
                out = loss = None; opt.zero_grad(); torch.cuda.empty_cache(); skipped += 1
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1; run = 0.98 * run + 0.02 * loss.item()
            if step % 200 == 0:
                print(f"ep{ep} step {step}/{steps} loss {run:.4f} skipped {skipped} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(f"{ROOT}/models", exist_ok=True)
    torch.save(model.state_dict(), f"{ROOT}/models/grid_{a.traj}_{a.tag}.pt")
    os.makedirs(f"{ROOT}/preds/{a.traj}", exist_ok=True)
    json.dump(evals(), open(f"{ROOT}/preds/{a.traj}/essn_{a.tag}.json", "w"))
    print("done", a.tag, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
