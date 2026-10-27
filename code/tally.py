"""TALLY: sequential hypothesis testing over the answers in hand.

One learned function, one probabilistic model.

  learned   a cross-encoder reads ONE kind of input, a text pair (claim, text), where claim = "question Answer: a":
              e(a, p)      evidence of passage p for "a is correct" against "the correct answer is not in hand"
                           (a log-likelihood ratio); p = "" gives the proposal prior beta(a) = e(a, "")
              s2(a, x)     informativeness of the next round for a, read from the pair (claim, next query x)
              rho(x)       forecast of the proposal prior of the next new draft, read from ("question Answer: unknown", x)
  derived   (no further parameters)
              filter       stick-breaking open-set posterior over A_t + {bot}; every candidate accumulates the evidence of
                           EVERY passage in E_t (exchangeable in the passages, hence re-verification is intrinsic)
              planning     E[v_{t+k} | H_t] under the same model: hypothesis-conditioned Gaussian LLR increments
                           N(+-k s2/2, k s2) (the only family whose value is its own log-likelihood ratio) and a discovery
                           prior rho_k = 1-(1-rho)^k; computed exactly over the truth and by antithetic MC over the noise
              stop/select  v_t = max_a pi_t(a) vs max_k [V_t(k) - lambda dC]; return argmax_a pi_t(a)
  training  maximum likelihood of the joint model on complete rollouts: cross-entropy of the posterior, Gaussian
            likelihood of the observed (detached) evidence increments, cross-entropy of the proposal forecast.

Ablations (trained):  single  (only the latest draft is in hand)      cached (a candidate keeps the evidence of the round it
                      was last produced)      closed (no bot outcome)      constS (informativeness / discovery forecast
                      not conditioned on the next query)
Planning-only variants written from the same model: noDisc (rho = 0), noDisamb (s2 = 0 for the answers in hand).

usage: python tally.py train --tag tally_m [--ablate single] [--seed 0]
       python tally.py predict --tag tally_m --traj llama31-8b --src_traj qwen25-7b
"""
import argparse, gzip, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.utils.checkpoint
from transformers import AutoModel, AutoTokenizer
from common import ROOT, DATASETS, T, load_traj, split, normalize

ENC = "BAAI/bge-base-en-v1.5"
DEVT = os.environ.get("DEV", "cuda")
ADT = torch.float16 if DEVT == "cuda" else torch.bfloat16
MAXLEN = 192
EVAL_SPLITS = ("calib", "test", "calib2", "cert", "test2")
NMC = 2048  # antithetic -> 4096 noise draws per state
LOGS2_MIN, LOGS2_MAX = -6.0, 4.0
_Z = np.random.default_rng(0).standard_normal((NMC, T + 3))
ZMC = np.concatenate([_Z, -_Z])  # common random numbers for every state and every lambda


def claim(q, a):
    return f"{q} Answer: {a or 'unknown'}"


# ------------------------------------------------------------------------------------------------ question structure
def prep(tr, corpus, ab):
    """Everything the model reads for one trajectory: the distinct candidate strings (in order of first appearance),
    the passages (in order of arrival) and the next query of every round. No other signal is used."""
    R = tr["rounds"]
    keys = [normalize(r["answer"]) for r in R]
    strings, birth, text = [], {}, {}
    for t, k in enumerate(keys):
        if k not in birth:
            birth[k] = t; strings.append(k); text[k] = R[t]["answer"]
    pids, npass = [], []
    for r in R:
        pids += r["new_pids"]; npass.append(len(pids))
    nq = []
    for t in range(T + 1):
        x = R[t].get("next_query") or (R[t + 1]["query"] if t < T else "") or tr["question"]
        nq.append(x)
    hand = []  # A_t as indices into strings, in birth order
    for t in range(T + 1):
        if "single" in ab:
            hand.append([strings.index(keys[t])])
        else:
            hand.append([i for i, k in enumerate(strings) if birth[k] <= t])
    last = [{i: max(u for u in range(t + 1) if keys[u] == strings[i]) for i in hand[t]} for t in range(T + 1)]
    ptxt = [f"{corpus[p]['title']}: {corpus[p]['text']}" for p in pids]
    ev_a, ev_b = [], []
    for k in strings:
        c = claim(tr["question"], text[k])
        ev_a += [c] * (1 + len(pids)); ev_b += [""] + ptxt
    lk_a, lk_b, lk_idx = [], [], []  # look-ahead pairs: (claim of a in A_t, next query) and (bot claim, next query)
    for t in range(T):
        for i in hand[t]:
            lk_idx.append((t, i)); lk_a.append(claim(tr["question"], text[strings[i]])); lk_b.append(nq[t])
        lk_idx.append((t, -1)); lk_a.append(claim(tr["question"], "")); lk_b.append(nq[t])
    f1 = {i: max(R[u]["f1"] for u in range(T + 1) if keys[u] == strings[i]) for i in range(len(strings))}
    return {"strings": strings, "birth": [birth[k] for k in strings], "npass": npass, "hand": hand, "last": last,
            "keys": keys, "pairs": (ev_a + lk_a, ev_b + lk_b), "nev": len(ev_a), "P": len(pids), "lk_idx": lk_idx,
            "f1": f1, "key": f"{tr['dataset']}::{tr['qid']}"}


# ------------------------------------------------------------------------------------------------ model
class Tally(nn.Module):
    def __init__(self, ablate=()):
        super().__init__()
        self.ab = set(ablate)
        self.enc = AutoModel.from_pretrained(ENC)
        d = self.enc.config.hidden_size
        self.head = nn.Linear(d, 3)  # [evidence LLR, log informativeness, logit discovery forecast]
        nn.init.normal_(self.head.weight, std=0.02)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([0.0, -1.0, -1.0]))
        self.const = nn.Parameter(torch.tensor([-1.0, -1.0, -1.0]))  # constS ablation: [log s2 in hand, log s2 bot, logit rho]

    def _cls(self, ids, am, tt):
        return self.enc(input_ids=ids, attention_mask=am, token_type_ids=tt).last_hidden_state[:, 0]

    def encode(self, enc_in, chunk=64):
        N, outs = enc_in["input_ids"].shape[0], []
        for i in range(0, N, chunk):
            p = [enc_in[k][i:i + chunk] for k in ("input_ids", "attention_mask", "token_type_ids")]
            if self.training and N > chunk:
                outs.append(torch.utils.checkpoint.checkpoint(self._cls, *p, use_reentrant=False))
            else:
                outs.append(self._cls(*p))
        return self.head(torch.cat(outs)).float()

    def forward(self, qs, enc_in):
        """qs: list of prep() dicts whose pairs were tokenised in order into enc_in. Returns, per question, the evidence
        matrix E (n_strings x (1 + P)) and the look-ahead quantities."""
        H = self.encode(enc_in)
        out, o = [], 0
        for q in qs:
            n, P = len(q["strings"]), q["P"]
            Eq = H[o:o + q["nev"], 0].view(n, 1 + P)
            Lq = H[o + q["nev"]:o + len(q["pairs"][0])]
            o += len(q["pairs"][0])
            logs2, rho = {}, {}
            for j, (t, i) in enumerate(q["lk_idx"]):
                if "constS" in self.ab:
                    ls = self.const[0] if i >= 0 else self.const[1]
                    lr = self.const[2]
                else:
                    ls, lr = Lq[j, 1], Lq[j, 2]
                logs2[(t, i)] = ls.clamp(LOGS2_MIN, LOGS2_MAX)
                if i < 0:
                    rho[t] = lr
            out.append({"E": Eq, "logs2": logs2, "rho": rho})
        return out

    def logits(self, q, o, t):
        """Open-set posterior logits at round t over hand[t] (+ bot as the last entry unless 'closed').
        Stick-breaking prior: the j-th proposal takes the fraction sigmoid(beta_j) of the mass that is still unexplained;
        every passage of E_t (cached ablation: E_{last round the candidate was produced}) multiplies by exp(e(a, p))."""
        idx = q["hand"][t]
        E = o["E"]
        if "cached" in self.ab:
            ev = torch.stack([E[i, 1:1 + q["npass"][q["last"][t][i]]].sum() for i in idx])
        else:
            ev = E[idx, 1:1 + q["npass"][t]].sum(-1)
        beta = E[idx, 0]
        if "closed" in self.ab:
            return beta + ev
        l1 = F.logsigmoid(-beta)
        prefix = torch.cumsum(l1, 0) - l1
        return torch.cat([F.logsigmoid(beta) + prefix + ev, l1.sum().view(1)])


def build(trajs, corpus, tok, dev, ab):
    qs = [prep(tr, corpus, ab) for tr in trajs]
    a = [x for q in qs for x in q["pairs"][0]]
    b = [x for q in qs for x in q["pairs"][1]]
    enc = tok(a, b, truncation="only_second", max_length=MAXLEN, padding=True, return_tensors="pt")
    return qs, {k: v.to(dev) for k, v in enc.items()}


# ------------------------------------------------------------------------------------------------ training objective
def gauss_nll(x, y, s2):
    """negative log-likelihood of an observed LLR increment x under the hypothesis-conditioned model
    y * N(+s2/2, s2) + (1-y) * N(-s2/2, s2) (soft label y = F1 of the candidate)."""
    lp = -0.5 * torch.log(2 * math.pi * s2) - (x - s2 / 2) ** 2 / (2 * s2)
    ln = -0.5 * torch.log(2 * math.pi * s2) - (x + s2 / 2) ** 2 / (2 * s2)
    return -(y * lp + (1 - y) * ln)


def loss_fn(model, qs, outs):
    ab = model.ab
    l_post, l_inc, l_disc = [], [], []
    for q, o in zip(qs, outs):
        E = o["E"]
        for t in range(T + 1):
            idx = q["hand"][t]
            y = torch.tensor([q["f1"][i] for i in idx], device=E.device)
            lg = model.logits(q, o, t)
            if "closed" in ab:
                if y.sum() <= 0:
                    continue
                tgt = y / y.sum()
            else:
                z = y.max()
                tgt = torch.cat([z * y / y.sum().clamp(min=1e-6), (1 - z).view(1)])
            l_post.append(-(tgt * F.log_softmax(lg, -1)).sum())
            if t == T:
                continue
            n0, n1 = q["npass"][t], q["npass"][t + 1]
            if n1 > n0:  # in-hand increments of the next round
                for j, i in enumerate(idx):
                    x = E[i, 1 + n0:1 + n1].sum().detach()
                    l_inc.append(gauss_nll(x, y[j], o["logs2"][(t, i)].exp()))
            if "closed" in ab:
                continue
            nxt = q["strings"].index(q["keys"][t + 1])
            newborn = nxt not in idx
            if newborn:  # the newborn's evidence against bot after E_{t+1}, and its proposal prior
                xb = E[nxt, 1:1 + n1].sum().detach()
                l_inc.append(gauss_nll(xb, torch.tensor(q["f1"][nxt], device=E.device), o["logs2"][(t, -1)].exp()))
                r_t = torch.sigmoid(E[nxt, 0]).detach()
            else:
                r_t = torch.zeros((), device=E.device)
            l_disc.append(F.binary_cross_entropy_with_logits(o["rho"][t], r_t))
    m = lambda xs: torch.stack(xs).mean() if xs else torch.zeros((), device=outs[0]["E"].device)
    lp, li, ld = m(l_post), m(l_inc), m(l_disc)
    return lp + li + ld, (lp.item(), li.item(), ld.item())


# ------------------------------------------------------------------------------------------------ planning (derived)
def plan(w, logs2_a, logs2_b, rho, k, closed=False, disc=True, disamb=True):
    """E[v_{t+k} | H_t] under the model. w: posterior logits (bot last unless closed). Exact over the identity of the
    correct answer (A_t, a newborn b, or still not in hand), antithetic MC over the Gaussian LLR noise."""
    w = np.asarray(w, dtype=np.float64)
    p = np.exp(w - w.max()); p /= p.sum()
    pa = p if closed else p[:-1]
    n = len(pa)
    s2a = np.exp(np.asarray(logs2_a, dtype=np.float64)) * k if disamb else np.zeros(n)
    if closed or not disc:
        rk, pb = 0.0, 0.0 if closed else p[-1]
    else:
        rk, pb = 1 - (1 - rho) ** k, p[-1]
    s2b = math.exp(logs2_b) * k
    prior = np.concatenate([pa, [pb * rk, pb * (1 - rk)]])
    lprior = np.log(np.maximum(prior, 1e-300))
    Za, Zb = ZMC[:, :n], ZMC[:, n]
    V = 0.0
    for h in range(n + 2):
        if prior[h] <= 0:
            continue
        Da = np.where(np.arange(n) == h, s2a / 2, -s2a / 2) + np.sqrt(s2a) * Za
        Db = (s2b / 2 if h == n else -s2b / 2) + math.sqrt(s2b) * Zb
        L = np.concatenate([lprior[:n] + Da, (lprior[n] + Db)[:, None], np.full((len(ZMC), 1), lprior[n + 1])], 1)
        if prior[n] <= 0:
            L[:, n] = -np.inf
        if prior[n + 1] <= 0:
            L[:, n + 1] = -np.inf
        L = L - L.max(1, keepdims=True)
        P = np.exp(L); P /= P.sum(1, keepdims=True)
        V += prior[h] * P[:, :n + 1].max(1).mean()
    return float(V)


def predict(model, trajs, corpus, tok, dev, bs=8):
    model.eval()
    if os.environ.get("SMOKE"):
        trajs = trajs[:int(os.environ["SMOKE"])]
    ab = model.ab
    res = {v: {} for v in ("full", "noDisc", "noDisamb")}
    with torch.no_grad(), torch.autocast(DEVT, dtype=ADT):
        for i in range(0, len(trajs), bs):
            qs, enc = build(trajs[i:i + bs], corpus, tok, dev, ab)
            outs = model(qs, enc)
            for q, o in zip(qs, outs):
                S, suff, Vs = [], [], {v: np.zeros((T + 1, T + 1)) for v in res}
                for t in range(T + 1):
                    w = model.logits(q, o, t).float().cpu().numpy()
                    p = np.exp(w - w.max()); p /= p.sum()
                    pa = p if "closed" in ab else p[:-1]
                    idx = q["hand"][t]
                    S.append([[int(q["last"][t][c]), round(float(v), 4)] for c, v in zip(idx, pa)])
                    suff.append(round(1.0 if "closed" in ab else 1.0 - float(p[-1]), 4))
                    for v in res:
                        Vs[v][t, 0] = pa.max()
                    if t == T:
                        continue
                    la = np.array([float(o["logs2"][(t, c)]) for c in idx])
                    lb = float(o["logs2"][(t, -1)])
                    rho = float(torch.sigmoid(o["rho"][t])) if t in o["rho"] else 0.0
                    for k in range(1, T - t + 1):
                        Vs["full"][t, k] = plan(w, la, lb, rho, k, "closed" in ab)
                        Vs["noDisc"][t, k] = plan(w, la, lb, rho, k, "closed" in ab, disc=False)
                        Vs["noDisamb"][t, k] = plan(w, la, lb, rho, k, "closed" in ab, disamb=False)
                for v in res:
                    res[v][q["key"]] = {"V": Vs[v].round(5).tolist(), "S": S, "suff": suff, "lik_used": False}
    model.train()
    return res


# ------------------------------------------------------------------------------------------------ main
def load_data(tag, dss):
    corpus, items = {}, {}
    for ds in dss:
        with gzip.open(f"{ROOT}/data/proc/{ds}/corpus.jsonl.gz", "rt", encoding="utf-8") as f:
            corpus[ds] = [json.loads(l) for l in f]
        items[ds] = load_traj(tag, ds)
    return corpus, items


class _Corpus:
    """corpus lookup across datasets keyed by the trajectory's dataset (pids are per-dataset indices)"""
    def __init__(self, corpus, ds):
        self.c, self.ds = corpus, ds

    def __getitem__(self, p):
        return self.c[self.ds][p]


def run_predict(model, items, corpus, tok, dev, eds):
    res = {v: {} for v in ("full", "noDisc", "noDisamb")}
    for ds in eds:
        for sp in EVAL_SPLITS:
            tr = split(items[ds], sp)
            if tr:
                r = predict(model, tr, _Corpus(corpus, ds), tok, dev)
                for v in res:
                    res[v].update(r[v])
    return res


def save(res, traj, name, plan_variants):
    os.makedirs(f"{ROOT}/preds/{traj}", exist_ok=True)
    json.dump(res["full"], open(f"{ROOT}/preds/{traj}/essn_{name}.json", "w"))
    if plan_variants:  # planning-only variants of the same model: essn_<pre>_noDisc<sfx>.json
        base, _, rest = name.partition("_m")
        for v in ("noDisc", "noDisamb"):
            json.dump(res[v], open(f"{ROOT}/preds/{traj}/essn_{base}_{v}{rest}.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--tag", default="tally_m")
    ap.add_argument("--traj", default="qwen25-7b")
    ap.add_argument("--src_traj", default="qwen25-7b")
    ap.add_argument("--ablate", default="")
    ap.add_argument("--train_ds", default=",".join(DATASETS))
    ap.add_argument("--eval_ds", default=",".join(DATASETS))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ntrain", type=int, default=0)
    ap.add_argument("--init", default="")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    ab = [x for x in a.ablate.split(",") if x]
    dev = DEVT
    tds, eds = a.train_ds.split(","), a.eval_ds.split(",")
    corpus, items = load_data(a.traj, sorted(set(eds) if a.mode == "predict" else set(tds) | set(eds)))
    tok = AutoTokenizer.from_pretrained(ENC)
    model = Tally(ab).to(dev)
    main_model = not ab
    if a.mode == "predict":
        model.load_state_dict(torch.load(f"{ROOT}/models/{a.src_traj}_{a.tag}.pt", map_location=dev))
        res = run_predict(model, items, corpus, tok, dev, eds)
        save(res, a.traj, f"{a.tag}_from_{a.src_traj}", False)
        print("done predict", flush=True)
        return
    if a.init:
        model.load_state_dict(torch.load(f"{ROOT}/models/{a.init}.pt", map_location=dev))
    train = [(ds, tr) for ds in tds for tr in split(items[ds], "train")[:a.ntrain or None]]
    if os.environ.get("SMOKE"):
        train = train[:int(os.environ["SMOKE"])]
    enc_p = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    oth_p = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": 2e-5}, {"params": oth_p, "lr": 5e-4}], weight_decay=0.01)
    steps = a.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler(DEVT, enabled=DEVT == "cuda")
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        random.shuffle(train)
        for ds, tr in train:
            qs, enc = build([tr], _Corpus(corpus, ds), tok, dev, model.ab)
            with torch.autocast(DEVT, dtype=ADT):
                outs = model(qs, enc)
            loss, parts = loss_fn(model, qs, outs)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % 200 == 0:
                print(f"ep{ep} step {step}/{steps} loss {loss.item():.4f} post/inc/disc {np.round(parts, 4)} "
                      f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(f"{ROOT}/models", exist_ok=True)
    torch.save(model.state_dict(), f"{ROOT}/models/{a.traj}_{a.tag}.pt")
    res = run_predict(model, items, corpus, tok, dev, eds)
    save(res, a.traj, a.tag, main_model)
    print("done", a.tag, f"{(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
