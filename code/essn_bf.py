"""ESSN-BF: the Evidence-Sufficiency State Network as a learned Bayesian evidence filter.

The network IS a filter over the open hypothesis set H_t = A_t (distinct drafts so far) + {bot} ("the correct answer
has not been found yet"). Its recurrent state is the log-evidence vector eta_t over H_t, and its only recurrence is
evidence accumulation:

    eta_t(h) = sum_{p in E_t + {null}} phi(h, p)  +  alpha * log p_LLM(h | E_t)          pi_t = softmax(eta_t)

  * phi(h, p): ONE learned evidence potential, a cross-encoder over (claim of h, passage p). The claim of a candidate a
    is "q Answer: a"; the claim of bot is the bare question "q Answer:" (relevance that no candidate explains).
    phi is pairwise, so every pair is scored once and cached: re-verifying every candidate against ALL current
    evidence (Prop. re-verification) is exact at incremental cost.
  * log p_LLM(h | E_t): the generator as a joint reader of the whole evidence. For a candidate it is the generator's
    own likelihood of that answer under the round-t prompt; for bot it is the generator's mass on answers NOT in hand,
    log(1 - sum_a p_LLM(a | E_t)). The generator is thus an expert over the same open hypothesis set.
  * Look-ahead: in a POMDP the value is a function of the belief. V_t(k) = sigma(g_k(u_t)), where u_t is the
    (permutation-invariant) shape of the belief pi_t, its change since t-1, the round, and the question.

Decisions: value of stopping v_t = max_a pi_t(a); selection argmax_a pi_t(a); sufficiency 1 - pi_t(bot);
stop iff v_t >= max_k V_t(k) - lambda * (C_{t+k} - C_t).

Training (equal weights, no auxiliary labels): graded open-set cross-entropy of pi_t; martingale coherence of pi_t;
realised-selection and tower targets for V_t(k).

usage:
  python essn_bf.py train --tag bf [--ablate noGen,noPhi,latestEv,noBot,noCoh,single,featProfile] --seed 0
  python essn_bf.py predict --tag bf --traj qwen3-4b --src_traj qwen25-7b
"""
import argparse, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.utils.checkpoint
from transformers import AutoModel, AutoTokenizer
from common import ROOT, DATASETS, T, load_traj, split, normalize

ENC = "BAAI/bge-base-en-v1.5"
DEVT = os.environ.get("DEV", "cuda")
ADT = torch.float16 if DEVT == "cuda" else torch.bfloat16
MAXLEN = 192
TOPK = 3  # belief shape: the three largest candidate probabilities (permutation invariant)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def hypotheses(rounds, single=False):
    """distinct drafts in order of appearance with their birth round; single=True keeps only the latest draft per
    round (the single-answer special case |A_t| = 1)."""
    hs, seen = [], {}
    for t, r in enumerate(rounds):
        n = normalize(r["answer"])
        if n not in seen:
            seen[n] = len(hs)
            hs.append({"round": t, "key": n, "f1": r["f1"]})
    return hs, seen


class Filter(nn.Module):
    def __init__(self, ablate=()):
        super().__init__()
        self.ab = set(ablate)
        self.enc = AutoModel.from_pretrained(ENC)
        d = self.enc.config.hidden_size
        self.phi = nn.Linear(d, 1)                                 # scalar evidence potential ("scalar" ablation)
        self.psi = nn.Sequential(nn.Linear(d, 128), nn.GELU())     # pair -> additive evidence message
        self.rho = mlp(128, 128, 1)                                # non-linear read-out of the evidence state
        self.psi_g = nn.Sequential(nn.Linear(4, 128), nn.GELU())  # generator reading -> additive evidence message
        self.cov = nn.Linear(128, 1)                               # coverage read-out of the question's evidence state
        self.log_alpha = nn.Parameter(torch.zeros(()))             # weight of the generator expert (total log-lik)
        self.beta = nn.Parameter(torch.zeros(()))                  # weight of its per-token log-lik (length prior)
        self.qproj = nn.Linear(d, 32)                              # question representation for the look-ahead
        udim = TOPK + 1 + 2 + (T + 1) + 32
        self.g = mlp(udim, 128, T)                                 # g_k, k = 1..T

    def encode(self, enc_in, chunk=48):
        N = enc_in["input_ids"].shape[0]
        f = lambda ids, am, tt: self.enc(input_ids=ids, attention_mask=am, token_type_ids=tt).last_hidden_state[:, 0]
        if not self.training or N <= chunk:
            return f(enc_in["input_ids"], enc_in["attention_mask"], enc_in["token_type_ids"])
        outs = [torch.utils.checkpoint.checkpoint(f, enc_in["input_ids"][i:i + chunk], enc_in["attention_mask"][i:i + chunk],
                                                  enc_in["token_type_ids"][i:i + chunk], use_reentrant=False)
                for i in range(0, N, chunk)]
        return torch.cat(outs)

    def forward(self, b):
        """b: one question. returns per round t: log-posterior over (alive candidates + bot), V_t(1..T) logits."""
        cls = self.encode(b["enc"]).float()
        nh, npas = b["nh"], b["np"]
        phi = self.phi(cls).view(nh + 1, npas)                     # rows: candidates..., bot ; cols: null + passages
        msg = self.psi(cls).view(nh + 1, npas, -1)
        qv = self.qproj(cls.view(nh + 1, npas, -1)[nh, 0])         # bot claim x null passage = the question alone
        alpha = self.log_alpha.exp()
        logpis, Vs, prev, covs = [], [], None, []
        for t in range(T + 1):
            alive = b["alive"][t]                                  # candidate indices alive at t (+ bot appended)
            rows = torch.tensor(alive + [nh], device=cls.device)
            if "latestEv" in self.ab:                              # ablation: only the evidence of the current round
                cols = b["new_cols"][t]
            else:
                cols = b["cols"][t]                                # null + every passage retrieved up to t
            eta = torch.zeros(len(rows), device=cls.device)
            covs.append(self.cov(msg[nh, b["cols"][t]].sum(0)).squeeze())
            if "scalar" in self.ab:                                # linear read-out: product of passage experts
                eta = eta + phi[rows][:, cols].sum(1)
            elif "noPhi" not in self.ab:                           # additive evidence state, non-linear read-out
                m = msg[rows][:, cols].sum(1)
                if "genMsg" in self.ab:                            # generator readings of rows 0..t enter the same state
                    m = m + self.psi_g(b["gmsg"][t].to(cls.device)).sum(1)
                eta = eta + self.rho(m).squeeze(-1)
            if "noGen" not in self.ab and "noAlpha" not in self.ab:
                gt = b["gen"][t].to(cls.device)
                eta = eta + alpha * gt[:, 0] + self.beta * gt[:, 1]
            if "noBot" in self.ab:
                eta = torch.cat([eta[:-1], eta.new_full((1,), -30.0)])
            logpi = torch.log_softmax(eta, 0)
            logpis.append(logpi)
            pi = logpi.exp()
            pc = pi[:-1]
            top = torch.sort(pc, descending=True).values[:TOPK]
            top = torch.cat([top, top.new_zeros(TOPK - len(top))])
            v, bot = pc.max(), pi[-1]
            dv, db = (v - prev[0], bot - prev[1]) if prev is not None else (v * 0, bot * 0)
            prev = (v, bot)
            u = torch.cat([top, bot.view(1), torch.stack([dv, db]), F.one_hot(torch.tensor(t), T + 1).float().to(cls.device), qv])
            Vs.append(self.g(u))
        self.covs = torch.stack(covs)
        return logpis, torch.stack(Vs)


LIKFULL = {}


def lik_full(traj, ds):
    import gzip
    if (traj, ds) not in LIKFULL:
        p = f"{ROOT}/data/lik_full/{traj}/{ds}.json.gz"
        LIKFULL[(traj, ds)] = json.load(gzip.open(p, "rt", encoding="utf-8")) if os.path.exists(p) else {}
    return LIKFULL[(traj, ds)]


def field(tr, hs, traj):
    """F[s][i] = (sum log p_LLM(h_i | prompt_s), n tokens) for every round s and every hypothesis of the trajectory"""
    L = lik_full(traj, tr["dataset"]).get(tr["qid"])
    R = tr["rounds"]
    if not L:
        return None
    pos = {h["key"]: i for i, h in enumerate(hs)}
    F = []
    for s in range(T + 1):
        row = {}
        for ci, sv, n in L["rows"][s]:
            k = normalize(R[ci]["answer"])
            if k in pos:
                row[pos[k]] = (sv, n)
        F.append(row)
    return F


def build(tr, corpus, tok, dev, ab, traj="qwen25-7b"):
    R = tr["rounds"]
    hs, idx = hypotheses(R)
    nh = len(hs)
    pids, arr = [], []
    for t, r in enumerate(R):
        for p in r["new_pids"]:
            pids.append(p); arr.append(t)
    pas = [""] + [f"{corpus[p]['title']}: {corpus[p]['text']}" for p in pids]
    arr = [0] + arr
    claims = [f"{tr['question']} Answer: {h['key'] and R[h['round']]['answer'] or 'unknown'}" for h in hs] + [f"{tr['question']} Answer:"]
    ta, tb = [], []
    for c in claims:
        ta += [c] * len(pas); tb += pas
    enc = tok(ta, tb, truncation="only_second", max_length=MAXLEN, padding=True, return_tensors="pt")
    enc = {k: v.to(dev) for k, v in enc.items()}
    alive, cols, new_cols, gen, ys, gmsg, extra = [], [], [], [], [], [], []
    Fd = field(tr, hs, traj) if "genMsg" in ab else None
    for t, r in enumerate(R):
        cur = idx[normalize(r["answer"])]
        al = [cur] if "single" in ab else [i for i, h in enumerate(hs) if h["round"] <= t]
        alive.append(al)
        cols.append([j for j, a in enumerate(arr) if a <= t])
        new_cols.append([0] + [j for j, a in enumerate(arr) if a == t and j > 0])
        L = r.get("lik", {})
        lp, nt = [], []
        for i in al:
            ci = hs[i]["round"]
            # the lik row of round t scores every distinct draft by (most recent) round index; map by text
            s, nn_ = None, 1
            for cj, (sv, n) in L.items():
                if normalize(R[cj]["answer"]) == hs[i]["key"]:
                    s, nn_ = sv, max(n, 1); break
            lp.append(s if s is not None else -30.0); nt.append(nn_)
        lp = torch.tensor(lp, dtype=torch.float).clamp(min=-30.0)
        mass = lp.exp().sum().clamp(max=1 - 1e-4)
        lb = torch.log1p(-mass).view(1)
        gen.append(torch.stack([torch.cat([lp, lb]), torch.cat([lp / torch.tensor(nt, dtype=torch.float), lb])], 1))
        ys.append(torch.tensor([hs[i]["f1"] for i in al], device=dev))
        if "genMsg" in ab:
            # generator readings of every hypothesis in hand under every evidence state s <= t (causal at t: a draft
            # born at t is scored under the cached prompts of rounds < t); bot reads the mass outside A_t
            gm = torch.zeros(len(al) + 1, t + 1, 4)
            for s_ in range(t + 1):
                row = Fd[s_] if Fd else {}
                lps = []
                for j, i in enumerate(al):
                    sv, n = row.get(i, (-30.0, 1))
                    sv = max(sv, -30.0); lps.append(sv)
                    gm[j, s_] = torch.tensor([sv / 5, sv / max(n, 1), math.log(n + 1) / 3, 0.0])
                mass = min(sum(math.exp(x) for x in lps), 1 - 1e-4)
                lb = math.log1p(-mass)
                gm[-1, s_] = torch.tensor([lb / 5, lb, 0.0, 1.0])
            gmsg.append(gm)
        # extra generator tokens at round t: every draft born at t is also scored under the t earlier prompts
        nb = [i for i, h in enumerate(hs) if h["round"] == t]
        extra.append(sum(t * (len(tok(R[t]["answer"] or "unknown", add_special_tokens=False)["input_ids"]) + 2) for _ in nb)
                     if "genMsg" in ab else 0)
    return {"enc": enc, "nh": nh, "np": len(pas), "alive": alive, "cols": cols, "new_cols": new_cols, "gen": gen,
            "ys": ys, "hs": hs, "cost": [r["cum_tok"] for r in R], "g": torch.tensor([r["gold_all"] for r in R], device=dev),
            "gmsg": gmsg, "extra": extra}


def losses(logpis, V, b, ab, cov=None):
    ys = b["ys"]
    # graded open-set target: mass z_t = max quality in hand spread over candidates by quality, 1 - z_t on bot
    l_post = []
    for lp, y in zip(logpis, ys):
        z = y.max()
        tgt = torch.cat([z * y / y.sum().clamp(min=1e-6), (1 - z).view(1)])
        l_post.append(-(tgt * lp).sum())
    l_post = torch.stack(l_post).mean()
    # realised quality of the answer the filter itself returns at every round (stop-gradient)
    sel = torch.stack([y[int(lp[:-1].detach().argmax())] for lp, y in zip(logpis, ys)])
    vnow = torch.stack([lp[:-1].detach().exp().max() for lp in logpis])
    Vs = torch.sigmoid(V.detach())
    tgt, tow, m = torch.zeros_like(V), torch.zeros_like(V), torch.zeros_like(V)
    for t in range(T):
        for k in range(1, T - t + 1):
            tgt[t, k - 1] = sel[t + k]
            tow[t, k - 1] = vnow[t + 1] if k == 1 else Vs[t + 1, k - 2]
            m[t, k - 1] = 1
    l_prof = (F.binary_cross_entropy_with_logits(V, tgt, reduction="none") * m).sum() / m.sum()
    l_coh = torch.zeros((), device=V.device)
    if "noCoh" not in ab:
        l_prof = l_prof + (F.binary_cross_entropy_with_logits(V, tow, reduction="none") * m).sum() / m.sum()
        # martingale coherence: pi_t = E[P pi_{t+1} | H_t]; P keeps candidates in hand, moves newborn mass to bot
        lc = []
        for t in range(T):
            nxt = logpis[t + 1].detach().exp()
            pos = {i: n for n, i in enumerate(b["alive"][t + 1])}
            if any(i not in pos for i in b["alive"][t]):
                continue
            kept = nxt[torch.tensor([pos[i] for i in b["alive"][t]], device=V.device)]
            tg = torch.cat([kept, (1 - kept.sum()).clamp(min=0).view(1)])
            lc.append(-(tg * logpis[t]).sum())
        if lc:
            l_coh = torch.stack(lc).mean()
    l_ev = torch.zeros((), device=V.device)
    if "ev" in ab and cov is not None and (b["g"] >= 0).all():
        # evidence coverage: does the evidence state of the question contain all annotated supporting documents?
        l_ev = F.binary_cross_entropy_with_logits(cov.float(), b["g"].float())
    return l_post + l_prof + l_coh + l_ev, (l_post.item(), l_prof.item(), l_coh.item(), l_ev.item())


def predict(model, trajs, corpus, tok, dev, ab, traj="qwen25-7b"):
    model.eval()
    out = {}
    with torch.no_grad(), torch.autocast(DEVT, dtype=ADT):
        for tr in trajs:
            b = build(tr, corpus, tok, dev, ab, traj)
            logpis, V = model(b)
            V = torch.sigmoid(V.float()).cpu().numpy()
            M = np.zeros((T + 1, T + 1))
            S, suff = [], []
            for t, lp in enumerate(logpis):
                pi = lp.float().exp().cpu().numpy()
                M[t, 0] = pi[:-1].max()
                M[t, 1:T - t + 1] = V[t, :T - t]
                S.append([[int(b["hs"][i]["round"]), round(float(p), 4)] for i, p in zip(b["alive"][t], pi[:-1])])
                suff.append(round(1 - float(pi[-1]), 4))
            out[f"{tr['dataset']}::{tr['qid']}"] = {"V": M.round(4).tolist(), "S": S, "suff": suff,
                                                     "lik_used": "noGen" not in ab, "extra_tok": b["extra"]}
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--tag", default="bf")
    ap.add_argument("--traj", default="qwen25-7b")
    ap.add_argument("--src_traj", default="qwen25-7b")
    ap.add_argument("--ablate", default="")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default="calib,test,calib2")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed)
    ab = set(x for x in a.ablate.split(",") if x)
    dev = DEVT
    tok = AutoTokenizer.from_pretrained(ENC)
    import gzip
    corpus, items = {}, {}
    for ds in DATASETS:
        with gzip.open(f"{ROOT}/data/proc/{ds}/corpus.jsonl.gz", "rt", encoding="utf-8") as f:
            corpus[ds] = [json.loads(l) for l in f]
        items[ds] = load_traj(a.traj, ds)
    model = Filter(ab).to(dev)
    evals = lambda: {k: v for ds in DATASETS for sp in a.splits.split(",")
                     for k, v in predict(model, split(items[ds], sp)[:int(os.environ.get("SMOKE", 10 ** 9))],
                                         corpus[ds], tok, dev, ab, a.traj).items()}
    if a.mode == "predict":
        model.load_state_dict(torch.load(f"{ROOT}/models/essn_{a.src_traj}_{a.tag}.pt", map_location=dev))
        os.makedirs(f"{ROOT}/preds/{a.traj}", exist_ok=True)
        json.dump(evals(), open(f"{ROOT}/preds/{a.traj}/essn_{a.tag}_from_{a.src_traj}.json", "w"))
        return
    train = [tr for ds in DATASETS for tr in split(items[ds], "train")]
    if os.environ.get("SMOKE"):
        train = train[:int(os.environ["SMOKE"])]
    enc_p = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    oth_p = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": 2e-5}, {"params": oth_p, "lr": 5e-4}], weight_decay=0.01)
    steps = a.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler(DEVT, enabled=DEVT == "cuda")
    step, t0, run = 0, time.time(), []
    for ep in range(a.epochs):
        random.shuffle(train)
        for tr in train:
            b = build(tr, corpus[tr["dataset"]], tok, dev, ab, a.traj)
            with torch.autocast(DEVT, dtype=ADT):
                logpis, V = model(b)
            loss, parts = losses([lp.float() for lp in logpis], V.float(), b, ab, model.covs)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1; run.append(parts)
            if step % 200 == 0:
                print(f"ep{ep} {step}/{steps} {np.mean(run[-200:], 0).round(4)} alpha {model.log_alpha.exp().item():.3f} "
                      f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(f"{ROOT}/models", exist_ok=True)
    torch.save(model.state_dict(), f"{ROOT}/models/essn_{a.traj}_{a.tag}.pt")
    os.makedirs(f"{ROOT}/preds/{a.traj}", exist_ok=True)
    json.dump(evals(), open(f"{ROOT}/preds/{a.traj}/essn_{a.tag}.json", "w"))
    print("done", a.tag, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
