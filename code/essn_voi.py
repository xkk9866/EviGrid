"""ESSN (v12): answer-in-hand posterior with a value-of-information head.

One network, three parts, each one an object of the decision problem:

  1. Two readers of the same evidence, for every hypothesis h in H_t = A_t (distinct drafts so far) + {bot}
       evidence reader   M_t(h) = sum_{p in E_t + {null}} psi(h, p)      (cross-encoder, every pair scored once,
                                                                         so re-verifying old drafts is exact)
       generator reader  g_t(h) = log p_LLM("Answer: h" | prompt_t)      (same prompt that produced a_t;
                                                                         bot reads the mass outside A_t)
  2. Open-set answer posterior   pi_t = softmax_h  phi([M_t(h); g_t(h)])
       sufficiency s_t = 1 - pi_t(bot), selection argmax_a pi_t(a), value of stopping v_t = max_a pi_t(a)
  3. Value-of-information head. Continuing can raise the value of the returned answer in exactly two ways
     (strict cases of the submartingale property): disambiguation of the answers in hand, whose headroom is
     s_t - v_t, and discovery of a correct answer not yet in hand, whose headroom is 1 - s_t:
       V_t(k) = v_t + delta_t(k) (s_t - v_t) + gamma_t(k) (1 - s_t),   delta, gamma in [0, 1], non-decreasing in k
     so v_t <= V_t(1) <= ... <= V_t(T-t) <= 1 for the LEARNED model, by construction. The network only learns the
     two rates (how much of each headroom k more rounds are expected to close) from the shape of its belief.

Decision: stop at the first t with v_t >= max_k V_t(k) - lambda (C_{t+k} - C_t); return argmax_a pi_t(a).
Training (equal weights, no auxiliary labels): graded open-set cross-entropy of pi_t; realised quality of the
network's own future selection for V_t(k); martingale coherence of pi_t and the tower property of V_t.

usage:
  python essn_voi.py train --tag voi [--ablate noGen,noEv,single,cached,free,mono,noCoh] --seed 0
  python essn_voi.py predict --tag voi --traj qwen3-4b --src_traj qwen25-7b
"""
import argparse, gzip, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.utils.checkpoint
from transformers import AutoModel, AutoTokenizer
from common import ROOT, DATASETS, T, load_traj, split, normalize

ENC = os.environ.get("ENC", "BAAI/bge-base-en-v1.5")
DEVT = os.environ.get("DEV", "cuda")
ADT = torch.float16 if DEVT == "cuda" else torch.bfloat16
MAXLEN, D_M, TOPK, LC = 192, 128, 3, 48


def log1mexp(x):
    """log(1 - exp(x)) for x < 0"""
    x = x.clamp(max=-1e-6)
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def hypotheses(rounds):
    """distinct drafts in order of appearance, with their birth round"""
    hs, idx = [], {}
    for t, r in enumerate(rounds):
        n = normalize(r["answer"])
        if n not in idx:
            idx[n] = len(hs)
            hs.append({"round": t, "key": n, "f1": r["f1"]})
    return hs, idx


class ESSN(nn.Module):
    def __init__(self, ablate=()):
        super().__init__()
        self.ab = set(ablate)
        self.enc = AutoModel.from_pretrained(ENC)
        self.sep_id = AutoTokenizer.from_pretrained(ENC).sep_token_id
        d = self.enc.config.hidden_size
        if "cls" in self.ab:                                        # ablation: pooled [CLS] relevance messages
            self.psi = nn.Sequential(nn.Linear(d, D_M), nn.GELU())
            self.norm = nn.LayerNorm(D_M)
            edim = D_M
        else:                                                       # requirement-level grounding
            self.pa, self.pb, self.ph = nn.Linear(d, D_M), nn.Linear(d, D_M), nn.Linear(d, D_M)
            self.sup = mlp(D_M + 1, 128, 1)                         # support of claim token j by passage p
            self.crit = mlp(D_M, 128, 1)                            # criticality of claim token j
            edim = D_M + 3
        self.edim = edim
        self.phi = mlp(edim + 3, 128, 1)                            # reader fusion -> posterior logit
        self.qproj = nn.Linear(edim, 32)                            # evidence reading of the question (bot row)
        udim = TOPK + 1 + 2 + (T + 1) + 32
        self.rate = mlp(udim, 128, 2 * T)                           # VOI head: rate increments (delta, gamma)
        self.free = mlp(udim, 128, T)                               # 'free' ablation: unconstrained V_t(k)

    def pairs(self, enc_in):
        """per claim-passage pair: support logits of the claim tokens (N, LC) and their projected states (N, LC, D_M)"""
        H = self.enc(**enc_in).last_hidden_state
        tt, am = enc_in["token_type_ids"], enc_in["attention_mask"]
        pmask = (tt == 1) & (am == 1) & (enc_in["input_ids"] != self.sep_id)
        Hc = H[:, 1:1 + LC]
        a = F.normalize(self.pa(Hc), dim=-1)
        bb = F.normalize(self.pb(H), dim=-1)
        A = torch.einsum("nir,njr->nij", a, bb).masked_fill(~pmask[:, None, :], -1.0).amax(-1)  # best alignment
        A = torch.where(pmask.any(1, keepdim=True), A, torch.zeros_like(A))                     # null passage
        hc = self.ph(Hc)
        return self.sup(torch.cat([hc, A.unsqueeze(-1)], -1)).squeeze(-1).float(), hc.float()

    def encode(self, enc_in, chunk=32):
        if "cls" in self.ab:
            f = lambda ids, am, tt: self.enc(input_ids=ids, attention_mask=am, token_type_ids=tt).last_hidden_state[:, 0]
        else:
            f = lambda ids, am, tt: self.pairs({"input_ids": ids, "attention_mask": am, "token_type_ids": tt})
        N = enc_in["input_ids"].shape[0]
        args = lambda i: (enc_in["input_ids"][i:i + chunk], enc_in["attention_mask"][i:i + chunk],
                          enc_in["token_type_ids"][i:i + chunk])
        if not self.training or N <= chunk:
            outs = [f(*args(i)) for i in range(0, N, chunk)]
        else:
            outs = [torch.utils.checkpoint.checkpoint(f, *args(i), use_reentrant=False) for i in range(0, N, chunk)]
        if "cls" in self.ab:
            return torch.cat(outs)
        return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])

    def reading(self, S, Hc, cl):
        """evidence reading of one hypothesis from its pairs with the current evidence.
        S: (P, LC) support logits, Hc: (P, LC, D_M), cl: claim length. A requirement (claim token) is covered if
        SOME passage supports it (noisy-OR over evidence); the claim is grounded if its critical requirements are
        covered (criticality-weighted AND)."""
        S, Hc = S[:, :cl], Hc[:, :cl]
        h = Hc.mean(0)
        w = F.softplus(self.crit(h).squeeze(-1)) + 1e-3
        wn = w / w.sum()
        logk = log1mexp(F.logsigmoid(-S).sum(0))              # log coverage of every requirement
        k = logk.exp()
        land = (w * logk).sum() / w.sum()                     # weighted AND (normalised log-probability)
        smin = -0.1 * torch.log((wn * torch.exp(-k / 0.1)).sum())
        return torch.cat([(wn[:, None] * h).sum(0), torch.stack([(wn * k).sum(), smin, land.clamp(min=-10) / 5])])

    def forward(self, b):
        """b: one question. Returns per round: log-posterior over (alive hypotheses + bot), V_t(1..T), rates."""
        dev = b["enc"]["input_ids"].device
        nh, npas = b["nh"], b["np"]
        if "cls" in self.ab:
            msg = self.psi(self.encode(b["enc"]).float()).view(nh + 1, npas, D_M)
        else:
            Sg, Hg = self.encode(b["enc"])
            Sg, Hg = Sg.view(nh + 1, npas, LC), Hg.view(nh + 1, npas, LC, D_M)
        logpis, Vs, rates, prev = [], [], [], None
        for t in range(T + 1):
            rows = b["alive"][t] + [nh]
            # re-verification: every hypothesis is read against ALL of E_t; 'cached': against E at its birth round
            ecols = [b["cols"][b["birth"][i]] if ("cached" in self.ab and i < nh) else b["cols"][t] for i in rows]
            if "cls" in self.ab:
                M = self.norm(torch.stack([msg[i, torch.tensor(c, device=dev)].sum(0) for i, c in zip(rows, ecols)]))
            else:
                M = torch.stack([self.reading(Sg[i, torch.tensor(c, device=dev)], Hg[i, torch.tensor(c, device=dev)],
                                              b["cl"][i]) for i, c in zip(rows, ecols)])
            rows = torch.tensor(rows, device=dev)
            if "noEv" in self.ab:
                M = M * 0
            g = b["gen"][t].to(dev)
            if "noGen" in self.ab:
                g = g * 0
            isbot = torch.zeros(len(rows), 1, device=dev); isbot[-1] = 1
            eta = self.phi(torch.cat([M, g, isbot], -1)).squeeze(-1)
            logpi = torch.log_softmax(eta, 0)
            logpis.append(logpi)
            pi = logpi.exp()
            v, s = pi[:-1].max(), 1 - pi[-1]
            top = torch.sort(pi[:-1], descending=True).values[:TOPK]
            top = torch.cat([top, top.new_zeros(TOPK - len(top))])
            dv, ds = (v - prev[0], s - prev[1]) if prev is not None else (v * 0, s * 0)
            prev = (v, s)
            u = torch.cat([top, pi[-1:], torch.stack([dv, ds]), F.one_hot(torch.tensor(t, device=dev), T + 1).float(),
                           self.qproj(M[-1])])
            if "free" in self.ab:
                V = torch.sigmoid(self.free(u))
                rt = torch.zeros(2, T, device=dev)
            else:
                inc = F.softplus(self.rate(u)).view(2, T)
                rt = 1 - torch.exp(-torch.cumsum(inc, -1))           # delta_t(k), gamma_t(k), non-decreasing in k
                if "mono" in self.ab:                                # one rate for the whole headroom 1 - v_t
                    V = v + rt[0] * (1 - v)
                else:
                    V = v + rt[0] * (s - v) + rt[1] * (1 - s)
            Vs.append(V)
            rates.append(rt)
        return logpis, torch.stack(Vs), torch.stack(rates)


def build(tr, corpus, tok, dev, ab):
    R = tr["rounds"]
    hs, idx = hypotheses(R)
    nh = len(hs)
    pids, arr = [], []
    for t, r in enumerate(R):
        for p in r["new_pids"]:
            pids.append(p); arr.append(t)
    pas = [""] + [f"{corpus[p]['title']}: {corpus[p]['text']}" for p in pids]
    arr = [0] + arr
    claims = [f"{tr['question']} Answer: {R[h['round']]['answer'] or 'unknown'}" for h in hs] + [f"{tr['question']} Answer:"]
    ta, tb = [], []
    for c in claims:
        ta += [c] * len(pas); tb += pas
    enc = tok(ta, tb, truncation="only_second", max_length=MAXLEN, padding="max_length", return_tensors="pt")
    enc = {k: v.to(dev) for k, v in enc.items()}
    cl = [max(1, min(len(tok(c, add_special_tokens=False)["input_ids"]), LC)) for c in claims]
    alive, cols, gen, ys = [], [], [], []
    for t, r in enumerate(R):
        al = [idx[normalize(r["answer"])]] if "single" in ab else [i for i, h in enumerate(hs) if h["round"] <= t]
        alive.append(al)
        cols.append([j for j, a in enumerate(arr) if a <= t])
        src = t
        lp, nt = [], []
        for i in al:
            # generator reading under the prompt of round t (re-verification) or of the birth round (cached)
            u = hs[i]["round"] if "cached" in ab else src
            L = R[u].get("lik", {})
            sv, n = -30.0, 1
            for cj, (s_, n_) in L.items():
                if normalize(R[cj]["answer"]) == hs[i]["key"]:
                    sv, n = s_, max(n_, 1); break
            lp.append(max(sv, -30.0)); nt.append(n)
        lp = torch.tensor(lp)
        mass = lp.exp().sum().clamp(max=1 - 1e-4)
        lb = torch.log1p(-mass).view(1)
        tot = torch.cat([lp, lb])
        per = torch.cat([lp / torch.tensor(nt, dtype=torch.float), lb])
        gen.append(torch.stack([tot / 5, per / 2], 1))
        ys.append(torch.tensor([hs[i]["f1"] for i in al], device=dev))
    return {"enc": enc, "nh": nh, "np": len(pas), "alive": alive, "cols": cols, "gen": gen, "ys": ys, "hs": hs, "cl": cl,
            "birth": [h["round"] for h in hs], "keys": [[hs[i]["key"] for i in al] for al in alive]}


def losses(logpis, V, b, ab):
    ys = b["ys"]
    # graded open-set target: z_t = best quality in hand, spread over candidates by quality, 1 - z_t on bot
    l_post = []
    for lp, y in zip(logpis, ys):
        z = y.max()
        l_post.append(-(torch.cat([z * y / y.sum().clamp(min=1e-6), (1 - z).view(1)]) * lp).sum())
    l_post = torch.stack(l_post).mean()
    # V_t(k): realised quality of the answer the network itself returns after k more rounds (stop-gradient)
    sel = torch.stack([y[int(lp[:-1].detach().argmax())] for lp, y in zip(logpis, ys)])
    vnow = torch.stack([lp[:-1].detach().exp().max() for lp in logpis])
    Vd = V.detach()
    tgt, tow, m = torch.zeros_like(V), torch.zeros_like(V), torch.zeros_like(V)
    for t in range(T):
        for k in range(1, T - t + 1):
            tgt[t, k - 1] = sel[t + k]
            tow[t, k - 1] = vnow[t + 1] if k == 1 else Vd[t + 1, k - 2]
            m[t, k - 1] = 1
    Vc = V.clamp(1e-5, 1 - 1e-5)
    l_val = (F.binary_cross_entropy(Vc, tgt, reduction="none") * m).sum() / m.sum()
    l_coh = torch.zeros((), device=V.device)
    if "noCoh" not in ab:
        # tower property V_t(k) = E[V_{t+1}(k-1) | H_t] and martingale coherence of pi_t: E[P pi_{t+1} | H_t] = pi_t,
        # where P keeps the mass of hypotheses already in hand and moves newborn mass to bot
        l_tow = (F.binary_cross_entropy(Vc, tow, reduction="none") * m).sum() / m.sum()
        lc = []
        for t in range(T):
            nxt = logpis[t + 1].detach().exp()
            pos = {k: n for n, k in enumerate(b["keys"][t + 1])}
            if any(k not in pos for k in b["keys"][t]):
                continue
            kept = nxt[torch.tensor([pos[k] for k in b["keys"][t]], device=V.device)]
            lc.append(-(torch.cat([kept, (1 - kept.sum()).clamp(min=0).view(1)]) * logpis[t]).sum())
        l_coh = l_tow + (torch.stack(lc).mean() if lc else 0.0)
    return l_post + l_val + l_coh, (l_post.item(), l_val.item(), float(l_coh))


def predict(model, trajs, corpus, tok, dev, ab):
    model.eval()
    out = {}
    with torch.no_grad(), torch.autocast(DEVT, dtype=ADT):
        for tr in trajs:
            b = build(tr, corpus, tok, dev, ab)
            logpis, V, rt = model(b)
            V, rt = V.float().cpu().numpy(), rt.float().cpu().numpy()
            M = np.zeros((T + 1, T + 1))
            S, suff, rates = [], [], []
            for t, lp in enumerate(logpis):
                pi = lp.float().exp().cpu().numpy()
                M[t, 0] = pi[:-1].max()
                M[t, 1:T - t + 1] = V[t, :T - t]
                S.append([[int(b["hs"][i]["round"]), round(float(p), 4)] for i, p in zip(b["alive"][t], pi[:-1])])
                suff.append(round(1 - float(pi[-1]), 4))
                rates.append(rt[t, :, :T - t].round(4).tolist())
            out[f"{tr['dataset']}::{tr['qid']}"] = {"V": M.round(4).tolist(), "S": S, "suff": suff, "rates": rates,
                                                     "lik_used": "noGen" not in ab}
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--tag", default="voi")
    ap.add_argument("--traj", default="qwen25-7b")
    ap.add_argument("--src_traj", default="qwen25-7b")
    ap.add_argument("--ablate", default="")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default="calib,test,calib2,cert,test2")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed)
    ab = set(x for x in a.ablate.split(",") if x)
    dev = DEVT
    tok = AutoTokenizer.from_pretrained(ENC)
    corpus, items = {}, {}
    for ds in DATASETS:
        with gzip.open(f"{ROOT}/data/proc/{ds}/corpus.jsonl.gz", "rt", encoding="utf-8") as f:
            corpus[ds] = [json.loads(l) for l in f]
        items[ds] = load_traj(a.traj, ds)
    model = ESSN(ab).to(dev)
    lim = int(os.environ.get("SMOKE", 10 ** 9))
    evals = lambda: {k: v for ds in DATASETS for sp in a.splits.split(",")
                     for k, v in predict(model, split(items[ds], sp)[:lim], corpus[ds], tok, dev, ab).items()}
    if a.mode == "predict":
        model.load_state_dict(torch.load(f"{ROOT}/models/essn_{a.src_traj}_{a.tag}.pt", map_location=dev))
        os.makedirs(f"{ROOT}/preds/{a.traj}", exist_ok=True)
        json.dump(evals(), open(f"{ROOT}/preds/{a.traj}/essn_{a.tag}_from_{a.src_traj}.json", "w"))
        return
    train = [tr for ds in DATASETS for tr in split(items[ds], "train")][:lim]
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
            b = build(tr, corpus[tr["dataset"]], tok, dev, ab)
            with torch.autocast(DEVT, dtype=ADT):
                logpis, V, _ = model(b)
            loss, parts = losses([lp.float() for lp in logpis], V.float(), b, ab)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1; run.append(parts)
            if step % 400 == 0:
                print(f"ep{ep} {step}/{steps} {np.mean(run[-400:], 0).round(4)} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(f"{ROOT}/models", exist_ok=True)
    torch.save(model.state_dict(), f"{ROOT}/models/essn_{a.traj}_{a.tag}.pt")
    os.makedirs(f"{ROOT}/preds/{a.traj}", exist_ok=True)
    json.dump(evals(), open(f"{ROOT}/preds/{a.traj}/essn_{a.tag}.json", "w"))
    print("done", a.tag, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
