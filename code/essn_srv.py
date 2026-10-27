"""ESSN: Evidence-Sufficiency State Network (v3: open-set answer posterior).

One network, one forward chain:
  claim-conditioned pair encoding -> requirement-support convolution (CNN on interaction maps)
  -> noisy-AND-of-OR sufficiency, evaluated for EVERY candidate answer seen so far against the
     CURRENT evidence (contrastive evidential posterior over candidates)
  -> recurrent belief state -> ONE open-set answer posterior pi_t over (verified candidates + "not yet found"):
     sufficiency = 1 - pi_t(not yet found); selection = argmax_a pi_t(a); stop = sufficiency now vs. the
     look-ahead profile V_t(k) of future sufficiency, net of cost.

usage:
  python essn.py train   --tag full [--ablate noCNN,noAND,noEvSup,noLLM,noGRU,noProfile,noContrast,noReverify,latestTarget,twoHead,simple,decision]
  python essn.py predict --tag full --traj qwen3-14b --src_traj qwen25-7b      (zero-shot transfer)
"""
import argparse, gzip, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.utils.checkpoint
from transformers import AutoModel, AutoTokenizer
from common import ROOT, DATASETS, T, load_traj, split, normalize

ENC = "BAAI/bge-base-en-v1.5"
DEVT = os.environ.get("DEV", "cuda")  # "cpu" only for smoke tests
ADT = torch.float16 if DEVT == "cuda" else torch.bfloat16
MAXLEN, LC_MAX, K = 192, 64, T
MAXC = T + 1  # every distinct draft is a candidate (no cap)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def log1mexp(x):
    """log(1 - exp(x)) for x < 0, numerically stable."""
    x = x.clamp(max=-1e-6)
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def lik_feats(R, rowp, ci, src):
    """generator-side verification of candidate ci, as seen at evidence state src:
    mean token log-likelihood under E_src, relative posterior among the candidates of row src, its mean over the
    rows since the candidate appeared (stability of support), and its change since then. src = t under
    re-verification; src = ci (birth round) for cached scoring."""
    L = R[src].get("lik", {}) if rowp[src] else {}
    if ci not in L:
        return [0.0, 0.0, 0.0, 0.0]
    s, n = L[ci]
    hist = [rowp[u][ci] for u in range(ci, src + 1) if ci in rowp[u]]
    return [max(s / max(n, 1), -10.0) / 5, rowp[src][ci], sum(hist) / len(hist), rowp[src][ci] - hist[0]]


def candidates(rounds, t, contrast=True):
    """all distinct drafts 0..t (most recent first, the current draft first); contrast=False keeps only the
    current draft. No filtering: an empty/'unknown' draft is simply a candidate with little support."""
    out, seen = [], set()
    for i in range(t, -1, -1):
        n = normalize(rounds[i]["answer"])
        if n in seen:
            continue
        seen.add(n)
        out.append(i)
        if len(out) == (MAXC if contrast else 1):
            break
    return out  # round indices; the candidate text is rounds[i]["answer"]


class ESSN(nn.Module):
    def __init__(self, ablate=()):
        super().__init__()
        self.ab = set(ablate)
        self.enc = AutoModel.from_pretrained(ENC)
        # memory is bounded by chunked checkpointing in support_chunked()
        self.sep_id = AutoTokenizer.from_pretrained(ENC).sep_token_id
        d, r = self.enc.config.hidden_size, 128
        self.pa, self.pb, self.hp = nn.Linear(d, r), nn.Linear(d, r), nn.Linear(d, r)
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 16, 3, padding=1), nn.GELU())
        self.sup = mlp(r + 16 + 1, 128, 1)
        self.crit = mlp(d, 128, 1)
        self.cls_suff = mlp(d, 128, 1)  # noAND ablation only
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.cdim = r + 7  # per-candidate sufficiency summary
        self.xdim = self.cdim + 6 + 6 + (T + 1)
        self.inp = nn.Linear(self.xdim, 256)
        self.gru = nn.GRUCell(256, 256)
        self.mlp_b = mlp(self.xdim, 256, 256)  # noGRU ablation
        self.head = mlp(256 + self.xdim, 256, K + 1)
        self.beta = nn.Parameter(torch.tensor([1.0] + [0.3] * K))
        self.qhead = mlp(self.cdim + 9 + 256, 256, 1)
        self.bot = mlp(256 + self.xdim, 256, 1)
        self.log_kappa = nn.Parameter(torch.tensor(math.log(10.0)))  # sharpness of the soft stop rule (learned)  # logit of "the correct answer is not (yet) among the verified candidates"

    # ---- pair level: requirement support logits -------------------------------------------------
    def support_chunked(self, enc_in, chunk=32):
        """Memory-bounded verification: claim-passage pairs are processed in chunks under activation
        checkpointing, so peak memory depends on the chunk size only, not on how many candidates and passages
        a question has. Numerically identical to support(enc_in)."""
        N = enc_in["input_ids"].shape[0]
        if not self.training or N <= chunk:
            return self.support(enc_in)
        outs = []
        for i in range(0, N, chunk):
            part = {k: v[i:i + chunk] for k, v in enc_in.items()}
            outs.append(torch.utils.checkpoint.checkpoint(
                lambda ids, am, tt: self.support({"input_ids": ids, "attention_mask": am, "token_type_ids": tt}),
                part["input_ids"], part["attention_mask"], part["token_type_ids"], use_reentrant=False))
        return tuple(torch.cat([o[j] for o in outs]) for j in range(3))

    def support(self, enc_in):
        H = self.enc(**enc_in).last_hidden_state
        N, L, _ = H.shape
        tt, am = enc_in["token_type_ids"], enc_in["attention_mask"]
        pmask = (tt == 1) & (am == 1) & (enc_in["input_ids"] != self.sep_id)
        Hc = H[:, 1:1 + LC_MAX]
        a = F.normalize(self.pa(Hc), dim=-1)
        b = F.normalize(self.pb(H), dim=-1)
        A = torch.einsum("nir,njr->nij", a, b) * pmask[:, None, :]
        has_p = pmask.any(1)
        neg = torch.finfo(A.dtype).min
        raw = A.masked_fill(~pmask[:, None, :], neg).amax(-1)
        raw = torch.where(has_p[:, None], raw, torch.zeros_like(raw))
        if "noCNN" in self.ab:
            cm = torch.zeros(N, LC_MAX, 16, device=H.device)
        else:
            C = self.conv(A.unsqueeze(1))
            cm = C.masked_fill(~pmask[:, None, None, :], neg).amax(-1).transpose(1, 2)
            cm = torch.where(has_p[:, None, None], cm, torch.zeros_like(cm))
        s = self.sup(torch.cat([self.hp(Hc), cm, raw.unsqueeze(-1)], -1)).squeeze(-1)
        return s.float(), Hc.float(), H[:, 0].float()

    # ---- claim level: noisy-AND-of-OR -----------------------------------------------------------
    def sufficiency(self, s, Hc, cls, idx, new, cl):
        st = s[idx, :cl]
        h = Hc[idx, :cl].mean(0)
        w = F.softplus(self.crit(h).squeeze(-1)) + 1e-3
        wn = w / w.sum()
        if "noAND" in self.ab:
            c = torch.sigmoid(st).mean(0)
            logP = -F.softplus(-self.cls_suff(cls[idx].mean(0))).squeeze()
        else:
            logc = log1mexp(F.logsigmoid(-st).sum(0))
            c = logc.exp()
            logP = (w * logc).sum()
        if new:
            new_cov = (wn * (1 - torch.exp(F.logsigmoid(-st[new]).sum(0)))).sum()
        else:
            new_cov = torch.zeros((), device=s.device)
        cmean = (wn * c).sum()
        cmin = -0.1 * torch.log((wn * torch.exp(-c / 0.1)).sum())
        logitP = (logP - log1mexp(logP.clamp(max=-1e-4))).clamp(-15, 15)
        summ = torch.cat([(wn[:, None] * self.hp(h)).sum(0),
                          torch.stack([cmean, cmin, new_cov, logP.clamp(min=-30) / 10, logitP / 5, w.sum().log() / 5,
                                       torch.tensor(float(len(idx)) / 16, device=s.device)])])
        return summ, logP, logitP, cmean, cmin

    def forward(self, batch):
        s, Hc, cls = self.support_chunked(batch["enc"])
        dev = s.device
        B = batch["B"]
        X, logitPs, Qs = [], [], []
        for q in range(B):
            prev = None
            xs = []
            for t in range(T + 1):
                cands = batch["cands"][q][t]
                outs = [self.sufficiency(s, Hc, cls, c["idx"], c["new"], c["cl"]) for c in cands]
                summ = torch.stack([o[0] for o in outs])
                logPc = torch.stack([o[1] for o in outs])
                pi = torch.softmax(logPc / self.log_tau.exp(), 0)
                ent = -(pi * torch.log(pi + 1e-9)).sum()
                margin = (logPc[0] - logPc[1:].max()).clamp(-30, 30) / 10 if len(cands) > 1 else torch.zeros((), device=dev)
                cmean, cmin = outs[0][3], outs[0][4]
                d_mean = cmean - prev[0] if prev is not None else torch.zeros((), device=dev)
                d_min = cmin - prev[1] if prev is not None else torch.zeros((), device=dev)
                prev = (cmean, cmin)
                f = batch["feat"][q][t]
                if "noLLM" in self.ab:
                    f = f.clone(); f[:2] = 0
                pi_cur = pi[0]
                if "noContrast" in self.ab:
                    pi_cur, ent, margin = torch.ones((), device=dev), ent * 0, margin * 0
                onehot = F.one_hot(torch.tensor(t, device=dev), T + 1).float()
                x = torch.cat([summ[0], torch.stack([d_mean, d_min, pi_cur, ent, margin,
                                                     torch.tensor(len(cands) / MAXC, device=dev)]), f, onehot])
                xs.append(x)
                logitPs.append(outs[0][2])
                cf = torch.stack([c["cfeat"] for c in cands]).to(dev)
                if "noLLM" in self.ab:
                    cf = cf.clone(); cf[:, 3:5] = 0
                cf = torch.cat([pi[:, None], cf[:, 1:]], 1)
                Qs.append((q, t, summ, cf))
            X.append(torch.stack(xs))
        X = torch.stack(X)
        logitP = torch.stack(logitPs).view(B, T + 1)
        if "noGRU" in self.ab:
            Hs = self.mlp_b(X)
        else:
            hs, hcur = [], torch.zeros(B, 256, device=dev)
            for t in range(T + 1):
                hcur = self.gru(torch.tanh(self.inp(X[:, t])), hcur)
                hs.append(hcur)
            Hs = torch.stack(hs, 1)
        V = self.head(torch.cat([Hs, X], -1)) + self.beta * logitP.unsqueeze(-1)
        # open-set answer posterior: logits over the verified candidates plus the null outcome (last entry)
        Q = [torch.cat([self.qhead(torch.cat([summ, cf, Hs[q, t].expand(len(summ), -1)], -1)).squeeze(-1),
                        self.bot(torch.cat([Hs[q, t], X[q, t]], -1))]) for q, t, summ, cf in Qs]
        if not ({"freeHead", "twoHead", "latestTarget"} & self.ab):
            V = self.submartingale_profile(V, Q, B)
        return V, logitP, Q

    def submartingale_profile(self, G, Q, B):
        """Look-ahead profile that satisfies the submartingale property (Prop. 2) BY CONSTRUCTION:
        V_t(0) = v_t = max_a pi_t(a) (value of stopping, read from the posterior) and
        V_t(k) = 1 - (1 - v_t) * exp(-sum_{j<=k} softplus(g_j)),  so  v_t <= V_t(1) <= ... <= 1.
        The network only learns the non-negative expected gain of further retrieval. Returned as logits."""
        v = torch.stack([torch.softmax(q.float(), -1)[:-1].max() for q in Q]).view(B, T + 1)
        log1mv = torch.log1p(-v.clamp(max=1 - 1e-6))                                  # log(1 - v)
        cum = torch.cumsum(F.softplus(G[:, :, 1:].float()), -1)                         # non-negative increments
        Lk = torch.cat([log1mv.unsqueeze(-1), log1mv.unsqueeze(-1) - cum], -1)         # log(1 - V_t(k))
        return log1mexp(Lk.clamp(max=-1e-6)) - Lk                                      # logit V_t(k)


# ------------------------------------------------------------------------------------------------
class Data:
    def __init__(self, tag, dss):
        self.items, self.corpus = {}, {}
        for ds in dss:
            with gzip.open(f"{ROOT}/data/proc/{ds}/corpus.jsonl.gz", "rt", encoding="utf-8") as f:
                self.corpus[ds] = [json.loads(l) for l in f]
            self.items[ds] = load_traj(tag, ds)


def build_batch(trajs, corpus, tok, dev, contrast=True, reverify=True, latest_target=False, use_lik=True):
    """reverify=False (noReverify): an earlier candidate keeps the verification it received when it was
    produced (its own evidence E_ci and new-evidence set), i.e. cached scoring instead of re-verification.
    latest_target=True: look-ahead profile supervised with the latest draft's quality instead of z."""
    texts_a, texts_b, allc, feats, meta = [], [], [], [], []
    for tr in trajs:
        cp = corpus[tr["dataset"]]
        R = tr["rounds"]
        # generator-side likelihood field: per round, the generator's relative posterior over the candidates
        # it scored under the same evidence (softmax of total log-likelihoods); empty if not available / ablated
        rowp = []
        for r in R:
            L = r.get("lik", {}) if use_lik else {}
            if L:
                m = max(s for s, _ in L.values())
                w = {ci: math.exp(s - m) for ci, (s, _) in L.items()}
                Z = sum(w.values())
                rowp.append({ci: v / Z for ci, v in w.items()})
            else:
                rowp.append({})
        ev, qc, qf, qm, ev_at = [], [], [], [], []
        for t, r in enumerate(R):
            ev = ev + r["new_pids"]
            ev_at.append(ev)
            cl_list = []
            rounds_c = candidates(R, t, contrast)
            for ci in rounds_c:
                src = t if (reverify or ci == t) else ci  # evidence the candidate is verified against
                pas = [""] + [f"{cp[p]['title']}: {cp[p]['text']}" for p in ev_at[src]]
                nnew = len(R[src]["new_pids"])
                claim = f"{tr['question']} Answer: {R[ci]['answer'] or 'unknown'}"
                start = len(texts_a)
                texts_a += [claim] * len(pas)
                texts_b += pas
                clen = min(len(tok(claim, add_special_tokens=False)["input_ids"]), LC_MAX)
                cl_list.append({"idx": list(range(start, start + len(pas))),
                                "new": list(range(len(pas) - nnew, len(pas))) if nnew else [],
                                "cl": clen,
                                "cfeat": torch.tensor([0.0, float(ci == t), (t - ci) / T, R[ci]["lp_mean"] / 5, R[ci]["lp_min"] / 10]
                                                      + lik_feats(R, rowp, ci, src)),
                                "f1": R[ci]["f1"]})
            qc.append(cl_list)
            qm.append(rounds_c)
            pr = list(rowp[t].values())
            ent = -sum(p * math.log(p + 1e-9) for p in pr) if pr else 0.0
            qf.append(torch.tensor([r["lp_mean"] / 5, r["lp_min"] / 10, r["same_prev"], float(len(ev)) / 15,
                                    ent, max(pr) if pr else 0.0], dtype=torch.float, device=dev))
        allc.append(qc); feats.append(qf); meta.append(qm)
    enc = tok(texts_a, texts_b, truncation="only_second", max_length=MAXLEN, padding=True, return_tensors="pt")
    if enc["input_ids"].shape[1] < LC_MAX + 2:
        enc = tok(texts_a, texts_b, truncation="only_second", max_length=MAXLEN, padding="max_length", return_tensors="pt")
    enc = {k: v.to(dev) for k, v in enc.items()}
    # value of stopping at round t = quality of the best candidate available at t
    z = torch.tensor([[max(c["f1"] for c in qc[t]) for t in range(T + 1)] for qc in allc], device=dev)
    if latest_target:
        z = torch.tensor([[r["f1"] for r in tr["rounds"]] for tr in trajs], device=dev)
    g = torch.tensor([[r["gold_all"] for r in tr["rounds"]] for tr in trajs], device=dev)
    qy = [torch.tensor([c["f1"] for c in allc[q][t]], device=dev) for q in range(len(trajs)) for t in range(T + 1)]
    # normalised cumulative cost of stopping at each round (for the decision-focused objective)
    cost = torch.tensor([[r["cum_tok"] / tr["rounds"][T]["cum_tok"] for r in tr["rounds"]] for tr in trajs], device=dev)
    keys = [[normalize(tr["rounds"][ci]["answer"]) for ci in meta[q][t]] for q, tr in enumerate(trajs) for t in range(T + 1)]
    return {"enc": enc, "cands": allc, "feat": feats, "B": len(trajs), "meta": meta, "cost": cost, "keys": keys}, z, g, qy


def selected_quality(Q, qy, B):
    """quality f(argmax_a pi_s(a)) of the answer the network itself would return at every round s
    (stop-gradient): the value that the stop-and-select policy actually realises."""
    sel = torch.zeros(B, T + 1, device=qy[0].device)
    for i, (q, y) in enumerate(zip(Q, qy)):
        sel[i // (T + 1), i % (T + 1)] = y[int(q[:-1].detach().argmax())]
    return sel


def coherence_targets(Q, keys, B):
    """Martingale projection of the next-round posterior onto the current candidate set.
    A coherent open-set posterior satisfies E[P pi_{t+1} | H_t] = pi_t, where P keeps the mass of every
    candidate already in hand (A_t is contained in A_{t+1}) and moves the mass of candidates first produced at
    round t+1, together with pi_{t+1}(not-found), to the not-found outcome of round t: from the viewpoint of
    round t they are answers that have not been found yet. Returned as stop-gradient targets for t < T."""
    out = {}
    for q in range(B):
        for t in range(T):
            i, j = q * (T + 1) + t, q * (T + 1) + t + 1
            nxt = torch.softmax(Q[j].detach().float(), -1)
            pos = {k: n for n, k in enumerate(keys[j])}
            idx = torch.tensor([pos[k] for k in keys[i]], device=nxt.device)
            kept = nxt[idx]
            out[i] = torch.cat([kept, (1.0 - kept.sum()).clamp(min=0).view(1)])
    return out


def loss_fn(V, logitP, Q, z, g, qy, ab, cost=None, log_kappa=None, keys=None):
    # look-ahead targets: realised quality of the answer selected after k more rounds (selected-answer value);
    # the latestTarget ablation keeps the latest draft's quality (z already holds it, see build_batch)
    if "latestTarget" not in ab:
        z = selected_quality(Q, qy, V.shape[0])
    tgt = torch.zeros_like(V)
    mask = torch.zeros_like(V)
    for t in range(T + 1):
        for k in range(T + 1 - t):
            tgt[:, t, k] = z[:, t + k]
            mask[:, t, k] = 1
    posterior = "twoHead" not in ab and "latestTarget" not in ab
    if posterior:  # the value of stopping now IS max_a pi_t(a), read from the posterior; the profile models the future
        mask[:, :, 0] = 0
    if "noProfile" in ab:
        mask[:, :, 1:] = 0
    l_prof = (F.binary_cross_entropy_with_logits(V, tgt, reduction="none") * mask).sum() / mask.sum().clamp(min=1)
    gm = (g >= 0).float()
    l_ev = (F.binary_cross_entropy_with_logits(logitP, g.clamp(min=0), reduction="none") * gm).sum() / gm.sum().clamp(min=1)
    if "noEvSup" in ab:
        l_ev = l_ev * 0
    if "twoHead" in ab:  # ablation: independent per-candidate sigmoid values instead of one shared posterior
        l_q = torch.stack([F.binary_cross_entropy_with_logits(q[:-1].float(), y, reduction="sum") for q, y in zip(Q, qy)]).sum() \
            / sum(len(y) for y in qy)
    else:  # soft-target cross-entropy of the open-set answer posterior over (candidates, not-yet-found)
        ls = []
        for q, y in zip(Q, qy):
            zt = y.max()
            tgt = torch.cat([zt * y / y.sum().clamp(min=1e-6), (1 - zt).view(1)])
            ls.append(-(tgt * F.log_softmax(q.float(), -1)).sum())
        l_q = torch.stack(ls).mean()
    l_coh = torch.zeros((), device=V.device)
    if "coh" in ab and keys is not None and "twoHead" not in ab:
        # temporal coherence: the posterior is a martingale (cross-entropy to the projected next posterior)
        # and the look-ahead profile obeys the tower property V_t(k) = E[V_{t+1}(k-1) | H_t], V_{t+1}(0) = v_{t+1}.
        # Both targets estimate the same quantities as the realised-outcome targets above (equal weights).
        B = V.shape[0]
        ct = coherence_targets(Q, keys, B)
        lc = [-(tg * F.log_softmax(Q[i].float(), -1)).sum() for i, tg in ct.items()]
        l_post = torch.stack(lc).mean() if lc else l_coh
        vnext = torch.stack([torch.softmax(Q[q * (T + 1) + t].detach().float(), -1)[:-1].max()
                             for q in range(B) for t in range(T + 1)]).view(B, T + 1)
        Vs = torch.sigmoid(V.detach())
        tt = torch.zeros_like(V); mm = torch.zeros_like(V)
        for t in range(T):
            for k in range(1, T + 1 - t):
                tt[:, t, k] = vnext[:, t + 1] if k == 1 else Vs[:, t + 1, k - 1]
                mm[:, t, k] = 1
        if "noProfile" in ab:
            mm = mm * 0
        l_tow = (F.binary_cross_entropy_with_logits(V, tt, reduction="none") * mm).sum() / mm.sum().clamp(min=1)
        l_coh = l_post + l_tow
    l_dec = torch.zeros((), device=V.device)
    if "decision" in ab and cost is not None:
        l_dec = -decision_utility(V, Q, qy, cost, log_kappa)
    # likelihood terms + (decision-focused variant) the exact expected utility of the policy, equal weights
    return l_prof + l_q + l_ev + l_dec + l_coh, (l_prof.item(), l_q.item(), l_ev.item(), l_coh.item())


def decision_utility(V, Q, qy, cost, log_kappa, n_lambda=4):
    """Exact expected utility E[f(selected answer)] - lambda * E[cost] of the soft stop-and-select policy on a
    full rollout. The stop hazard at round t is sigma(kappa * (v_t - c_t)) with the network's own decision
    quantities: v_t = max_a pi_t(a) (value of stopping) and c_t = max_k [V_t(k) - lambda * (C_{t+k} - C_t)]
    (net value of continuing); the answer is drawn from pi_t. lambda is sampled log-uniformly so that one
    network is trained for the whole accuracy-cost frontier. kappa is a learned sharpness."""
    B = V.shape[0]
    Vs = torch.sigmoid(V)
    post = [torch.softmax(q.float(), -1) for q in Q]
    v = torch.stack([p[:-1].max() for p in post]).view(B, T + 1)           # value of stopping
    qexp = torch.stack([(p[:-1] * y).sum() for p, y in zip(post, qy)]).view(B, T + 1)  # E[f] of the drawn answer
    kappa = log_kappa.exp()
    total = 0.0
    for _ in range(n_lambda):
        lam = float(torch.exp(torch.empty(1).uniform_(math.log(0.002), math.log(2.0))))
        surv = torch.ones(B, device=V.device)
        u = torch.zeros(B, device=V.device)
        for t in range(T + 1):
            if t < T:
                cont = torch.stack([Vs[:, t, k] - lam * (cost[:, t + k] - cost[:, t]) for k in range(1, T - t + 1)], -1).max(-1).values
                h = torch.sigmoid(kappa * (v[:, t] - cont))
            else:
                h = torch.ones(B, device=V.device)
            p_stop = surv * h
            u = u + p_stop * (qexp[:, t] - lam * cost[:, t])
            surv = surv * (1 - h)
        total = total + u.mean()
    return total / n_lambda


def predict(model, trajs, corpus, tok, dev, contrast, bs=4, opts={}):
    model.eval()
    if os.environ.get("SMOKE"):
        trajs = trajs[:int(os.environ["SMOKE"])]
    out = {}
    with torch.no_grad(), torch.autocast(DEVT, dtype=ADT):
        for i in range(0, len(trajs), bs):
            chunk = trajs[i:i + bs]
            batch, _, _, _ = build_batch(chunk, corpus, tok, dev, contrast, **opts)
            V, logitP, Q = model(batch)
            V = torch.sigmoid(V.float()).cpu().numpy()
            P = torch.sigmoid(logitP.float()).cpu().numpy()
            for j, tr in enumerate(chunk):
                sel = []
                for t in range(T + 1):
                    ql = Q[j * (T + 1) + t].float()
                    if "twoHead" in model.ab:
                        qv = torch.sigmoid(ql[:-1]).cpu().numpy()
                    else:
                        post = torch.softmax(ql, -1).cpu().numpy()
                        qv = post[:-1]
                        if "latestTarget" not in model.ab:
                            V[j, t, 0] = post[:-1].max()  # stop value = expected quality of the selected answer
                    sel.append([[int(r), round(float(v), 4)] for r, v in zip(batch["meta"][j][t], qv)])
                suff = [round(1.0 - float(torch.softmax(Q[j * (T + 1) + t].float(), -1)[-1]), 4) for t in range(T + 1)]
                out[f"{tr['dataset']}::{tr['qid']}"] = {"V": V[j].round(4).tolist(), "P": P[j].round(4).tolist(), "S": sel, "suff": suff,
                                                         "lik_used": bool(opts.get("use_lik", True) and any(r.get("lik") for r in tr["rounds"]))}
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--traj", default="qwen25-7b")
    ap.add_argument("--src_traj", default="qwen25-7b")
    ap.add_argument("--ablate", default="")
    ap.add_argument("--train_ds", default=",".join(DATASETS))
    ap.add_argument("--eval_ds", default=",".join(DATASETS))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    ab = [x for x in a.ablate.split(",") if x]
    if "simple" in ab:  # strong simplification: candidate cache + cross-encoder [CLS] ranker + MLP stop predictor
        ab = sorted(set(ab) | {"noCNN", "noAND", "noGRU"})
    contrast = "noContrast" not in ab
    opts = {"reverify": "noReverify" not in ab, "latest_target": "latestTarget" in ab, "use_lik": "noLik" not in ab}
    dev = DEVT
    if os.environ.get("MEMFRAC"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["MEMFRAC"]))
    tds, eds = a.train_ds.split(","), a.eval_ds.split(",")
    data = Data(a.traj, sorted(set(eds) if a.mode == "predict" else set(tds) | set(eds)))
    tok = AutoTokenizer.from_pretrained(ENC)
    model = ESSN(ab).to(dev)
    if a.mode == "predict":
        model.load_state_dict(torch.load(f"{ROOT}/models/essn_{a.src_traj}_{a.tag}.pt", map_location=dev), strict=False)
        preds = {}
        for ds in eds:
            for sp in ("calib", "test", "calib2"):
                preds.update(predict(model, split(data.items[ds], sp), data.corpus, tok, dev, contrast, opts=opts))
        os.makedirs(f"{ROOT}/preds/{a.traj}", exist_ok=True)
        json.dump(preds, open(f"{ROOT}/preds/{a.traj}/essn_{a.tag}_from_{a.src_traj}.json", "w"))
        print("done predict", flush=True)
        return
    train = [tr for ds in tds for tr in split(data.items[ds], "train")]
    if os.environ.get("SMOKE"):
        train = train[:int(os.environ["SMOKE"])]
    enc_p = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    oth_p = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": 2e-5}, {"params": oth_p, "lr": 5e-4}], weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler(DEVT, enabled=DEVT == "cuda")
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        random.shuffle(train)
        for i in range(0, len(train), a.bs):
            batch, z, g, qy = build_batch(train[i:i + a.bs], data.corpus, tok, dev, contrast, **opts)
            with torch.autocast(DEVT, dtype=ADT):
                V, logitP, Q = model(batch)
            loss, parts = loss_fn(V.float(), logitP.float(), Q, z, g, qy, ab, batch["cost"], model.log_kappa, batch["keys"])
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % 100 == 0:
                print(f"ep{ep} step {step}/{steps} loss {loss.item():.4f} parts {np.round(parts, 4)} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(f"{ROOT}/models", exist_ok=True)
    torch.save(model.state_dict(), f"{ROOT}/models/essn_{a.traj}_{a.tag}.pt")
    preds = {}
    for ds in eds:
        for sp in ("calib", "test", "calib2"):
            preds.update(predict(model, split(data.items[ds], sp), data.corpus, tok, dev, contrast, opts=opts))
    os.makedirs(f"{ROOT}/preds/{a.traj}", exist_ok=True)
    json.dump(preds, open(f"{ROOT}/preds/{a.traj}/essn_{a.tag}.json", "w"))
    print("done", a.tag, flush=True)


if __name__ == "__main__":
    main()
