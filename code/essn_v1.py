"""ESSN: Evidence-Sufficiency State Network.

One network, one forward chain:
  claim-conditioned pair encoding -> requirement-support convolution (CNN on interaction maps)
  -> noisy-AND-of-OR sufficiency (coverage + critical-fact support) -> recurrent belief state
  -> look-ahead value profile V_t(k), whose k=0 entry is anchored on the sufficiency layer.

usage:
  python essn.py train --tag full [--ablate noCNN,noAND,...] [--train_ds hotpotqa,2wiki,musique,nq]
  (training automatically writes predictions for calib/test of all datasets to preds/<tag>.json)
"""
import argparse, gzip, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from common import ROOT, DATASETS, T, load_traj, split

ENC = "BAAI/bge-base-en-v1.5"
MAXLEN, LC_MAX, K = 192, 64, T


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def log1mexp(x):
    """log(1 - exp(x)) for x < 0, numerically stable."""
    x = x.clamp(max=-1e-6)
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


class ESSN(nn.Module):
    def __init__(self, ablate=()):
        super().__init__()
        self.ab = set(ablate)
        self.enc = AutoModel.from_pretrained(ENC)
        self.enc.gradient_checkpointing_enable()
        self.sep_id = AutoTokenizer.from_pretrained(ENC).sep_token_id
        d, r = self.enc.config.hidden_size, 128
        self.pa, self.pb, self.hp = nn.Linear(d, r), nn.Linear(d, r), nn.Linear(d, r)
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 16, 3, padding=1), nn.GELU())
        self.sup = mlp(r + 16 + 1, 128, 1)
        self.crit = mlp(d, 128, 1)
        self.cls_suff = mlp(d, 128, 1)  # only used by the noAND ablation
        self.xdim = r + 11 + (T + 1)
        self.inp = nn.Linear(self.xdim, 256)
        self.gru = nn.GRUCell(256, 256)
        self.mlp_b = mlp(self.xdim, 256, 256)  # noGRU ablation
        self.head = mlp(256 + self.xdim, 256, K + 1)
        self.beta = nn.Parameter(torch.tensor([1.0] + [0.3] * K))

    # ---- pair level -------------------------------------------------------------
    def support(self, enc_in, lc):
        H = self.enc(**enc_in).last_hidden_state.float()  # N,L,d
        N, L, _ = H.shape
        tt, am = enc_in["token_type_ids"], enc_in["attention_mask"]
        pmask = (tt == 1) & (am == 1) & (enc_in["input_ids"] != self.sep_id)
        Lc = LC_MAX
        Hc = H[:, 1:1 + Lc]  # claim tokens are always positions 1..lc
        a = F.normalize(self.pa(Hc), dim=-1)
        b = F.normalize(self.pb(H), dim=-1)
        A = torch.einsum("nir,njr->nij", a, b) * pmask[:, None, :]  # N,Lc,L
        has_p = pmask.any(1)
        neg = torch.finfo(A.dtype).min
        raw = A.masked_fill(~pmask[:, None, :], neg).amax(-1)
        raw = torch.where(has_p[:, None], raw, torch.zeros_like(raw))
        if "noCNN" in self.ab:
            cm = torch.zeros(N, Lc, 16, device=H.device)
        else:
            C = self.conv(A.unsqueeze(1))  # N,16,Lc,L
            cm = C.masked_fill(~pmask[:, None, None, :], neg).amax(-1).transpose(1, 2)
            cm = torch.where(has_p[:, None, None], cm, torch.zeros_like(cm))
        s = self.sup(torch.cat([self.hp(Hc), cm, raw.unsqueeze(-1)], -1)).squeeze(-1)  # N,Lc
        return s, Hc, H[:, 0]

    # ---- question level ---------------------------------------------------------
    def forward(self, batch):
        s, Hc, cls = self.support(batch["enc"], None)
        dev = s.device
        B = batch["B"]
        X, logPs, logitPs, covs = [], [], [], []
        for q in range(B):
            prev = None
            xs = []
            for t in range(T + 1):
                idx = batch["slots"][q][t]  # indices of pairs of this state; [0] = null slot
                new = batch["new"][q][t]  # indices (within idx) of passages retrieved this round
                cl = batch["clen"][q][t]
                st = s[idx, :cl]  # P,cl
                h = Hc[idx, :cl].mean(0)  # cl,d
                w = F.softplus(self.crit(h).squeeze(-1)) + 1e-3
                wn = w / w.sum()
                if "noAND" in self.ab:
                    c = torch.sigmoid(st).mean(0)
                    logP = -F.softplus(-self.cls_suff(cls[idx].mean(0))).squeeze()
                else:
                    logq = F.logsigmoid(-st).sum(0)  # log prod(1-sigma)
                    logc = log1mexp(logq)
                    c = logc.exp()
                    logP = (w * logc).sum()
                if new:
                    cn = 1 - torch.exp(F.logsigmoid(-st[new]).sum(0))
                    new_cov = (wn * cn).sum()
                else:
                    new_cov = torch.zeros((), device=dev)
                cmean = (wn * c).sum()
                cmin = -0.1 * torch.log((wn * torch.exp(-c / 0.1)).sum())
                d_mean = cmean - prev[0] if prev is not None else torch.zeros((), device=dev)
                d_min = cmin - prev[1] if prev is not None else torch.zeros((), device=dev)
                prev = (cmean, cmin)
                logitP = logP - log1mexp(logP.clamp(max=-1e-4))
                f = batch["feat"][q][t]
                if "noLLM" in self.ab:
                    f = f.clone(); f[:2] = 0
                onehot = F.one_hot(torch.tensor(t, device=dev), T + 1).float()
                x = torch.cat([(wn[:, None] * self.hp(h)).sum(0),
                               torch.stack([cmean, cmin, d_mean, d_min, new_cov, logP.clamp(min=-30) / 10, logitP.clamp(-15, 15) / 5]),
                               f, onehot])
                xs.append(x)
                logPs.append(logP)
                logitPs.append(logitP)
                covs.append(c.detach())
            X.append(torch.stack(xs))
        X = torch.stack(X)  # B,T+1,xdim
        logitP = torch.stack(logitPs).view(B, T + 1).clamp(-15, 15)
        if "noGRU" in self.ab:
            Hs = self.mlp_b(X)
        else:
            hs, hcur = [], torch.zeros(B, 256, device=dev)
            for t in range(T + 1):
                hcur = self.gru(torch.tanh(self.inp(X[:, t])), hcur)
                hs.append(hcur)
            Hs = torch.stack(hs, 1)
        V = self.head(torch.cat([Hs, X], -1)) + self.beta * logitP.unsqueeze(-1)  # B,T+1,K+1 logits
        return V, logitP, covs


# ---------------------------------------------------------------------------------
class Data:
    def __init__(self, tag, dss):
        self.items, self.corpus = {}, {}
        for ds in dss:
            with gzip.open(f"{ROOT}/data/proc/{ds}/corpus.jsonl.gz", "rt", encoding="utf-8") as f:
                self.corpus[ds] = [json.loads(l) for l in f]
            self.items[ds] = load_traj(tag, ds)


def build_batch(trajs, corpus, tok, dev):
    texts_a, texts_b, slots, news, clens, feats = [], [], [], [], [], []
    for tr in trajs:
        cp = corpus[tr["dataset"]]
        ev, qs, qn, qc, qf = [], [], [], [], []
        for r in tr["rounds"]:
            ev = ev + r["new_pids"]
            claim = f"{tr['question']} Answer: {r['answer'] or 'unknown'}"
            start = len(texts_a)
            pas = [""] + [f"{cp[p]['title']}: {cp[p]['text']}" for p in ev]
            texts_a += [claim] * len(pas)
            texts_b += pas
            qs.append(list(range(start, start + len(pas))))
            nnew = len(r["new_pids"])
            qn.append(list(range(len(pas) - nnew, len(pas))) if nnew else [])
            ca = tok(claim, add_special_tokens=False)["input_ids"]
            qc.append(min(len(ca), LC_MAX))
            qf.append(torch.tensor([r["lp_mean"] / 5, r["lp_min"] / 10, r["same_prev"], float(len(ev)) / 15], dtype=torch.float))
        slots.append(qs); news.append(qn); clens.append(qc); feats.append([f.to(dev) for f in qf])
    enc = tok(texts_a, texts_b, truncation="only_second", max_length=MAXLEN, padding=True, return_tensors="pt")
    if enc["input_ids"].shape[1] < LC_MAX + 2:
        enc = tok(texts_a, texts_b, truncation="only_second", max_length=MAXLEN, padding="max_length", return_tensors="pt")
    enc = {k: v.to(dev) for k, v in enc.items()}
    y = torch.tensor([[r["f1"] for r in tr["rounds"]] for tr in trajs], device=dev)
    g = torch.tensor([[r["gold_all"] for r in tr["rounds"]] for tr in trajs], device=dev)
    return {"enc": enc, "slots": slots, "new": news, "clen": clens, "feat": feats, "B": len(trajs)}, y, g


def loss_fn(V, logitP, y, g, ab):
    B = y.shape[0]
    tgt = torch.zeros_like(V)
    mask = torch.zeros_like(V)
    for t in range(T + 1):
        for k in range(T + 1 - t):
            tgt[:, t, k] = y[:, t + k]
            mask[:, t, k] = 1
    l_prof = (F.binary_cross_entropy_with_logits(V, tgt, reduction="none") * mask).sum() / mask.sum()
    if "noProfile" in ab:
        l_prof = F.binary_cross_entropy_with_logits(V[:, :, 0], y)
    gm = (g >= 0).float()
    l_ev = (F.binary_cross_entropy_with_logits(logitP, g.clamp(min=0), reduction="none") * gm).sum() / gm.sum().clamp(min=1)
    if "noEvSup" in ab:
        l_ev = l_ev * 0
    z = V[:, :, 0]
    dz = z.unsqueeze(2) - z.unsqueeze(1)
    dy = y.unsqueeze(2) - y.unsqueeze(1)
    pm = (dy > 0.5).float()
    l_rank = (F.softplus(-dz) * pm).sum() / pm.sum().clamp(min=1)
    return l_prof + 0.3 * l_ev + 0.1 * l_rank, (l_prof.item(), l_ev.item(), l_rank.item())


def predict(model, trajs, corpus, tok, dev, bs=4):
    model.eval()
    out = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(trajs), bs):
            chunk = trajs[i:i + bs]
            batch, _, _ = build_batch(chunk, corpus, tok, dev)
            V, logitP, covs = model(batch)
            V = torch.sigmoid(V.float()).cpu().numpy()
            P = torch.sigmoid(logitP.float()).cpu().numpy()
            for j, tr in enumerate(chunk):
                out[f"{tr['dataset']}::{tr['qid']}"] = {"V": V[j].round(4).tolist(), "P": P[j].round(4).tolist()}
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--traj", default="qwen25-7b")
    ap.add_argument("--ablate", default="")
    ap.add_argument("--train_ds", default=",".join(DATASETS))
    ap.add_argument("--eval_ds", default=",".join(DATASETS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src_traj", default="qwen25-7b")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    ab = [x for x in a.ablate.split(",") if x]
    dev = "cuda"
    tds, eds = a.train_ds.split(","), a.eval_ds.split(",")
    data = Data(a.traj, sorted(set(eds) if a.mode == "predict" else set(tds) | set(eds)))
    tok = AutoTokenizer.from_pretrained(ENC)
    model = ESSN(ab).to(dev)
    if a.mode == "predict":  # zero-shot transfer: controller trained on --src_traj, applied to --traj
        model.load_state_dict(torch.load(f"{ROOT}/models/essn_{a.src_traj}_{a.tag}.pt", map_location=dev))
        preds = {}
        for ds in eds:
            for sp in ("calib", "test"):
                preds.update(predict(model, split(data.items[ds], sp), data.corpus, tok, dev))
        os.makedirs(f"{ROOT}/preds/{a.traj}", exist_ok=True)
        json.dump(preds, open(f"{ROOT}/preds/{a.traj}/essn_{a.tag}_from_{a.src_traj}.json", "w"))
        print("done predict", flush=True)
        return
    train = [tr for ds in tds for tr in split(data.items[ds], "train")]
    enc_p = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    oth_p = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": 2e-5}, {"params": oth_p, "lr": 5e-4}], weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * max(0.0, 1 - s / steps))
    scaler = torch.cuda.amp.GradScaler()
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        random.shuffle(train)
        for i in range(0, len(train), a.bs):
            batch, y, g = build_batch(train[i:i + a.bs], data.corpus, tok, dev)
            with torch.autocast("cuda", dtype=torch.float16):
                V, logitP, _ = model(batch)
            loss, parts = loss_fn(V.float(), logitP.float(), y, g, ab)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % 100 == 0:
                print(f"ep{ep} step {step}/{steps} loss {loss.item():.4f} parts {np.round(parts, 4)} {time.time() - t0:.0f}s", flush=True)
    preds = {}
    for ds in eds:
        for sp in ("calib", "test"):
            preds.update(predict(model, split(data.items[ds], sp), data.corpus, tok, dev))
    os.makedirs(f"{ROOT}/preds/{a.traj}", exist_ok=True)
    json.dump(preds, open(f"{ROOT}/preds/{a.traj}/essn_{a.tag}.json", "w"))
    torch.save(model.state_dict(), f"{ROOT}/models/essn_{a.traj}_{a.tag}.pt")
    print("done", a.tag, flush=True)


if __name__ == "__main__":
    main()
