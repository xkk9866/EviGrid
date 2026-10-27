"""Learned stopping baselines re-implemented on the shared trajectories (same BERT-base backbone).

  stoprag  : Stop-RAG (Park et al., NeurIPS'25 W). Value network over question [SEP] accumulated retrieved docs;
             the stopping reward is the quality of the answer generated from the accumulated evidence.
             Two heads Q(stop), Q(cont); Q(cont) trained with forward-view Q(lambda) targets
             (following the official repo: n-step returns bootstrapped with max(Q1,Q2) + MC tail).
             Stop when Q(stop) - Q(cont) >= tau.
  simrag   : SIM-RAG critic (Yang et al., SIGIR'25). Input = question [SEP] answer [SEP] docs.
             Binary label = current answer is correct. Stop when p(accept) >= tau.
  adaptive : Adaptive-RAG (Jeong et al., NAACL'24). Question-only 3-way complexity classifier
             (A: no retrieval, B: single step, C: multi step), labels from which strategy answers
             correctly, dataset-bias fallback.

  rerank   : re-verification reranker for "existing stopper + answer reranking" baselines: a cross-encoder that
             scores every candidate of the ESSN candidate set against the CURRENT evidence (target = candidate F1).

usage: python baselines.py {stoprag|simrag|adaptive|rerank} [--traj qwen25-7b]
"""
import argparse, gzip, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from common import ROOT, DATASETS, T, load_traj, split
from essn import candidates

ENC = "BAAI/bge-base-en-v1.5"
DEVT = os.environ.get("DEV", "cuda")  # "cpu" only for smoke tests
ADT = torch.float16 if DEVT == "cuda" else torch.bfloat16
MULTIHOP = {"hotpotqa", "2wiki", "musique"}


class Clf(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.enc = AutoModel.from_pretrained(ENC)
        self.enc.gradient_checkpointing_enable()
        d = self.enc.config.hidden_size
        self.heads = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, n_out))

    def forward(self, enc):
        return self.heads(self.enc(**enc).last_hidden_state[:, 0].float())


def state_text(tr, t, corpus, with_answer, ans_round=None):
    ev = [p for r in tr["rounds"][:t + 1] for p in r["new_pids"]]
    docs = " [SEP] ".join(f"{corpus[p]['title']}: {' '.join(corpus[p]['text'].split()[:40])}" for p in ev)
    a = tr["question"]
    ar = t if ans_round is None else ans_round
    b = (f"Answer: {tr['rounds'][ar]['answer']} [SEP] " if with_answer else "") + (docs or "(no documents)")
    return a, b


def adaptive_label(tr):
    y = [r["f1"] for r in tr["rounds"]]
    if y[0] >= 0.5:
        return 0
    if y[1] >= 0.5:
        return 1
    if max(y) >= 0.5:
        return 2
    return 2 if tr["dataset"] in MULTIHOP else 1


def encode_q(tok, qs, dev):
    e = tok(qs, truncation=True, max_length=64, padding=True, return_tensors="pt")
    return {k: v.to(dev) for k, v in e.items()}


def encode(tok, pairs, dev, maxlen=512):
    e = tok([a for a, _ in pairs], [b for _, b in pairs], truncation="only_second", max_length=maxlen,
            padding=True, return_tensors="pt")
    return {k: v.to(dev) for k, v in e.items()}


def qlambda_targets(q1, q2, y, lam):
    """Forward-view Q(lambda) target for Q(cont) at every t (Stop-RAG, Eq. in official code).
    q1,q2: (B,T+1) detached predictions; y: (B,T+1) rewards of stopping at each t."""
    B, H = y.shape
    tgt = torch.zeros(B, H - 1, device=y.device)
    for t in range(H - 1):
        n_max = H - 1 - t
        mc = y[:, t + 1:].max(1).values
        acc = torch.zeros(B, device=y.device)
        for n in range(1, n_max + 1):
            seen = y[:, t + 1:t + n].max(1).values if n > 1 else torch.full((B,), -1.0, device=y.device)
            boot = torch.maximum(q1[:, t + n], q2[:, t + n]) if t + n < H - 1 else q1[:, t + n]
            g = torch.maximum(seen, boot)
            if n < n_max:
                acc = acc + (1 - lam) * lam ** (n - 1) * g
        tgt[:, t] = acc + lam ** (n_max - 1) * mc if n_max >= 1 else mc
    return tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method")
    ap.add_argument("--traj", default="qwen25-7b")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--train_ds", default=",".join(DATASETS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--also_traj", default="")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = DEVT
    corp, items = {}, {}
    for ds in DATASETS:
        if not os.path.exists(f"{ROOT}/traj/{a.traj}/{ds}.jsonl.gz"):
            continue
        with gzip.open(f"{ROOT}/data/proc/{ds}/corpus.jsonl.gz", "rt", encoding="utf-8") as f:
            corp[ds] = [json.loads(l) for l in f]
        items[ds] = load_traj(a.traj, ds)
    tok = AutoTokenizer.from_pretrained(ENC)
    n_out = {"stoprag": 2, "simrag": 1, "adaptive": 3, "rerank": 1}[a.method]
    model = Clf(n_out).to(dev)
    train = [tr for ds in a.train_ds.split(",") for tr in split(items[ds], "train")]
    if os.environ.get("PREDICT_ONLY"):  # transfer: load the controller trained on --traj, predict on --also_traj
        model.load_state_dict(torch.load(f"{ROOT}/models/{a.method}{a.tag}_{a.traj}.pt", map_location=dev))
        train = []
    if os.environ.get("SMOKE"):
        train = train[:int(os.environ["SMOKE"])]
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": 2e-5}, {"params": model.heads.parameters(), "lr": 5e-4}],
                            weight_decay=0.01)
    steps = max(1, a.epochs * math.ceil(len(train) / a.bs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler(DEVT, enabled=DEVT == "cuda")
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        random.shuffle(train)
        for i in range(0, len(train), a.bs):
            chunk = train[i:i + a.bs]
            if a.method == "adaptive":
                enc = encode_q(tok, [tr["question"] for tr in chunk], dev)
                with torch.autocast(DEVT, dtype=ADT):
                    out = model(enc)
                loss = F.cross_entropy(out.float(), torch.tensor([adaptive_label(tr) for tr in chunk], device=dev))
            elif a.method == "rerank":
                # re-verification reranker: every candidate of the candidate set scored against the CURRENT evidence
                trip = [(tr, t, ci) for tr in chunk for t in range(T + 1) for ci in candidates(tr["rounds"], t)]
                enc = encode(tok, [state_text(tr, t, corp[tr["dataset"]], True, ci) for tr, t, ci in trip], dev)
                y = torch.tensor([tr["rounds"][ci]["f1"] for tr, t, ci in trip], device=dev)
                with torch.autocast(DEVT, dtype=ADT):
                    out = model(enc).float()[:, 0]
                loss = F.binary_cross_entropy_with_logits(out, y)
            else:
                pairs = [state_text(tr, t, corp[tr["dataset"]], a.method == "simrag") for tr in chunk for t in range(T + 1)]
                enc = encode(tok, pairs, dev)
                y = torch.tensor([[r["f1"] for r in tr["rounds"]] for tr in chunk], device=dev)
                with torch.autocast(DEVT, dtype=ADT):
                    out = model(enc).float().view(len(chunk), T + 1, -1)
                if a.method == "simrag":
                    loss = F.binary_cross_entropy_with_logits(out[..., 0], (y >= 0.5).float())
                else:
                    q1, q2 = torch.sigmoid(out[..., 0]), torch.sigmoid(out[..., 1])
                    lam = 1.0 - 0.5 * min(1.0, step / max(1, steps * 0.5))  # anneal MC -> Q(0.5)
                    tgt = qlambda_targets(q1.detach(), q2.detach(), y, lam)
                    loss = F.mse_loss(q1, y) + F.mse_loss(q2[:, :T], tgt)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % 100 == 0:
                print(f"{a.method} ep{ep} {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    os.makedirs(f"{ROOT}/models", exist_ok=True)
    if not os.environ.get("PREDICT_ONLY"):
        torch.save(model.state_dict(), f"{ROOT}/models/{a.method}{a.tag}_{a.traj}.pt")
    for tg in ([] if os.environ.get("PREDICT_ONLY") else [a.traj]) + [x for x in a.also_traj.split(",") if x]:
        if tg != a.traj:
            items = {ds: load_traj(tg, ds) for ds in DATASETS if os.path.exists(f"{ROOT}/traj/{tg}/{ds}.jsonl.gz")}
        predict_all(model, items, corp, tok, dev, a, tg)


def predict_all(model, items, corp, tok, dev, a, tg):
    preds = {}
    with torch.no_grad(), torch.autocast(DEVT, dtype=ADT):
        for ds in items:
            ev = split(items[ds], "calib,test,calib2,cert,test2,test3")
            if os.environ.get("SMOKE"):
                ev = ev[:2 * int(os.environ["SMOKE"])]
            for i in range(0, len(ev), 8):
                chunk = ev[i:i + 8]
                if a.method == "adaptive":
                    out = torch.softmax(model(encode_q(tok, [tr["question"] for tr in chunk], dev)).float(), -1)
                    for j, tr in enumerate(chunk):
                        preds[f"{ds}::{tr['qid']}"] = out[j].cpu().numpy().round(4).tolist()
                elif a.method == "rerank":
                    trip = [(tr, t, ci) for tr in chunk for t in range(T + 1) for ci in candidates(tr["rounds"], t)]
                    sc = torch.sigmoid(model(encode(tok, [state_text(tr, t, corp[ds], True, ci) for tr, t, ci in trip], dev)).float()[:, 0]).cpu().numpy()
                    for (tr, t, ci), v in zip(trip, sc):
                        preds.setdefault(f"{ds}::{tr['qid']}", {"S": [[] for _ in range(T + 1)]})["S"][t].append([int(ci), round(float(v), 4)])
                else:
                    pairs = [state_text(tr, t, corp[ds], a.method == "simrag") for tr in chunk for t in range(T + 1)]
                    out = torch.sigmoid(model(encode(tok, pairs, dev)).float()).view(len(chunk), T + 1, -1).cpu().numpy()
                    for j, tr in enumerate(chunk):
                        preds[f"{ds}::{tr['qid']}"] = out[j].round(4).tolist()
    os.makedirs(f"{ROOT}/preds/{tg}", exist_ok=True)
    suffix = "" if tg == a.traj else f"_from_{a.traj}"
    json.dump(preds, open(f"{ROOT}/preds/{tg}/{a.method}{a.tag}{suffix}.json", "w"))
    print("done", a.method, flush=True)


if __name__ == "__main__":
    main()
