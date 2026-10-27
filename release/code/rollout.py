"""Full-horizon iterative-RAG rollouts with an INT4 (AWQ) LLM served by vLLM.

For every question we run t = 0 (closed book) and t = 1..T retrieval rounds and log, per round:
draft answer, answer-token log-probs, next query, newly retrieved passages, token cost and
(for calib/test) 4 sampled answers + an LLM sufficiency judgement.  All stopping policies are
then evaluated offline on exactly the same trajectories.

usage: python rollout.py <dataset> <model_dir> <tag>
"""
import gzip, json, math, os, re, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from vllm import LLM, SamplingParams

ROOT = os.path.expanduser("~/research/esrag")
T, K_PER_ROUND, MAX_WORDS = 5, 3, 110
N_SPLIT = {"train": 1200, "calib": 300, "test": 500}
if os.environ.get("NO_TRAIN"):
    N_SPLIT["train"] = 0
if os.environ.get("LIMIT"):
    N_SPLIT = {k: int(os.environ["LIMIT"]) for k in N_SPLIT}
QPREFIX = "Represent this sentence for searching relevant passages: "

SYS = ("You answer questions using retrieved Wikipedia evidence. Think briefly, then give a short answer "
       "(an entity, date, number, yes/no or short phrase). If the evidence is not enough, give your best guess "
       "and ask for the missing information.")
FMT = ("Reply in exactly this format:\nReasoning: <at most two sentences>\nAnswer: <only the answer itself: an entity, date, number, yes/no or short phrase, never a full sentence>\n"
       "Next query: <a search query for the most important missing fact>")
JUDGE = ("Question: {q}\n\nEvidence:\n{ev}\n\nIs the evidence above sufficient to answer the question? "
         "Reply with only Yes or No.")


def load(ds):
    if os.environ.get("FRESH3"):  # third evaluation split (extend_splits3.py): questions no experiment has used
        with gzip.open(f"{ROOT}/data/proc/{ds}/questions_fresh3.jsonl.gz", "rt", encoding="utf-8") as f:
            return [json.loads(l) for l in f], load_corpus(ds)
    if os.environ.get("FRESH2"):
        # final evaluation splits never touched during development (extend_splits.py): cert + test2;
        # SPLITS restricts the splits, SUBSET caps the number of test2 questions (secondary settings)
        with gzip.open(f"{ROOT}/data/proc/{ds}/questions_fresh.jsonl.gz", "rt", encoding="utf-8") as f:
            qs = [json.loads(l) for l in f]
        sps = os.environ.get("SPLITS", "cert,test2").split(",")
        cap = int(os.environ.get("SUBSET", 10 ** 9))
        cnt = {}
        out = []
        for q in qs:
            if q["split"] in sps and cnt.get(q["split"], 0) < (cap if q["split"] == "test2" else 10 ** 9):
                cnt[q["split"]] = cnt.get(q["split"], 0) + 1
                out.append(q)
        return out, load_corpus(ds)
    with gzip.open(f"{ROOT}/data/proc/{ds}/questions.jsonl.gz", "rt", encoding="utf-8") as f:
        qs = [json.loads(l) for l in f]
    out, cnt = [], {k: 0 for k in N_SPLIT}
    if os.environ.get("FRESH"):
        # fresh calibration split for certification: development questions never used in any experiment
        # (test questions 500..799 of the official development set; the test split uses 0..499)
        tq = [q for q in qs if q["split"] == "test"][N_SPLIT["test"]:N_SPLIT["test"] + 300]
        return [{**q, "split": "calib2"} for q in tq], load_corpus(ds)
    only = os.environ.get("ONLY")
    for q in qs:
        if only and q["split"] != only:
            continue
        if cnt[q["split"]] < N_SPLIT[q["split"]]:
            cnt[q["split"]] += 1
            out.append(q)
    return out, load_corpus(ds)


def load_corpus(ds):
    with gzip.open(f"{ROOT}/data/proc/{ds}/corpus.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


class BM25:
    """Okapi BM25 (k1=0.9, b=0.4, Pyserini defaults) over title + text, as a sparse matrix product."""
    def __init__(self, corpus):
        import scipy.sparse as sp
        from collections import Counter
        self.tokre = re.compile(r"[a-z0-9]+")
        self.vocab, rows, cols, vals, dl = {}, [], [], [], []
        for i, p in enumerate(corpus):
            c = Counter(self.tok(p["title"] + " " + p["text"]))
            dl.append(sum(c.values()))
            for w, n in c.items():
                rows.append(i); cols.append(self.vocab.setdefault(w, len(self.vocab))); vals.append(n)
        tf = sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(len(corpus), len(self.vocab)))
        dl = np.array(dl, dtype=np.float32)
        df = np.bincount(cols, minlength=len(self.vocab)).astype(np.float32)
        self.idf = np.log(1 + (len(corpus) - df + 0.5) / (df + 0.5)).astype(np.float32)
        k1, b = 0.9, 0.4
        tf = tf.tocoo()
        norm = k1 * (1 - b + b * dl[tf.row] / dl.mean())
        w = tf.data * (k1 + 1) / (tf.data + norm)
        self.M = sp.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).T.tocsr()  # vocab x docs

    STOP = set("a an the of in on at to for and or is was are were be by with as from that this which who what when "
               "where how did does do".split())

    def tok(self, s):
        return [w for w in self.tokre.findall(s.lower()) if w not in self.STOP]

    def search(self, queries, exclude, k):
        res = []
        for q, ex in zip(queries, exclude):
            ids = [self.vocab[w] for w in set(self.tok(q)) if w in self.vocab]
            if not ids:
                res.append([]); continue
            sc = np.asarray(self.M[ids].T @ self.idf[ids]).ravel()
            top = np.argpartition(-sc, min(k + 45, len(sc) - 1))[:k + 45]
            top = top[np.argsort(-sc[top])]
            res.append([int(c) for c in top if int(c) not in ex][:k])
        return res


class Retriever:
    def __init__(self, ds):
        self.emb = torch.from_numpy(np.load(f"{ROOT}/data/index/{ds}.npy")).cuda()
        self.tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
        self.enc = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5", torch_dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def search(self, queries, exclude, k):
        res = []
        for i in range(0, len(queries), 128):
            b = self.tok([QPREFIX + q for q in queries[i:i + 128]], padding=True, truncation=True,
                         max_length=64, return_tensors="pt").to("cuda")
            h = torch.nn.functional.normalize(self.enc(**b).last_hidden_state[:, 0], dim=-1)
            top = (h @ self.emb.T).topk(k + 45, dim=-1).indices.cpu().tolist()
            for j, cand in enumerate(top):
                ex = exclude[i + j]
                res.append([c for c in cand if c not in ex][:k])
        return res


def passage(p):
    w = p["text"].split()
    return f"{p['title']}: {' '.join(w[:MAX_WORDS])}"


def evidence_block(pids, corpus):
    return "\n".join(f"[{i + 1}] {passage(corpus[p])}" for i, p in enumerate(pids)) if pids else "(none)"


def chat(tok, user):
    return tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


def parse(text):
    def grab(tag):
        m = re.search(tag + r"\s*:\s*(.*)", text, flags=re.I)
        return m.group(1).strip() if m else ""
    ans = grab("Answer")
    ans = ans.split("\n")[0].strip()
    if not ans or ans.lower().startswith("next query"):
        ans = "unknown"
    return grab("Reasoning"), ans, grab("Next query").split("\n")[0].strip()


def answer_logprobs(out):
    """mean / min log-prob of the tokens that form the answer span."""
    toks, lps, pos, text = out.token_ids, out.logprobs, [], ""
    for tid, lp in zip(toks, lps):
        d = lp[tid]
        start = len(text)
        text += d.decoded_token or ""
        pos.append((start, len(text), d.logprob))
    m = re.search(r"Answer\s*:\s*([^\n]*)", text, flags=re.I)
    if not m:
        return -5.0, -10.0
    s, e = m.span(1)
    sel = [lp for a, b, lp in pos if b > s and a < e]
    if not sel:
        return -5.0, -10.0
    return float(np.mean(sel)), float(np.min(sel))


def quant_kwargs():
    """INT4 weights: AWQ checkpoints, or 4-bit bitsandbytes quantisation at load time (QUANT=bnb)."""
    if os.environ.get("QUANT") == "bnb":
        return {"quantization": "bitsandbytes", "load_format": "bitsandbytes"}
    return {"quantization": "awq_marlin"}


def main(ds, model_dir, tag):
    qs, corpus = load(ds)
    llm = LLM(model_dir, **quant_kwargs(), gpu_memory_utilization=float(os.environ.get("GPU_UTIL", 0.8)),
              max_model_len=6144, enable_prefix_caching=True, max_num_seqs=64,
              enforce_eager=False, seed=0)
    tok = llm.get_tokenizer()
    ret = BM25(corpus) if os.environ.get("RETRIEVER") == "bm25" else Retriever(ds)
    greedy = SamplingParams(temperature=0.0, max_tokens=128, logprobs=0, stop=["<|im_end|>", "<|eot_id|>"])
    sample = SamplingParams(temperature=0.7, top_p=0.95, n=4, max_tokens=128, seed=1)
    judge = SamplingParams(temperature=0.0, max_tokens=1, logprobs=10)
    yes_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ["Yes", " Yes", "yes"]}
    no_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ["No", " No", "no"]}

    n = len(qs)
    ev = [[] for _ in range(n)]
    next_q = [q["question"] for q in qs]
    trajs = [{**q, "dataset": ds, "rounds": []} for q in qs]
    evals = [] if os.environ.get("NOSAMPLE") else [i for i, q in enumerate(qs) if q["split"] != "train"]
    t0 = time.time()
    for t in range(T + 1):
        new = [[] for _ in range(n)]
        if t >= 1:
            queries = [qs[i]["question"] if t == 1 else (next_q[i] or qs[i]["question"]) for i in range(n)]
            new = ret.search(queries, [set(e) for e in ev], K_PER_ROUND)
            for i in range(n):
                ev[i] = ev[i] + new[i]
        prompts = [chat(tok, f"Question: {qs[i]['question']}\n\nEvidence:\n{evidence_block(ev[i], corpus)}\n\n{FMT}")
                   for i in range(n)]
        outs = llm.generate(prompts, greedy, use_tqdm=False)
        samp = llm.generate([prompts[i] for i in evals], sample, use_tqdm=False) if evals else []
        jprompts = [chat(tok, JUDGE.format(q=qs[i]["question"], ev=evidence_block(ev[i], corpus))) for i in evals]
        jouts = llm.generate(jprompts, judge, use_tqdm=False) if evals else []
        smap = {i: s for i, s in zip(evals, samp)}
        jmap = {i: j for i, j in zip(evals, jouts)}
        for i, o in enumerate(outs):
            c = o.outputs[0]
            reason, ans, nq = parse(c.text)
            lp_mean, lp_min = answer_logprobs(c)
            r = {"t": t, "query": (qs[i]["question"] if t == 1 else next_q[i]) if t else "",
                 "new_pids": new[i], "new_titles": [corpus[p]["title"] for p in new[i]],
                 "reasoning": reason, "answer": ans, "next_query": nq, "lp_mean": lp_mean, "lp_min": lp_min,
                 "ptoks": len(o.prompt_token_ids), "ctoks": len(c.token_ids)}
            if i in smap:
                r["samples"] = [parse(s.text)[1] for s in smap[i].outputs]
                r["sample_ctoks"] = sum(len(s.token_ids) for s in smap[i].outputs)
                lp = jmap[i].outputs[0].logprobs[0]
                py = sum(math.exp(v.logprob) for k, v in lp.items() if k in yes_ids)
                pn = sum(math.exp(v.logprob) for k, v in lp.items() if k in no_ids)
                r["judge_yes"] = py / (py + pn + 1e-9)
                r["judge_ptoks"] = len(jmap[i].prompt_token_ids)
            trajs[i]["rounds"].append(r)
            next_q[i] = nq
        print(f"[{ds}] round {t} done, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(f"{ROOT}/traj/{tag}", exist_ok=True)
    with gzip.open(f"{ROOT}/traj/{tag}/{ds}.jsonl.gz", "wt", encoding="utf-8") as f:
        for tr in trajs:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")
    print("saved", ds, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
