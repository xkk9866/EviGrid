"""(lik_full) Generator-side re-verification: the causal candidate x evidence-state likelihood field.

At every round t and for every candidate a in A_t (all distinct drafts so far, ESSN's candidate set) we score
    log p_LLM( "Answer: a" | exactly the generation prompt of round t )
with the generator itself. The prefix is token-identical to the prompt that produced draft a_t, so in deployment
the KV cache of the generation call is reused and the extra cost is only the candidate tokens, which we record
and charge. Row t of the field compares all candidates under the SAME evidence E_t.

usage: python lik_score.py <model_dir> --traj qwen25-7b
writes data/lik/<traj>/<ds>.json.gz : {qid: {"rows": [[[cand_round, sum_logprob, n_tokens], ...] per t], "tok": [...]}}
"""
import argparse, gzip, json, os
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from common import ROOT, DATASETS, T, load_traj
from rollout import chat, evidence_block, FMT, quant_kwargs
from essn import candidates
FULL = bool(os.environ.get("FULL"))  # score every distinct draft of the trajectory under every round's prompt
OUT = "lik_full" if FULL else "lik"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--traj", default="qwen25-7b")
    ap.add_argument("--ds", default=",".join(DATASETS))
    a = ap.parse_args()
    llm = LLM(a.model, **quant_kwargs(), gpu_memory_utilization=float(os.environ.get("GPU_UTIL", 0.55)),
              max_model_len=6144, enable_prefix_caching=True, seed=0, max_num_batched_tokens=1024, max_num_seqs=16)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    for ds in a.ds.split(","):
        path = f"{ROOT}/traj/{a.traj}/{ds}.jsonl.gz"
        if not os.path.exists(path):
            continue
        with gzip.open(f"{ROOT}/data/proc/{ds}/corpus.jsonl.gz", "rt", encoding="utf-8") as f:
            corpus = [json.loads(l) for l in f]
        trajs = load_traj(a.traj, ds)
        if os.environ.get("SMOKE"):
            trajs = trajs[:int(os.environ["SMOKE"])]
        jobs, meta = [], []
        for qi, tr in enumerate(trajs):
            ev = []
            for t, r in enumerate(tr["rounds"]):
                ev = ev + r["new_pids"]
                # token-identical to the generation prompt of round t (rollout.py), then the forced answer slot
                prefix = chat(tok, f"Question: {tr['question']}\n\nEvidence:\n{evidence_block(ev, corpus)}\n\n{FMT}") + "Answer:"
                pids = tok.encode(prefix, add_special_tokens=False)
                for ci in (candidates(tr["rounds"], T, True) if FULL else candidates(tr["rounds"], t, True)):
                    ans = tr["rounds"][ci]["answer"].strip() or "unknown"
                    fids = tok.encode(prefix + " " + ans, add_special_tokens=False)
                    k = 0
                    while k < min(len(pids), len(fids)) and pids[k] == fids[k]:
                        k += 1
                    jobs.append(TokensPrompt(prompt_token_ids=fids))
                    meta.append((qi, t, ci, k, len(fids)))
        outs = llm.generate(jobs, sp, use_tqdm=False)
        res = {}
        for (qi, t, ci, k, n), o in zip(meta, outs):
            lps = o.prompt_logprobs
            ids = o.prompt_token_ids
            s = sum(lps[i][ids[i]].logprob for i in range(max(k, 1), n))
            q = res.setdefault(trajs[qi]["qid"], {"rows": [[] for _ in range(T + 1)], "tok": [0] * (T + 1)})
            q["rows"][t].append([ci, round(s, 4), n - k])
            q["tok"][t] += n - k + 2  # candidate tokens + the forced "Answer:" marker
        os.makedirs(f"{ROOT}/data/{OUT}/{a.traj}", exist_ok=True)
        with gzip.open(f"{ROOT}/data/{OUT}/{a.traj}/{ds}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(res, f)
        print(ds, len(res), "questions scored", flush=True)


if __name__ == "__main__":
    main()
