"""Trajectories of an RL-trained search agent (Search-R1) in the shared retrieval environment.

The agent keeps its own policy: it thinks, writes its own <search> queries and reads <information> blocks.
Three additions make its trajectory usable by stop-and-select controllers, none of which changes what the agent
sees or does:
  * answer probe   after every search turn t (and before the first one) the draft a_t is read off a side branch
                   ctx_t + "<answer>" (greedy, prefix KV shared); the probe never enters the agent's context.
  * native stop    the first turn at which the agent itself emits <answer> is recorded (its own stopping decision
                   and answer). To obtain the full T-round rollout needed for offline evaluation, the answer is
                   removed from the context and the agent is asked to continue with "<search>" (budget forcing);
                   rounds after the native stop are therefore only used by controllers that decide to continue.
  * generator reading   log p(" a" | ctx_t + "<answer>") of every candidate in hand, written in the format of
                   lik_score.py (data/lik/<tag>/<ds>.json.gz); its tokens are recorded and charged.
Retrieval: the shared dense retriever, top-3 unseen passages per search (as for every other controller).
INT4: 4-bit bitsandbytes weights at load time.

usage: python rollout_agent.py <dataset> <model_dir> <tag>     (env: SPLITS, SUBSET, FRESH2 as in rollout.py)
"""
import gzip, json, math, os, re, sys, time
import numpy as np
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from rollout import ROOT, T, K_PER_ROUND, MAX_WORDS, load, Retriever

PROMPT = ("Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get "
          "new information. After reasoning, if you find you lack some knowledge, you can call a search engine by "
          "<search> query </search> and it will return the top searched results between <information> and "
          "</information>. You can search as many times as your want. If you find no further external knowledge "
          "needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. "
          "For example, <answer> Beijing </answer>. Question: {q}\n")


def docs(pids, corpus):
    return "".join(f"Doc {i + 1}(Title: {corpus[p]['title']}) {' '.join(corpus[p]['text'].split()[:MAX_WORDS])}\n"
                   for i, p in enumerate(pids))


def main(ds, model_dir, tag):
    qs, corpus = load(ds)
    llm = LLM(model_dir, quantization="bitsandbytes", load_format="bitsandbytes",
              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", 0.8)), max_model_len=8192,
              enable_prefix_caching=True, max_num_seqs=64, seed=0)
    tok = llm.get_tokenizer()
    ret = Retriever(ds)
    turn = SamplingParams(temperature=0.0, max_tokens=512, stop=["</search>", "</answer>"], include_stop_str_in_output=True)
    force = SamplingParams(temperature=0.0, max_tokens=64, stop=["</search>"], include_stop_str_in_output=True)
    probe = SamplingParams(temperature=0.0, max_tokens=32, stop=["</answer>"], logprobs=0)
    score = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    n = len(qs)

    def qtext(q):
        q = q.strip()
        return q if q.endswith("?") else q + "?"
    base = [PROMPT.format(q=qtext(q["question"])) for q in qs]
    if tok.chat_template:
        base = [tok.apply_chat_template([{"role": "user", "content": b}], add_generation_prompt=True, tokenize=False) for b in base]
    ctx = base[:]
    ev = [[] for _ in range(n)]
    trajs = [{**q, "dataset": ds, "rounds": [], "native_stop": None, "native_answer": None} for q in qs]
    lik = {q["qid"]: {"rows": [[] for _ in range(T + 1)], "tok": [0] * (T + 1)} for q in qs}
    from common import normalize
    t0 = time.time()
    for t in range(T + 1):
        new, gen_text, ptoks, ctoks = [[] for _ in range(n)], [""] * n, [0] * n, [0] * n
        if t >= 1:
            outs = llm.generate(ctx, turn, use_tqdm=False)
            queries, redo = [""] * n, []
            for i, o in enumerate(outs):
                c = o.outputs[0]
                txt = c.text
                ptoks[i] += len(o.prompt_token_ids); ctoks[i] += len(c.token_ids)
                if txt.rstrip().endswith("</search>") and "<search>" in txt:
                    queries[i] = txt.rsplit("<search>", 1)[1].replace("</search>", "").strip()
                    gen_text[i] = txt
                else:
                    if "<answer>" in txt and trajs[i]["native_stop"] is None:
                        trajs[i]["native_stop"] = t - 1
                        trajs[i]["native_answer"] = txt.split("<answer>", 1)[1].replace("</answer>", "").strip()
                    gen_text[i] = txt.split("<answer>", 1)[0].rstrip() + "\n<search>"
                    redo.append(i)
            if redo:  # budget forcing: the agent writes the query it would issue next
                fo = llm.generate([ctx[i] + gen_text[i] for i in redo], force, use_tqdm=False)
                for i, o in zip(redo, fo):
                    c = o.outputs[0]
                    ptoks[i] += len(o.prompt_token_ids); ctoks[i] += len(c.token_ids)
                    gen_text[i] += c.text if c.text.rstrip().endswith("</search>") else c.text + " </search>"
                    queries[i] = c.text.replace("</search>", "").strip() or qs[i]["question"]
            new = ret.search([q or qs[i]["question"] for i, q in enumerate(queries)], [set(e) for e in ev], K_PER_ROUND)
            for i in range(n):
                ev[i] = ev[i] + new[i]
                ctx[i] = ctx[i] + gen_text[i] + f"\n\n<information>{docs(new[i], corpus)}</information>\n\n"
        pr = llm.generate([c + "<answer>" for c in ctx], probe, use_tqdm=False)
        for i, o in enumerate(pr):
            c = o.outputs[0]
            ans = c.text.replace("</answer>", "").strip().split("\n")[0].strip() or "unknown"
            lps = [lp[tid].logprob for tid, lp in zip(c.token_ids, c.logprobs)] if c.logprobs else [-5.0]
            trajs[i]["rounds"].append({"t": t, "query": queries[i] if t else "", "new_pids": new[i],
                                       "new_titles": [corpus[p]["title"] for p in new[i]], "reasoning": gen_text[i][-400:],
                                       "answer": ans, "next_query": "", "lp_mean": float(np.mean(lps)), "lp_min": float(np.min(lps)),
                                       "ptoks": ptoks[i], "ctoks": ctoks[i], "probe_toks": len(c.token_ids) + 1})
        # generator reading of every candidate in hand under the current context (shared prefix)
        jobs, meta = [], []
        for i in range(n):
            R = trajs[i]["rounds"]
            prefix = ctx[i] + "<answer>"
            pids = tok.encode(prefix, add_special_tokens=False)
            seen = set()
            for ci in range(t, -1, -1):
                k_ = normalize(R[ci]["answer"])
                if k_ in seen:
                    continue
                seen.add(k_)
                fids = tok.encode(prefix + " " + R[ci]["answer"], add_special_tokens=False)
                k = 0
                while k < min(len(pids), len(fids)) and pids[k] == fids[k]:
                    k += 1
                jobs.append(TokensPrompt(prompt_token_ids=fids)); meta.append((i, ci, k, len(fids)))
        for (i, ci, k, m), o in zip(meta, llm.generate(jobs, score, use_tqdm=False)):
            s = sum(o.prompt_logprobs[j][o.prompt_token_ids[j]].logprob for j in range(max(k, 1), m))
            L = lik[qs[i]["qid"]]
            L["rows"][t].append([ci, round(s, 4), m - k]); L["tok"][t] += m - k + 1
        print(f"[{ds}] round {t} done, {time.time() - t0:.0f}s", flush=True)
    for tr in trajs:
        if tr["native_stop"] is None:
            tr["native_stop"] = T
    os.makedirs(f"{ROOT}/traj/{tag}", exist_ok=True)
    with gzip.open(f"{ROOT}/traj/{tag}/{ds}.jsonl.gz", "wt", encoding="utf-8") as f:
        for tr in trajs:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")
    os.makedirs(f"{ROOT}/data/lik/{tag}", exist_ok=True)
    with gzip.open(f"{ROOT}/data/lik/{tag}/{ds}.json.gz", "wt", encoding="utf-8") as f:
        json.dump(lik, f)
    print("saved", ds, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
