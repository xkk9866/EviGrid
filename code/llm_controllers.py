"""Evidence-coverage stopping baselines that need LLM calls, run on the SHARED trajectories (calib + test).

  HALT (Roh & Han, Findings of EMNLP 2026), deployable variant:
      hop claims are generated zero-shot from the question; at every round each claim is verified against the
      newly retrieved passages with label scoring over {match, partial, null} (argmax of label log-probability,
      as in HALT); each claim keeps the strongest label seen so far (match > partial > null); the default rule
      stops when every claim is `match` (all_match). HALT fine-tunes a 3B LoRA verifier on HotpotQA discrepancy
      labels; we use the same INT4 generator zero-shot as verifier (noted in the paper).
  S2G-Judge (Li et al., ACL 2026), stop-only:
      context-only structured sufficiency judgement with gap items (category/target/slot/description schema).
      S2G-RAG also rewrites the next query from the gap items; under our shared-trajectory protocol only the
      judge's stop decision is used (noted in the paper). The original judge is a LoRA-distilled 3B model; we use
      the prompted (unfine-tuned) judge, which the S2G paper reports as its "unfine-tuned judge" ablation.

All verifier / judge tokens are recorded and charged as extra cost in evaluate.py.
usage: python llm_controllers.py <model_dir> [--traj qwen25-7b]
"""
import argparse, gzip, json, math, os, re
import numpy as np
from vllm import LLM, SamplingParams
from common import ROOT, DATASETS, T, load_traj, split

CLAIM_PROMPT = ("Question: {q}\n\nList the atomic factual claims (one per reasoning hop) that must be established from "
                "evidence to answer this question. Use placeholders such as [X] for unknown entities. Output 1 to 4 lines, "
                "one claim per line, no numbering, nothing else.")
VERIFY_PROMPT = ("Claim: {c}\n\nPassage: {p}\n\nDoes the passage support the claim? Answer with exactly one word: "
                 "match (the passage states the fact needed by the claim), partial (it mentions the relevant entity or "
                 "relation but not the full fact), or null (unrelated).")
S2G_PROMPT = ("You judge whether retrieved evidence is sufficient to answer a question. Use ONLY the evidence below, "
              "not your own knowledge.\n\nQuestion: {q}\n\nEvidence:\n{ev}\n\nFirst line: 'Sufficient: Yes' or "
              "'Sufficient: No'. If No, list the missing information as gap items, one per line, in the form "
              "'category | target | slot | description' with category in {{bridge_entity, attribute, relation, "
              "evidence_span, other}}.")
LABELS = ["match", "partial", "null"]
RANK = {"match": 2, "partial": 1, "null": 0}


def chat(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def passage(cp, pid, words=110):
    p = cp[pid]
    return f"{p['title']}: {' '.join(p['text'].split()[:words])}"


def label_scores(out, tok):
    """log-probability of each label's first token (labels have distinct first tokens)."""
    lp = out.outputs[0].logprobs[0]
    best = {}
    for lab in LABELS:
        ids = {tok.encode(v, add_special_tokens=False)[0] for v in (lab, " " + lab, lab.capitalize())}
        best[lab] = max([lp[i].logprob for i in ids if i in lp] or [-1e9])
    return max(best, key=best.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--traj", default="qwen25-7b")
    a = ap.parse_args()
    llm = LLM(a.model, quantization="awq_marlin", gpu_memory_utilization=float(os.environ.get("GPU_UTIL", 0.8)),
              max_model_len=6144, enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    gen = SamplingParams(temperature=0.0, max_tokens=128)
    one = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    halt, s2g = {}, {}
    for ds in DATASETS:
        with gzip.open(f"{ROOT}/data/proc/{ds}/corpus.jsonl.gz", "rt", encoding="utf-8") as f:
            cp = [json.loads(l) for l in f]
        sps = os.environ.get("SPLITS", "calib,test").split(",")
        trajs = [tr for tr in load_traj(a.traj, ds) if tr["split"] in sps]
        if os.environ.get("SMOKE"):
            trajs = trajs[:int(os.environ["SMOKE"])]
        # ---------------- HALT: claims
        cprompts = [chat(tok, CLAIM_PROMPT.format(q=tr["question"])) for tr in trajs]
        couts = llm.generate(cprompts, gen, use_tqdm=False)
        claims = []
        for o in couts:
            lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", l).strip() for l in o.outputs[0].text.split("\n")]
            claims.append([l for l in lines if len(l) > 3][:4] or ["the answer to the question"])
        # verify every (claim, new passage) pair of every round
        jobs = []
        for qi, tr in enumerate(trajs):
            for t, r in enumerate(tr["rounds"]):
                for ci, c in enumerate(claims[qi]):
                    for pid in r["new_pids"]:
                        jobs.append((qi, t, ci, chat(tok, VERIFY_PROMPT.format(c=c, p=passage(cp, pid)))))
        vouts = llm.generate([j[3] for j in jobs], one, use_tqdm=False)
        state = [[[0] * len(claims[qi]) for _ in range(T + 1)] for qi in range(len(trajs))]
        vtok = [[0] * (T + 1) for _ in trajs]
        for (qi, t, ci, _), o in zip(jobs, vouts):
            state[qi][t][ci] = max(state[qi][t][ci], RANK[label_scores(o, tok)])
            vtok[qi][t] += len(o.prompt_token_ids) + 1
        for qi, tr in enumerate(trajs):
            best = [0] * len(claims[qi])
            cov, part = [], []
            for t in range(T + 1):
                best = [max(x, y) for x, y in zip(best, state[qi][t])]
                cov.append(sum(b == 2 for b in best) / len(best))
                part.append(sum(b >= 1 for b in best) / len(best))
            vtok[qi][0] += len(couts[qi].prompt_token_ids) + len(couts[qi].outputs[0].token_ids)
            halt[f"{ds}::{tr['qid']}"] = {"claims": claims[qi], "match_frac": cov, "partial_frac": part, "tok": vtok[qi]}
        # ---------------- S2G-Judge (stop-only)
        sjobs, sidx = [], []
        for qi, tr in enumerate(trajs):
            ev = []
            for t, r in enumerate(tr["rounds"]):
                ev = ev + r["new_pids"]
                evs = "\n".join(f"[{i + 1}] {passage(cp, p)}" for i, p in enumerate(ev)) or "(none)"
                sjobs.append(chat(tok, S2G_PROMPT.format(q=tr["question"], ev=evs)))
                sidx.append((qi, t))
        souts = llm.generate(sjobs, SamplingParams(temperature=0.0, max_tokens=128, logprobs=20), use_tqdm=False)
        yes_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ["Yes", " Yes"]}
        no_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ["No", " No"]}
        res = {}
        for (qi, t), o in zip(sidx, souts):
            c = o.outputs[0]
            p_yes = 0.5
            for pos, lp in enumerate(c.logprobs[:8]):  # the Yes/No token right after "Sufficient:"
                y = sum(math.exp(v.logprob) for k, v in lp.items() if k in yes_ids)
                n = sum(math.exp(v.logprob) for k, v in lp.items() if k in no_ids)
                if y + n > 0.5:
                    p_yes = y / (y + n)
                    break
            gaps = sum(1 for l in c.text.split("\n")[1:] if l.count("|") >= 3)
            r = res.setdefault(qi, {"p": [0.0] * (T + 1), "gaps": [0] * (T + 1), "tok": [0] * (T + 1)})
            r["p"][t], r["gaps"][t], r["tok"][t] = p_yes, gaps, len(o.prompt_token_ids) + len(c.token_ids)
        for qi, tr in enumerate(trajs):
            s2g[f"{ds}::{tr['qid']}"] = res[qi]
        print(ds, "done", flush=True)
    os.makedirs(f"{ROOT}/preds/{a.traj}", exist_ok=True)
    for name, d in (("halt", halt), ("s2g", s2g)):
        fn = f"{ROOT}/preds/{a.traj}/{name}{os.environ.get('OUTTAG', '')}.json"
        if os.environ.get("MERGE") and os.path.exists(fn):  # add the new splits to the existing predictions
            d = {**json.load(open(fn)), **d}
        json.dump(d, open(fn, "w"))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
