"""Controller overhead: wall-clock latency and peak memory of ESSN vs. a single-sequence controller
(Stop-RAG / SIM-RAG architecture) per stopping decision, batch size 1, on one RTX 4090.
LLM generation time per round is estimated from the measured rollout wall time."""
import json, re, time, glob, os
import numpy as np, torch
from transformers import AutoTokenizer
from common import ROOT, T, load_traj, split
from essn import ESSN, Data, build_batch, ENC
from baselines import Clf, state_text, encode

dev = "cuda"
data = Data("qwen25-7b", ["hotpotqa"])
test = split(data.items["hotpotqa"], "test")[:100]
tok = AutoTokenizer.from_pretrained(ENC)
res = {}
m = ESSN([]).to(dev).eval()
m.load_state_dict(torch.load(f"{ROOT}/models/essn_qwen25-7b_full.pt", map_location=dev), strict=False)
torch.cuda.reset_peak_memory_stats()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    for tr in test[:5]:
        m(build_batch([tr], data.corpus, tok, dev)[0])
    torch.cuda.synchronize(); t0 = time.time()
    for tr in test:
        m(build_batch([tr], data.corpus, tok, dev)[0])
    torch.cuda.synchronize()
# one forward processes all T+1 states of a question; report per decision (state)
res["ESSN_ms_per_decision"] = 1000 * (time.time() - t0) / (len(test) * (T + 1))
res["ESSN_peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
c = Clf(2).to(dev).eval()
torch.cuda.reset_peak_memory_stats()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    torch.cuda.synchronize(); t0 = time.time()
    for tr in test:
        for t in range(T + 1):
            c(encode(tok, [state_text(tr, t, data.corpus["hotpotqa"], True)], dev))
    torch.cuda.synchronize()
res["SingleSeq_ms_per_decision"] = 1000 * (time.time() - t0) / (len(test) * (T + 1))
res["SingleSeq_peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
# LLM time per round from rollout logs (wall seconds / (questions * rounds))
for ds in ["hotpotqa", "2wiki", "musique"]:
    log = open(f"{ROOT}/logs/roll_qwen25-7b_{ds}.log").read()
    secs = [int(x) for x in re.findall(r"round 5 done, (\d+)s", log)]
    n = len(load_traj("qwen25-7b", ds))
    res[f"LLM_ms_per_round_{ds}_batched"] = 1000 * secs[-1] / (n * (T + 1)) if secs else None
print(json.dumps(res, indent=1))
json.dump(res, open(f"{ROOT}/results/bench.json", "w"), indent=1)
