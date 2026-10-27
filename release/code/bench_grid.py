"""Controller cost of EviGrid: wall-clock per stop decision and encoder FLOPs, measured on the test split.

Per question and round t the controller needs (i) the cross-encoder readings of the cells that are new at round t
(the new passages against every answer in hand, and a new answer against every passage in hand; earlier cells are
cached) and (ii) one pass of the axial transformer and the readout. We time both on the device given by DEV
(default cuda) with batch size 1 (one question at a time, as in deployment) and report the mean over decisions.
usage: python bench_grid.py <tag> <n_questions_per_dataset> <out.json>
"""
import json, sys, time
import numpy as np, torch
import grid as G
from common import ROOT, DATASETS, T, load_traj, split

tag, nq, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
dev = G.DEVT
tok = G.AutoTokenizer.from_pretrained(G.ENC)
model = G.EviGrid([]).to(dev)
model.load_state_dict(torch.load(f"{ROOT}/models/grid_qwen25-7b_{tag}.pt", map_location=dev))
model.eval()
sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
res = {}
for ds in DATASETS:
    cp = G._Corpus(ds)
    te = split(load_traj("qwen25-7b", ds), "test2")[:nq]
    enc_ms, head_ms, ncell, ntok = [], [], [], []
    with torch.no_grad(), torch.autocast(dev, dtype=G.ADT):
        for tr in te:
            g = G.prep(tr, cp, [])
            nr = len(g["rows"]) + 1
            nc = len(g["col_round"])
            a, b = g["pairs"]
            # full encoder output once, to time the head alone (encode is replaced by a lookup)
            H = model.encode(a, b, tok, dev)
            real_encode = model.encode
            model.encode = lambda *_a, **_k: H
            sync(); t0 = time.perf_counter(); model(g, tok, dev); sync()
            head_ms.append((time.perf_counter() - t0) * 1000 / (T + 1))  # one forward covers the T+1 decisions of the question
            model.encode = real_encode
            done = set()
            for t in range(T + 1):
                vis = [i for i, k in enumerate(g["rows"]) if g["birth"][k] <= t] + [nr - 1]
                need = {(i, j) for i in vis for j in range(g["cut"][t])}
                new = sorted(need - done)
                done |= need
                if not new:
                    continue
                idx = [i * nc + j for i, j in new]
                sync(); t0 = time.perf_counter()
                model.encode([a[i] for i in idx], [b[i] for i in idx], tok, dev)
                sync(); enc_ms.append((time.perf_counter() - t0) * 1000)
                ncell.append(len(idx))
                ntok.append(sum(min(len(tok(b[i], add_special_tokens=False)["input_ids"]), G.MAXLEN) for i in idx) + 24 * len(idx))
    pdec = len(te) * (T + 1)
    res[ds] = {"encoder_ms_per_decision": float(np.sum(enc_ms) / pdec), "head_ms_per_decision": float(np.mean(head_ms)),
               "cells_per_decision": float(np.sum(ncell) / pdec), "encoder_tokens_per_decision": float(np.sum(ntok) / pdec),
               "encoder_gflops_per_decision": float(np.sum(ntok) / pdec * 2 * 110e6 / 1e9)}
    print(ds, {k: round(v, 2) for k, v in res[ds].items()}, flush=True)
json.dump({"device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "res": res}, open(out, "w"), indent=1)
