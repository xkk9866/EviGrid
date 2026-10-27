"""Encode pooled corpora with BGE-base-en-v1.5 (fp16) -> data/index/<ds>.npy"""
import gzip, json, os, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel

ROOT = os.path.expanduser("~/research/esrag")
MODEL = "BAAI/bge-base-en-v1.5"


def load_corpus(ds):
    with gzip.open(f"{ROOT}/data/proc/{ds}/corpus.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


@torch.no_grad()
def encode(texts, tok, model, bs=256, max_len=256):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to("cuda")
        h = model(**b).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h, dim=-1).half().cpu())
        if i % (bs * 100) == 0:
            print(i, len(texts), flush=True)
    return torch.cat(out).numpy()


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    os.makedirs(f"{ROOT}/data/index", exist_ok=True)
    for ds in sys.argv[1:]:
        corpus = load_corpus(ds)
        emb = encode([f"{p['title']}: {p['text']}" for p in corpus], tok, model)
        np.save(f"{ROOT}/data/index/{ds}.npy", emb)
        print(ds, emb.shape, flush=True)
