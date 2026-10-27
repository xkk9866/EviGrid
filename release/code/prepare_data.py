"""Build question splits and pooled corpora for HotpotQA / 2Wiki / MuSiQue.

Output per dataset (data/proc/<ds>/):
  questions.jsonl : {qid, question, answers:[..], split: train|calib|test, gold_titles:[..]}
  corpus.jsonl    : {pid, title, text}
Corpus = de-duplicated pool of all paragraphs attached to every dev question plus the
selected train questions (distractor-pooled corpus, following IRCoT / Stop-RAG).
"""
import gzip, json, os, random, sys
import pandas as pd

HUB = os.path.expanduser("~/.cache/huggingface/hub")
ROOT = "Y:/xkk_workspace/workspace/WWW/9022ERAG/data"
RAW = f"{ROOT}/raw"
N_TRAIN, N_CALIB, N_TEST = 2000, 300, 1000
random.seed(13)


def write(ds, questions, paras):
    out = f"{ROOT}/proc/{ds}"
    os.makedirs(out, exist_ok=True)
    with gzip.open(f"{out}/questions.jsonl.gz", "wt", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with gzip.open(f"{out}/corpus.jsonl.gz", "wt", encoding="utf-8") as f:
        for i, (t, x) in enumerate(paras):
            f.write(json.dumps({"pid": i, "title": t, "text": x}, ensure_ascii=False) + "\n")
    print(ds, "questions", len(questions), "corpus", len(paras), flush=True)


class Pool:
    def __init__(self):
        self.seen, self.items = set(), []

    def add(self, title, text):
        k = (title, text[:200])
        if k not in self.seen:
            self.seen.add(k)
            self.items.append((title, text))


def split_rows(dev_rows, train_rows):
    random.shuffle(dev_rows)
    random.shuffle(train_rows)
    test, calib = dev_rows[:N_TEST], dev_rows[N_TEST:N_TEST + N_CALIB]
    return train_rows[:N_TRAIN], calib, test


def hf_ctx(row):
    c = row["context"]
    return [(t, " ".join(s)) for t, s in zip(c["title"], c["sentences"])]


def wiki_style(ds, snap, train_files, dev_file):
    dev = pd.read_parquet(f"{snap}/{dev_file}")
    train = pd.concat([pd.read_parquet(f"{snap}/{f}") for f in train_files])
    dev_rows = [r for _, r in dev.iterrows()]
    tr_rows = [r for _, r in train.sample(n=N_TRAIN * 2, random_state=13).iterrows()]
    tr, ca, te = split_rows(dev_rows[:], tr_rows)
    pool, qs = Pool(), []
    for r in dev_rows + tr:
        for t, x in hf_ctx(r):
            pool.add(t, x)
    for name, rows in (("train", tr), ("calib", ca), ("test", te)):
        for r in rows:
            qs.append({"qid": str(r["id"]), "question": r["question"], "answers": [r["answer"]],
                       "split": name, "gold_titles": sorted(set(r["supporting_facts"]["title"].tolist()))})
    write(ds, qs, pool.items)


def musique():
    ld = lambda s: [json.loads(l) for l in open(f"{RAW}/MuSiQue/musique_ans_v1.0_{s}.jsonl", encoding="utf-8")]
    dev, train = ld("dev"), ld("train")
    tr, ca, te = split_rows(dev[:], train[:])
    pool, qs = Pool(), []
    for r in dev + tr:
        for p in r["paragraphs"]:
            pool.add(p["title"], p["paragraph_text"])
    for name, rows in (("train", tr), ("calib", ca), ("test", te)):
        for r in rows:
            qs.append({"qid": r["id"], "question": r["question"], "answers": [r["answer"]] + r["answer_aliases"],
                       "split": name, "gold_titles": sorted({p["title"] for p in r["paragraphs"] if p["is_supporting"]})})
    write("musique", qs, pool.items)


if __name__ == "__main__":
    which = sys.argv[1:] or ["hotpotqa", "2wiki", "musique"]
    if "hotpotqa" in which:
        wiki_style("hotpotqa", f"{HUB}/datasets--hotpot_qa/snapshots/1908d6afbbead072334abe2965f91bd2709910ab",
                   ["distractor/train-00000-of-00002.parquet", "distractor/train-00001-of-00002.parquet"],
                   "distractor/validation-00000-of-00001.parquet")
    if "2wiki" in which:
        wiki_style("2wiki", f"{HUB}/datasets--framolfese--2WikiMultihopQA/snapshots/fe713bfbd1afbca1a65246741a75890405d56a3a",
                   ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
                   "data/validation-00000-of-00001.parquet")
    if "musique" in which:
        musique()
