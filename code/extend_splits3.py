"""Third evaluation split `test3`: questions that no experiment has used (not train, not the development pool, not cert, not test2).

HotpotQA / 2WikiMultihopQA: development questions fresh[2500:4000] (cert and test2 took fresh[:2500]); same replay of the random
state as prepare_data.py / extend_splits.py (asserted).
MuSiQue: every development question is already used, so test3 = the 800 training-set questions that were pooled into the corpus
but never rolled out or used to train a controller (questions.jsonl.gz, split 'train', positions 1200..1999).
Output: data/proc/<ds>/questions_fresh3.jsonl.gz
"""
import gzip, json, os, random
import prepare_data as P
import extend_splits as X

N3 = 1500


def emit_wiki(ds, dev_rows, conv):
    old = [json.loads(l) for l in gzip.open(f"{P.ROOT}/proc/{ds}/questions.jsonl.gz", "rt", encoding="utf-8")]
    old_test = [q["qid"] for q in old if q["split"] == "test"]
    old_cal = [q["qid"] for q in old if q["split"] == "calib"]
    te, ca = dev_rows[:P.N_TEST], dev_rows[P.N_TEST:P.N_TEST + P.N_CALIB]
    assert [conv(r)["qid"] for r in te] == old_test and [conv(r)["qid"] for r in ca] == old_cal, ds
    fresh = te[800:] + dev_rows[P.N_TEST + P.N_CALIB:]
    used = X.N_CERT[ds] + X.N_TEST2[ds]
    prev = {json.loads(l)["qid"] for l in gzip.open(f"{P.ROOT}/proc/{ds}/questions_fresh.jsonl.gz", "rt", encoding="utf-8")}
    out = [{**conv(r), "split": "test3"} for r in fresh[used:used + N3]]
    assert not ({q["qid"] for q in out} & prev) and not ({q["qid"] for q in out} & {q["qid"] for q in old}), ds
    with gzip.open(f"{P.ROOT}/proc/{ds}/questions_fresh3.jsonl.gz", "wt", encoding="utf-8") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(ds, "test3", len(out))


def wiki_style(ds, snap, train_files, dev_file):
    import pandas as pd
    dev = pd.read_parquet(f"{snap}/{dev_file}")
    train = pd.concat([pd.read_parquet(f"{snap}/{f}") for f in train_files])
    dev_rows = [r for _, r in dev.iterrows()]
    tr_rows = [r for _, r in train.sample(n=P.N_TRAIN * 2, random_state=13).iterrows()]
    random.shuffle(dev_rows)
    random.shuffle(tr_rows)
    emit_wiki(ds, dev_rows, X.wiki_conv)


def musique():
    old = [json.loads(l) for l in gzip.open(f"{P.ROOT}/proc/musique/questions.jsonl.gz", "rt", encoding="utf-8")]
    train = [q for q in old if q["split"] == "train"]
    out = [{**q, "split": "test3"} for q in train[1200:2000]]
    with gzip.open(f"{P.ROOT}/proc/musique/questions_fresh3.jsonl.gz", "wt", encoding="utf-8") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print("musique test3", len(out), "(unused training-set questions)")


if __name__ == "__main__":
    random.seed(13)
    wiki_style("hotpotqa", f"{P.HUB}/datasets--hotpot_qa/snapshots/1908d6afbbead072334abe2965f91bd2709910ab",
               ["distractor/train-00000-of-00002.parquet", "distractor/train-00001-of-00002.parquet"],
               "distractor/validation-00000-of-00001.parquet")
    wiki_style("2wiki", f"{P.HUB}/datasets--framolfese--2WikiMultihopQA/snapshots/fe713bfbd1afbca1a65246741a75890405d56a3a",
               ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
               "data/validation-00000-of-00001.parquet")
    musique()
