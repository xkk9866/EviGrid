"""Fresh, never-touched evaluation splits for the final (frozen-model) evaluation.

Replays the random state of prepare_data.py exactly (same seed, same call order), checks that the regenerated
calib/test question ids equal the ones in data/proc, and takes the development questions that no experiment has
used so far: test[800:1000] (never rolled out) and the development questions after the first 1,300.
  cert  : 500 (HotpotQA, 2Wiki) / 300 (MuSiQue) questions for risk certification
  test2 : 2,000 (HotpotQA, 2Wiki) / all remaining (MuSiQue) questions for the final test
Their paragraphs are already in the pooled corpora (prepare_data pools the paragraphs of ALL development
questions), so retrieval for existing trajectories is unchanged.
Output: data/proc/<ds>/questions_fresh.jsonl.gz
"""
import gzip, json, os, random
import pandas as pd
import prepare_data as P

N_CERT = {"hotpotqa": 500, "2wiki": 500, "musique": 300}
N_TEST2 = {"hotpotqa": 2000, "2wiki": 2000, "musique": 10 ** 9}


def emit(ds, dev_rows, conv):
    old = [json.loads(l) for l in gzip.open(f"{P.ROOT}/proc/{ds}/questions.jsonl.gz", "rt", encoding="utf-8")]
    old_test = [q["qid"] for q in old if q["split"] == "test"]
    old_cal = [q["qid"] for q in old if q["split"] == "calib"]
    te, ca = dev_rows[:P.N_TEST], dev_rows[P.N_TEST:P.N_TEST + P.N_CALIB]
    assert [conv(r)["qid"] for r in te] == old_test and [conv(r)["qid"] for r in ca] == old_cal, ds
    fresh = te[800:] + dev_rows[P.N_TEST + P.N_CALIB:]
    out = [{**conv(r), "split": "cert"} for r in fresh[:N_CERT[ds]]] + \
          [{**conv(r), "split": "test2"} for r in fresh[N_CERT[ds]:N_CERT[ds] + N_TEST2[ds]]]
    with gzip.open(f"{P.ROOT}/proc/{ds}/questions_fresh.jsonl.gz", "wt", encoding="utf-8") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(ds, "fresh", len(fresh), "cert", sum(q["split"] == "cert" for q in out), "test2", sum(q["split"] == "test2" for q in out))


def wiki_conv(r):
    return {"qid": str(r["id"]), "question": r["question"], "answers": [r["answer"]],
            "gold_titles": sorted(set(r["supporting_facts"]["title"].tolist()))}


def wiki_style(ds, snap, train_files, dev_file):
    dev = pd.read_parquet(f"{snap}/{dev_file}")
    train = pd.concat([pd.read_parquet(f"{snap}/{f}") for f in train_files])
    dev_rows = [r for _, r in dev.iterrows()]
    tr_rows = [r for _, r in train.sample(n=P.N_TRAIN * 2, random_state=13).iterrows()]
    random.shuffle(dev_rows)
    random.shuffle(tr_rows)
    emit(ds, dev_rows, wiki_conv)


def musique():
    ld = lambda s: [json.loads(l) for l in open(f"{P.RAW}/MuSiQue/musique_ans_v1.0_{s}.jsonl", encoding="utf-8")]
    dev, train = ld("dev"), ld("train")
    random.shuffle(dev)
    random.shuffle(train)
    emit("musique", dev, lambda r: {"qid": r["id"], "question": r["question"], "answers": [r["answer"]] + r["answer_aliases"],
                                    "gold_titles": sorted({p["title"] for p in r["paragraphs"] if p["is_supporting"]})})


if __name__ == "__main__":
    random.seed(13)
    wiki_style("hotpotqa", f"{P.HUB}/datasets--hotpot_qa/snapshots/1908d6afbbead072334abe2965f91bd2709910ab",
               ["distractor/train-00000-of-00002.parquet", "distractor/train-00001-of-00002.parquet"],
               "distractor/validation-00000-of-00001.parquet")
    wiki_style("2wiki", f"{P.HUB}/datasets--framolfese--2WikiMultihopQA/snapshots/fe713bfbd1afbca1a65246741a75890405d56a3a",
               ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
               "data/validation-00000-of-00001.parquet")
    musique()
