# EviGrid

EviGrid selects an answer and decides whether to continue retrieving evidence. It compares answer drafts against the accumulated evidence in a grid, then estimates how much another retrieval step could improve the answer.

## Code layout

| Path | Purpose |
| --- | --- |
| `code/grid.py` | Grid model, training and prediction |
| `code/common.py` | Dataset definitions, trajectory loading and paths |
| `code/prepare_data.py`, `code/rollout.py` | Corpus preparation and retrieval trajectories |
| `code/lik_score.py` | Answer-likelihood features |
| `code/calibrate.py` | Calibration of answer values and retrieval gains |
| `code/grid_ens.py`, `code/paired_ci.py` | Ensembles and paired uncertainty estimates |
| `code/evaluate.py` | Answer-quality and retrieval-cost evaluation |
| `release/code/`, `local/` | Additional code snapshots and utilities |

The commands below use `code/`.

## Environment and inputs

The controller uses PyTorch, Transformers, NumPy and SciPy. Data preparation also uses pandas. Retrieval and reader stages have additional dependencies in their respective scripts.

Run the examples in Bash from the repository root:

```bash
export ESRAG_ROOT="$PWD"
```

Prepare HotpotQA, 2WikiMultihopQA and MuSiQue from their original sources. In `code/prepare_data.py`, set `ROOT` to your local **data directory** and check the dataset-cache paths. This is separate from `ESRAG_ROOT`, which points to the **project directory**.

The controller expects corpora under `data/proc/<dataset>/corpus.jsonl.gz` and trajectories under `traj/<trajectory-name>/<dataset>.jsonl.gz`. Trajectories contain the retrieval states, answer drafts and split labels. Generate them with the rollout and scoring scripts before training the controller.

## Train and calibrate

With trajectories named `qwen25-7b` prepared locally:

```bash
python code/grid.py train --tag g_full --traj qwen25-7b --seed 0
mkdir -p results
TRAJ=qwen25-7b python code/calibrate.py g_full
```

Training writes `models/grid_qwen25-7b_g_full.pt` and predictions to `preds/qwen25-7b/essn_g_full.json`. Calibration produces `essn_g_full_cal.json` in the same prediction directory and a summary under `results/`.

To apply the trained controller to prepared trajectories:

```bash
python code/grid.py predict --tag g_full --traj qwen25-7b --src_traj qwen25-7b
```

Use `code/evaluate.py` for answer-quality and cost comparisons, selecting the prediction tag and splits through its command-line options. The default grid uses passage granularity; the `GRAN` environment variable selects the granularity for other configurations.

Datasets, trajectories, trained weights and experiment outputs are prepared locally. The repository includes the implementation and experiment scripts, without manuscript files or publication-figure generators.

## License

A license for the project-specific code has not yet been specified.
