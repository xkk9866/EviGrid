# EviGrid

Joint answer selection and adaptive stopping for retrieval-augmented generation.

This repository publishes the project's implementation, training, retrieval, evaluation and experiment-launch code. Manuscripts, submission documents, publication figures and their typesetting scripts are excluded.

## Code map

- `code/grid.py`: answer-evidence grid controller.
- `code/calibrate.py`, `grid_ens.py`, `evaluate.py`, `paired_ci.py`: calibration, ensembles and evaluation.
- `code/prepare_data.py`, `rollout.py`, `lik_score.py`: data and retrieval trajectories.
- `release/code/`: preserved code snapshot; generated release data and manuscript files are not included.

## Environment and data

The code uses Python and a GPU-oriented PyTorch/Transformers stack. Inspect the imports and command-line options of the selected entry point before installing dependencies. Historical snapshots and comparison implementations are retained; they are not all required for one experiment.

Download benchmark datasets and pretrained models from their original providers. The source release does not contain benchmark corpora, generated trajectories, trained weights or retrieval indexes. Some launch scripts use the original experiment directory layout, GPU assignments and model-cache paths; set these for your own machine before running them. No claim is made that a new machine reproduces the experiments without preparing these inputs.

## Remote execution

Remote helpers that previously embedded login details now read `SSH_HOST`, `SSH_PORT` (default 22), `SSH_USER` and optional `SSH_PASSWORD` from the environment. Set these locally, or use your SSH agent/key configuration. Do not commit credentials or local server information files.

## Included snapshots and third-party code

Source subdirectories retain their original relative paths. Where `repos/` is present, the included upstream source retains its existing LICENSE and NOTICE files. Those licenses apply to their respective components. No new license grant for the project's own code has been added in this release.

## Release checks

Python source files were parsed for syntax, publication files were excluded, and the release was checked for embedded SSH credentials and common token/private-key patterns. Training and large-scale retrieval experiments were not rerun as part of packaging.
