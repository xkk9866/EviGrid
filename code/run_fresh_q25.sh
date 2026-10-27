#!/bin/bash
# fresh certification/test splits for the main generator: rollouts (with samples + judge), generator readings, merge
cd ~/research/esrag/code
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GPU_UTIL=${GPU_UTIL:-0.45}
M=/home/yaokang/research/testimony_rag/models/Qwen2.5-7B-Instruct-AWQ
for ds in hotpotqa 2wiki musique; do
  [ -f ../traj/qwen25-7b_f/$ds.jsonl.gz ] || FRESH2=1 ../.venv/bin/python -u rollout.py $ds $M qwen25-7b_f > ../logs/v12/roll_q25f_$ds.log 2>&1
done
../.venv/bin/python -u lik_score.py $M --traj qwen25-7b_f > ../logs/v12/lik_q25f.log 2>&1
../.venv/bin/python -u merge_traj.py qwen25-7b qwen25-7b_f > ../logs/v12/merge_q25f.log 2>&1 && touch ../traj/qwen25-7b/.merged_fresh
