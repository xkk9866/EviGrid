#!/bin/bash
# Full v3 experiment queue on ONE GPU (default 6). INT4 generators only (Qwen2.5-7B-AWQ main, Qwen3-4B-AWQ transfer).
# Three multi-hop datasets (HotpotQA, 2Wiki, MuSiQue). Usage: bash run_v3.sh [GPU]
# Lane A (exclusive first, then ESSN jobs sequentially, each needs >10 GB);
# Lane B (small cross-encoder baselines, ~7 GB) runs concurrently with lane A's ESSN jobs.
G=${1:-6}
cd ~/research/esrag/code
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M7=/home/yaokang/research/testimony_rag/models/Qwen2.5-7B-Instruct-AWQ
L=../logs/v3; mkdir -p $L ../preds_v2_archive
# old v2 predictions were trained with NQ in the training mix -> archive, never mixed with v3 results
[ -d ../preds/qwen25-7b ] && mv ../preds/qwen25-7b ../preds_v2_archive/qwen25-7b_$(date +%m%d%H%M)
[ -d ../preds/qwen3-4b ] && mv ../preds/qwen3-4b ../preds_v2_archive/qwen3-4b_$(date +%m%d%H%M)
rm -f ../results/V3_*

run() { echo "[$(date +%T)] start $*" >> $L/queue.log; "$@" > $L/$(echo "$*" | tr ' /' '__' | cut -c1-90).log 2>&1; rc=$?; echo "[$(date +%T)] end($rc) $*" >> $L/queue.log; return $rc; }
# ESSN jobs share the GPU with lane B; a job that fails (e.g. OOM) is re-run alone after lane B has finished
FAILED=$L/failed_essn.txt; : > $FAILED
essn() { run ../.venv/bin/python -u essn.py "$@" || echo "$*" >> $FAILED; }

lane_a() {
  # 1) LLM-based coverage stoppers (HALT, S2G-Judge) need the whole GPU for vLLM -> run first
  GPU_UTIL=${GPU_UTIL:-0.85} run ../.venv/bin/python -u llm_controllers.py $M7 --traj qwen25-7b
  GPU_UTIL=${GPU_UTIL:-0.85} run ../.venv/bin/python -u llm_controllers.py $M7 --traj qwen3-4b
  touch ../results/V3_LLM_DONE
  if [ "${SEQ:-0}" = 1 ]; then while [ ! -f ../results/V3_BASE_DONE ]; do sleep 60; done; fi
  # 2) ESSN: decomposition variants first (seed 0), then seeds, then architecture ablations
  for v in full noReverify twoHead latestTarget noContrast; do
    ab="--ablate $v"; [ $v = full ] && ab=""
    essn train --tag $v $ab --seed 0
  done
  for sd in 1 2; do
    for v in full noReverify twoHead latestTarget noContrast; do
      ab="--ablate $v"; [ $v = full ] && ab=""
      essn train --tag ${v}_s$sd $ab --seed $sd
    done
  done
  for v in noProfile noCNN noAND noEvSup noLLM noGRU; do
    essn train --tag $v --ablate $v --seed 0
  done
  essn predict --tag full --traj qwen3-4b
  while [ ! -f ../results/V3_BASE_DONE ]; do sleep 60; done
  cp $FAILED $FAILED.round1
  while read -r args; do [ -n "$args" ] && run ../.venv/bin/python -u essn.py $args; done < $FAILED.round1
  touch ../results/V3_ESSN_DONE
}

lane_b() {
  while [ ! -f ../results/V3_LLM_DONE ]; do sleep 60; done
  [ "${SEQ:-0}" = 1 ] || export MEMFRAC=0.35
  run ../.venv/bin/python -u baselines.py stoprag --also_traj qwen3-4b
  run ../.venv/bin/python -u baselines.py simrag --also_traj qwen3-4b
  run ../.venv/bin/python -u baselines.py rerank --bs 1 --also_traj qwen3-4b
  run ../.venv/bin/python -u baselines.py adaptive
  for sd in 1 2; do
    run ../.venv/bin/python -u baselines.py stoprag --seed $sd --tag _s$sd
    run ../.venv/bin/python -u baselines.py simrag --seed $sd --tag _s$sd
  done
  touch ../results/V3_BASE_DONE
}

if [ "${SEQ:-0}" = 1 ]; then
  # shared GPU with little free memory: run everything sequentially (lane B finishes before the ESSN jobs)
  lane_b & lane_a; wait
else
  lane_a &
  lane_b &
  wait
fi
# 3) evaluation, analysis, overhead benchmark (CPU except bench)
V="full,noReverify,twoHead,latestTarget,noContrast,noProfile,noCNN,noAND,noEvSup,noLLM,noGRU"
run ../.venv/bin/python evaluate.py --traj qwen25-7b --essn $V
run ../.venv/bin/python evaluate.py --traj qwen3-4b --essn_preds_from qwen25-7b
run ../.venv/bin/python analysis.py --traj qwen25-7b
run ../.venv/bin/python bench.py
touch ../results/V3_ALL_DONE
