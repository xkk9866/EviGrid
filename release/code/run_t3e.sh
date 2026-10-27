#!/bin/bash
# Independent evaluation on `test3` (frozen models and operating points) on idle GPUs 3, 4, 5 (one dataset per GPU).
cd ~/research/esrag/code
L=../logs/t3; mkdir -p $L
mkdir $L/lock 2>/dev/null || { echo "already running"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=../.venv/bin/python
M=/home/yaokang/research/testimony_rag/models/Qwen2.5-7B-Instruct-AWQ
declare -A G=( [hotpotqa]=4 [2wiki]=2 [musique]=2 )
# something outside this script occasionally SIGKILLs jobs on the shared server: retry each job up to 4 times
job() { out=$1; shift; for a in 1 2 3 4; do [ -f "$out" ] && return 0; "$@"; sleep 5; done; }
# ---- 1. rollouts, one dataset per GPU
for ds in hotpotqa 2wiki musique; do
  ( job ../traj/qwen25-7b_t3/$ds.jsonl.gz env CUDA_VISIBLE_DEVICES=${G[$ds]} FRESH3=1 GPU_UTIL=0.45 bash -c "$P -u rollout.py $ds $M qwen25-7b_t3 >> $L/roll_$ds.log 2>&1" ) &
done
wait
# ---- 2. generator readings of every draft, one dataset per GPU
for ds in hotpotqa 2wiki musique; do
  ( job ../data/lik/qwen25-7b_t3/$ds.json.gz env CUDA_VISIBLE_DEVICES=${G[$ds]} GPU_UTIL=0.5 bash -c "$P -u lik_score.py $M --traj qwen25-7b_t3 --ds $ds >> $L/lik_$ds.log 2>&1" ) &
done
wait
touch $L/ROLL_DONE
# ---- 3. combined trajectory set qwen25-7b3 = qwen25-7b + test3 (original files untouched)
if [ ! -d ../traj/qwen25-7b3 ]; then
  cp -r ../traj/qwen25-7b ../traj/qwen25-7b3; mkdir -p ../data/lik; cp -r ../data/lik/qwen25-7b ../data/lik/qwen25-7b3
  $P merge_traj.py qwen25-7b3 qwen25-7b_t3 > $L/merge.log 2>&1
fi
# ---- 4. predictions of the frozen controllers on test3, merged into copies of the existing prediction files
export CUDA_VISIBLE_DEVICES=4
mkdir -p ../preds/qwen25-7b3
for tag in p_full p_full_s1 p_full_s2; do
  [ -f ../preds/qwen25-7b_t3/essn_${tag}_from_qwen25-7b.json ] || $P -u grid.py predict --tag $tag --traj qwen25-7b_t3 --src_traj qwen25-7b > $L/pred_$tag.log 2>&1
done
for m in simrag stoprag; do
  [ -f ../preds/qwen25-7b_t3/${m}_from_qwen25-7b.json ] || PREDICT_ONLY=1 $P -u baselines.py $m --traj qwen25-7b --also_traj qwen25-7b_t3 > $L/pred_$m.log 2>&1
done
$P - <<'PY' > $L/mergepreds.log 2>&1
import json, os
R = os.path.expanduser("~/research/esrag/preds")
pairs = [(f"essn_{t}.json", f"essn_{t}_from_qwen25-7b.json") for t in ("p_full", "p_full_s1", "p_full_s2")] + [(f"{m}.json", f"{m}_from_qwen25-7b.json") for m in ("simrag", "stoprag")]
for base, new in pairs:
    a = json.load(open(f"{R}/qwen25-7b/{base}")); b = json.load(open(f"{R}/qwen25-7b_t3/{new}"))
    assert not (set(a) & set(b)); a.update(b)
    json.dump(a, open(f"{R}/qwen25-7b3/{base}", "w")); print(base, len(a))
PY
# ---- 5. ensemble, cross-fitted calibration (fitted on the development pool only), evaluation (CPU)
export TRAJ=qwen25-7b3 CUDA_VISIBLE_DEVICES=
$P grid_ens.py p_full,p_full_s1,p_full_s2 p_ens > $L/ens.log 2>&1
ESRAG_ROOT=/home/yaokang/research/esrag $P calibrate.py crossfit p_full,p_full_s1,p_full_s2,p_ens > $L/cf.log 2>&1
ev() { $P _ev15.py --traj qwen25-7b3 --dev calib,test,calib2 --test test3 --cert cert "$@"; }
ev --essn p_full_cf,p_full_s1_cf,p_full_s2_cf,p_full,p_full_s1,p_full_s2 --main p_full_cf --out ../results/t3_all.json > $L/eval_all.log 2>&1 &
ev --essn p_ens_cf,p_ens --main p_ens_cf --out ../results/t3_ens.json > $L/eval_ens.log 2>&1 &
wait
touch $L/T3_DONE
