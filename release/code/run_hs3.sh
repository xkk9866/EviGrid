#!/bin/bash
# HALT / S2G-Judge for calib2 (GPU 2) and test3 (GPU 4), then merged into preds/qwen25-7b3/{halt,s2g}.json
cd ~/research/esrag/code
L=../logs/t3
mkdir $L/lock_hs3 2>/dev/null || { echo already; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
M=/home/yaokang/research/testimony_rag/models/Qwen2.5-7B-Instruct-AWQ
run() { gpu=$1; sp=$2; for a in 1 2 3 4 5 6; do
  [ -f ../preds/qwen25-7b3/halt_$sp.json ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu SPLITS=$sp OUTTAG=_$sp GPU_UTIL=0.55 ../.venv/bin/python -u llm_controllers.py $M --traj qwen25-7b3 >> $L/hs_$sp.log 2>&1
  sleep 20; done; }
run 2 calib2 &
run 4 test3 &
wait
python3 - <<'PY'
import json, os
R = os.path.expanduser("~/research/esrag/preds/qwen25-7b3")
for m in ("halt", "s2g"):
    d = json.load(open(f"{R}/{m}.json"))
    for sp in ("calib2", "test3"):
        d.update(json.load(open(f"{R}/{m}_{sp}.json")))
    json.dump(d, open(f"{R}/{m}.json", "w")); print(m, len(d))
PY
touch $L/HS_DONE2
