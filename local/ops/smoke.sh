cd ~/research/esrag/code
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p ../logs/v13
SMOKE=40 timeout 1500 ../.venv/bin/python -u tally.py train --tag tally_smk --train_ds hotpotqa --eval_ds hotpotqa --traj qwen25-7b --epochs 1 > ../logs/v13/smoke.log 2>&1
echo rc=$?
tail -20 ../logs/v13/smoke.log
ls -la ../preds/qwen25-7b/essn_tally_smk*.json
