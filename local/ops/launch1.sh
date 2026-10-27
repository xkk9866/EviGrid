cd ~/research/esrag/code
L=../logs/v13; mkdir -p $L
E="CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
env $E CUDA_VISIBLE_DEVICES=3 nohup ../.venv/bin/python -u tally.py train --tag tally_m --seed 0 > $L/tally_m.log 2>&1 &
env $E CUDA_VISIBLE_DEVICES=7 nohup bash -c "../.venv/bin/python -u baselines.py stoprag --seed 2 --tag '_s2' > $L/stoprag_s2.log 2>&1; ../.venv/bin/python -u baselines.py simrag --seed 2 --tag '_s2' > $L/simrag_s2.log 2>&1" > /dev/null 2>&1 &
sleep 90
tail -3 $L/tally_m.log; tail -3 $L/stoprag_s2.log
nvidia-smi --query-gpu=index,memory.used --format=csv
