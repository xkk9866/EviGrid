cd ~/research/esrag/code
L=../logs/v13; P=../.venv/bin/python
M=$(cat v12_main.txt)
E="CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
env $E CUDA_VISIBLE_DEVICES=7 nohup $P -u essn.py train --tag e12_free --ablate simple,freeHead,coh,minimal > $L/e12_free.log 2>&1 &
env $E CUDA_VISIBLE_DEVICES=7 nohup $P -u essn.py train --tag e12_single --ablate $M,noContrast > $L/e12_single.log 2>&1 &
env $E CUDA_VISIBLE_DEVICES=6 nohup $P -u essn.py train --tag e12_cached --ablate $M,noReverify > $L/e12_cached.log 2>&1 &
env $E CUDA_VISIBLE_DEVICES=6 nohup $P -u essn.py train --tag e12_noLik --ablate $M,noLik > $L/e12_noLik.log 2>&1 &
sleep 120; tail -n 2 $L/e12_*.log; nvidia-smi --query-gpu=index,memory.used --format=csv
