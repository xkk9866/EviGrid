cd ~/research/esrag/code
L=../logs/v13
E="CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
P=../.venv/bin/python
Q25=/home/yaokang/research/testimony_rag/models/Qwen2.5-7B-Instruct-AWQ
mkdir -p ../preds_v12_backup && cp -n ../preds/qwen25-7b/halt.json ../preds/qwen25-7b/s2g.json ../preds_v12_backup/
# HALT / S2G on the fresh splits (vLLM, INT4)
env $E CUDA_VISIBLE_DEVICES=4 GPU_UTIL=0.72 SPLITS=cert,test2 MERGE=1 nohup $P -u llm_controllers.py $Q25 > $L/llm_fresh.log 2>&1 &
# re-verification reranker and Adaptive-RAG on GPU 7 (after the seed-2 baselines finish)
env $E CUDA_VISIBLE_DEVICES=7 nohup bash -c "while pgrep -f 'baselines.py s' >/dev/null; do sleep 20; done; $P -u baselines.py rerank > $L/rerank.log 2>&1; $P -u baselines.py adaptive > $L/adaptive.log 2>&1" > /dev/null 2>&1 &
sleep 5; ps -eo pid,etime,cmd | grep -E "llm_controllers|baselines|tally" | grep -v grep | cut -c1-150
