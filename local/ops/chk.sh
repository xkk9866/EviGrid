cd ~/research/esrag
date
tail -4 logs/v13/tally_m.log; tail -2 logs/v13/stoprag_s2.log; tail -2 logs/v13/simrag_s2.log 2>/dev/null
ls -la preds/qwen25-7b/ | grep -E "tally_m|_s2"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
