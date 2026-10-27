cd ~/research/esrag
date
python3 - <<'EOF'
import json
from collections import Counter
d = json.load(open('logs/v12/status.json'))
print(Counter(d.values()))
print([k for k, v in d.items() if v != 'done'])
EOF
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
ps -eo pid,user,etime,cmd | grep -E 'python|vllm' | grep -v grep | cut -c1-220
echo "--- traj"; ls traj
echo "--- preds qwen25-7b"; ls preds/qwen25-7b | tr '\n' ' '
echo; echo "--- preds dirs"; ls preds
echo "--- results"; ls results
