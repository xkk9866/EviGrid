cd ~/research/esrag
python3 - <<'EOF'
import json
d = json.load(open('logs/v12/status.json'))
print("done:", [k for k, v in d.items() if v == 'done'])
EOF
ls -la traj/qwen25-7b traj/qwen25-7b_f
cat code/v12_main.txt; echo
tail -30 logs/v12/sched.log
ls .venv/bin/python && .venv/bin/python -c "import torch,transformers;print(torch.__version__,transformers.__version__)"
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, 'code')
from common import load_traj
from collections import Counter
for ds in ["hotpotqa", "2wiki", "musique"]:
    tr = load_traj("qwen25-7b", ds)
    print(ds, len(tr), Counter(t["split"] for t in tr), "lik" in tr[0]["rounds"][1], tr[0]["rounds"][1].get("next_query"))
EOF
ls ~/.cache/huggingface/hub | grep -i bge
