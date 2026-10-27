"""Small patches that add the split `test3` and a TRAJ switch (run once, locally, before uploading)."""
import re
C = r"Y:\xkk_workspace\workspace\WWW\9022ERAG\code"
def sub(f, pairs):
    s = open(f"{C}\\{f}", encoding="utf-8").read()
    for a, b in pairs:
        assert a in s, (f, a[:60])
        s = s.replace(a, b)
    open(f"{C}\\{f}", "w", encoding="utf-8").write(s)
sub("rollout.py", [('    if os.environ.get("FRESH2"):',
 '    if os.environ.get("FRESH3"):  # third evaluation split (extend_splits3.py): questions no experiment has used\n'
 '        with gzip.open(f"{ROOT}/data/proc/{ds}/questions_fresh3.jsonl.gz", "rt", encoding="utf-8") as f:\n'
 '            return [json.loads(l) for l in f], load_corpus(ds)\n'
 '    if os.environ.get("FRESH2"):')])
sub("grid.py", [('EVAL_SPLITS = ("calib", "test", "calib2", "cert", "test2")', 'EVAL_SPLITS = ("calib", "test", "calib2", "cert", "test2", "test3")')])
sub("baselines.py", [('ev = split(items[ds], "calib,test,calib2,cert,test2")', 'ev = split(items[ds], "calib,test,calib2,cert,test2,test3")')])
sub("calibrate.py", [('from common import ROOT, DATASETS, T, load_traj, split', 'import os\nfrom common import ROOT, DATASETS, T, load_traj, split\nTRAJ = os.environ.get("TRAJ", "qwen25-7b")'),
                     ('load_traj("qwen25-7b", ds)', 'load_traj(TRAJ, ds)')])
s = open(f"{C}\\calibrate.py", encoding="utf-8").read()
s = s.replace('preds/qwen25-7b/', 'preds/{TRAJ}/').replace('"calib,test,calib2,cert,test2"', '"calib,test,calib2,cert,test2,test3"')
open(f"{C}\\calibrate.py", "w", encoding="utf-8").write(s)
sub("grid_ens.py", [('import json, sys, numpy as np', 'import json, os, sys, numpy as np\nTRAJ = os.environ.get("TRAJ", "qwen25-7b")'),
                    ('../preds/qwen25-7b/essn_{t}.json', '../preds/{TRAJ}/essn_{t}.json'), ('../preds/qwen25-7b/essn_{out}.json', '../preds/{TRAJ}/essn_{out}.json')])
print("patched")
