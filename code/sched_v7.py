"""Memory-aware scheduler for the v6 experiment (submartingale-consistent value head) on several shared GPUs.

Every job declares the GPU memory it needs. Every 30 s the scheduler reads the free memory of each allowed GPU
(nvidia-smi, so other users' processes are respected), subtracts what it has just launched there, and starts the
next pending job whose dependencies are done on any GPU with room. Several jobs may share a GPU. A failed job
(e.g. OOM because another user grew) is re-queued up to MAX_TRY times. When all jobs are done, the CPU-side
evaluation, analysis and the latency benchmark run.

usage: python sched_v3.py 2,3,5
"""
import os, subprocess, sys, time, json

GPUS = [int(g) for g in sys.argv[1].split(",")]
ROOT = os.path.expanduser("~/research/esrag")
CODE = f"{ROOT}/code"
LOG = f"{ROOT}/logs/v7"
PY = f"{ROOT}/.venv/bin/python"
M7 = "/home/yaokang/research/testimony_rag/models/Qwen2.5-7B-Instruct-AWQ"
MAX_TRY = 3
os.makedirs(LOG, exist_ok=True)

# name, command, GB needed, dependencies
JOBS = []
def job(name, cmd, gb, deps=(), env=None):
    JOBS.append({"name": name, "cmd": cmd, "gb": gb, "deps": list(deps), "env": env or {}, "state": "pending", "try": 0})

for sd in (0, 1, 2):
    for v in ("full", "noReverify", "noContrast", "simple", "noLik"):
        tag = v if sd == 0 else f"{v}_s{sd}"
        ab = "" if v == "full" else f"--ablate {v}"
        job(f"essn_{tag}", f"{PY} -u essn.py train --tag {tag} {ab} --seed {sd}", 9)
job("essn_predict_4b", f"{PY} -u essn.py predict --tag full --traj qwen3-4b", 6, deps=["essn_full"])
job("essn_predict_4b_nolik", f"{PY} -u essn.py predict --tag noLik --ablate noLik --traj qwen3-4b", 6, deps=["essn_noLik"])


def outputs(jb):
    """files a job produces; a job whose outputs all exist is skipped (resume after a restart)"""
    P7, P4, n = f"{ROOT}/preds/qwen25-7b", f"{ROOT}/preds/qwen3-4b", jb["name"]
    if n.startswith("lik_"):
        d = n[4:]
        return [f"{ROOT}/data/lik/qwen3-4b/hotpotqa.json.gz"] if d == "4b" else [f"{ROOT}/data/lik/qwen25-7b/{d}.json.gz"]
    if n == "llm_7b":
        return [f"{P7}/halt.json", f"{P7}/s2g.json"]
    if n == "llm_4b":
        return [f"{P4}/halt.json", f"{P4}/s2g.json"]
    if n in ("stoprag", "simrag", "rerank"):
        return [f"{P7}/{n}.json", f"{P4}/{n}_from_qwen25-7b.json"]
    if n.startswith(("stoprag_s", "simrag_s")) or n == "adaptive":
        return [f"{P7}/{n}.json"]
    if n == "essn_predict_4b":
        return [f"{P4}/essn_full_from_qwen25-7b.json"]
    if n == "essn_predict_4b_nolik":
        return [f"{P4}/essn_noLik_from_qwen25-7b.json"]
    if n == "essn_predict_4b_decision":
        return [f"{P4}/essn_decision_from_qwen25-7b.json"]
    if n.startswith("essn_"):
        return [f"{P7}/{n}.json"]
    return []


def alive(jb):
    """is this job's command already running (launched by a previous scheduler instance)?"""
    pat = " ".join(jb["cmd"].split()[1:]) + "$"  # exact argument list, end-anchored (so "full" != "full_s1")
    return subprocess.run(["pgrep", "-f", "--", pat], capture_output=True).returncode == 0


for jb in JOBS:
    if outputs(jb) and all(os.path.exists(f) for f in outputs(jb)):
        jb["state"] = "done"
    elif alive(jb):
        jb["state"] = "adopted"  # keep the running process; it is done when its outputs appear


def free_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    res = {}
    for line in out.strip().split("\n"):
        i, used, tot = [int(x) for x in line.split(",")]
        res[i] = (tot - used) / 1024, used / 1024, tot / 1024
    return res


def log(msg):
    with open(f"{LOG}/sched.log", "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


running = []  # (job, proc, gpu, t_start)
recent = {g: [] for g in GPUS}  # (t, gb) launched recently, not yet visible in nvidia-smi
log(f"scheduler start on GPUs {GPUS}, {len(JOBS)} jobs, {sum(j['state'] == 'done' for j in JOBS)} already done")
while True:
    # reap finished jobs
    for r in running[:]:
        jb, p, g, t0 = r
        if p.poll() is not None:
            running.remove(r)
            if p.returncode == 0:
                jb["state"] = "done"
                log(f"done   {jb['name']} on GPU{g} ({(time.time() - t0) / 60:.0f} min)")
            else:
                jb["try"] += 1
                jb["state"] = "pending" if jb["try"] < MAX_TRY else "failed"
                log(f"FAIL   {jb['name']} on GPU{g} rc={p.returncode} try={jb['try']} -> {jb['state']}")
    for jb in JOBS:
        if jb["state"] == "adopted":
            if all(os.path.exists(f) for f in outputs(jb)):
                jb["state"] = "done"; log(f"done   {jb['name']} (adopted)")
            elif not alive(jb):
                jb["state"] = "pending"; log(f"adopted {jb['name']} vanished -> pending")
    if all(j["state"] in ("done", "failed") for j in JOBS):
        break
    mem = free_gb()
    now = time.time()
    for g in GPUS:
        recent[g] = [(t, gb) for t, gb in recent[g] if now - t < 240]
    done = {j["name"] for j in JOBS if j["state"] == "done"}
    for jb in JOBS:
        if jb["state"] != "pending" or any(d not in done for d in jb["deps"]):
            continue
        for g in sorted(GPUS, key=lambda x: -mem[x][0]):
            avail = mem[g][0] - sum(gb for _, gb in recent[g]) - 1.0  # 1 GB safety margin
            if avail >= jb["gb"]:
                env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(g), HF_HUB_OFFLINE="1",
                           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", **jb["env"])
                if "VLLM_GB" in jb["env"]:  # vLLM's fraction is of the whole card, including other users' memory
                    used, tot = mem[g][1], mem[g][2]
                    env["GPU_UTIL"] = f"{min(0.97, (used + jb['gb']) / tot):.3f}"
                out = open(f"{LOG}/{jb['name']}.log", "w")
                p = subprocess.Popen(jb["cmd"].split(), cwd=CODE, env=env, stdout=out, stderr=subprocess.STDOUT)
                jb["state"] = "running"
                running.append((jb, p, g, time.time()))
                recent[g].append((time.time(), jb["gb"]))
                mem[g] = (mem[g][0] - jb["gb"], mem[g][1] + jb["gb"], mem[g][2])
                log(f"start  {jb['name']} on GPU{g} (free before {avail + 1.0:.1f} GB)")
                break
    with open(f"{LOG}/status.json", "w") as f:
        json.dump({j["name"]: j["state"] for j in JOBS}, f, indent=0)
    time.sleep(30)

log("all training jobs finished: " + ", ".join(f"{j['name']}={j['state']}" for j in JOBS if j["state"] != "done"))
V = "full,noLik,noReverify,noContrast,simple"
for name, cmd in [("eval_main", f"{PY} evaluate.py --traj qwen25-7b --essn {V}"),
                  ("eval_transfer", f"{PY} evaluate.py --traj qwen3-4b --essn_preds_from qwen25-7b --essn full,noLik"),
                  ("analysis", f"{PY} analysis.py --traj qwen25-7b")]:
    rc = subprocess.run(cmd.split(), cwd=CODE, stdout=open(f"{LOG}/{name}.log", "w"), stderr=subprocess.STDOUT).returncode
    log(f"{name} rc={rc}")
# latency benchmark on the GPU with most free memory
g = max(GPUS, key=lambda x: free_gb()[x][0])
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(g), HF_HUB_OFFLINE="1")
rc = subprocess.run([PY, "bench.py"], cwd=CODE, env=env, stdout=open(f"{LOG}/bench.log", "w"), stderr=subprocess.STDOUT).returncode
log(f"bench rc={rc}")
open(f"{ROOT}/results/V7_ALL_DONE", "w").write("done\n")
log("ALL DONE")
