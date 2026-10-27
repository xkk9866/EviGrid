"""Memory-aware scheduler for the v9 experiment (temporally coherent answer posterior; fresh calibration split;
in-domain training on three INT4 generators of two model families). Same mechanics as sched_v7.py.

usage: python sched_v9.py 4,5
"""
import os, subprocess, sys, time, json

GPUS = [int(g) for g in sys.argv[1].split(",")]
ROOT = os.path.expanduser("~/research/esrag")
CODE = f"{ROOT}/code"
LOG = f"{ROOT}/logs/v9"
PY = f"{ROOT}/.venv/bin/python"
MODELS = {"qwen25-7b": ("/home/yaokang/research/testimony_rag/models/Qwen2.5-7B-Instruct-AWQ", {}),
          "qwen3-4b": ("/home/yaokang/models/Qwen3-4B-AWQ", {}),
          "llama31-8b": ("/home/yaokang/research/testimony_rag/models/Meta-Llama-3.1-8B-Instruct", {"QUANT": "bnb"})}
MAX_TRY = 3
os.makedirs(LOG, exist_ok=True)
DS = ["hotpotqa", "2wiki", "musique"]

JOBS = []
def job(name, cmd, gb, deps=(), env=None, out=()):
    JOBS.append({"name": name, "cmd": cmd, "gb": gb, "deps": list(deps), "env": env or {}, "state": "pending",
                 "try": 0, "out": list(out)})

# ---- 1. rollouts: fresh calibration split for every generator, training rollouts for the new generators
parts = {"qwen25-7b": ["c2"], "qwen3-4b": ["tr", "c2"], "llama31-8b": ["all", "c2"]}
PENV = {"c2": {"FRESH": "1"}, "tr": {"ONLY": "train"}, "all": {}}
merge_deps = {}
for g, (mdir, menv) in MODELS.items():
    for pt in parts[g]:
        ptag = f"{g}_{pt}"
        for ds in DS:
            job(f"roll_{ptag}_{ds}", f"{PY} -u rollout.py {ds} {mdir} {ptag}", 17,
                env={**menv, **PENV[pt], "VLLM_GB": "1"}, out=[f"{ROOT}/traj/{ptag}/{ds}.jsonl.gz"])
        job(f"lik_{ptag}", f"{PY} -u lik_score.py {mdir} --traj {ptag}", 14, deps=[f"roll_{ptag}_{d}" for d in DS],
            env={**menv, "VLLM_GB": "1"}, out=[f"{ROOT}/data/lik/{ptag}/{d}.json.gz" for d in DS])
        merge_deps.setdefault(g, []).append(f"lik_{ptag}")
    job(f"merge_{g}", f"{PY} -u merge_traj.py {g} " + " ".join(f"{g}_{p}" for p in parts[g]), 0,
        deps=merge_deps[g], out=[f"{ROOT}/traj/{g}/.merged_v9"])

# ---- 2. controllers. ESSN main = coherent posterior + lightweight encoder + free profile head.
MAIN = "simple,freeHead,coh"
VARS7 = {"essn": MAIN, "essn_noCoh": "simple,freeHead", "essn_structEnc": "freeHead,coh",
         "essn_noContrast": MAIN + ",noContrast", "essn_noReverify": MAIN + ",noReverify", "essn_noLik": MAIN + ",noLik"}
VARSX = {"essn": MAIN, "essn_noCoh": "simple,freeHead"}
for g in MODELS:
    vs = VARS7 if g == "qwen25-7b" else VARSX
    for sd in (0, 1, 2):
        sfx = "" if sd == 0 else f"_s{sd}"
        for v, ab in vs.items():
            tag = f"v9{v[4:]}{sfx}"
            job(f"{g}_{v}{sfx}", f"{PY} -u essn.py train --traj {g} --tag {tag} --ablate {ab} --seed {sd}", 9,
                deps=[f"merge_{g}"], out=[f"{ROOT}/preds/{g}/essn_{tag}.json"])
        for m in ("stoprag", "simrag"):
            job(f"{g}_{m}{sfx}", f"{PY} -u baselines.py {m} --traj {g} --seed {sd} --tag {sfx}", 6,
                deps=[f"merge_{g}"], out=[f"{ROOT}/preds/{g}/{m}{sfx}.json"])
    if g != "qwen25-7b":  # zero-shot transfer of the Qwen2.5-7B controller (no retraining)
        job(f"{g}_transfer", f"{PY} -u essn.py predict --traj {g} --src_traj qwen25-7b --tag v9 --ablate {MAIN}", 6,
            deps=[f"merge_{g}", "qwen25-7b_essn"], out=[f"{ROOT}/preds/{g}/essn_v9_from_qwen25-7b.json"])


def outputs(jb):
    return jb["out"]


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
                cmd = jb["cmd"] + (f" && touch {jb['out'][0]}" if jb["name"].startswith("merge_") else "")
                p = subprocess.Popen(["bash", "-c", cmd], cwd=CODE, env=env, stdout=out, stderr=subprocess.STDOUT)
                jb["state"] = "running"
                running.append((jb, p, g, time.time()))
                recent[g].append((time.time(), jb["gb"]))
                mem[g] = (mem[g][0] - jb["gb"], mem[g][1] + jb["gb"], mem[g][2])
                log(f"start  {jb['name']} on GPU{g} (free before {avail + 1.0:.1f} GB)")
                break
    with open(f"{LOG}/status.json", "w") as f:
        json.dump({j["name"]: j["state"] for j in JOBS}, f, indent=0)
    time.sleep(30)

log("all jobs finished: " + ", ".join(f"{j['name']}={j['state']}" for j in JOBS if j["state"] != "done"))
open(f"{ROOT}/results/V9_DONE", "w").write("ok")
