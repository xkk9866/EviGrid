"""v12 experiment queue on ONE GPU (memory-aware, resumable, robust to other users' jobs on the same card).

  rollouts   fresh certification/test splits for the main generator; a second model family (Llama-3.1-8B), a smaller
             generator (Qwen3-4B), a second retriever (BM25) and an RL-trained search agent (Search-R1)
  training   ESSN (main + ablations, three seeds), learned baselines (three seeds), LLM controllers on the fresh splits,
             in-domain controllers for the other backbones, zero-shot transfer and few-shot adaptation
vLLM jobs are started only when no other job of this queue holds the card, with a memory fraction computed from the
memory actually in use (vLLM counts other processes against its budget).

usage: python sched_v12.py 2
"""
import os, subprocess, sys, time, json

GPU = int(sys.argv[1])
ROOT = os.path.expanduser("~/research/esrag")
CODE = f"{ROOT}/code"
LOG = f"{ROOT}/logs/v12"
PY = f"{ROOT}/.venv/bin/python"
Q25 = "/home/yaokang/research/testimony_rag/models/Qwen2.5-7B-Instruct-AWQ"
L31 = "/home/yaokang/research/testimony_rag/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
Q34 = "/home/yaokang/models/Qwen3-4B-AWQ"
SR1 = "/home/yaokang/models/SearchR1-qwen2.5-7b-em-ppo"
DS = ["hotpotqa", "2wiki", "musique"]
MAX_TRY = 2
MAIN = open(f"{CODE}/v12_main.txt").read().strip()    # ablation flags of the main model (fixed before this queue)
os.makedirs(LOG, exist_ok=True)

JOBS = []
def job(name, cmd, gb, deps=(), env=None, out=(), vllm=False, prio=5):
    JOBS.append({"name": name, "cmd": cmd, "gb": gb, "deps": list(deps), "env": env or {}, "state": "pending",
                 "try": 0, "out": list(out), "vllm": vllm, "prio": prio})


def rollout_set(tag, model, parts, env0, prio, agent=False):
    """parts: list of (part_name, env) -> rollouts per dataset, generator readings, merge into <tag>"""
    done = []
    for pname, penv in parts:
        ptag = f"{tag}_{pname}"
        for ds in DS:
            scr = "rollout_agent.py" if agent else "rollout.py"
            job(f"roll_{ptag}_{ds}", f"{PY} -u {scr} {ds} {model} {ptag}", 11, env={**env0, **penv}, vllm=True,
                out=[f"{ROOT}/traj/{ptag}/{ds}.jsonl.gz"], prio=prio)
        if not agent:
            job(f"lik_{ptag}", f"{PY} -u lik_score.py {model} --traj {ptag}", 11, deps=[f"roll_{ptag}_{d}" for d in DS],
                env=env0, vllm=True, out=[f"{ROOT}/data/lik/{ptag}/{d}.json.gz" for d in DS], prio=prio)
            done.append(f"lik_{ptag}")
        else:
            done += [f"roll_{ptag}_{d}" for d in DS]
    job(f"merge_{tag}", f"{PY} -u merge_traj.py {tag} " + " ".join(f"{tag}_{p}" for p, _ in parts), 0, deps=done,
        out=[f"{ROOT}/traj/{tag}/.merged_v12"], prio=prio)
    return f"merge_{tag}"


NS = {"NOSAMPLE": "1"}
FR = lambda n=None: {"FRESH2": "1", "SPLITS": "test2", **({"SUBSET": str(n)} if n else {})}
# ---- rollouts (priority: main generator first)
m_q25 = rollout_set("qwen25-7b", Q25, [("f", {"FRESH2": "1"})], {}, 0)
m_l31 = rollout_set("llama31-8b", L31, [("tr", {"ONLY": "train", **NS}), ("ca", {"ONLY": "calib", **NS}), ("te", {**FR(1000), **NS})], {}, 5)
m_q34 = rollout_set("qwen3-4b", Q34, [("tr", {"ONLY": "train", "LIMIT": "300", **NS}), ("te", {**FR(1000), **NS})], {}, 9)
m_bm = rollout_set("qwen25-7b-bm25", Q25, [("tr", {"ONLY": "train", "LIMIT": "300", **NS}), ("ca", {"ONLY": "calib", **NS}),
                                         ("te", {**FR(1000), **NS})], {"RETRIEVER": "bm25"}, 9)
sr1_ok = os.path.exists(f"{SR1}/model-00007-of-00007.safetensors") and not os.path.exists(f"{SR1}/model-00007-of-00007.safetensors.part")
m_sr1 = rollout_set("searchr1", SR1, [("tr", {"ONLY": "train"}), ("ca", {"ONLY": "calib"}), ("te", FR(1000))], {}, 7, agent=True) \
    if sr1_ok else None

# ---- main generator: ESSN (main + ablations) and learned baselines, three seeds
P25 = f"{ROOT}/preds/qwen25-7b"
drop = lambda s, *xs: ",".join(x for x in s.split(",") if x not in xs)
VARS = {"m": MAIN, "free": drop(MAIN, "voi"), "mono": drop(MAIN, "voi", "freeHead"), "noLik": MAIN + ",noLik",
        "single": MAIN + ",noContrast", "cached": MAIN + ",noReverify", "noCoh": drop(MAIN, "coh"),
        "fullIn": drop(MAIN, "minimal"), "structEnc": drop(MAIN, "simple")}
ORDER = ["m", "free", "single", "cached", "noLik", "mono", "noCoh", "fullIn", "structEnc"]
KEY = {"m", "free", "single", "cached"}
for sd in (0, 1, 2):
    sfx = "" if sd == 0 else f"_s{sd}"
    for v in ORDER:
        pr = 1 if v == "m" else (3 if sd == 0 else (4 if v in KEY else 11))
        job(f"e12_{v}{sfx}", f"{PY} -u essn.py train --tag e12_{v}{sfx} --ablate {VARS[v]} --seed {sd}", 7.5,
            deps=[m_q25], out=[f"{P25}/essn_e12_{v}{sfx}.json"], prio=pr)
    for m in ("stoprag", "simrag"):
        job(f"{m}{sfx}", f"{PY} -u baselines.py {m} --seed {sd} --tag '{sfx}'", 7, deps=[m_q25], out=[f"{P25}/{m}{sfx}.json"], prio=1)
job("rerank", f"{PY} -u baselines.py rerank", 7, deps=[m_q25], out=[f"{P25}/rerank.json"], prio=1)
job("adaptive", f"{PY} -u baselines.py adaptive", 4, deps=[m_q25], out=[f"{P25}/adaptive.json"], prio=1)
job("llm_fresh", f"{PY} -u llm_controllers.py {Q25}", 12, deps=[m_q25], env={"SPLITS": "cert,test2", "MERGE": "1"}, vllm=True,
    out=[f"{LOG}/llm_fresh.ok"], prio=2)
PRI = {"llama31-8b": (6, 6), "searchr1": (8, 8), "qwen3-4b": (10, 10), "qwen25-7b-bm25": (10, 10)}

# ---- other backbones: zero-shot transfer, few-shot adaptation (100 questions per dataset), in-domain training
for tgt, mg, indomain in (("llama31-8b", m_l31, True), ("qwen3-4b", m_q34, False), ("qwen25-7b-bm25", m_bm, False),
                          ("searchr1", m_sr1, True)):
    if mg is None:
        continue
    PT = f"{ROOT}/preds/{tgt}"
    p0, p1 = PRI[tgt]
    job(f"zs_{tgt}", f"{PY} -u essn.py predict --traj {tgt} --src_traj qwen25-7b --tag e12_m --ablate {MAIN}", 5,
        deps=[mg, "e12_m"], out=[f"{PT}/essn_e12_m_from_qwen25-7b.json"], prio=p0)
    for m in ("stoprag", "simrag"):
        job(f"zs_{m}_{tgt}", f"{PY} -u baselines.py {m} --tag '' --also_traj {tgt}", 5, deps=[mg, m],
            env={"PREDICT_ONLY": "1"}, out=[f"{PT}/{m}_from_qwen25-7b.json"], prio=p0)
    job(f"fs_{tgt}", f"{PY} -u essn.py train --traj {tgt} --tag e12_fs --ablate {MAIN} --init essn_qwen25-7b_e12_m "
                     f"--ntrain 100 --epochs 3", 7.5, deps=[mg, "e12_m"], out=[f"{PT}/essn_e12_fs.json"], prio=p0)
    if indomain:
        for sd in (0, 1, 2):
            sfx = "" if sd == 0 else f"_s{sd}"
            pr = p1 if sd == 0 else 12
            job(f"id_{tgt}{sfx}", f"{PY} -u essn.py train --traj {tgt} --tag e12_m{sfx} --ablate {MAIN} --seed {sd}", 7.5,
                deps=[mg], out=[f"{PT}/essn_e12_m{sfx}.json"], prio=pr)
            for m in ("stoprag", "simrag"):
                job(f"id_{m}_{tgt}{sfx}", f"{PY} -u baselines.py {m} --traj {tgt} --seed {sd} --tag '{sfx}'", 7, deps=[mg],
                    out=[f"{PT}/{m}{sfx}.json"], prio=pr)


def alive(jb):
    pat = " ".join(jb["cmd"].split()[1:]).replace("'", "")
    pat = "[" + pat[0] + "]" + pat[1:] + "$"
    return subprocess.run(["pgrep", "-f", "--", pat], capture_output=True).returncode == 0


for jb in JOBS:
    if jb["out"] and all(os.path.exists(f) for f in jb["out"]):
        jb["state"] = "done"
    elif alive(jb):
        jb["state"] = "adopted"


def mem():
    out = subprocess.run(["nvidia-smi", "-i", str(GPU), "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    used, tot = [int(x) for x in out.strip().split(",")]
    return used / 1024, tot / 1024


def log(msg):
    with open(f"{LOG}/sched.log", "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


running, recent = [], []
log(f"scheduler start GPU{GPU} MAIN={MAIN}: {len(JOBS)} jobs, {sum(j['state'] == 'done' for j in JOBS)} done")
while True:
    for r in running[:]:
        jb, p, t0 = r
        if p.poll() is not None:
            running.remove(r)
            ok = p.returncode == 0 and all(os.path.exists(f) for f in jb["out"] if not f.endswith((".ok", ".merged_v12")))
            if ok:
                jb["state"] = "done"; log(f"done   {jb['name']} ({(time.time() - t0) / 60:.0f} min)")
            else:
                jb["try"] += 1
                jb["state"] = "pending" if jb["try"] < MAX_TRY else "failed"
                log(f"FAIL   {jb['name']} rc={p.returncode} try={jb['try']} -> {jb['state']}")
    for jb in JOBS:
        if jb["state"] == "adopted":
            if all(os.path.exists(f) for f in jb["out"]):
                jb["state"] = "done"; log(f"done   {jb['name']} (adopted)")
            elif not alive(jb):
                jb["state"] = "pending"; log(f"adopted {jb['name']} vanished -> pending")
    if all(j["state"] in ("done", "failed") for j in JOBS) or \
            (not running and not any(j["state"] == "adopted" for j in JOBS) and
             all(j["state"] in ("done", "failed") or any(JOBS[[x["name"] for x in JOBS].index(d)]["state"] == "failed"
                                                          for d in j["deps"]) for j in JOBS)):
        break
    now = time.time()
    recent = [(t, gb) for t, gb in recent if now - t < 300]
    used, tot = mem()
    free = tot - used - sum(gb for _, gb in recent) - 1.0
    done = {j["name"] for j in JOBS if j["state"] == "done"}
    mine = len(running) + sum(j["state"] == "adopted" for j in JOBS)
    vllm_running = any(jb["vllm"] for jb, _, _ in running)
    cand = sorted([j for j in JOBS if j["state"] == "pending" and all(d in done for d in j["deps"])], key=lambda j: j["prio"])
    for jb in cand:
        if vllm_running:
            break
        if jb["vllm"]:
            if mine > 0 or free < jb["gb"]:
                break  # strict priority: drain the card for the vLLM job
            env_extra = {"GPU_UTIL": f"{min(0.95, (used + jb['gb']) / tot):.3f}"}
        else:
            if free < jb["gb"] or mine >= 3:
                continue
            env_extra = {}
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(GPU), HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", **jb["env"], **env_extra)
        out = open(f"{LOG}/{jb['name']}.log", "w")
        cmd = jb["cmd"]
        if jb["out"] and jb["out"][0].endswith((".ok", ".merged_v12")):
            cmd += f" && touch {jb['out'][0]}"
        p = subprocess.Popen(["bash", "-c", cmd], cwd=CODE, env=env, stdout=out, stderr=subprocess.STDOUT)
        jb["state"] = "running"
        running.append((jb, p, time.time()))
        recent.append((time.time(), jb["gb"]))
        free -= jb["gb"]; mine += 1
        log(f"start  {jb['name']} (free {free + jb['gb']:.1f} GB{', util ' + env_extra['GPU_UTIL'] if env_extra else ''})")
        if jb["vllm"]:
            break
    with open(f"{LOG}/status.json", "w") as f:
        json.dump({j["name"]: j["state"] for j in JOBS}, f, indent=0)
    time.sleep(30)

log("finished: " + ", ".join(f"{j['name']}={j['state']}" for j in JOBS if j["state"] != "done"))
open(f"{ROOT}/results/V12_DONE", "w").write("ok")
