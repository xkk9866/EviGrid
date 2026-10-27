"""Parallel ranged download of a HF model repo through the mirror (resumable).
usage: python dl_hf.py <repo_id> <out_dir> [n_conn]"""
import json, os, sys, time, urllib.request, concurrent.futures as cf

EP = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
repo, out = sys.argv[1], sys.argv[2]
NC = int(sys.argv[3]) if len(sys.argv) > 3 else 16
CH = 32 * 1024 * 1024
os.makedirs(out, exist_ok=True)
_op = urllib.request.build_opener()
_op.addheaders = [("User-Agent", "curl/7.68.0")]
urllib.request.install_opener(_op)
meta = json.load(urllib.request.urlopen(f"{EP}/api/models/{repo}", timeout=60))
files = [s["rfilename"] for s in meta["siblings"] if not s["rfilename"].startswith(".")]
SIZES = {x["path"]: x["size"] for x in json.load(urllib.request.urlopen(f"{EP}/api/models/{repo}/tree/main", timeout=60))
         if x.get("type") == "file"}


def size_of(url):
    for _ in range(10):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                cr = r.headers.get("Content-Range")
                n = int(cr.split("/")[-1]) if cr else int(r.headers["Content-Length"])
                return n, url
        except Exception:
            time.sleep(3)
    raise RuntimeError(url)


def get_range(url, a, b, path):
    for _ in range(30):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == b - a + 1:
                with open(path, "r+b") as f:
                    f.seek(a); f.write(data)
                return
        except Exception:
            time.sleep(5)
    raise RuntimeError(f"{path} {a}")


for fn in files:
    dst = os.path.join(out, fn)
    url = f"{EP}/{repo}/resolve/main/{fn}"
    n, final = SIZES.get(fn) or size_of(url)[0], url
    if n < 50 * 1024 * 1024:  # small file: one plain request
        if not (os.path.exists(dst) and os.path.getsize(dst) == n):
            for _ in range(20):
                try:
                    data = urllib.request.urlopen(url, timeout=120).read()
                    open(dst, "wb").write(data); break
                except Exception:
                    time.sleep(5)
        continue
    if os.path.exists(dst) and os.path.getsize(dst) == n and not os.path.exists(dst + ".part"):
        continue
    if not os.path.exists(dst) or os.path.getsize(dst) != n:
        with open(dst, "wb") as f:
            f.truncate(n)
    open(dst + ".part", "w").close()
    done_file = dst + ".chunks"
    done = set(json.load(open(done_file))) if os.path.exists(done_file) else set()
    rng = [(a, min(a + CH, n) - 1) for a in range(0, n, CH) if a not in done]
    t0 = time.time()
    with cf.ThreadPoolExecutor(NC) as ex:
        futs = {ex.submit(get_range, final, a, b, dst): a for a, b in rng}
        for fu in cf.as_completed(futs):
            fu.result()
            done.add(futs[fu])
            json.dump(sorted(done), open(done_file, "w"))
    os.remove(dst + ".part")
    if os.path.exists(done_file):
        os.remove(done_file)
    print(fn, n, f"{n / max(time.time() - t0, 1e-3) / 1e6:.2f} MB/s", flush=True)
print("ALL DONE", flush=True)
