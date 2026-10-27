import gzip,json,sys,numpy as np
sys.path.insert(0,"code")
import os; os.environ["ESRAG_ROOT"]="."
from common import scores, normalize
for ds in ["hotpotqa","2wiki","musique"]:
    tr=[json.loads(l) for l in gzip.open(f"local/traj/q25/{ds}.jsonl.gz","rt",encoding="utf-8")]
    lik=json.load(gzip.open(f"local/lik_q25/{ds}.json.gz","rt"))
    te=[t for t in tr if t["split"]=="test2"]
    R={k:[] for k in ["last5","best_inhand5","lik_sel5","vote5","lik_sum5","oracle_any"]}
    for t in te:
        f=[scores(r["answer"],t["answers"])[1] for r in t["rounds"]]
        R["last5"].append(f[5]); R["best_inhand5"].append(max(f))
        L=lik.get(t["qid"])
        row={int(ci):(s,n) for ci,s,n in L["rows"][5]} if L else {}
        if row:
            ci=max(row,key=lambda c:row[c][0]/max(row[c][1],1)); R["lik_sel5"].append(f[ci])
        else: R["lik_sel5"].append(f[5])
        # majority vote among non-unknown drafts
        cnt={}
        for i,r in enumerate(t["rounds"]):
            n=normalize(r["answer"])
            if n and n!="unknown": cnt.setdefault(n,[]).append(i)
        if cnt:
            best=max(cnt,key=lambda n:(len(cnt[n]),max(cnt[n]))); R["vote5"].append(f[cnt[best][-1]])
        else: R["vote5"].append(f[5])
        # lik summed over rows
        acc={}
        for tt in range(6):
            rr={int(ci):(s,n) for ci,s,n in L["rows"][tt]} if L else {}
            if not rr: continue
            m=max(s for s,_ in rr.values()); Z=sum(np.exp(s-m) for s,_ in rr.values())
            for ci,(s,n) in rr.items():
                k=normalize(t["rounds"][ci]["answer"]); acc[k]=acc.get(k,0)+np.exp(s-m)/Z
        R["lik_sum5"].append(max((f[i] for i,r in enumerate(t["rounds"]) if normalize(r["answer"])==max(acc,key=acc.get)),default=f[5]) if acc else f[5])
    print(ds,{k:round(100*np.mean(v),1) for k,v in R.items() if v})
    print(" lik row example", L["rows"][5][:3], L["tok"])
