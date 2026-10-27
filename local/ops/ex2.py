import json
P = json.load(open("../preds/qwen25-7b/essn_e12_m.json"))
k = "hotpotqa::5ae3a1aa5542994393b9e72d"
x = P[k]
print({a: (b if a in ("S", "V", "suff") else str(b)[:200]) for a, b in x.items()})
