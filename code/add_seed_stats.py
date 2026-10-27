"""Add three-seed mean +- s.d. rows for EviGrid to the result tables of both manuscripts (WWW and Elsevier).
Values come from result files only: v14/f_test2_all.json (main evaluation), v14/f_test3_all.json (test3),
v14/sel_seeds.json (Sel@5). Seed keys: raw = ESSN-p_full{,_s1,_s2}; calibrated = ESSN, ESSN-p_full_s{1,2}_cf.
Idempotent: rows are only inserted once.
usage: python add_seed_stats.py <results_dir> <paper_dir>"""
import json, sys
import numpy as np

RES, PAP = sys.argv[1:3]
DS = ["hotpotqa", "2wiki", "musique"]
RAWK = ["ESSN-p_full", "ESSN-p_full_s1", "ESSN-p_full_s2"]
CALK = ["ESSN", "ESSN-p_full_s1_cf", "ESSN-p_full_s2_cf"]
MAIN = json.load(open(f"{RES}/v14/f_test2_all.json"))["main"]
T3 = json.load(open(f"{RES}/v14/f_test3_all.json"))["main"]
SEL = json.load(open(f"{RES}/v14/sel_seeds.json"))
PM = r"{\scriptsize$\pm$"


def ms(src, ds, keys, m):
    v = np.array([src[ds][k][m] for k in keys], float)
    return v.mean(), v.std(ddof=1)


def cell(src, ds, keys, m):
    mu, sd = ms(src, ds, keys, m)
    if m == "Tok":
        return f"{mu / 1000:.2f}k{PM}{sd / 1000:.2f}}}"
    return f"{mu:.1f}{PM}{sd:.1f}}}"


def row(name, src, keys, metrics):
    return name + " & " + " & ".join(cell(src, ds, keys, m) for ds in DS for m in metrics) + r" \\"


def seed_rows(src):
    met = ["EM", "F1", "Tok", "AUC"]
    return [row(r"\textbf{\method{}}, raw readout (3 seeds)", src, RAWK, met),
            row(r"\textbf{\method{}}, calibrated (3 seeds)", src, CALK, met)]


def patch(path, subs, inserts):
    txt = open(path, encoding="utf8").read()
    if "(3 seeds)" in txt:
        print("skip (already patched)", path)
        return
    for old, new in subs:
        assert txt.count(old) == 1, (path, old[:60])
        txt = txt.replace(old, new)
    for anchor, rows in inserts:
        lines = txt.split("\n")
        idx = [i for i, l in enumerate(lines) if l.startswith(anchor)]
        assert len(idx) == 1, (path, anchor)
        lines[idx[0] + 1:idx[0] + 1] = rows
        txt = "\n".join(lines)
    open(path, "w", encoding="utf8").write(txt)
    print("patched", path)


# ------------------------------------------------------------------ captions
MAIN_OLD = "Results are point estimates. "
MAIN_NEW = (r"Rows marked ``3 seeds'' give the mean $\pm$ standard deviation of \method{} over three training seeds; "
            r"all other rows are point estimates. ")
T3_OLD = "Results are point estimates, with the best deployable single-model quality scores in bold."
T3_NEW = (r"Rows marked ``3 seeds'' give the mean $\pm$ standard deviation over three training seeds; "
          r"unmarked \method{} rows use seed~0, and baselines are point estimates. Bold marks the best deployable single-model quality scores.")
ABL_NEW = (r"The ``3 seeds'' row gives the full model's mean $\pm$ standard deviation over three training seeds; "
           r"ablated variants were trained with seed~0 only.")
abl = [r"\method{} (full, 3 seeds) & " + " & ".join(cell(MAIN, ds, CALK, m) for ds in DS for m in ("F1", "AUC")) + r" \\"]

STAT_OLD = ("Learned controllers use three training seeds. The main table reports the prespecified seed~0, and {ref} reports seed means and "
            "ensemble results. All tables use point estimates without appended seed standard deviations; variability across seeds is described in "
            "the text and figures.")
STAT_NEW = (r"\method{} is trained with three seeds. Result tables report the prespecified seed~0, the three-seed mean $\pm$ standard deviation "
            r"and the three-seed ensemble; learned baselines and ablated variants are reported for seed~0.")

# ------------------------------------------------------------------ WWW
patch(f"{PAP}/WWW/appendix.tex", [(MAIN_OLD, MAIN_NEW)], [(r"\textbf{\method{}}, calibrated & 49.8", seed_rows(MAIN))])
patch(f"{PAP}/WWW/body.tex",
      [(T3_OLD, T3_NEW),
       (r"Section~\ref{sec:rq1} reports full-model means across training seeds.", ABL_NEW),
       (STAT_OLD.format(ref=r"Section~\ref{sec:rq1}"), STAT_NEW)],
      [(r"\textbf{\method{}}, calibrated & 46.1", seed_rows(T3)), (r"\method{} (full) & 61.4", abl)])
# ------------------------------------------------------------------ Elsevier
patch(f"{PAP}/els/tables/main.tex", [(MAIN_OLD, MAIN_NEW)], [(r"\textbf{\method{}}, calibrated & 49.8", seed_rows(MAIN))])
patch(f"{PAP}/els/tables/test3.tex", [(T3_OLD, T3_NEW)], [(r"\textbf{\method{}}, calibrated & 46.1", seed_rows(T3))])
patch(f"{PAP}/els/tables/ablation.tex", [(r"Table~\ref{tab:seeds} reports full-model means across training seeds.", ABL_NEW)],
      [(r"\method{} (full) & 61.4", abl)])
patch(f"{PAP}/els/sec_exp.tex", [(STAT_OLD.format(ref=r"Table~\ref{tab:seeds}"), STAT_NEW)], [])

# els seed table: restore mean +- s.d. in every single-model cell
p = f"{PAP}/els/tables/seeds.tex"
txt = open(p, encoding="utf8").read()
if r"\pm" not in txt.split(r"\midrule", 1)[1]:
    txt = txt.replace("Tables report point estimates; seed variability is described in the text and calibration figure.",
                      r"Single-model rows give the mean $\pm$ standard deviation over the three seeds.")
    rep = {}
    for lab, keys in (("raw", RAWK), ("calibrated", CALK)):
        for name, m, d in (("F1", "F1", 1), ("Tok (k)", "Tok", 2), ("AUC", "AUC", 1)):
            vals = []
            for ds in DS:
                mu, sd = ms(MAIN, ds, keys, m)
                if m == "Tok":
                    mu, sd = mu / 1000, sd / 1000
                vals.append(f"${mu:.{d}f}\\pm{sd:.{d}f}$")
            rep[(lab, name)] = vals
    out, cur = [], None
    for l in txt.split("\n"):
        parts = [c.strip() for c in l.rstrip(" \\").split("&")]
        if len(parts) == 5 and parts[1] in ("F1", "Tok (k)", "AUC"):
            cur = parts[0] or cur
            if (cur, parts[1]) in rep:
                l = f"{parts[0]} & {parts[1]} & " + " & ".join(rep[(cur, parts[1])]) + r" \\"
        if len(parts) == 5 and parts[1] == "Sel@5 (F1)":
            l = f"{parts[0]} & {parts[1]} & " + " & ".join(f"${np.mean(SEL[ds]):.1f}\\pm{np.std(SEL[ds], ddof=1):.1f}$" for ds in DS) + r" \\"
        out.append(l)
    open(p, "w", encoding="utf8").write("\n".join(out))
    print("patched", p)
