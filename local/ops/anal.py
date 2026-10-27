import fitz, re, glob, os, collections
for f in sorted(glob.glob("sample/els_refs/*.pdf")):
    d = fitz.open(f)
    txt = "\n".join(p.get_text() for p in d)
    open(f[:-4] + ".txt", "w", encoding="utf8").write(txt)
    figs = sorted(set(int(x) for x in re.findall(r"\n\s*Fig(?:ure|\.)\s*(\d+)[\.:]", txt)))
    tabs = sorted(set(int(x) for x in re.findall(r"\n\s*Table\s*(\d+)[\.:]?\s*\n?", txt) if int(x) < 40))
    secs = re.findall(r"\n(\d(?:\.\d)?)\.?\s+([A-Z][A-Za-z ,:\-]{3,60})\n", txt)
    refs = len(re.findall(r"\n\[\d+\]", txt))
    words = len(txt.split())
    lim = len(re.findall(r"limitation", txt, re.I))
    print("=" * 100)
    print(os.path.basename(f), "pages", len(d), "words", words, "figs", figs[-1:] , "tables", tabs[-1:], "refs~", refs, "limitation-mentions", lim)
    print("  sections:", [f"{a} {b.strip()}" for a, b in secs][:40])
    ab = re.search(r"(?is)abstract(.{0,1500})", txt)
    print("  abstract:", " ".join(ab.group(1).split())[:1100] if ab else "")
    hi = re.search(r"(?is)highlights(.{0,700})", txt)
    if hi: print("  highlights:", " ".join(hi.group(1).split())[:600])
    for cap in re.findall(r"\n\s*((?:Fig\.|Figure|Table)\s*\d+[\.:][^\n]{0,160})", txt)[:16]:
        print("   cap:", cap.strip())
    intro = re.search(r"(?is)\n1\.?\s+Introduction\n(.{0,4500})", txt)
    if intro:
        print("  intro:", " ".join(intro.group(1).split())[:3200])
    con = re.search(r"(?is)\n\d\.?\s+Conclusions?[^\n]*\n(.{0,1800})", txt)
    if con:
        print("  conclusion:", " ".join(con.group(1).split())[:1500])
