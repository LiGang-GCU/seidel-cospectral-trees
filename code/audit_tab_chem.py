"""Claim-audit: reconcile tab:chem (paper table 3) with chem_boundary_n*.json
and with OEIS A000602 skeleton counts."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
tex = (ROOT / "paper" / "sections" / "6_chemistry.tex").read_text(encoding="utf-8")

m = re.search(r"\\label\{tab:chem\}.*?\\end\{tabular\}", tex, re.S)
block = m.group(0)
rows = re.findall(r"(\d+) & ([\d{,}]+) & ([\d{,}]+) & (\d+)\\\\", block)
print("parsed rows:", len(rows))

ok = True
for n_s, skel_s, fp_s, fail_s in rows:
    n = int(n_s)
    skel = int(skel_s.replace("{,}", "").replace(",", ""))
    fp = int(fp_s.replace("{,}", "").replace(",", ""))
    fail = int(fail_s)
    d = json.load(open(ROOT / "results" / f"chem_boundary_n{n}.json"))
    match = (d["n_alkane_trees"] == skel
             and d["walk_collision_classes"] == fp
             and d["noncospectral_classes"] == fail)
    ok &= match
    print(f"n={n}: table=({skel},{fp},{fail}) "
          f"json=({d['n_alkane_trees']},{d['walk_collision_classes']},"
          f"{d['noncospectral_classes']})", "OK" if match else "MISMATCH")
print("TABLE vs JSON:", "ALL CONSISTENT" if ok else "PROBLEM")

a000602 = {4: 2, 5: 3, 6: 5, 7: 9, 8: 18, 9: 35, 10: 75, 11: 159, 12: 355,
           13: 802, 14: 1858, 15: 4347, 16: 10359, 17: 24894, 18: 60523,
           19: 148284, 20: 366319, 21: 910726, 22: 2278658, 23: 5731580,
           24: 14490245}
ok2 = True
for n, v in a000602.items():
    d = json.load(open(ROOT / "results" / f"chem_boundary_n{n}.json"))
    if d["n_alkane_trees"] != v:
        ok2 = False
        print(f"A000602 MISMATCH at n={n}: json={d['n_alkane_trees']} oeis={v}")
print("skeleton counts vs OEIS A000602:", "ALL MATCH" if ok2 else "MISMATCH")

# n=23 counterexample pair: verify charpolys are tree-shaped (deg-sum = 2n-2)
d23 = json.load(open(ROOT / "results" / "chem_boundary_n23.json"))
cls = d23["classes"][0]
for mem in cls["members"]:
    ds = mem["degseq"]
    print(f"n=23 member: |V|={len(ds)}, deg-sum={sum(ds)} "
          f"(need 2*22=44)", "OK" if len(ds) == 23 and sum(ds) == 44 else "BAD")
