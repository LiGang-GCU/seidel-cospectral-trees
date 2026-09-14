"""Generate LaTeX tables for the paper and re-verify the n=22 counterexample
walk data exactly (integer matrix powers)."""
import json
import sys
from fractions import Fraction
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dgs_census import charpoly_int  # noqa: E402

RES = "results"
OUT = "paper/sections/tab_census.tex"
OUT2 = "paper/sections/tab_walks.tex"

# ---------- census table ----------
files = ["walk_census_n8-12.json", "walk_census_n13-15.json", "walk_census_n16-18.json"]
rows = []
for f in files:
    d = json.load(open(f"{RES}/{f}"))
    for n_str, rec in sorted(d.items(), key=lambda kv: int(kv[0])):
        n = rec["n"]
        rows.append((n, rec["n_trees"],
                     rec["classes"]["WALK"]["collision_classes"],
                     rec["classes"]["SSPEC"]["collision_classes"],
                     rec["classes"]["GEN"]["collision_classes"],
                     rec["classes"]["ASPEC"]["collision_classes"],
                     rec["partition_equality"]["SSPEC==GEN"],
                     rec["partition_equality"]["WALK==SSPEC"]))
for n in (19, 20, 21, 22):
    d = json.load(open(f"{RES}/walk_census_big_n{n}.json"))
    cc = d["collision_classes"]
    rows.append((n, d["n_trees"], cc["WALK"], cc["SSPEC"], cc["GEN"], None,
                 d["partition_equality"]["SSPEC==GEN"],
                 d["partition_equality"]["WALK==SSPEC"]))

# small orders n<=7: verify directly that no SSPEC/GEN collisions exist.
# Exact arithmetic only -- integral characteristic polynomials via
# Faddeev-LeVerrier and big-integer walk counts; no eigenvalues.
def fingerprint_pair(T):
    n_ = T.number_of_nodes()
    A = nx.to_numpy_array(T).astype(int)
    cp_A = tuple(charpoly_int(A.tolist()))
    Ab = np.ones((n_, n_), dtype=int) - np.eye(n_, dtype=int) - A
    cp_Ab = tuple(charpoly_int(Ab.tolist()))
    S = np.ones((n_, n_), dtype=int) - np.eye(n_, dtype=int) - 2 * A
    cp_S = tuple(charpoly_int(S.tolist()))
    adj = [sorted(T.neighbors(v)) for v in range(n_)]
    vec = [1] * n_
    W = [n_]
    for _ in range(n_ - 1):
        vec = [sum(vec[u] for u in adj[i]) for i in range(n_)]
        W.append(sum(vec))
    return (cp_A, cp_Ab), cp_S, tuple(W)


small_total = 0
for n_ in range(1, 8):
    fps_gen, fps_s, fps_w = dict(), dict(), dict()
    for T in nx.nonisomorphic_trees(n_):
        small_total += 1
        g6 = nx.to_graph6_bytes(T, header=False).decode().strip()
        gen, s, w = fingerprint_pair(T)
        fps_gen.setdefault(gen, set()).add(g6)
        fps_s.setdefault(s, set()).add(g6)
        fps_w.setdefault(w, set()).add(g6)
    coll_gen = sum(1 for v in fps_gen.values() if len(v) > 1)
    coll_s = sum(1 for v in fps_s.values() if len(v) > 1)
    coll_w = sum(1 for v in fps_w.values() if len(v) > 1)
    print(f"n={n_}: trees checked (exact), GEN collisions={coll_gen}, "
              f"SSPEC collisions={coll_s}, WALK collisions={coll_w}")
    assert coll_gen == 0 and coll_s == 0 and coll_w == 0

tot_trees = sum(r[1] for r in rows) + small_total
print("total trees n<=22:", tot_trees)
assert tot_trees == 9114285, tot_trees

lines = []
lines.append("\\begin{table}[htbp]")
lines.append("\\centering")
lines.append("\\caption{Census of all unlabeled trees on $n=8,\\dots,22$ vertices. "
             "For each fingerprint we give the number of non-trivial collision classes "
             "(classes containing at least two non-isomorphic trees). "
             "ASPEC collision counts for $n\\ge 19$ were not needed and were not computed.}")
lines.append("\\label{tab:census}")
lines.append("\\begin{tabular}{rrrrrrcc}")
lines.append("\\toprule")
lines.append("$n$ & \\#trees & WALK & SSPEC & GEN & ASPEC & SSPEC$=$GEN & WALK$=$SSPEC\\\\")
lines.append("\\midrule")
for (n, t, w, s, g, a, sg, ws) in rows:
    astr = f"{a:,}" if a is not None else "--"
    lines.append(f"{n} & {t:,} & {w:,} & {s:,} & {g:,} & {astr} & "
                 f"{'yes' if sg else 'no'} & {'yes' if ws else 'no'}\\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", OUT)

# ---------- counterexample walk verification ----------
d = json.load(open(f"{RES}/n22_counterexample.json"))[0]
g1, g2 = d["g6"]
T1 = nx.from_graph6_bytes(g1.encode())
T2 = nx.from_graph6_bytes(g2.encode())
n = T1.number_of_nodes()
assert n == 22 and T2.number_of_nodes() == 22

def adjlist(T):
    return [sorted(T.neighbors(v)) for v in range(n)]

def walks_pure(al, kmax):
    m = len(al)
    v = [1] * m
    out = []
    for k in range(kmax + 1):
        out.append(sum(v))
        nv = [0] * m
        for i in range(m):
            for j in al[i]:
                nv[j] += v[i]
        v = nv
    return out

W1 = walks_pure(adjlist(T1), 200)
W2 = walks_pure(adjlist(T2), 200)
# exact big-integer check: walks agree for k=0..200; since both sequences
# satisfy linear recurrences of order <= 22 (Cayley-Hamilton), the
# difference satisfies one of order <= 44, hence agreement on k=0..43
# already proves agreement for ALL k.  We check far beyond that.
assert W1 == W2, "walk sequences differ -- re-examine!"
print("walk counts agree for ALL k (verified exactly to k=200;"
      " recurrence argument extends to all k)")
print("W0..W21 =", W1[:22])
expected = [22, 42, 108, 276, 744, 1938, 5244, 13686, 37044, 96708, 261768,
            683430, 1849908, 4829898, 13073580, 34133892, 92393736, 241232034,
            652967772, 1704843798, 4614677748, 12048538260]
assert W1[:22] == expected, W1[:22]
print("W37:", W1[37], "vs", W2[37])

# charpoly coefficients from JSON
cp1 = [int(x) for x in d["members"][0]["adj_cp"]]
cp2 = [int(x) for x in d["members"][1]["adj_cp"]]
# matchings m(T,k) = |cp[2k]|
m1 = [abs(cp1[2 * k]) for k in range(1, 6)]
m2 = [abs(cp2[2 * k]) for k in range(1, 6)]
print("m(T1,k) k=1..5:", m1)
print("m(T2,k) k=1..5:", m2)
assert m1[2] == 796 and m2[2] == 788

# main eigenvalues: eigenvalue lambda is main iff 1 not orthogonal to eigenspace;
# for trees check via walk generating function poles. Use numeric check on the
# distinct eigenvalues.
ev1 = np.linalg.eigvalsh(nx.to_numpy_array(T1))
ev2 = np.linalg.eigvalsh(nx.to_numpy_array(T2))
from collections import Counter
def main_evs(A_num, evs):
    ones = np.ones(len(evs))
    res = Counter()
    w, V = np.linalg.eigh(A_num)
    comp = V.T @ ones
    for val, c in zip(w, comp):
        if abs(c) > 1e-9:
            res[round(val, 6)] += 1
    return res
A1f = nx.to_numpy_array(T1)
A2f = nx.to_numpy_array(T2)
me1 = main_evs(A1f, ev1)
me2 = main_evs(A2f, ev2)
print("main eigenvalues T1:", dict(me1))
print("main eigenvalues T2:", dict(me2))
assert me1 == me2

# ---------- walk table LaTeX ----------
lines = []
lines.append("\\begin{table}[htbp]")
lines.append("\\centering")
lines.append("\\caption{The unique counterexample pair at $n=22$. "
             "Left: the total walk counts $W_k(T_1)=W_k(T_2)$ for $k=0,\\dots,21$ "
             "(exact integers). Right: matching numbers $m(T_i,k)$, i.e.\\ the absolute "
             "values of the even characteristic-polynomial coefficients; the first "
             "disagreement is at $k=3$ (bold).}")
lines.append("\\label{tab:walks}")
lines.append("\\begin{tabular}{rr@{\\qquad\\qquad}rrr}")
lines.append("\\toprule")
lines.append("$k$ & $W_k$ & $k$ & $m(T_1,k)$ & $m(T_2,k)$\\\\")
lines.append("\\midrule")
for k in range(22):
    if 1 <= k <= 5:
        bold1 = f"\\textbf{{{m1[k-1]}}}" if k == 3 else str(m1[k-1])
        bold2 = f"\\textbf{{{m2[k-1]}}}" if k == 3 else str(m2[k-1])
        lines.append(f"{k} & {W1[k]:,} & {k} & {bold1} & {bold2}\\\\")
    else:
        lines.append(f"{k} & {W1[k]:,} & & &\\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")
open(OUT2, "w").write("\n".join(lines) + "\n")
print("wrote", OUT2)
print("ALL CHECKS PASSED")
