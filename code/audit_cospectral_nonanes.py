"""Recompute every cospectral nonane class exactly (audit of tab:nonanes).

There are five cospectral classes among the 35 nonanes, all of them pairs.
An earlier draft of the manuscript reported four, because the writer in
cospectral_alkane_search.py serialised only classes[:4] -- a display
truncation reported as an exhaustive count. Both have since been fixed.

This script recomputes the classes from scratch in exact integer arithmetic
and records, for each, whether the two members also share their total walk
counts -- the property the nonane subsection once attributed to them. They
do not: cospectrality fixes the closed-walk counts tr(A^k) but not
W_k = 1^T A^k 1, which also depends on the main-eigenvalue weights.
"""
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dgs_census import charpoly_int  # noqa: E402

ROOT = HERE.parent


def walks(G, kmax):
    """Exact W_k = 1^T A^k 1 for k = 0..kmax."""
    n = G.number_of_nodes()
    adj = [sorted(G.neighbors(v)) for v in range(n)]
    v = [1] * n
    out = [n]
    for _ in range(kmax):
        v = [sum(v[u] for u in adj[i]) for i in range(n)]
        out.append(sum(v))
    return tuple(out)


def main():
    trees = [T for T in nx.nonisomorphic_trees(9)
         if max(d for _, d in T.degree()) <= 4]
    groups = defaultdict(list)
    for T in trees:
        cp = tuple(charpoly_int(nx.to_numpy_array(T).astype(int).tolist()))
        g6 = nx.to_graph6_bytes(T, header=False).strip().decode()
        groups[cp].append(g6)

    classes = [(cp, mem) for cp, mem in groups.items() if len(mem) > 1]
    classes.sort(key=lambda t: t[1])
    n_pairs = sum(len(m) - 1 for _, m in classes)

    print("alkane skeletons n=9 :", len(trees))
    print("cospectral classes   :", len(classes))
    print("cospectral pairs     :", n_pairs)
    print()

    recs = []
    for i, (cp, mem) in enumerate(classes, 1):
        Gs = [nx.from_graph6_bytes(g.encode()) for g in mem]
        wieners = [int(nx.wiener_index(G)) for G in Gs]
        hosoya = sum(abs(int(c)) for c in cp)
        ws = [walks(G, 9) for G in Gs]
        walks_equal = len(set(ws)) == 1
        sep = len(set(wieners)) == len(wieners)
        recs.append({
            "class": i,
            "members_g6": mem,
            "wiener": wieners,
            "hosoya": hosoya,
            "walk_counts": [[str(x) for x in w] for w in ws],
            "walks_equal": walks_equal,
            "wiener_separates": sep,
        })
        print("class %d: %s" % (i, mem))
        print("   wiener     : %s   separates=%s" % (wieners, sep))
        print("   hosoya : %d (shared, spectrum-determined)" % hosoya)
        print("   walks equal   : %s" % walks_equal)
        if not walks_equal:
            first = next(k for k in range(10) if ws[0][k] != ws[1][k])
            print("   walks differ  : first at k=%d, %s vs %s"
                  % (first, ws[0][first], ws[1][first]))

    out = {
        "n": 9,
        "n_alkane_skeletons": len(trees),
        "cospectral_classes": len(classes),
        "cospectral_pairs": n_pairs,
        "arithmetic": ("exact integer characteristic polynomials "
            "(Faddeev-LeVerrier) and big-integer walk counts; "
                   "no floating point"),
        "note": ("Cospectral members share every adjacency-spectrum "
              "invariant: characteristic and matching polynomial, Hosoya "
              "index, closed-walk moments tr(A^k). They do NOT in general "
              "share total walk counts 1^T A^k 1, which depend on the main "
              "eigenvalue weights as well. The walks_equal field records "
              "this per class."),
        "classes": recs,
    }
    dst = ROOT / "results" / "cospectral_nonanes_exact.json"
    io.open(dst, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, indent=1) + "\n")
    print()
    print("->", dst)


if __name__ == "__main__":
    main()
