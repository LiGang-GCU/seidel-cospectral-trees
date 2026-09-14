"""Mechanism study for Result 3 (Seidel spectrum == generalized spectrum
on trees).

For every Seidel-cospectral tree class (n=17: 129 classes, n=18: 334),
test whether class members are SEIDEL-SWITCHING EQUIVALENT as graphs:
  exists signing sigma in {+1,-1}^n (global sign fixed) such that
  switch(T1, sigma) ~= T2 (isomorphic).

The answer decides the proof route:
  - if YES everywhere: equivalence may run through switching classes
    restricted to trees (two-graph viewpoint);
  - if NO: the equality of class collections is a deeper numerical
    coincidence -> moment/walk-count route.

Vectorized signing enumeration: batch sign matrices, adjacency of the
switched graph via sign mask, degree-multiset prefilter, iso check only
for survivors.
"""
import json
import time
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"


def switch_adj_batch(A, signs):
    """A: (n,n) 0/1 adjacency. signs: (B,n) +-1.
    Returns adjacency of switched graphs: (B,n,n)."""
    M = (signs[:, :, None] * signs[:, None, :]) > 0      # same-sign mask
    comp = 1.0 - A - np.eye(len(A))                       # complement adj
    return M * A[None] + (~M) * comp[None]


def switching_equivalent(T1, T2, batch=4096):
    """Test exists signing: switch(T1,s) isomorphic to T2."""
    n = T1.number_of_nodes()
    A = nx.to_numpy_array(T1)
    deg2 = sorted(d for _, d in T2.degree())
    m = n - 1  # fix sigma_0 = +1
    for start in range(0, 2 ** m, batch):
        B = min(batch, 2 ** m - start)
        bits = ((np.arange(start, start + B)[:, None]
                 >> np.arange(m)[None]) & 1)
        signs = np.where(bits == 0, 1.0, -1.0)
        signs = np.column_stack([np.ones(B), signs])     # sigma_0 = +1
        Hs = switch_adj_batch(A, signs)
        degs = np.sort(Hs.sum(2), axis=1)
        match = (degs == np.array(deg2)[None]).all(1)
        for idx in np.flatnonzero(match):
            H = nx.from_numpy_array(Hs[idx])
            if nx.is_isomorphic(H, T2):
                return True, int(start + idx)
    return False, -1


def analyze(n, classes):
    t0 = time.time()
    rows = []
    n_sw = 0
    for ci, cls in enumerate(classes):
        Gs = [nx.from_graph6_bytes(g.encode()) for g in cls]
        rec = {"n": n, "class": ci, "size": len(cls),
               "degseq_same": len({tuple(sorted(dict(G.degree()).values()))
                                     for G in Gs}) == 1,
               "diameters": sorted(nx.diameter(G) for G in Gs)}
        sw = {}
        for i, j in combinations(range(len(Gs)), 2):
            eq, _ = switching_equivalent(Gs[i], Gs[j])
            sw[f"{i}-{j}"] = eq
        rec["switching_equiv"] = sw
        rec["all_switching_equiv"] = all(sw.values())
        n_sw += rec["all_switching_equiv"]
        rows.append(rec)
        if (ci + 1) % 25 == 0:
            print(f"  n={n}: {ci+1}/{len(classes)} classes "
                  f"({time.time()-t0:.0f}s, sw-equiv so far: {n_sw})",
                  flush=True)
    ok = sum(r["all_switching_equiv"] for r in rows)
    print(f"[n={n}] classes={len(classes)} all-switching-equiv={ok} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return rows


def main():
    kp = json.loads((RESULTS / "matrix_census_key_pairs.json").read_text())
    out = {}
    for ns in ("17", "18"):
        classes = kp[ns]["seidel_blind_classes"]
        out[ns] = analyze(int(ns), classes)
    dest = RESULTS / "seidel_mechanism.json"
    dest.write_text(json.dumps(out, indent=1))
    # summary
    for ns, rows in out.items():
        tot = len(rows)
        sw = sum(r["all_switching_equiv"] for r in rows)
        deg = sum(r["degseq_same"] for r in rows)
        print(f"n={ns}: {tot} classes; switching-equivalent {sw}/{tot}; "
              f"same degree sequence {deg}/{tot}")
    print("wrote", dest)


if __name__ == "__main__":
    main()
