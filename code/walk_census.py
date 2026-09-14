"""Walk-count census on trees (Result 3, sharpened conjecture).

Sharpened claim: on trees, the walk-count sequence W_k = 1^T A^k 1
(k = 0..n-1) determines the adjacency spectrum. Equivalent check:
partition equality among
  WALK : exact tuple (W_0..W_{n-1})
  ASPEC: exact adjacency charpoly
  SSPEC: exact Seidel charpoly
  GEN  : (adjacency charpoly, complement charpoly)

All fingerprints exact (Python bigints / integer Faddeev-LeVerrier).
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dgs_census import charpoly_int  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def fps(G):
    n = G.number_of_nodes()
    A = np.array(nx.to_numpy_array(G).astype(int).tolist(), dtype=object)
    one = [1] * n
    v = list(one)
    W = [n]
    for _ in range(n - 1):
        v = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
        W.append(sum(v))
    Ai = A.astype(int)
    adj_cp = charpoly_int(Ai.tolist())
    comp = (np.ones((n, n)) - np.eye(n) - Ai).astype(int)
    comp_cp = charpoly_int(comp.tolist())
    seidel = (np.ones((n, n)) - np.eye(n) - 2 * Ai).astype(int)
    s_cp = charpoly_int(seidel.tolist())
    return {"WALK": tuple(W), "ASPEC": adj_cp, "SSPEC": s_cp,
            "GEN": (adj_cp, comp_cp)}


def partitions_equal(p1, p2):
    """Do two fingerprint dicts {idx: fp} induce the same partition?"""
    g1 = defaultdict(set)
    for i, f in p1.items():
        g1[f].add(i)
    g2 = defaultdict(set)
    for i, f in p2.items():
        g2[f].add(i)
    return {frozenset(s) for s in g1.values()} == {frozenset(s) for s in g2.values()}


def main():
    import sys
    ns = [int(x) for x in sys.argv[1:]] or list(range(8, 19))
    kinds = ["WALK", "ASPEC", "SSPEC", "GEN"]
    out = {}
    for n in ns:
        t0 = time.time()
        trees = list(nx.nonisomorphic_trees(n))
        fp = {k: {} for k in kinds}
        for i, G in enumerate(trees):
            f = fps(G)
            for k in kinds:
                fp[k][i] = f[k]
            if (i + 1) % 20000 == 0:
                print(f"  n={n} {i+1}/{len(trees)} ({time.time()-t0:.0f}s)",
                      flush=True)
        row = {"n": n, "n_trees": len(trees), "classes": {}}
        for k in kinds:
            groups = defaultdict(list)
            for i, f in fp[k].items():
                groups[f].append(i)
            coll = [v for v in groups.values() if len(v) > 1]
            row["classes"][k] = {
                "n_classes": len(groups),
                "collision_classes": len(coll),
                "blind_pairs": sum(len(v) * (len(v) - 1) // 2 for v in coll)}
        # pairwise partition equality
        pe = {}
        for i, k1 in enumerate(kinds):
            for k2 in kinds[i + 1:]:
                pe[f"{k1}=={k2}"] = partitions_equal(fp[k1], fp[k2])
        row["partition_equality"] = pe
        out[str(n)] = row
        print(f"[n={n}] " + " ".join(
            f"{k}:{row['classes'][k]['collision_classes']}" for k in kinds)
            + "  " + json.dumps(pe) + f"  ({time.time()-t0:.0f}s)", flush=True)
        dest = RESULTS / f"walk_census_n{ns[0]}-{ns[-1]}.json"
        dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
