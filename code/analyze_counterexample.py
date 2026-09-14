"""Locate the n=22 counterexample: the WALK class whose members have
different Seidel/adjacency spectra. Fast numeric pre-scan, exact verify.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dgs_census import charpoly_int  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def mats(g):
    T = nx.from_graph6_bytes(g.encode())
    n = T.number_of_nodes()
    A = nx.to_numpy_array(T)
    I = np.eye(n)
    return A, (np.ones((n, n)) - I - 2 * A), (np.ones((n, n)) - I - A)


def main():
    d = json.loads((RESULTS / "walk_census_big_n22.json").read_text())
    W = d["classes"]["WALK"]
    print(f"{len(W)} WALK collision classes", flush=True)

    suspects = []
    for ci, cls in enumerate(W):
        fps = set()
        for g in cls:
            A, S, C = mats(g)
            fps.add((tuple(np.round(np.linalg.eigvalsh(A), 8)),
                     tuple(np.round(np.linalg.eigvalsh(S), 8))))
        if len(fps) > 1:
            suspects.append((ci, cls))
        if (ci + 1) % 3000 == 0:
            print(f"  scanned {ci+1}/{len(W)}, suspects {len(suspects)}",
                  flush=True)
    print(f"suspect classes: {len(suspects)}")

    out = []
    for ci, cls in suspects:
        print(f"\n=== class #{ci}: {len(cls)} trees ===")
        rec = {"class_index": ci, "g6": cls, "members": []}
        for g in cls:
            A, S, C = mats(g)
            Ai = A.astype(int)
            adj_cp = charpoly_int(Ai.tolist())
            s_cp = charpoly_int(S.astype(int).tolist())
            c_cp = charpoly_int(C.astype(int).tolist())
            T = nx.from_graph6_bytes(g.encode())
            info = {"g6": g,
                    "degseq": sorted(d2 for _, d2 in T.degree()),
                    "diameter": nx.diameter(T),
                    "adj_ev": np.round(np.linalg.eigvalsh(A), 6).tolist(),
                    "adj_cp": [str(c) for c in adj_cp],
                    "seidel_cp": [str(c) for c in s_cp],
                    "comp_cp": [str(c) for c in c_cp]}
            rec["members"].append(info)
            print(f"  {g}")
            print(f"    degseq={info['degseq']} diam={info['diameter']}")
            print(f"    adj ev: {info['adj_ev']}")
            print(f"    adj cp: {adj_cp}")
        # exact walk-tuple equality re-verification
        Ws = []
        for g in cls:
            A, _, _ = mats(g)
            Ai = A.astype(int)
            n = len(Ai)
            v = [1] * n
            w = [n]
            for _ in range(n - 1):
                v = [sum(Ai[i][j] * v[j] for j in range(n)) for i in range(n)]
                w.append(sum(v))
            Ws.append(tuple(w))
        rec["walk_tuples_identical"] = len(set(Ws)) == 1
        print(f"  exact walk tuples identical: {rec['walk_tuples_identical']}")
        if rec["walk_tuples_identical"]:
            print(f"  W = {list(Ws[0])}")
        out.append(rec)

    dest = RESULTS / "n22_counterexample.json"
    dest.write_text(json.dumps(out, indent=1))
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
