"""Matrix-spectrum boundary search on trees (Stage A').

Question: smallest n with a blind pair (identical spectrum, non-isomorphic)
for a given matrix M in {N, DIST, DL, DQ}.
Known: DL/DQ have NO blind pairs for n<=18 (all trees); DIST first pair n=17.

Two-pass streaming protocol (memory-safe for n=22, ~5.6M trees):
  pass 1: enumerate trees, batch eigvalsh of M, fingerprint =
          int32(round(ev*1e6)) bytes; count per fingerprint.
  pass 2: re-enumerate; keep graph6 only for colliding fingerprints.
  exact:  within each numeric collision group, verify pairwise with the
          exact charpoly (integer FL for DIST/DL/DQ; rational FL for N).

Usage: python n_boundary_search.py <n> --matrix DL [--outdir PATH]
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matrix_census import charpoly_rat, dist_based  # noqa: E402
from dgs_census import charpoly_int  # noqa: E402

CHUNK = 20000


def norm_adj(G):
    A = nx.to_numpy_array(G, dtype=float)
    d = A.sum(1)
    return A / np.sqrt(np.outer(d, d))


def num_mat(G, kind):
    if kind == "N":
        return norm_adj(G)
    return dist_based(G, kind.lower())


def exact_fp(G, kind):
    if kind == "N":
        n = G.number_of_nodes()
        A = nx.to_numpy_array(G, dtype=float)
        d = A.sum(1)
        Mrat = [[Fraction(1, int(d[i])) * int(A[i, j]) for j in range(n)]
                for i in range(n)]
        return charpoly_rat(Mrat)
    return charpoly_int(dist_based(G, kind.lower()).astype(int).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--matrix", default="DL", choices=["N", "DIST", "DL", "DQ"])
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent.parent / "results"))
    a = ap.parse_args()
    n, kind = a.n, a.matrix
    t0 = time.time()

    # ---- pass 1: fingerprint counts ----
    counts = defaultdict(int)
    total = 0
    buf = []
    for G in nx.nonisomorphic_trees(n):
        buf.append(num_mat(G, kind))
        total += 1
        if len(buf) >= CHUNK:
            for ev in np.linalg.eigvalsh(np.array(buf)):
                counts[np.round(ev * 1e6).astype(np.int32).tobytes()] += 1
            buf.clear()
        if total % 200000 == 0:
            print(f"[n={n}/{kind}] pass1 {total} ({time.time()-t0:.0f}s)", flush=True)
    if buf:
        for ev in np.linalg.eigvalsh(np.array(buf)):
            counts[np.round(ev * 1e6).astype(np.int32).tobytes()] += 1
    buf = None
    hot = {k for k, c in counts.items() if c > 1}
    print(f"[n={n}/{kind}] pass1 done: {total} trees, {len(counts)} distinct fps, "
          f"{len(hot)} colliding fps ({time.time()-t0:.0f}s)", flush=True)
    del counts

    # ---- pass 2: collect graph6 of colliding trees ----
    groups = defaultdict(list)
    buf = []
    for G in nx.nonisomorphic_trees(n):
        buf.append(G)
        if len(buf) >= CHUNK:
            evs = np.linalg.eigvalsh(np.array([num_mat(x, kind) for x in buf]))
            for ev, G2 in zip(evs, buf):
                key = np.round(ev * 1e6).astype(np.int32).tobytes()
                if key in hot:
                    groups[key].append(
                        nx.to_graph6_bytes(G2, header=False).decode().strip())
            buf.clear()
    if buf:
        evs = np.linalg.eigvalsh(np.array([num_mat(x, kind) for x in buf]))
        for ev, G2 in zip(evs, buf):
            key = np.round(ev * 1e6).astype(np.int32).tobytes()
            if key in hot:
                groups[key].append(nx.to_graph6_bytes(G2, header=False).decode().strip())
    print(f"[n={n}/{kind}] pass2 done: {sum(len(v) for v in groups.values())} trees "
          f"in {len(groups)} numeric collision groups ({time.time()-t0:.0f}s)",
          flush=True)

    # ---- exact verification ----
    exact_pairs = []
    for key, g6s in groups.items():
        sub = defaultdict(list)
        for g in g6s:
            sub[exact_fp(nx.from_graph6_bytes(g.encode()), kind)].append(g)
        for v in sub.values():
            if len(v) > 1:
                exact_pairs.append(sorted(v))
    exact_pairs.sort()

    out = {"n": n, "matrix": kind, "n_trees": total,
           "numeric_collision_groups": len(groups),
           "exact_blind_pairs": sum(len(p) * (len(p) - 1) // 2 for p in exact_pairs),
           "collision_classes": exact_pairs,
           "runtime_s": round(time.time() - t0, 1)}
    dest = Path(a.outdir) / f"boundary_{kind}_n{n}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"[n={n}/{kind}] EXACT blind pairs: {out['exact_blind_pairs']} "
          f"({out['runtime_s']}s) -> {dest}", flush=True)


if __name__ == "__main__":
    main()

