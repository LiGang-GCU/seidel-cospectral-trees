"""Matrix-spectrum discrimination census on trees (Stage 1 of the
"replace-the-Laplacian" research line).

For every unlabeled tree on n vertices, compute the spectrum of each
candidate matrix M = f(A) and count blind pairs: pairs of non-isomorphic
trees with identical spectrum of M.

Two-phase protocol (same rigor as the n<=18 census paper):
  Phase 1 (numeric): eigenvalues via numpy, rounded to 6 decimals, group.
  Phase 2 (exact):   within each numeric collision group, verify pairwise
                     with exact characteristic polynomials (integer or
                     rational Faddeev-LeVerrier). Groups that split are
                     split; blind-pair counts use exact classes only.

Candidate matrices (all spectra exact-comparable):
  A        adjacency
  L        Laplacian D - A
  Q        signless Laplacian D + A        (= A_alpha at alpha=1/2, pencil t=1)
  S        Seidel J - I - 2A
  Aa25     D + 3A   (A_alpha, alpha=1/4, scaled x4)
  Aa67     A + 2D   (A_alpha, alpha=2/3, scaled x3; = degree pencil t=2)
  N        normalized adjacency D^{-1/2} A D^{-1/2} (rational charpoly via
           similar matrix D^{-1} A)
  DIST     distance matrix (on trees, resistance matrix == distance matrix,
           so DIST also covers the resistance spectrum)

Run:  python matrix_census.py <n> [n2 n3 ...]
Output: results/matrix_census_n{n}.json
"""
import json
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dgs_census import charpoly_int  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------- matrices
def build_mats(G):
    """Return dict name -> (numeric_matrix, kind) where kind in
    {'int', 'rat'} selects the exact charpoly routine."""
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, dtype=float)
    d = A.sum(1)
    D = np.diag(d)
    mats = {
        "A":    (A, "int"),
        "L":    (D - A, "int"),
        "Q":    (D + A, "int"),
        "S":    (np.ones((n, n)) - np.eye(n) - 2 * A, "int"),
        "Aa25": (D + 3.0 * A, "int"),
        "Aa67": (A + 2.0 * D, "int"),
        "N":    (A / np.sqrt(np.outer(d, d)), "rat"),  # symm D^{-1/2}AD^{-1/2}
        # NOTE: eigvalsh requires a SYMMETRIC matrix; an earlier version used
        # D^{-1}A (non-symmetric) for the numeric phase, which silently read
        # only one triangle and produced false "0 blind" results. Fixed.
        "DIST": (None, "dist"),
        "DL":   (None, "dl"),                        # distance Laplacian Tr-D
        "DQ":   (None, "dq"),                        # distance signless Tr+D
    }
    return mats


def dist_based(G, kind):
    Dm = dist_matrix(G)
    tr = Dm.sum(1)
    if kind == "dist":
        return Dm
    if kind == "dl":
        return np.diag(tr) - Dm
    if kind == "dq":
        return np.diag(tr) + Dm
    raise ValueError(kind)


def dist_matrix(G):
    n = G.number_of_nodes()
    adj = [[] for _ in range(n)]
    for u, v in G.edges():
        adj[u].append(v)
        adj[v].append(u)
    M = np.zeros((n, n), dtype=float)
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = [s]
        for u in q:
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        M[s] = dist
    return M


# ------------------------------------------------------- exact charpolys
def charpoly_rat(M):
    """Faddeev-LeVerrier over Fractions. M: 2-D list/np of Fraction-compatible."""
    n = len(M)
    A = [[Fraction(str(M[i][j])) if not isinstance(M[i][j], Fraction) else M[i][j]
          for j in range(n)] for i in range(n)]

    def matmul(X, Y):
        return [[sum(X[i][k] * Y[k][j] for k in range(n)) for j in range(n)]
                for i in range(n)]

    def trace(X):
        return sum(X[i][i] for i in range(n))

    coeffs = [Fraction(1)]
    B = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for k in range(1, n + 1):
        AB = matmul(A, B)
        ck = -trace(AB) / k
        coeffs.append(ck)
        B = [[AB[i][j] + (ck if i == j else Fraction(0)) for j in range(n)]
             for i in range(n)]
    return tuple(coeffs)


def exact_fp(name, Mnum, G):
    if name in ("DIST", "DL", "DQ"):
        return charpoly_int(dist_based(G, name.lower()).astype(int).tolist())
    if name == "N":
        n = G.number_of_nodes()
        A = nx.to_numpy_array(G, dtype=float)
        d = A.sum(1)
        Mrat = [[Fraction(1, int(d[i])) * int(A[i, j]) for j in range(n)]
                for i in range(n)]
        return charpoly_rat(Mrat)
    return charpoly_int(Mnum.astype(int).tolist())  # python ints: no overflow


# ---------------------------------------------------------------- census
def census_n(n, skip_exact=()):
    t0 = time.time()
    trees = list(nx.nonisomorphic_trees(n))
    t_enum = time.time() - t0
    N = len(trees)

    names = ["A", "L", "Q", "S", "Aa25", "Aa67", "N", "DIST", "DL", "DQ"]
    num_groups = {k: defaultdict(list) for k in names}
    CHUNK = 20000
    for c0 in range(0, N, CHUNK):
        chunk = trees[c0:c0 + CHUNK]
        m = len(chunk)
        stacks = {k: np.empty((m, n, n)) for k in names}
        for j, G in enumerate(chunk):
            mats = build_mats(G)
            for k in names:
                stacks[k][j] = (dist_based(G, k.lower()) if k in ("DIST", "DL", "DQ")
                                else mats[k][0])
        for k in names:
            evs = np.linalg.eigvalsh(stacks[k])
            for j in range(m):
                num_groups[k][tuple(np.round(evs[j], 6))].append(c0 + j)
        print(f"  [n={n}] {c0+m}/{N} numeric ({time.time()-t0:.0f}s)",
              flush=True)

    out = {"n": n, "n_trees": N, "enum_s": round(t_enum, 1), "matrices": {}}
    exact_classes = {}
    for k in names:
        classes = []
        for grp in num_groups[k].values():
            if len(grp) == 1 or k in skip_exact:
                classes.append(grp)
                continue
            # exact re-verification inside numeric group
            sub = defaultdict(list)
            for i in grp:
                fp = exact_fp(k, build_mats(trees[i])[k][0], trees[i])
                sub[fp].append(i)
            classes.extend(sub.values())
        exact_classes[k] = classes
        if k in skip_exact:
            print(f"  [n={n}] {k}: exact verification SKIPPED (already "
                  f"certified by T1 census)", flush=True)
        coll = [c for c in classes if len(c) > 1]
        pairs = sum(len(c) * (len(c) - 1) // 2 for c in coll)
        out["matrices"][k] = {
            "classes": len(classes),
            "collision_classes": len(coll),
            "blind_pairs": pairs,
            "max_class_size": max((len(c) for c in coll), default=1),
        }
        print(f"  [n={n}] {k:5s} classes={len(classes)} "
              f"collision_classes={len(coll)} blind_pairs={pairs}", flush=True)

    # intermediate save before complementarity (crash/timeout safety)
    out["runtime_to_exact_s"] = round(time.time() - t0, 1)
    dest = RESULTS / f"matrix_census_n{n}.json"
    dest.write_text(json.dumps(out, indent=1))

    # complementarity vs L. Numeric pre-filter per member, exact only on ties.
    def num_fp(k, G):
        M = dist_based(G, k.lower()) if k in ("DIST", "DL", "DQ") else build_mats(G)[k][0]
        return tuple(np.round(np.linalg.eigvalsh(M), 6))

    def still_blind_pairs(k, cls, other):
        """#pairs of cls (blind under k) still blind under `other`."""
        sig_num = defaultdict(list)
        for i in cls:
            sig_num[num_fp(other, trees[i])].append(i)
        still = 0
        for sub in sig_num.values():
            if len(sub) == 1:
                continue
            sig_ex = defaultdict(list)
            for i in sub:
                fp = exact_fp(other, build_mats(trees[i])[other][0], trees[i])
                sig_ex[fp].append(i)
            still += sum(len(s) * (len(s) - 1) // 2 for s in sig_ex.values())
        return still

    L_blind = [c for c in exact_classes["L"] if len(c) > 1]
    comp = {}
    for k in names:
        if k == "L":
            continue
        resolved = 0
        for c in L_blind:
            tot = len(c) * (len(c) - 1) // 2
            resolved += tot - still_blind_pairs("L", c, k)
        resolved_back = 0
        M_blind = [c for c in exact_classes[k] if len(c) > 1]
        for c in M_blind:
            tot = len(c) * (len(c) - 1) // 2
            resolved_back += tot - still_blind_pairs(k, c, "L")
        comp[k] = {"L_blind_resolved_by_M": resolved,
                   "M_blind_resolved_by_L": resolved_back}
        print(f"  [n={n}] comp {k:5s}: L-blind resolved {resolved}, "
              f"{k}-blind resolved by L {resolved_back}", flush=True)
    out["vs_L"] = comp
    out["runtime_s"] = round(time.time() - t0, 1)

    dest = RESULTS / f"matrix_census_n{n}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"[n={n}] done ({out['runtime_s']}s) -> {dest}", flush=True)
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    skip = set()
    for a in sys.argv[1:]:
        if a.startswith("--skip-exact="):
            skip = set(a.split("=", 1)[1].split(","))
    for n in [int(x) for x in args]:
        census_n(n, skip_exact=skip)
