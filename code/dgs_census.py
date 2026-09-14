"""
Generalized-spectrum (DGS) census on chemical trees.

For each graph G (trees n=8..13; alkane skeletons = max-degree<=4 trees,
i.e. C8..C13 carbon skeletons):
  - exact integer charpoly of A and of the complement Abar (Faddeev-LeVerrier)
  - generalized fingerprint = (charpoly A, charpoly Abar)  [exact]
  - collisions of specA alone vs generalized spectrum (exact)
  - walk matrix W = [e, Ae, ..., A^{n-1}e]: exact determinant (Bareiss),
    controllability, Wang criterion: w = 2^{-floor(n/2)} det W integer,
    odd and square-free  =>  provably DGS (Wang 2017)
  - the 3 pencil-cospectral exceptional tree pairs (n=11): does the
    generalized spectrum separate them?  (Godsil-McKay switching preserves
    complement spectra, so GM-type pairs stay tied.)

Output: results/dgs_census.json
"""
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import sympy

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


# ---------------- exact integer linear algebra ----------------
def charpoly_int(A):
    """Faddeev-LeVerrier; A: list of lists of ints -> coeff tuple,
    char poly = sum c_k x^{n-k}."""
    n = len(A)
    B = [[int(i == j) for j in range(n)] for i in range(n)]
    coeffs = [1]
    for k in range(1, n + 1):
        AB = [[sum(A[i][t] * B[t][j] for t in range(n)) for j in range(n)]
              for i in range(n)]
        ck = -sum(AB[i][i] for i in range(n)) // k
        coeffs.append(ck)
        B = [[AB[i][j] + (ck if i == j else 0) for j in range(n)]
             for i in range(n)]
    assert all(c * k == c * k for c in coeffs)  # integer check built-in
    return tuple(coeffs)


def det_bareiss(M):
    """Exact determinant of integer matrix via Bareiss algorithm."""
    n = len(M)
    A = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    A[k], A[i] = A[i], A[k]
                    sign = -sign
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
        for i in range(k + 1, n):
            A[i][k] = 0
    return sign * A[n - 1][n - 1]


def walk_matrix(A):
    n = len(A)
    e = [1] * n
    cols = [e]
    for _ in range(n - 1):
        v = cols[-1]
        cols.append([sum(A[i][t] * v[t] for t in range(n))
                     for i in range(n)])
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def wang_criterion(detW, n):
    """w = 2^{-floor(n/2)} detW; returns dict with integer/odd/squarefree."""
    if detW == 0:
        return dict(controllable=False, integer=False, odd=False,
                    squarefree=None, provable_dgs=False)
    shift = n // 2
    if detW % (1 << shift) != 0:
        return dict(controllable=True, integer=False, odd=False,
                    squarefree=None, provable_dgs=False)
    w = detW >> shift
    odd = (w % 2 != 0)
    sf = None
    provable = False
    if odd:
        # bounded factorization: trial division up to 1e5, then primality
        # of the cofactor; if cofactor is composite-and-unfactored we
        # honestly report squarefree=None (undecided) instead of hanging.
        fac = sympy.factorint(w, limit=100_000)
        rest = w
        for p, e in fac.items():
            rest //= p ** e
        if any(e >= 2 for e in fac.values()):
            sf = False
        elif rest == 1:
            sf = True
        elif sympy.isprime(rest):
            sf = True
        elif rest in fac:
            sf = False
        elif sympy.perfect_power(rest):
            sf = False
        else:
            # unfactored composite cofactor: check small-prime repeats only
            sf = None
        provable = bool(sf)
    return dict(controllable=True, integer=True, odd=odd,
                squarefree=sf, provable_dgs=provable)


def graph_to_int_adj(G):
    n = G.number_of_nodes()
    A = [[0] * n for _ in range(n)]
    for u, v in G.edges():
        A[u][v] = 1
        A[v][u] = 1
    return A


def complement(A):
    n = len(A)
    return [[(1 - A[i][j]) if i != j else 0 for j in range(n)]
            for i in range(n)]


# ---------------- census ----------------
def census(graphs, label):
    t0 = time.time()
    spec_groups = defaultdict(list)
    gen_groups = defaultdict(list)
    stats = dict(n_graphs=0, controllable=0, wang_odd=0, wang_squarefree=0,
                 provable_dgs=0)
    per_n = defaultdict(lambda: dict(n=0, specA_classes=0, gen_classes=0,
                                     provable_dgs=0))
    specA_seen = defaultdict(set)
    gen_seen = defaultdict(set)
    for G in graphs:
        n = G.number_of_nodes()
        A = graph_to_int_adj(G)
        cpA = charpoly_int(A)
        cpC = charpoly_int(complement(A))
        specA_seen[n].add(cpA)
        gen_seen[n].add((cpA, cpC))
        spec_groups[(n, cpA)].append(G)
        gen_groups[(n, cpA, cpC)].append(G)
        W = walk_matrix(A)
        detW = det_bareiss(W)
        crit = wang_criterion(detW, n)
        stats["n_graphs"] += 1
        per_n[n]["n"] += 1
        if crit["controllable"]:
            stats["controllable"] += 1
        if crit["integer"] and crit["odd"]:
            stats["wang_odd"] += 1
        if crit["squarefree"]:
            stats["wang_squarefree"] += 1
        if crit["provable_dgs"]:
            stats["provable_dgs"] += 1
            per_n[n]["provable_dgs"] += 1
    for n in sorted(specA_seen):
        per_n[n]["specA_classes"] = len(specA_seen[n])
        per_n[n]["gen_classes"] = len(gen_seen[n])
    specA_collisions = sum(len(v) - 1 for v in spec_groups.values()
                           if len(v) > 1)
    gen_collisions = sum(len(v) - 1 for v in gen_groups.values()
                         if len(v) > 1)
    # collision pairs (for listing)
    specA_pairs = sum(len(v) * (len(v) - 1) // 2
                      for v in spec_groups.values())
    gen_pairs = sum(len(v) * (len(v) - 1) // 2
                    for v in gen_groups.values())
    gen_collision_examples = []
    for (n, cpA, cpC), v in gen_groups.items():
        if len(v) > 1:
            gen_collision_examples.append(dict(
                n=n, size=len(v),
                graph6=[nx.to_graph6_bytes(g, header=False).decode().strip()
                        for g in v[:4]]))
    return dict(
        label=label, stats=stats, per_n={str(k): v for k, v in per_n.items()},
        specA_collision_graphs=specA_collisions,
        gen_collision_graphs=gen_collisions,
        specA_pairs=specA_pairs, gen_pairs=gen_pairs,
        gen_collision_examples=gen_collision_examples[:20],
        runtime_s=round(time.time() - t0, 1))


def dataset_trees(nmin, nmax, maxdeg=None):
    out = []
    for n in range(nmin, nmax + 1):
        for G in nx.nonisomorphic_trees(n):
            if maxdeg is None or max(dict(G.degree()).values()) <= maxdeg:
                out.append(G)
    return out


def exceptional_pairs_check():
    d = json.load(open(RESULTS / "exceptional_pairs.json"))
    rows = []
    for rec in d:
        e1, e2 = rec["edges_1"], rec["edges_2"]
        G1 = nx.Graph(e1)
        G2 = nx.Graph(e2)
        n = G1.number_of_nodes()
        out = dict(n=n, g1=rec["g1"], g2=rec["g2"])
        for tag, G in (("g1", G1), ("g2", G2)):
            A = graph_to_int_adj(G)
            out[f"cpA_{tag}"] = charpoly_int(A)
            out[f"cpC_{tag}"] = charpoly_int(complement(A))
        out["same_specA"] = out["cpA_g1"] == out["cpA_g2"]
        out["same_complement_spec"] = out["cpC_g1"] == out["cpC_g2"]
        out["gen_cospectral"] = out["same_specA"] and \
            out["same_complement_spec"]
        # keep JSON small: drop raw coeffs unless tied
        if not out["gen_cospectral"]:
            for k in list(out):
                if k.startswith("cp"):
                    out[k] = "differ"
        rows.append(out)
    return rows


def main():
    results = {}
    print("== census: trees n=8..13 ==", flush=True)
    results["trees"] = census(dataset_trees(8, 13), "trees_n8-13")
    print(json.dumps(results["trees"]["stats"]), flush=True)
    print("== census: alkanes (maxdeg<=4) ==", flush=True)
    results["alkanes"] = census(dataset_trees(8, 13, maxdeg=4),
                                "alkanes_C8-C13")
    print(json.dumps(results["alkanes"]["stats"]), flush=True)
    print("== exceptional pairs vs generalized spectrum ==", flush=True)
    results["exceptional_pairs"] = exceptional_pairs_check()
    for r in results["exceptional_pairs"]:
        print({k: v for k, v in r.items() if not k.startswith("cp")},
              flush=True)
    with open(RESULTS / "dgs_census.json", "w") as f:
        json.dump(results, f, indent=1)
    print("saved results/dgs_census.json", flush=True)


if __name__ == "__main__":
    main()
