"""Test on ALL graphs n<=7 (graph atlas): does Seidel-cospectral coincide
with generalized-cospectral (spec A + spec complement) beyond trees?

Also counts blind pairs per relation for A, S, gen, L on general graphs.
"""
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dgs_census import charpoly_int  # noqa: E402


def mats(G):
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, dtype=int).tolist()
    Ac = [[(1 - A[i][j]) if i != j else 0 for j in range(n)] for i in range(n)]
    d = [sum(r) for r in A]
    L = [[d[i] * int(i == j) - A[i][j] for j in range(n)] for i in range(n)]
    S = [[(1 - 2 * A[i][j]) if i != j else 0 for j in range(n)] for i in range(n)]
    return A, Ac, L, S


def main():
    for n in (6, 7):
        graphs = [G for G in nx.graph_atlas_g()
                  if G.number_of_nodes() == n and nx.is_connected(G)]
        fps = {"A": defaultdict(set), "L": defaultdict(set),
               "S": defaultdict(set), "GEN": defaultdict(set)}
        for i, G in enumerate(graphs):
            A, Ac, L, S = mats(G)
            fps["A"][charpoly_int(A)].add(i)
            fps["L"][charpoly_int(L)].add(i)
            fps["S"][charpoly_int(S)].add(i)
            fps["GEN"][(charpoly_int(A), charpoly_int(Ac))].add(i)
        counts = {k: sum(len(v) * (len(v) - 1) // 2 for v in d.values())
                  for k, d in fps.items()}
        # relation-level comparison (pairwise), not class-set equality
        def pair_set(d):
            out = set()
            for v in d.values():
                v = sorted(v)
                for a in range(len(v)):
                    for b in range(a + 1, len(v)):
                        out.add((v[a], v[b]))
            return out
        s_pairs = pair_set(fps["S"])
        g_pairs = pair_set(fps["GEN"])
        a_pairs = pair_set(fps["A"])
        print(f"n={n}: connected={len(graphs)} blind pairs {counts}")
        print(f"   GEN pairs={len(g_pairs)}, S pairs={len(s_pairs)}, "
              f"GEN<=S: {g_pairs <= s_pairs}, S<=GEN: {s_pairs <= g_pairs}")
        print(f"   A pairs={len(a_pairs)}, A<=S: {a_pairs <= s_pairs}, "
              f"GEN<=A: {g_pairs <= a_pairs}")


if __name__ == "__main__":
    main()
