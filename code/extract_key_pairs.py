"""Extract and cross-check the key collision pairs from the matrix census.

1. Seidel-blind tree pairs at n=17,18  vs  generalized-cospectral classes
   (gen_pairs_n{n}.json) — test whether the two relations COINCIDE exactly.
2. Distance-blind tree pairs at n=17,18 (first distance-cospectral trees).

Output: results/matrix_census_key_pairs.json
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


def g6(G):
    return nx.to_graph6_bytes(G, header=False).decode().strip()


def seidel_classes(n):
    groups = defaultdict(list)
    trees = list(nx.nonisomorphic_trees(n))
    g6s = [g6(G) for G in trees]
    for i, G in enumerate(trees):
        A = nx.to_numpy_array(G, dtype=float)
        S = np.ones((n, n)) - np.eye(n) - 2 * A
        ev = tuple(np.round(np.linalg.eigvalsh(S), 6))
        groups[ev].append(i)
    classes = []
    for grp in groups.values():
        if len(grp) == 1:
            continue
        sub = defaultdict(list)
        for i in grp:
            A = nx.to_numpy_array(trees[i], dtype=float)
            S = (np.ones((n, n)) - np.eye(n) - 2 * A).astype(int).tolist()
            sub[charpoly_int(S)].append(i)
        classes.extend(v for v in sub.values() if len(v) > 1)
    return [[g6s[i] for i in c] for c in classes], g6s


def dist_pairs(n, g6s):
    trees = [nx.from_graph6_bytes(s.encode()) for s in g6s]
    groups = defaultdict(list)
    for i, G in enumerate(trees):
        adj = [[] for _ in range(n)]
        for u, v in G.edges():
            adj[u].append(v)
            adj[v].append(u)
        M = np.zeros((n, n), dtype=int)
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
        groups[tuple(np.round(np.linalg.eigvalsh(M.astype(float)), 6))].append(i)
    out = []
    for grp in groups.values():
        if len(grp) == 1:
            continue
        sub = defaultdict(list)
        for i in grp:
            adjm = nx.to_numpy_array(trees[i], dtype=int)
            # distance matrix again as ints for charpoly
            adj = [[] for _ in range(n)]
            for u, v in trees[i].edges():
                adj[u].append(v)
                adj[v].append(u)
            M = np.zeros((n, n), dtype=int)
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
            sub[charpoly_int(M.tolist())].append(i)
        out.extend([g6s[i] for i in v] for v in sub.values() if len(v) > 1)
    return out


def main():
    result = {}
    for n in (17, 18):
        s_classes, g6s = seidel_classes(n)
        gen = json.load(open(RESULTS / f"gen_pairs_n{n}.json"))["classes"]
        # normalize to frozensets of frozensets
        s_set = {frozenset(c) for c in s_classes}
        g_set = {frozenset(c["graph6"]) for c in gen}
        result[str(n)] = {
            "seidel_blind_classes": sorted(sorted(c) for c in s_classes),
            "gen_cospectral_classes": sorted(sorted(c["graph6"]) for c in gen),
            "counts": [len(s_classes), len(gen)],
            "seidel_subset_of_gen": s_set <= g_set,
            "gen_subset_of_seidel": g_set <= s_set,
            "identical": s_set == g_set,
            "seidel_minus_gen": sorted(sorted(c) for c in s_set - g_set),
            "gen_minus_seidel": sorted(sorted(c) for c in g_set - s_set),
        }
        print(f"n={n}: Seidel blind classes={len(s_classes)}, "
              f"gen classes={len(gen)}, identical={s_set == g_set}", flush=True)
        if s_set != g_set:
            print("  seidel\\gen:", result[str(n)]["seidel_minus_gen"][:3])
            print("  gen\\seidel:", result[str(n)]["gen_minus_seidel"][:3])
        dp = dist_pairs(n, g6s)
        result[str(n)]["distance_blind_pairs"] = dp
        print(f"n={n}: distance-blind pairs={len(dp)}", flush=True)
        for p in dp:
            print("   DIST pair:", p)
    dest = RESULTS / "matrix_census_key_pairs.json"
    dest.write_text(json.dumps(result, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
