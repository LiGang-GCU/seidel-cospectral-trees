"""Approximate substitutive naming of small alkane skeletons (n<=12).

Strategy: choose the longest path as parent chain (ties: max number of
substituents, then lowest locant set). Name each branch recursively as an
alkyl group for small groups (methyl/ethyl/propyl/isopropyl/butyl/isobutyl/
sec-butyl/tert-butyl/...). Output a readable semi-systematic name.
"""
import json
import sys
from pathlib import Path

import networkx as nx

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def longest_paths(G):
    """All diameter paths (as node lists), oriented both ways."""
    d = nx.diameter(G)
    paths = []
    for s in G:
        for t in G:
            if s < t and nx.shortest_path_length(G, s, t) == d:
                paths.append(nx.shortest_path(G, s, t))
    return paths


def branch_tree(G, attach, chain_set):
    """Subtree hanging off `attach` away from the chain; return (root, H)."""
    nbrs = [u for u in G.neighbors(attach) if u not in chain_set]
    H = nx.Graph()
    for u in nbrs:
        comp = nx.node_connected_component(
            nx.induced_subgraph(G, [v for v in G if v not in chain_set]), u)
        H.add_nodes_from(comp)
    H.add_edges_from((u, v) for u, v in G.subgraph(H.nodes()).edges())
    return H


# canonical small alkyl group names by (size, sorted degseq of rooted tree)
def alkyl_name(H, root):
    """Name the group H attached via `root` (root becomes the attachment C)."""
    n = H.number_of_nodes()
    if n == 1:
        return "methyl"
    if n == 2:
        return "ethyl"
    degs = sorted(dict(H.degree()).values())
    if n == 3:
        return "isopropyl" if H.degree(root) == 2 else "propyl"
    if n == 4:
        if max(degs) == 3:
            return "tert-butyl" if H.degree(root) == 3 else "isobutyl"
        return "sec-butyl" if H.degree(root) == 2 else "butyl"
    # fallback: nested description
    subs = []
    for u in H.neighbors(root):
        comp = nx.node_connected_component(
            nx.induced_subgraph(H, [v for v in H if v != root]), u)
        sub = nx.induced_subgraph(H, comp)
        subs.append(alkyl_name(sub, u))
    subs.sort()
    return "(" + ",".join(subs) + ")methyl" if H.degree(root) > 1 else \
        subs[0].replace("methyl", "methyl", 1) if False else "(" + subs[0] + ")"


def name_alkane(G):
    best = None
    for path in longest_paths(G):
        for cand in (path, path[::-1]):
            chain_set = set(cand)
            subs = []  # (locant, name)
            for i, v in enumerate(cand):
                for u in G.neighbors(v):
                    if u in chain_set:
                        continue
                    comp = nx.node_connected_component(
                        nx.induced_subgraph(G, [x for x in G if x not in chain_set]), u)
                    H = nx.induced_subgraph(G, comp)
                    subs.append((i + 1, alkyl_name(H, u)))
            n_sub = len(subs)
            locs = tuple(sorted(l for l, _ in subs))
            key = (-n_sub, locs)
            if best is None or key < best[0]:
                best = (key, cand, subs)
    _, chain, subs = best
    parent = {3: "propane", 4: "butane", 5: "pentane", 6: "hexane",
              7: "heptane", 8: "octane", 9: "nonane", 10: "decane",
              11: "undecane", 12: "dodecane"}[len(chain)]
    # group substituents
    from collections import defaultdict
    byname = defaultdict(list)
    for loc, nm in subs:
        byname[nm].append(loc)
    parts = []
    multi = {1: "", 2: "di", 3: "tri", 4: "tetra"}
    for nm in sorted(byname):
        locs = sorted(byname[nm])
        parts.append(f"{'-'.join(map(str, locs))}-{multi[len(locs)]}{nm}")
    return ",".join(parts) + ("-" if parts else "") + parent, len(chain), subs


data = json.load(open(BASE / "results" / "cospectral_alkane_search.json"))
print(f"first cospectral n = {data['first_n']}\n")
for ci, mem in enumerate(data["classes"]):
    print(f"class {ci+1}:")
    for m in mem:
        G = nx.from_graph6_bytes(m["g6"].encode())
        nm, clen, subs = name_alkane(G)
        print(f"  {m['g6']:12s} W={m['wiener']:4.0f}  ->  {nm}")
    print()
