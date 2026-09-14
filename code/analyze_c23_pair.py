"""Structural analysis of the C23 alkane-skeleton counterexample pair.

Parses the two graph6 strings, computes structural invariants, extracts the
branching core (iteratively pruned tree), draws both skeletons, and prints a
human-readable branching description.
"""
import json
from itertools import combinations

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\28945\Documents\kimi\workspace\matrix-spectrum-research"
REC = json.load(open(BASE + r"\results\chem_boundary_n23.json"))["classes"][0]
G6 = [m["g6"] for m in REC["members"]]


def prune_core(G):
    """Iteratively delete degree-1 vertices until none remain (branching core)."""
    H = G.copy()
    layers = []
    while True:
        leaves = [v for v in H if H.degree(v) <= 1]
        if not leaves:
            break
        layers.append(leaves)
        H.remove_nodes_from(leaves)
        if H.number_of_nodes() == 0:
            break
    return H, layers


def describe(G, name):
    deg = dict(G.degree())
    branch = sorted([v for v in G if deg[v] >= 3], key=lambda v: -deg[v])
    core, layers = prune_core(G)
    diam = nx.diameter(G)
    center = nx.center(G)
    wiener = nx.wiener_index(G)
    # distances between branch vertices
    bd = {}
    for u, v in combinations(branch, 2):
        bd[(u, v)] = nx.shortest_path_length(G, u, v)
    # degree-4 and degree-3 vertices positions
    d4 = [v for v in branch if deg[v] == 4]
    d3 = [v for v in branch if deg[v] == 3]
    print(f"\n===== {name} =====")
    print(f"  g6: {nx.to_graph6_bytes(G, header=False).decode().strip()}")
    print(f"  degseq tail: {sorted(deg.values())}")
    print(f"  diameter={diam}  radius={nx.radius(G)}  center={center}  Wiener={wiener}")
    print(f"  branch vertices: {len(branch)} (deg4: {len(d4)}, deg3: {len(d3)})")
    print(f"  pruning layers (leaf-stripping rounds): {len(layers)}, core size={core.number_of_nodes()}")
    if core.number_of_nodes() > 0:
        core_deg = sorted(core.degree(v) for v in core)
        print(f"  core degseq: {core_deg}, core g6: {nx.to_graph6_bytes(core, header=False).decode().strip()}")
    # pendant structure at each branch vertex: lengths of arms after removing branch vertices
    print("  branch-vertex local structure:")
    for v in branch:
        arms = []
        Gm = G.copy()
        nbrs = list(G.neighbors(v))
        Gm.remove_node(v)
        for u in nbrs:
            comp = nx.node_connected_component(Gm, u)
            # arm = path length until it ends or hits another branch vertex
            arm_len = 0
            cur, prev = u, v
            hit_branch = False
            while True:
                if cur in branch:
                    hit_branch = True
                    break
                nxt = [w for w in G.neighbors(cur) if w != prev]
                if not nxt:
                    arm_len += 1
                    break
                prev, cur = cur, nxt[0]
                arm_len += 1
            arms.append((arm_len, hit_branch))
        print(f"    v{v} deg={deg[v]}: arms={sorted(arms, reverse=True)}")
    return {"diam": diam, "wiener": wiener, "n_branch": len(branch)}


Gs = [nx.from_graph6_bytes(g.encode()) for g in G6]
infos = [describe(G, f"A{i+1}") for i, G in enumerate(Gs)]

# ---- draw both skeletons ----
import sys
sys.path.insert(0, BASE + r"\code")
from make_figures import tree_layout  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, G, ttl in zip(axes, Gs, ["A1 (deg tail ...3,3,3,4,4)", "A2 (deg tail ...3,3,3,3,3,3,4)"]):
    pos = tree_layout(G, nx.center(G)[0])
    deg = dict(G.degree())
    colors = [{1: "#aec7e8", 2: "#1f77b4", 3: "#ff7f0e", 4: "#d62728"}[deg[v]] for v in G]
    sizes = [{1: 60, 2: 90, 3: 180, 4: 260}[deg[v]] for v in G]
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.2, edge_color="#555555")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                           edgecolors="black", linewidths=0.6)
    ax.set_title(ttl, fontsize=12)
    ax.axis("off")
import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=c, label=f"deg {d}")
           for d, c in [(1, "#aec7e8"), (2, "#1f77b4"), (3, "#ff7f0e"), (4, "#d62728")]]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11)
fig.suptitle("C$_{23}$H$_{48}$ alkane skeletons: walk-equivalent but non-cospectral pair (n=23)",
             fontsize=14)
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
out = BASE + r"\figures\c23_pair.png"
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nfigure saved: {out}")

# save summary json
def hosoya(charpoly_strs):
    coeffs = [int(c) for c in charpoly_strs]
    # charpoly of forest: x^n - m1 x^{n-2} + m2 x^{n-4} - ... ; m_k = |coeff of x^{n-2k}|
    n = len(coeffs) - 1
    return sum(abs(coeffs[i]) for i in range(len(coeffs)) if i % 2 == 0)

hs = [hosoya(m["charpoly"]) for m in REC["members"]]
print(f"\nHosoya Z: A1={hs[0]}  A2={hs[1]}  (diff {hs[0]-hs[1]})")
summ = {"A1": {"g6": G6[0], "wiener": infos[0]["wiener"], "hosoya": hs[0]},
        "A2": {"g6": G6[1], "wiener": infos[1]["wiener"], "hosoya": hs[1]}}
json.dump({"note": "C23 pair: identical walks all orders, identical main spectrum; "
                   "Wiener 1060 vs 1036, Hosoya differs -> walk descriptors provably "
                   "insufficient for QSPR on C23H48",
           "members": summ, "walk_tuple_first24": REC["walk_tuple"]},
          open(BASE + r"\results\c23_pair_structure.json", "w"), indent=1)
print("done")
