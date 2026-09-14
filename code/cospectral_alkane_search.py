"""Find the smallest cospectral non-isomorphic alkane-skeleton pair.

Scans max-degree<=4 trees (carbon skeletons of acyclic molecules) from n=4
upward, groups by integral characteristic polynomial, and reports the first n
at which a cospectral pair (or class) appears, plus counts per n.
Also prints Wiener index / Hosoya Z / degree sequence for each member of the
first few cospectral classes, and draws the smallest pair.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dgs_census import charpoly_int  # noqa: E402
from make_figures import tree_layout  # noqa: E402

BASE = HERE.parent
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 14


def alkanes(n):
    for G in nx.nonisomorphic_trees(n):
        if max(d for _, d in G.degree()) <= 4:
            yield G


def hosoya(G, cp):
    return sum(abs(cp[i]) for i in range(0, len(cp), 2))


summary = []
first_hit = None
for n in range(4, NMAX + 1):
    bycp = defaultdict(list)
    tot = 0
    for G in alkanes(n):
        tot += 1
        cp = tuple(charpoly_int(nx.to_numpy_array(G).astype(int).tolist()))
        bycp[cp].append(nx.to_graph6_bytes(G, header=False).decode().strip())
    classes = [gs for gs in bycp.values() if len(gs) > 1]
    n_pairs = sum(len(gs) * (len(gs) - 1) // 2 for gs in classes)
    summary.append({"n": n, "n_alkane": tot,
                    "cospectral_classes": len(classes),
                    "cospectral_pairs": n_pairs})
    print(f"n={n}: {tot} alkane skeletons, {len(classes)} cospectral classes, "
          f"{n_pairs} pairs", flush=True)
    if classes and first_hit is None:
        first_hit = (n, classes)

out = {"summary": summary}
if first_hit:
    n, classes = first_hit
    detail = []
    for gs in classes:
        mem = []
        for g in gs:
            G = nx.from_graph6_bytes(g.encode())
            cp = tuple(charpoly_int(nx.to_numpy_array(G).astype(int).tolist()))
            mem.append({"g6": g,
                        "degseq": sorted(d for _, d in G.degree()),
                        "diameter": nx.diameter(G),
                        "wiener": nx.wiener_index(G),
                        "hosoya": hosoya(G, cp)})
        detail.append(mem)
    out["first_n"] = n
    out["classes"] = detail
    print(f"\nFIRST cospectral alkane class at n={n}:")
    for mem in detail:
        for m in mem:
            print(f"  {m['g6']}  deg={m['degseq']}  diam={m['diameter']}  "
                  f"W={m['wiener']}  Z={m['hosoya']}")
        print("  ---")
    # draw the first class
    gs = classes[0]
    k = len(gs)
    fig, axes = plt.subplots(1, k, figsize=(7 * k, 6))
    if k == 1:
        axes = [axes]
    for ax, g in zip(axes, gs):
        G = nx.from_graph6_bytes(g.encode())
        pos = tree_layout(G, nx.center(G)[0])
        deg = dict(G.degree())
        colors = [{1: "#aec7e8", 2: "#1f77b4", 3: "#ff7f0e", 4: "#d62728"}[deg[v]] for v in G]
        sizes = [{1: 80, 2: 120, 3: 220, 4: 300}[deg[v]] for v in G]
        nx.draw_networkx_edges(G, pos, ax=ax, width=1.4, edge_color="#555555")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                               edgecolors="black", linewidths=0.7)
        ax.set_title(g, fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Smallest cospectral alkane-skeleton pair (n={n})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fout = BASE / "figures" / f"cospectral_alkane_n{n}.png"
    fout.parent.mkdir(exist_ok=True)
    fig.savefig(fout, dpi=200, bbox_inches="tight")
    print(f"figure: {fout}")

(BASE / "results" / "cospectral_alkane_search.json").write_text(
    json.dumps(out, indent=1))
print("saved results/cospectral_alkane_search.json")
