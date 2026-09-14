"""Paper figures (Phase 2): F2 growth curve, F3 moment divergence,
F1 counterexample tree pair drawings. Outputs to figures/."""
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

setup_plot()

# ---------------------------------------------------------------- F2
NS = list(range(8, 23))
CLASSES = {  # collision classes (SSPEC==GEN); WALK diverges at n=22
    8: 0, 9: 0, 10: 0, 11: 0, 12: 1, 13: 2, 14: 6, 15: 17, 16: 49,
    17: 129, 18: 334, 19: 847, 20: 2209, 21: 5788, 22: 15235}
WALK22 = 15236

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ys = [CLASSES[n] for n in NS]
ax.semilogy(NS, [max(y, 0.5) for y in ys], "o-", color="#1f77b4",
            label="SSPEC = GEN collision classes")
ax.semilogy([22], [WALK22], "s", color="#d62728", markersize=9,
            label="WALK classes at $n=22$ (15,236)")
ax.annotate("walk-determination\nbreaks here", xy=(22, WALK22),
            xytext=(17.5, 30000), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#d62728"))
ax.set_xlabel("$n$ (number of vertices)")
ax.set_ylabel("collision classes on trees (log)")
ax.set_xticks(NS)
ax.legend(loc="lower right", fontsize=9)
fig.savefig(FIG / "fig_growth.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig_growth.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- F3
# Main / non-main spectrum comparison of the counterexample pair.
# Verified exactly: walk counts agree for ALL k (pure-integer check to
# k=200 + order-44 recurrence argument), so the main spectral measures
# coincide; the whole difference sits in the non-main eigenvalues.
sys.path.insert(0, str(ROOT / "code"))
g1 = "Up_K?E??I??@?@?A??G?O??C?G???I?????@???G"
g2 = "Up_KA?@?GA?@?OO???G?@??O??G?AA?????@???O"


def spectrum_split(g):
    """Split the spectrum into main / non-main eigenvalues.

    An eigenvalue is main iff the all-ones vector has a nonzero
    component in its EIGENSPACE, i.e. ||P_lambda 1||^2 > 0. Testing a
    single LAPACK eigenvector against 1 is basis-dependent and wrong at
    repeated eigenvalues: inside a multi-dimensional eigenspace the
    returned basis is arbitrary, so an individual vector can be
    orthogonal to 1 while the eigenspace is not. So cluster equal
    eigenvalues into eigenspaces first, then sum squared components
    over each cluster -- a basis-invariant quantity.
    """
    T = nx.from_graph6_bytes(g.encode())
    A = nx.to_numpy_array(T)
    w, V = np.linalg.eigh(A)
    comp = V.T @ np.ones(len(A))
    mult, weight = {}, {}
    for val, c in zip(w, comp):
        key = round(val, 6)
        mult[key] = mult.get(key, 0) + 1
        weight[key] = weight.get(key, 0.0) + float(c) ** 2
    main, nonmain = {}, {}
    for key, m in mult.items():
        tgt = main if weight[key] > 1e-16 else nonmain
        tgt[key] = m
    return main, nonmain


main1, nonmain1 = spectrum_split(g1)
main2, nonmain2 = spectrum_split(g2)
assert set(main1) == set(main2), "main eigenvalue sets differ!"

fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True)
for ax, (main, nonmain), name in zip(
        axes, ((main1, nonmain1), (main2, nonmain2)), ("$T_1$", "$T_2$")):
    for ev, mult in main.items():
        ax.stem([ev], [mult], linefmt="C0-", markerfmt="C0o", basefmt=" ")
    for ev, mult in nonmain.items():
        ax.stem([ev], [mult], linefmt="C3--", markerfmt="C3s", basefmt=" ")
    ax.set_ylabel(name, rotation=0, labelpad=14, va="center")
    ax.set_yticks([])
    ax.spines[["left", "top", "right"]].set_visible(False)
h1 = axes[0].stem([1e9], [0], linefmt="C0-", markerfmt="C0o", basefmt=" ",
                  label="main eigenvalue (shared)")
h2 = axes[0].stem([1e9], [0], linefmt="C3--", markerfmt="C3s", basefmt=" ",
                  label="non-main eigenvalue (differs)")
axes[0].set_xlim(-3.2, 3.2)
fig.legend(loc="upper center", fontsize=9, ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.04))
fig.subplots_adjust(hspace=0.35)
axes[1].set_xlabel("adjacency eigenvalue")
fig.savefig(FIG / "fig_divergence.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig_divergence.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- F1
def tree_layout(T, root):
    """Simple layered BFS layout, children spread by angular order."""
    import math
    depth = {root: 0}
    order = {root: 0.0}
    layers = {0: [root]}
    parent = {root: None}
    q = [root]
    while q:
        u = q.pop(0)
        ch = [v for v in T.neighbors(u) if v not in depth]
        for v in ch:
            depth[v] = depth[u] + 1
            parent[v] = u
            q.append(v)
            layers.setdefault(depth[v], []).append(v)
    pos = {}
    for d, nodes in layers.items():
        # order by parent's x to reduce crossings
        nodes.sort(key=lambda v: order.get(parent[v], 0.0))
        m = len(nodes)
        for i, v in enumerate(nodes):
            x = i - (m - 1) / 2
            order[v] = x
            pos[v] = (x, -d)
    return pos


def draw_tree(T, ax, title):
    # root at a center vertex for balance
    ctr = nx.center(T)[0]
    pos = tree_layout(T, ctr)
    deg = dict(T.degree())
    nx.draw_networkx_edges(T, pos, ax=ax, edge_color="#555555", width=1.2)
    nx.draw_networkx_nodes(T, pos, ax=ax, node_size=180,
                           node_color=[deg[v] for v in T.nodes()],
                           cmap=plt.cm.viridis, vmin=1, vmax=5)
    for v, (x, y) in pos.items():
        if deg[v] >= 4:
            ax.annotate(str(deg[v]), (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, g, name in zip(axes, (g1, g2), ("$T_1$", "$T_2$")):
    T = nx.from_graph6_bytes(g.encode())
    draw_tree(T, ax, name)
fig.savefig(FIG / "fig_pair.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig_pair.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print("figures written to", FIG)
for f in sorted(FIG.iterdir()):
    print(" ", f.name)
