"""EXP1: GNN expressivity separation on the walk-equivalent tree pair.

The pair T1, T2 (n=22) is walk-equivalent for ALL k (main spectral
measures identical) but not cospectral (m(T,3): 796 vs 788).

Part A: forward-pass separability of standard GNNs (GCN/GIN/SAGE,
constant node features, random init, 20 seeds, L=1..6 layers,
sum/mean readout). Expected: distinguishable (degree sequences differ).

Part B: spectral-feature models.
  - global walk counts W_0..W_K  -> identical inputs (exact check)
  - main spectral measure        -> identical
  - full spectrum / closed walks -> differ
  - heat-kernel trace (full spec) vs walk generating function (main)

Part C: learnable task. Train GIN vs WalkMLP to predict m(T,3) on
trees n=8..16; evaluate on the pair: WalkMLP provably cannot output
two different values for the pair; GIN can.

Output: /media/jiuzhang/sda/msr/exp1_gnn/results.json
"""
import json
import itertools
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, SAGEConv

OUT = Path("/media/jiuzhang/sda/msr/exp1_gnn")
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

G6_1 = "Up_K?E??I??@?@?A??G?O??C?G???I?????@???G"
G6_2 = "Up_KA?@?GA?@?OO???G?@??O??G?AA?????@???O"
M_TRUE = (796, 788)

results = {"device": DEV, "time": time.strftime("%F %T")}

# ----------------------------------------------------------------------
def walks_exact(g6, kmax):
    T = nx.from_graph6_bytes(g6.encode())
    al = [sorted(T.neighbors(v)) for v in range(T.number_of_nodes())]
    n = len(al)
    v = [1] * n
    out = []
    for _ in range(kmax + 1):
        out.append(sum(v))
        nv = [0] * n
        for i in range(n):
            for j in al[i]:
                nv[j] += v[i]
        v = nv
    return out


def main_measure(g6):
    T = nx.from_graph6_bytes(g6.encode())
    A = nx.to_numpy_array(T)
    w, V = np.linalg.eigh(A)
    comp = V.T @ np.ones(len(A))
    pairs = {}
    for val, c in zip(w, comp):
        g = c * c
        if g > 1e-9:
            pairs[round(val, 9)] = pairs.get(round(val, 9), 0.0) + g
    return sorted(pairs.items())


def full_spectrum(g6):
    T = nx.from_graph6_bytes(g6.encode())
    return np.sort(np.linalg.eigvalsh(nx.to_numpy_array(T)))


# ---------------- Part B (cheap, do first) -----------------------------
K = 40
W1 = walks_exact(G6_1, K)
W2 = walks_exact(G6_2, K)
walk_identical = (W1 == W2)

mm1, mm2 = main_measure(G6_1), main_measure(G6_2)
main_identical = len(mm1) == len(mm2) and all(
    abs(a - c) < 1e-6 and abs(b - d) < 1e-6
    for (a, b), (c, d) in zip(mm1, mm2))

sp1, sp2 = full_spectrum(G6_1), full_spectrum(G6_2)
spec_l2 = float(np.linalg.norm(sp1 - sp2))

# closed-walk counts (full spectrum) vs walk generating function (main)
ts = [0.5, 1.0, 2.0]
heat1 = [float(np.sum(np.exp(-t * sp1))) for t in ts]
heat2 = [float(np.sum(np.exp(-t * sp2))) for t in ts]
xs = [0.05, 0.1, 0.2]  # below 1/rho ~ 1/2.66
wgf1 = [sum(w * x**k for k, w in enumerate(W1)) for x in xs]
wgf2 = [sum(w * x**k for k, w in enumerate(W2)) for x in xs]

results["partB"] = {
    "walk_counts_k0..40_identical": walk_identical,
    "main_measure_identical": main_identical,
    "full_spectrum_l2_distance": spec_l2,
    "heat_kernel_trace_t": ts,
    "heat_trace_T1": heat1,
    "heat_trace_T2": heat2,
    "walk_gen_fn_x": xs,
    "walk_gen_fn_T1": wgf1,
    "walk_gen_fn_T2": wgf2,
    "walk_gen_fn_identical": all(abs(a - b) < 1e-12
                                 for a, b in zip(wgf1, wgf2)),
}

# ---------------- Part A -----------------------------------------------
def graph_to_data(g6):
    T = nx.from_graph6_bytes(g6.encode())
    edges = list(T.edges())
    ei = torch.tensor(edges + [(v, u) for u, v in edges],
                      dtype=torch.long).t().contiguous()
    x = torch.ones(T.number_of_nodes(), 1)
    return Data(x=x, edge_index=ei)


D1, D2 = graph_to_data(G6_1), graph_to_data(G6_2)


def build(model, L, hid=64):
    layers = nn.ModuleList()
    for i in range(L):
        if model == "gcn":
            layers.append(GCNConv(1 if i == 0 else hid, hid))
        elif model == "sage":
            layers.append(SAGEConv(1 if i == 0 else hid, hid))
        elif model == "gin":
            mlp = nn.Sequential(nn.Linear(1 if i == 0 else hid, hid),
                                nn.ReLU(), nn.Linear(hid, hid))
            layers.append(GINConv(mlp))
    return layers


def embed(layers, model, data, readout):
    x, ei = data.x.to(DEV), data.edge_index.to(DEV)
    for c in layers:
        x = F.relu(c(x, ei))
    if readout == "sum":
        return x.sum(0)
    return x.mean(0)


partA = {}
for model in ("gcn", "gin", "sage"):
    for readout in ("sum", "mean"):
        for L in (1, 2, 3, 4, 5, 6):
            dists = []
            for seed in range(20):
                torch.manual_seed(seed)
                layers = build(model, L).to(DEV)
                with torch.no_grad():
                    e1 = embed(layers, model, D1, readout)
                    e2 = embed(layers, model, D2, readout)
                d = float((e1 - e2).norm() / (e1.norm() + e2.norm() + 1e-12))
                dists.append(d)
            partA[f"{model}/{readout}/L{L}"] = {
                "mean": float(np.mean(dists)), "std": float(np.std(dists)),
                "min": float(np.min(dists))}
results["partA"] = partA

# ---------------- Part C -----------------------------------------------
def count_m3(T):
    """Number of 3-matchings, direct combinatorial count."""
    edges = list(T.edges())
    cnt = 0
    for a, b, c in itertools.combinations(edges, 3):
        s = {a[0], a[1], b[0], b[1], c[0], c[1]}
        if len(s) == 6:
            cnt += 1
    return cnt


def tree_dataset(nmin=8, nmax=16, cap_per_n=4000, seed=0, kmax=15):
    rng = np.random.RandomState(seed)
    data = []
    for n in range(nmin, nmax + 1):
        trees = list(nx.nonisomorphic_trees(n))
        if len(trees) > cap_per_n:
            idx = rng.choice(len(trees), cap_per_n, replace=False)
            trees = [trees[i] for i in idx]
        for T in trees:
            W = walks_exact(nx.to_graph6_bytes(T, header=False)
                            .decode().strip(), kmax)
            edges = list(T.edges())
            ei = torch.tensor(edges + [(v, u) for u, v in edges],
                              dtype=torch.long).t().contiguous()
            d = Data(x=torch.ones(n, 1), edge_index=ei,
                     y=torch.tensor([float(count_m3(T))]))
            d.walk_feat = torch.tensor(
                [np.log1p(w) for w in W] + [float(n)], dtype=torch.float)
            data.append(d)
    return data


class WalkMLP(nn.Module):
    def __init__(self, din, hid=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 1))

    def forward(self, feats):
        return self.net(feats).squeeze(-1)


class GINReg(nn.Module):
    def __init__(self, L=5, hid=64):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(L):
            mlp = nn.Sequential(nn.Linear(1 if i == 0 else hid, hid),
                                nn.ReLU(), nn.Linear(hid, hid))
            self.layers.append(GINConv(mlp))
        self.head = nn.Linear(hid, 1)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for c in self.layers:
            x = F.relu(c(x, ei))
        # sum readout over nodes (single graphs, use batch ptr)
        from torch_geometric.utils import scatter
        out = scatter(x, data.batch, dim=0, reduce="sum")
        return self.head(out).squeeze(-1)


def train_part_c():
    print("building dataset ...", flush=True)
    data = tree_dataset()
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(data))
    ntr = int(0.9 * len(data))
    train = [data[i] for i in idx[:ntr]]
    test = [data[i] for i in idx[ntr:]]
    print(f"dataset {len(data)} trees, train {len(train)}, test {len(test)}",
          flush=True)

    pair = []
    for g6, m in ((G6_1, M_TRUE[0]), (G6_2, M_TRUE[1])):
        T = nx.from_graph6_bytes(g6.encode())
        W = walks_exact(g6, 21)
        # pad walk features to dataset dim (n-1 up to 15 -> k=0..15 + n = 17 dims)
        d = graph_to_data(g6)
        d.y = torch.tensor([float(m)])
        wf = [np.log1p(w) for w in W[:16]] + [22.0]
        d.walk_feat = torch.tensor(wf, dtype=torch.float)
        pair.append(d)

    loader_tr = DataLoader(train, batch_size=256, shuffle=True)
    loader_te = DataLoader(test, batch_size=512)

    # --- WalkMLP ---
    Xtr = torch.stack([d.walk_feat for d in train])
    ytr = torch.cat([d.y for d in train])
    Xte = torch.stack([d.walk_feat for d in test])
    yte = torch.cat([d.y for d in test])
    Xp = torch.stack([d.walk_feat for d in pair])

    mlp = WalkMLP(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for ep in range(30):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 1024):
            j = perm[i:i + 1024]
            pred = mlp(Xtr[j].to(DEV))
            loss = F.l1_loss(pred, ytr[j].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mae_mlp = float(F.l1_loss(mlp(Xte.to(DEV)), yte.to(DEV)))
        pair_mlp = mlp(Xp.to(DEV)).cpu().tolist()

    # --- GIN ---
    gin = GINReg().to(DEV)
    opt = torch.optim.Adam(gin.parameters(), lr=1e-3)
    for ep in range(10):
        for batch in loader_tr:
            batch = batch.to(DEV)
            loss = F.l1_loss(gin(batch), batch.y)
            opt.zero_grad(); loss.backward(); opt.step()
    gin.eval()
    errs = []
    with torch.no_grad():
        for batch in loader_te:
            batch = batch.to(DEV)
            errs.append((gin(batch) - batch.y).abs().cpu())
        mae_gin = float(torch.cat(errs).mean())
        pair_gin = []
        for d in pair:
            dd = Data(x=d.x, edge_index=d.edge_index,
                      batch=torch.zeros(d.x.size(0), dtype=torch.long)).to(DEV)
            pair_gin.append(float(gin(dd)))
    return {
        "n_train": len(train), "n_test": len(test),
        "walkmlp_test_mae": mae_mlp,
        "gin_test_mae": mae_gin,
        "pair_true_m3": list(M_TRUE),
        "pair_pred_walkmlp": pair_mlp,
        "pair_pred_gin": pair_gin,
        "walkmlp_pair_outputs_identical": abs(pair_mlp[0] - pair_mlp[1]) < 1e-9,
        "gin_pair_outputs_differ": abs(pair_gin[0] - pair_gin[1]) > 1e-6,
    }


results["partC"] = train_part_c()

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=1)
print(json.dumps(results["partB"], indent=1))
print("partA sample:", {k: v for k, v in
                        list(results["partA"].items())[:4]})
print(json.dumps(results["partC"], indent=1))
print("DONE ->", OUT / "results.json")
