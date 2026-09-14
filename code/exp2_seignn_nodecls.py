"""EXP2: Seidel-channel GNN (SeiGCN) for node classification.

Theory hook: S = J - I - 2A, so Sx = (global sum) - x - 2Ax: a global
channel SIGNED by adjacency. Our census (general graphs n=6: Seidel
spectrum has 0 blind pairs vs adjacency spectrum's 450) shows this
channel carries information beyond A. The sign is what distinguishes
SeiGCN from a virtual-node (unsigned global channel) -- clean ablation.

Models (2 layers, hidden 64, dropout 0.5):
  MLP, GCN, GCN+VN (unsigned global channel), SeiGCN (signed, ours)
Matched parameter counts for GCN+VN vs SeiGCN (two matrices + bias).

Datasets:
  homophilic : Cora, CiteSeer, Pubmed (Planetoid public split, 10 seeds)
  heterophilic: Texas, Cornell, Wisconsin, Actor, Chameleon, Squirrel
               (geom-gcn 10 splits x 2 seeds)

Output: /media/jiuzhang/sda/msr/exp2_nodecls/results.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import (Actor, Planetoid, WebKB,
                                      WikipediaNetwork)

ROOT = "/media/jiuzhang/sda/msr/exp2_nodecls"
Path(ROOT).mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------ models
def a_hat(x, edge_index, n, self_loops=True):
    ei = edge_index
    if self_loops:
        sl = torch.arange(n, device=ei.device)
        ei = torch.cat([ei, sl.unsqueeze(0).repeat(2, 1)], dim=1)
    src, dst = ei[0], ei[1]
    deg = torch.zeros(n, device=ei.device)
    deg.index_add_(0, src, torch.ones(src.size(0), device=ei.device))
    w = deg.clamp(min=1).pow(-0.5)
    msg = x[src] * (w[src] * w[dst]).unsqueeze(1)
    out = torch.zeros_like(x)
    out.index_add_(0, dst, msg)
    return out


class SeiLayer(nn.Module):
    """out = W_a * A_hat x + W_s * S_hat x + b,  S_hat x = (Jx - x - 2Ax)/n."""

    def __init__(self, din, dout):
        super().__init__()
        self.lin_a = nn.Linear(din, dout, bias=False)
        self.lin_s = nn.Linear(din, dout, bias=False)
        self.bias = nn.Parameter(torch.zeros(dout))

    def forward(self, x, edge_index, n):
        xa = a_hat(x, edge_index, n, self_loops=True)
        ax = a_hat(x, edge_index, n, self_loops=False)
        xs = (x.sum(0, keepdim=True) - x - 2.0 * ax) / n
        return self.lin_a(xa) + self.lin_s(xs) + self.bias


class VNLayer(nn.Module):
    """out = W_a * A_hat x + W_v * mean(x) + b  (unsigned global channel)."""

    def __init__(self, din, dout):
        super().__init__()
        self.lin_a = nn.Linear(din, dout, bias=False)
        self.lin_v = nn.Linear(din, dout, bias=False)
        self.bias = nn.Parameter(torch.zeros(dout))

    def forward(self, x, edge_index, n):
        xa = a_hat(x, edge_index, n, self_loops=True)
        return (self.lin_a(xa) + self.lin_v(x.mean(0, keepdim=True))
                + self.bias)


class GCNLayer(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.lin = nn.Linear(din, dout)

    def forward(self, x, edge_index, n):
        return self.lin(a_hat(x, edge_index, n, self_loops=True))


class Net(nn.Module):
    def __init__(self, kind, din, dh, dout):
        super().__init__()
        self.kind = kind
        if kind == "mlp":
            self.l1 = nn.Linear(din, dh)
            self.l2 = nn.Linear(dh, dout)
        else:
            Layer = {"gcn": GCNLayer, "vn": VNLayer, "sei": SeiLayer}[kind]
            self.l1 = Layer(din, dh)
            self.l2 = Layer(dh, dout)

    def forward(self, x, edge_index, n):
        if self.kind == "mlp":
            h = F.dropout(F.relu(self.l1(x)), 0.5, self.training)
            return self.l2(h)
        h = F.dropout(x, 0.5, self.training)
        h = F.relu(self.l1(h, edge_index, n))
        h = F.dropout(h, 0.5, self.training)
        return self.l2(h, edge_index, n)


# ------------------------------------------------------------ training
def get_masks(data, family, name):
    """Return list of (train, val, test) boolean mask triples."""
    if data.train_mask.dim() == 2:
        return [(data.train_mask[:, i], data.val_mask[:, i],
                 data.test_mask[:, i])
                for i in range(data.train_mask.size(1))]
    if family == "planetoid":
        # same public split, 10 seeds (run_one seeds by split index)
        return [(data.train_mask, data.val_mask, data.test_mask)] * 10
    # single-mask hetero sets (WebKB etc.): 10 random 48/32/20 splits
    n = data.x.size(0)
    masks = []
    for s in range(10):
        g = torch.Generator().manual_seed(1000 + s)
        perm = torch.randperm(n, generator=g)
        ntr, nva = int(0.48 * n), int(0.32 * n)
        tr = torch.zeros(n, dtype=torch.bool)
        va = torch.zeros(n, dtype=torch.bool)
        te = torch.zeros(n, dtype=torch.bool)
        tr[perm[:ntr]] = True
        va[perm[ntr:ntr + nva]] = True
        te[perm[ntr + nva:]] = True
        masks.append((tr, va, te))
    return masks


def run_one(data, kind, mask_triples, epochs=300):
    n = data.x.size(0)
    x, ei, y = data.x.to(DEV), data.edge_index.to(DEV), data.y.to(DEV)
    nclass = int(data.y.max()) + 1
    accs = []
    for si, (tr, va, te) in enumerate(mask_triples):
        torch.manual_seed(42 + si)
        model = Net(kind, data.x.size(1), 64, nclass).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=0.01,
                               weight_decay=5e-4)
        tr, va, te = tr.to(DEV), va.to(DEV), te.to(DEV)
        best_val, best_test = 0.0, 0.0
        for ep in range(epochs):
            model.train()
            out = model(x, ei, n)
            loss = F.cross_entropy(out[tr], y[tr])
            opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(x, ei, n).argmax(1)
                va_acc = float((pred[va] == y[va]).float().mean())
                te_acc = float((pred[te] == y[te]).float().mean())
                if va_acc > best_val:
                    best_val, best_test = va_acc, te_acc
        accs.append(best_test)
    return accs


def load_dataset(family, name):
    # shim: old scipy unpickling under numpy>=1.24
    if not hasattr(np, "typeDict"):
        np.typeDict = np.sctypeDict
    if family == "planetoid":
        return Planetoid(f"{ROOT}/data", name, split="public")[0]
    if family == "webkb":
        return WebKB(f"{ROOT}/data", name)[0]
    if family == "wiki":
        try:
            return WikipediaNetwork(f"{ROOT}/data", name,
                                    geom_gcn_preprocess=True)[0]
        except TypeError:
            return WikipediaNetwork(f"{ROOT}/data", name)[0]
    try:
        return Actor(f"{ROOT}/data", geom_gcn_preprocess=True)[0]
    except TypeError:
        return Actor(f"{ROOT}/data")[0]


def main():
    results = {"device": DEV, "time": time.strftime("%F %T"),
               "datasets": {}}
    plan = []
    for name in ("Cora", "CiteSeer", "Pubmed"):
        plan.append(("planetoid", name))
    for name in ("Texas", "Cornell", "Wisconsin"):
        plan.append(("webkb", name))
    plan.append(("actor", "Actor"))
    for name in ("chameleon", "squirrel"):
        plan.append(("wiki", name))

    for family, name in plan:
        try:
            data = load_dataset(family, name)
            print(f"== {name}: n={data.x.size(0)} "
                  f"e={data.edge_index.size(1)} "
                  f"classes={int(data.y.max())+1} "
                  f"maskdim={data.train_mask.dim()}", flush=True)
        except Exception as ex:
            print(f"== {name}: LOAD FAILED: {ex!r}", flush=True)
            results["datasets"][name] = {"error": repr(ex)}
            continue
        masks = get_masks(data, family, name)
        entry = {}
        for kind in ("mlp", "gcn", "vn", "sei"):
            accs = run_one(data, kind, masks)
            entry[kind] = {"mean": float(np.mean(accs)),
                           "std": float(np.std(accs)),
                           "runs": len(accs)}
            print(f"   {kind:4s} {np.mean(accs):.4f} +- {np.std(accs):.4f}"
                  f"  ({len(accs)} runs)", flush=True)
        results["datasets"][name] = entry
        with open(f"{ROOT}/results.json", "w") as f:
            json.dump(results, f, indent=1)
    print("DONE ->", f"{ROOT}/results.json")


if __name__ == "__main__":
    main()
