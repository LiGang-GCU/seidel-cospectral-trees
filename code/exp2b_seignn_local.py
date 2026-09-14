"""EXP2b: Seidel-channel GNN on locally cached datasets (no downloads).

Same models as exp2 (MLP / GCN / GCN+VN / SeiGCN, 2 layers, hidden 64),
on server-local datasets:
  homophilic : Pubmed (public split), WikiCS (20 splits),
               Coauthor CS, Amazon Photo (random 60/20/20 splits)
  heterophilic: Texas (WebKB local), squirrel (geom-gcn splits),
                roman_empire, minesweeper (HeterophilicGraphDataset),
                Actor (local)

Output: /media/jiuzhang/sda/msr/exp2_nodecls/results2.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = "/media/jiuzhang/sda/msr/exp2_nodecls"
DATA = "/media/jiuzhang/sda/data/data"
Path(ROOT).mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# numpy>=1.24 vs old scipy unpickling shim
if not hasattr(np, "typeDict"):
    np.typeDict = np.sctypeDict


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
    def __init__(self, din, dout):
        super().__init__()
        self.lin_a = nn.Linear(din, dout, bias=False)
        self.lin_s = nn.Linear(din, dout, bias=False)
        self.bias = nn.Parameter(torch.zeros(dout))

    def forward(self, x, edge_index, n):
        xa = a_hat(x, edge_index, n, True)
        ax = a_hat(x, edge_index, n, False)
        xs = (x.sum(0, keepdim=True) - x - 2.0 * ax) / n
        return self.lin_a(xa) + self.lin_s(xs) + self.bias


class VNLayer(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.lin_a = nn.Linear(din, dout, bias=False)
        self.lin_v = nn.Linear(din, dout, bias=False)
        self.bias = nn.Parameter(torch.zeros(dout))

    def forward(self, x, edge_index, n):
        return (self.lin_a(a_hat(x, edge_index, n, True))
                + self.lin_v(x.mean(0, keepdim=True)) + self.bias)


class GCNLayer(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.lin = nn.Linear(din, dout)

    def forward(self, x, edge_index, n):
        return self.lin(a_hat(x, edge_index, n, True))


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


# ------------------------------------------------------------ data
def random_masks(n, k=10, ratios=(0.6, 0.2, 0.2)):
    masks = []
    for s in range(k):
        g = torch.Generator().manual_seed(1000 + s)
        perm = torch.randperm(n, generator=g)
        ntr = int(ratios[0] * n)
        nva = int(ratios[1] * n)
        tr = torch.zeros(n, dtype=torch.bool)
        va = torch.zeros(n, dtype=torch.bool)
        te = torch.zeros(n, dtype=torch.bool)
        tr[perm[:ntr]] = True
        va[perm[ntr:ntr + nva]] = True
        te[perm[ntr + nva:]] = True
        masks.append((tr, va, te))
    return masks


def load(name):
    from torch_geometric.datasets import (Amazon, Coauthor,
                                          HeterophilicGraphDataset,
                                          Planetoid, WebKB,
                                          WikipediaNetwork, WikiCS, Actor)
    if name == "Pubmed":
        d = Planetoid(f"{DATA}/Planetoid", "Pubmed", split="public")[0]
        return d, [(d.train_mask, d.val_mask, d.test_mask)] * 10
    if name == "WikiCS":
        d = WikiCS(f"{DATA}/WikiCS")[0]
        if d.train_mask.dim() == 2:
            return d, [(d.train_mask[:, i], d.val_mask[:, i],
                        d.test_mask[:, i])
                       for i in range(d.train_mask.size(1))]
        return d, [(d.train_mask, d.val_mask, d.test_mask)] * 10
    if name == "CoauthorCS":
        d = Coauthor(f"{DATA}/Coauthor", "CS")[0]
        return d, random_masks(d.x.size(0))
    if name == "AmazonPhoto":
        d = Amazon(f"{DATA}/Amazon", "Photo")[0]
        return d, random_masks(d.x.size(0))
    if name == "Texas":
        d = WebKB(f"{DATA}/WebKB", "texas")[0]
        if d.train_mask.dim() == 2:
            return d, [(d.train_mask[:, i], d.val_mask[:, i],
                        d.test_mask[:, i])
                       for i in range(d.train_mask.size(1))]
        return d, random_masks(d.x.size(0), ratios=(0.48, 0.32, 0.2))
    if name == "squirrel":
        d = WikipediaNetwork(f"{DATA}", "squirrel",
                             geom_gcn_preprocess=True)[0]
        return d, [(d.train_mask[:, i], d.val_mask[:, i],
                    d.test_mask[:, i])
                   for i in range(d.train_mask.size(1))]
    if name == "roman_empire":
        for nm in ("Roman-empire", "roman_empire"):
            try:
                d = HeterophilicGraphDataset(f"{DATA}", nm)[0]
                break
            except Exception:
                continue
        return d, [(d.train_mask[:, i], d.val_mask[:, i],
                    d.test_mask[:, i])
                   for i in range(d.train_mask.size(1))]
    if name == "minesweeper":
        for nm in ("Minesweeper", "minesweeper"):
            try:
                d = HeterophilicGraphDataset(f"{DATA}", nm)[0]
                break
            except Exception:
                continue
        return d, [(d.train_mask[:, i], d.val_mask[:, i],
                    d.test_mask[:, i])
                   for i in range(d.train_mask.size(1))]
    if name == "Actor":
        try:
            d = Actor(f"{DATA}", geom_gcn_preprocess=True)[0]
        except TypeError:
            d = Actor(f"{DATA}")[0]
        if d.train_mask.dim() == 2:
            return d, [(d.train_mask[:, i], d.val_mask[:, i],
                        d.test_mask[:, i])
                       for i in range(d.train_mask.size(1))]
        return d, random_masks(d.x.size(0), ratios=(0.48, 0.32, 0.2))
    raise ValueError(name)


def run_one(data, kind, mask_triples, epochs=300):
    n = data.x.size(0)
    x = data.x.to(DEV)
    if x.dtype != torch.float:
        x = x.float()
    ei, y = data.edge_index.to(DEV), data.y.to(DEV)
    nclass = int(y.max()) + 1
    accs = []
    for si, (tr, va, te) in enumerate(mask_triples):
        torch.manual_seed(42 + si)
        model = Net(kind, x.size(1), 64, nclass).to(DEV)
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


def main():
    results = {"device": DEV, "time": time.strftime("%F %T"),
               "datasets": {}}
    names = ["Pubmed", "WikiCS", "CoauthorCS", "AmazonPhoto",
             "Texas", "squirrel", "roman_empire", "minesweeper", "Actor"]
    for name in names:
        try:
            data, masks = load(name)
            print(f"== {name}: n={data.x.size(0)} "
                  f"e={data.edge_index.size(1)} "
                  f"classes={int(data.y.max())+1} "
                  f"splits={len(masks)}", flush=True)
        except Exception as ex:
            print(f"== {name}: LOAD FAILED: {ex!r}", flush=True)
            results["datasets"][name] = {"error": repr(ex)}
            continue
        entry = {}
        try:
            for kind in ("mlp", "gcn", "vn", "sei"):
                accs = run_one(data, kind, masks)
                entry[kind] = {"mean": float(np.mean(accs)),
                               "std": float(np.std(accs)),
                               "runs": len(accs)}
                print(f"   {kind:4s} {np.mean(accs):.4f} +- "
                      f"{np.std(accs):.4f}  ({len(accs)} runs)", flush=True)
        except Exception as ex:
            print(f"   TRAIN FAILED: {ex!r}", flush=True)
            entry["error"] = repr(ex)
        results["datasets"][name] = entry
        with open(f"{ROOT}/results2.json", "w") as f:
            json.dump(results, f, indent=1)
    print("DONE ->", f"{ROOT}/results2.json")


if __name__ == "__main__":
    main()
