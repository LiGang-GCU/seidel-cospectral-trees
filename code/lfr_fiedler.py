"""Stage C': distance-Fiedler spectral clustering vs classical spectral
clustering vs Louvain, on LFR benchmark graphs.

Hypothesis (from Stage 1 census): DL/DQ spectra encode global distance
structure invisible to adjacency/Laplacian spectra. If so, the eigenvectors
of the DISTANCE Laplacian DL = Tr - D (notably the distance Fiedler vector,
2nd smallest) may separate communities differently — possibly better when
communities are defined by distance patterns rather than pure edge density.

Protocol:
  - LFR benchmark graphs, n=300, mixing mu in {0.1..0.7}, 10 seeds each
  - embedding methods (true k given):
      Lspec : k smallest eigvecs of sym-norm Laplacian (Ng-Jordan-Weiss)
      DLspec: k smallest eigvecs of distance Laplacian DL
      DQspec: k largest eigvecs of distance signless DQ
      Louvain baseline (no k given)
  - k-means (k-means++ , 25 restarts) on row-normalized embeddings
  - NMI vs planted partition; mean +/- std over seeds

Pure numpy/networkx implementation (no sklearn on this machine).
"""
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
RNG = np.random.default_rng(20260905)


# ------------------------------------------------------------ clustering
def kmeans(X, k, restarts=25, iters=100, rng=RNG):
    best, best_inertia = None, np.inf
    n = len(X)
    for _ in range(restarts):
        # k-means++ init
        cen = [X[rng.integers(n)]]
        for _ in range(k - 1):
            d2 = np.min(((X[:, None, :] - np.array(cen)[None]) ** 2).sum(-1), axis=1)
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
            cen.append(X[rng.choice(n, p=p)])
        cen = np.array(cen)
        lab = np.zeros(n, int)
        for _ in range(iters):
            d = ((X[:, None, :] - cen[None]) ** 2).sum(-1)
            new = d.argmin(1)
            if np.array_equal(new, lab):
                break
            lab = new
            for j in range(k):
                pts = X[lab == j]
                if len(pts):
                    cen[j] = pts.mean(0)
        inertia = ((X - cen[lab]) ** 2).sum()
        if inertia < best_inertia:
            best, best_inertia = lab.copy(), inertia
    return best


def nmi(true, pred):
    true, pred = np.asarray(true), np.asarray(pred)
    n = len(true)
    _, t = np.unique(true, return_inverse=True)
    _, p = np.unique(pred, return_inverse=True)
    k_t, k_p = t.max() + 1, p.max() + 1
    C = np.zeros((k_t, k_p))
    np.add.at(C, (t, p), 1)
    P = C / n
    pt, pp = P.sum(1), P.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        I = np.nansum(P * np.log(P / (pt[:, None] * pp[None])))
    Ht = -(pt[pt > 0] * np.log(pt[pt > 0])).sum()
    Hp = -(pp[pp > 0] * np.log(pp[pp > 0])).sum()
    return float(I / np.sqrt(Ht * Hp)) if Ht > 0 and Hp > 0 else 1.0


# ------------------------------------------------------------ embeddings
def dist_matrix(A):
    n = len(A)
    adj = [np.flatnonzero(A[i]) for i in range(n)]
    M = np.zeros((n, n))
    for s in range(n):
        dist = np.full(n, -1)
        dist[s] = 0
        q = [s]
        for u in q:
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        M[s] = dist
    return M


def embed(A, kind, k):
    n = len(A)
    if kind == "Lspec":
        d = A.sum(1)
        Lsym = np.eye(n) - A / np.sqrt(np.outer(d, d))
        w, V = np.linalg.eigh(Lsym)
        U = V[:, np.argsort(w)[:k]]
    else:
        Dm = dist_matrix(A)
        tr = Dm.sum(1)
        M = np.diag(tr) - Dm if kind == "DLspec" else np.diag(tr) + Dm
        w, V = np.linalg.eigh(M)
        U = V[:, np.argsort(w)[:k]] if kind == "DLspec" else V[:, np.argsort(w)[-k:]]
    nr = np.linalg.norm(U, axis=1, keepdims=True)
    nr[nr == 0] = 1
    return U / nr  # Ng-Jordan-Weiss row normalization


# ------------------------------------------------------------ experiment
def one_graph(seed, mu, n=300):
    G = nx.LFR_benchmark_graph(
        n, tau1=2.5, tau2=1.5, mu=mu, average_degree=8,
        min_community=30, seed=seed, max_iters=300)
    comms = [frozenset(c) for c in
             nx.get_node_attributes(G, "community").values()]
    # planted partition label per node
    true = np.zeros(n, int)
    for ci, c in enumerate({frozenset(x) for x in
                            nx.get_node_attributes(G, 'community').values()}):
        for v in c:
            true[v] = ci
    A = nx.to_numpy_array(G)
    k = len(set(true))
    out = {}
    for kind in ("Lspec", "DLspec", "DQspec"):
        pred = kmeans(embed(A, kind, k), k)
        out[kind] = nmi(true, pred)
    lou = nx.community.louvain_communities(G, seed=seed)
    pred = np.zeros(n, int)
    for ci, c in enumerate(lou):
        for v in c:
            pred[v] = ci
    out["Louvain"] = nmi(true, pred)
    out["k_true"] = k
    return out


def _gen_one(args):
    seed, mu = args
    return one_graph(seed=seed, mu=mu)


def main():
    import signal
    t0 = time.time()
    mus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    reps = 10
    rows = []

    class GenTimeout(Exception):
        pass

    def handler(signum, frame):
        raise GenTimeout()

    signal.signal(signal.SIGALRM, handler)

    for mu in mus:
        for r in range(reps):
            ok = False
            for base in (1000, 5000, 9000):  # up to 3 seed families
                seed = base + 97 * r + int(mu * 1000)
                signal.alarm(60)
                try:
                    res = one_graph(seed=seed, mu=mu)
                    ok = True
                except Exception as ex2:
                    last = ex2
                finally:
                    signal.alarm(0)
                if ok:
                    break
            if not ok:
                print(f"  mu={mu} rep={r} FAILED: {last}", flush=True)
                continue
            res["mu"] = mu
            rows.append(res)
        print(f"mu={mu} done ({time.time()-t0:.0f}s)", flush=True)

    summ = {}
    for mu in mus:
        sub = [x for x in rows if x["mu"] == mu]
        summ[str(mu)] = {
            m: {"mean": float(np.mean([x[m] for x in sub])),
                "std": float(np.std([x[m] for x in sub])),
                "n": len(sub)}
            for m in ("Lspec", "DLspec", "DQspec", "Louvain")}
    dest = RESULTS / "lfr_fiedler.json"
    dest.write_text(json.dumps({"rows": rows, "summary": summ}, indent=1))
    print("\n== mean NMI ==")
    print(f"{'mu':>5} {'Lspec':>14} {'DLspec':>14} {'DQspec':>14} {'Louvain':>14}")
    for mu in mus:
        s = summ[str(mu)]
        print(f"{mu:>5} " + " ".join(
            f"{s[m]['mean']:.3f}±{s[m]['std']:.3f}" for m in
            ("Lspec", "DLspec", "DQspec", "Louvain")))
    print("wrote", dest)


if __name__ == "__main__":
    main()
