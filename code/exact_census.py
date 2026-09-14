"""EXACT census of trees n=1..22: modular charpoly fingerprints (no floats).

Fingerprints per tree (all exact, modular arithmetic):
  WALK : (W_0..W_{n-1}) mod two 24-bit primes
  SSPEC: charpoly(S) coefficients mod two primes, via Newton's identities
  GEN  : (charpoly(A), charpoly(A_comp)) mod two primes
  ASPEC: charpoly(A) mod two primes

Protocol per n:
  pass 1: count fingerprints (collision = count > 1)
  pass 2: collect graph6 of trees in hot classes
  pass 3: inside every hot class recompute EXACT big-integer charpolys /
          walk tuples, re-form classes, compare SSPEC vs GEN partitions.

Output: results/exact_census_n{n}.json + summary line per n.
Usage: python3 exact_census.py [lo hi]
"""
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dgs_census import charpoly_int  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
P1, P2 = 16777213, 16777199
PS = (P1, P2)
CHUNK = 4000


def traces_mod(M, p):
    # float64 BLAS matmul; entries < p < 2^24, products < 2^48, row sums
    # < 2^52.5, hence every partial value is an exactly-represented integer.
    n = M.shape[0]
    t = []
    P = M.astype(np.float64)
    Mf = M.astype(np.float64)
    for _ in range(n):
        t.append(int(P.trace()) % p)
        P = (P @ Mf) % p
    return t


def cp_fp(M):
    # Newton's identities divide by k, so the Fermat inverse pow(k, p-2, p)
    # below requires BOTH moduli to be prime: 16777213 and 16777199 are.
    # (An earlier version used 16777211 = 11 * 101 * 15101, for which the
    # Fermat inverse is wrong for every k >= 2.)
    n = M.shape[0]
    out = []
    for p in PS:
        t = traces_mod(M % p, p)
        e = [1] + [0] * n
        for k in range(1, n + 1):
            s = 0
            for i in range(1, k + 1):
                s += (1 if i % 2 == 1 else -1) * e[k - i] * t[i - 1]
            e[k] = (s % p) * pow(k, p - 2, p) % p
        out.append(tuple(e[1:]))
    return tuple(out)


def walk_fp(A):
    n = A.shape[0]
    outs = []
    for p in PS:
        Am = (A % p).astype(np.float64)
        v = np.ones(n, dtype=np.float64)
        w = [n % p]
        for _ in range(n - 1):
            v = (Am @ v) % p
            w.append(int(v.sum()) % p)
        outs.append(tuple(w))
    return tuple(outs)


def fps_of_g6(g6):
    G = nx.from_graph6_bytes(g6.encode())
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G).astype(np.int64)
    J = np.ones((n, n), dtype=np.int64)
    I = np.eye(n, dtype=np.int64)
    S = J - I - 2 * A
    Ac = J - I - A
    return (g6, walk_fp(A), cp_fp(S), (cp_fp(A), cp_fp(Ac)), cp_fp(A))


def exact_walks(g6):
    G = nx.from_graph6_bytes(g6.encode())
    n = G.number_of_nodes()
    al = [sorted(G.neighbors(v)) for v in range(n)]
    v = [1] * n
    W = [n]
    for _ in range(n - 1):
        nv = [0] * n
        for i in range(n):
            for j in al[i]:
                nv[j] += v[i]
        v = nv
        W.append(sum(v))
    return tuple(W)


def exact_keys(g6):
    G = nx.from_graph6_bytes(g6.encode())
    A = nx.to_numpy_array(G).astype(int).tolist()
    n = len(A)
    S = [[(1 if i != j else 0) - 2 * A[i][j] for j in range(n)]
         for i in range(n)]
    Ac = [[(1 if i != j else 0) - A[i][j] for j in range(n)]
          for i in range(n)]
    return (exact_walks(g6),
            tuple(charpoly_int(S)),
            (tuple(charpoly_int(A)), tuple(charpoly_int(Ac))),
            tuple(charpoly_int(A)))


def run(n, pool):
    t0 = time.time()
    g6s = [nx.to_graph6_bytes(G, header=False).decode().strip()
           for G in nx.nonisomorphic_trees(n)]
    walk_c, sspec_c, gen_c, aspec_c = (defaultdict(int) for _ in range(4))
    for res in pool.imap_unordered(fps_of_g6, g6s, chunksize=64):
        _, w, s, g, a = res
        walk_c[w] += 1
        sspec_c[s] += 1
        gen_c[g] += 1
        aspec_c[a] += 1
    hot = set()
    for d in (walk_c, sspec_c, gen_c, aspec_c):
        hot |= {f for f, c in d.items() if c > 1}
    n_hot = len(hot)
    # pass 2: collect members of hot classes
    keep = []
    hot_w = {f for f, c in walk_c.items() if c > 1}
    hot_s = {f for f, c in sspec_c.items() if c > 1}
    hot_g = {f for f, c in gen_c.items() if c > 1}
    hot_a = {f for f, c in aspec_c.items() if c > 1}
    for res in pool.imap_unordered(fps_of_g6, g6s, chunksize=64):
        g6, w, s, g, a = res
        if w in hot_w or s in hot_s or g in hot_g or a in hot_a:
            keep.append((g6, w, s, g, a))
    # pass 3: exact re-grouping
    def regroup(idx, keyfn):
        classes = defaultdict(list)
        for rec in keep:
            classes[rec[idx]].append(rec[0])
        exact_classes = []
        for _, members in classes.items():
            if len(members) < 2:
                continue
            sub = defaultdict(list)
            for g6 in members:
                sub[keyfn(g6)].append(g6)
            exact_classes.extend(v for v in sub.values() if len(v) > 1)
        return exact_classes

    w_cls = regroup(1, lambda g: exact_keys(g)[0])
    s_cls = regroup(2, lambda g: exact_keys(g)[1])
    g_cls = regroup(3, lambda g: exact_keys(g)[2])
    a_cls = regroup(4, lambda g: exact_keys(g)[3])
    # partition identity SSPEC vs GEN on the whole set:
    # the relations coincide iff the equivalence closures of hot classes
    # agree (outside hot classes everything is singleton in both)
    s_part = {g6: i for i, c in enumerate(s_cls) for g6 in c}
    g_part = {g6: i for i, c in enumerate(g_cls) for g6 in c}
    same_partition = True
    for c in s_cls:
        ids = {g_part.get(x, -1) for x in c}
        if len(ids) != 1 or -1 in ids and len(c) > 1:
            same_partition = False
    for c in g_cls:
        ids = {s_part.get(x, -1) for x in c}
        if len(ids) != 1 or -1 in ids and len(c) > 1:
            same_partition = False
    out = {"n": n, "n_trees": len(g6s),
           "walk_classes": len(w_cls), "sspec_classes": len(s_cls),
           "gen_classes": len(g_cls), "aspec_classes": len(a_cls),
           "sspec_eq_gen": same_partition,
           "runtime_s": round(time.time() - t0, 1)}
    (RESULTS / f"exact_census_n{n}.json").write_text(json.dumps(out, indent=1))
    print(f"[n={n}] trees={len(g6s)} WALK={len(w_cls)} SSPEC={len(s_cls)} "
          f"GEN={len(g_cls)} ASPEC={len(a_cls)} eq={same_partition} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 22
    with Pool(processes=7) as pool:
        for n in range(lo, hi + 1):
            run(n, pool)
