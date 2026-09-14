"""Big-n three-way equivalence check on trees (n=19..22).

No floating-point eigenvalues appear anywhere in this pipeline.
The protocol has two layers with different jobs:

  filter layer  -- exact modular fingerprints, used only to decide
      which trees are worth comparing exactly;
  verdict layer -- exact integer arithmetic (bigint walk counts,
   integer Faddeev-LeVerrier charpolys), run inside
     every candidate group; this alone decides the
    reported partitions.

The filter is sound in the only direction that matters. A modular
fingerprint of an exact integer invariant can produce false MERGES
(distinct integers sharing residues); pass 3 removes them by exact
recomputation. It can never produce a false SPLIT: equal integers
always have equal residues. Hence the reported answer does not
depend on the prime -- only the runtime does. Floating-point
eigenvalue buckets have the opposite, fatal failure mode: two
genuinely cospectral trees can round into different buckets, are
then never compared, and no in-bucket exact recheck can repair it.

For memory, pass 1 groups on a 64-bit digest of the exact key bytes
rather than on the bytes themselves. Same argument: equal keys have
equal digests (no false split), and a digest collision is one more
false merge, removed by pass 3.

  WALK : (W_0..W_{n-1})   mod p
  SSPEC: charpoly(S) coefficients, S = J-I-2A, mod p
  GEN  : (charpoly(A), charpoly(Abar)) coeffs, mod p

Faddeev-LeVerrier runs mod p in float64 so BLAS matmul does the work:
all entries stay below p and n*p^2 < 2^53, so every product and every
n-term dot product is exact in a float64 mantissa. The float64
arithmetic here is exact integer arithmetic, not an approximation.

Usage: python walk_census_big.py 19 [20 ...]
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dgs_census import charpoly_int  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

# One ~23-bit prime. Bound for exactness of the float64 matmul:
# n * p^2 < 2^53. With n <= 22 and p < 10^7: 22 * 10^14 = 2.2e15,
# comfortably under 2^53 = 9.007e15. Correctness of the reported
# partitions does not depend on this choice (see module docstring).
P1 = 9999991
CHUNK = 20000


def _fl_coeffs_mod(Mb, p, n):
    """Batched Faddeev-LeVerrier mod p.

    Mb: (B, n, n) float64, integer-valued, entries reduced mod p.
    Returns (B, n+1) int64 charpoly coefficients mod p, with
    charpoly = sum_k c_k x^(n-k) and c_0 = 1.

    The recurrence is the same over Z_p as over Q, and the true
    coefficients are integers, so the mod-p run agrees with the
    integer coefficients mod p. Requires k invertible mod p for
    k = 1..n, which holds since p is prime and p > n.
    """
    idx = np.arange(n)
    Bm = np.zeros_like(Mb)
    Bm[:, idx, idx] = 1.0
    out = np.empty((Mb.shape[0], n + 1), dtype=np.int64)
    out[:, 0] = 1
    for k in range(1, n + 1):
        AB = (Mb @ Bm) % p
        tr = AB[:, idx, idx].sum(axis=1) % p
        ck = (-tr * pow(k, -1, p)) % p
        out[:, k] = ck.astype(np.int64)
        Bm = AB
        Bm[:, idx, idx] = (Bm[:, idx, idx] + ck[:, None]) % p
    return out


def _walk_coeffs_mod(Ab, p, n):
    """Batched walk counts (W_0..W_{n-1}) mod p.

    Ab: (B, n, n) float64 adjacency (0/1). Returns (B, n) int64.
    Exact: v stays < p and A is 0/1, so A @ v < n*p < 2^53.
    """
    v = np.ones((Ab.shape[0], n, 1))
    out = np.empty((Ab.shape[0], n), dtype=np.int64)
    out[:, 0] = n % p
    for k in range(1, n):
        v = (Ab @ v) % p
        out[:, k] = (v.sum(axis=(1, 2)) % p).astype(np.int64)
    return out


def batch_mats(Gs, n):
    """Exact-modular fingerprint matrices: kind -> (B, m) int64."""
    B = len(Gs)
    A = np.empty((B, n, n))
    for i, G in enumerate(Gs):
        A[i] = nx.to_numpy_array(G, nodelist=sorted(G))
    I = np.eye(n)
    J = np.ones((n, n))
    p = P1
    walk = _walk_coeffs_mod(A % p, p, n)
    cpS = _fl_coeffs_mod((J - I - 2.0 * A) % p, p, n)
    cpA = _fl_coeffs_mod(A % p, p, n)
    cpC = _fl_coeffs_mod((J - I - A) % p, p, n)
    return {"WALK": walk,
            "SSPEC": cpS,
            "GEN": np.ascontiguousarray(np.hstack((cpA, cpC)))}


def batch_keys(Gs, n):
    """Exact byte keys per tree: kind -> list of bytes.

    Equal fingerprints always give equal keys (no false splits);
    distinct fingerprints always give distinct keys (byte-level).
    """
    return {k: [row.tobytes() for row in np.ascontiguousarray(M)]
            for k, M in batch_mats(Gs, n).items()}


def _digest(M):
    """64-bit FNV-style digest of each row of an (B, m) int64 array."""
    h = np.full(M.shape[0], 14695981039346656037, dtype=np.uint64)
    K = np.uint64(1099511628211)
    for j in range(M.shape[1]):
        h ^= M[:, j].astype(np.uint64)
        h *= K
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xff51afd7ed558ccd)
    h ^= h >> np.uint64(29)
    return h


def batch_digests(Gs, n):
    """kind -> (B,) uint64 digests of the exact-modular keys."""
    return {k: _digest(M) for k, M in batch_mats(Gs, n).items()}


def exact_fps(G):
    """Exact integer fingerprints (bigint; only inside hot classes)."""
    n = G.number_of_nodes()
    Ai = nx.to_numpy_array(G, nodelist=sorted(G)).astype(int)
    v = [1] * n
    W = [n]
    for _ in range(n - 1):
        v = [sum(int(Ai[i][j]) * v[j] for j in range(n))
             for i in range(n)]
        W.append(sum(v))
    comp = (np.ones((n, n), int) - np.eye(n, dtype=int) - Ai)
    seid = (np.ones((n, n), int) - np.eye(n, dtype=int) - 2 * Ai)
    return {"WALK": tuple(W),
            "SSPEC": charpoly_int(seid.tolist()),
            "GEN": (charpoly_int(Ai.tolist()),
                    charpoly_int(comp.tolist()))}


def run(n):
    t0 = time.time()
    kinds = ("WALK", "SSPEC", "GEN")

    # ---- pass 1: exact-modular digests for every tree ----
    parts = {k: [] for k in kinds}
    total = 0
    buf = []
    for G in nx.nonisomorphic_trees(n):
        buf.append(G)
        total += 1
        if len(buf) >= CHUNK:
            ds = batch_digests(buf, n)
            for k in kinds:
                parts[k].append(ds[k])
            buf.clear()
            if total % 400000 < CHUNK:
                print(f"  [n={n}] pass1 {total} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    if buf:
        ds = batch_digests(buf, n)
        for k in kinds:
            parts[k].append(ds[k])
        buf.clear()

    # digest per tree, in enumeration order (order is deterministic)
    dig = {}
    memb = {}
    hot_n = {}
    for k in kinds:
        a = np.concatenate(parts[k]) if parts[k] else np.empty(0, np.uint64)
        parts[k] = None
        u, c = np.unique(a, return_counts=True)
        hotv = u[c > 1]
        hot_n[k] = int(hotv.size)
        dig[k] = a
        memb[k] = np.isin(a, hotv)
        del u, c, hotv
    del parts
    print(f"[n={n}] pass1: {total} trees; hot keys: "
          + json.dumps(hot_n)
          + f" ({time.time()-t0:.0f}s)", flush=True)

    # ---- pass 2: keep graph6 of trees in any hot group ----
    # The enumeration order of nx.nonisomorphic_trees is deterministic,
    # so position i here is the same tree as position i in pass 1; no
    # fingerprint is recomputed.
    anyhot = memb["WALK"] | memb["SSPEC"] | memb["GEN"]
    keep = {k: defaultdict(list) for k in kinds}
    for i, G in enumerate(nx.nonisomorphic_trees(n)):
        if not anyhot[i]:
            continue
        g6 = nx.to_graph6_bytes(G, header=False).decode().strip()
        for k in kinds:
            if memb[k][i]:
                keep[k][int(dig[k][i])].append(g6)
    assert i + 1 == total, (i + 1, total)
    del dig, memb, anyhot
    print(f"[n={n}] pass2: " + json.dumps(
        {k: sum(len(v) for v in d.values()) for k, d in keep.items()})
        + f" ({time.time()-t0:.0f}s)", flush=True)

    # ---- pass 3: exact refine inside each collision group ----
    exact_cls = {}
    n_exact = 0
    for k in kinds:
        classes = []
        for g6s in keep[k].values():
            sub = defaultdict(list)
            for g in g6s:
                fp = exact_fps(nx.from_graph6_bytes(g.encode()))[k]
                sub[fp].append(g)
                n_exact += 1
            classes.extend(sorted(v) for v in sub.values()
                       if len(v) > 1)
        exact_cls[k] = sorted(classes)
    print(f"[n={n}] pass3: {n_exact} exact fingerprints "
          f"({time.time()-t0:.0f}s)", flush=True)

    eq = {f"{a}=={b}": sorted(map(sorted, exact_cls[a])) ==
          sorted(map(sorted, exact_cls[b]))
          for a, b in (("WALK", "SSPEC"), ("WALK", "GEN"),
                     ("SSPEC", "GEN"))}
    out = {"n": n, "n_trees": total,
           "protocol": "exact-modular filter + exact integer verdict "
                       "(no floating-point eigenvalues); false merges "
                       "removed by exact refinement, false splits "
                       "impossible by construction",
           "primes": [P1],
           "hot_keys": hot_n,
           "n_exact_fingerprints": n_exact,
           "collision_classes": {k: len(v)
                                 for k, v in exact_cls.items()},
           "partition_equality": eq,
           "classes": exact_cls,
           "runtime_s": round(time.time() - t0, 1)}
    dest = RESULTS / f"walk_census_big_n{n}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"[n={n}] DONE eq={eq} classes="
          f"{ {k: len(v) for k, v in exact_cls.items()} } -> {dest}",
          flush=True)


if __name__ == "__main__":
    assert 22 * P1 * P1 < 2 ** 53
    for n in [int(x) for x in sys.argv[1:]]:
        run(n)
