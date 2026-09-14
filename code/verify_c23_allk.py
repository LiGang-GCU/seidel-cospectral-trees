"""Exact all-k walk-equality witness for the C23H48 pair (Theorem chemboundary).

The chemistry census stores only W_0..W_{n-1}; for a NON-cospectral pair that
finite window does not imply all-k equality. Both walk sequences satisfy the
linear recurrence of their own characteristic polynomial (Cayley-Hamilton,
degree 23), hence both satisfy the common recurrence of order <= 46 given by
the product of the two characteristic polynomials. Agreement on 46 consecutive
values therefore propagates to every k. This script verifies far more than 46,
in exact Python big-integer arithmetic, and writes the witness to results/.
"""
import io
import json
import sys
from pathlib import Path

import networkx as nx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dgs_census import charpoly_int  # noqa: E402

ROOT = HERE.parent
KMAX = 200


def walks(G, kmax):
    """Exact W_k = 1^T A^k 1 for k = 0..kmax, pure Python ints."""
    n = G.number_of_nodes()
    adj = [sorted(G.neighbors(v)) for v in range(n)]
    v = [1] * n
    out = [n]
    for _ in range(kmax):
        v = [sum(v[u] for u in adj[i]) for i in range(n)]
        out.append(sum(v))
    return out


def main():
    src = json.load(io.open(ROOT / "results" / "chem_boundary_n23.json",
         encoding="utf-8"))
    cls = src["classes"][0]
    g6s = [m["g6"] for m in cls["members"]]
    assert len(g6s) == 2, g6s

    Gs = [nx.from_graph6_bytes(g.encode()) for g in g6s]
    for G in Gs:
        assert G.number_of_nodes() == 23
        assert max(d for _, d in G.degree()) <= 4

    W = [walks(G, KMAX) for G in Gs]
    agree = W[0] == W[1]
    first_diff = next((k for k in range(KMAX + 1) if W[0][k] != W[1][k]), None)

    cps = [tuple(charpoly_int(nx.to_numpy_array(G).astype(int).tolist()))
           for G in Gs]
    cospectral = cps[0] == cps[1]
    order = (len(cps[0]) - 1) + (len(cps[1]) - 1)
    cp_diff = next((i for i, (a, b) in enumerate(zip(*cps)) if a != b), None)

    stored = [int(x) for x in cls["walk_tuple"]]
    assert stored == W[0][:len(stored)], "census tuple disagrees with recompute"

    print("graphs        :", g6s[0])
    print("          ", g6s[1])
    print("cospectral    :", cospectral)
    print("charpoly diff : first differing coefficient index", cp_diff)
    print("recurrence ord: <=", order)
    print("W_k agree     : k=0..%d -> %s" % (KMAX, agree)
        + ("" if agree else " (first differ at k=%s)" % first_diff))
    print("values checked: %d >> %d  -> propagates to all k: %s"
     % (KMAX + 1, order, agree))
    print("W_200 digits  :", len(str(W[0][200])))
    print("census tuple  : %d values (W_0..W_%d) reproduced exactly"
    % (len(stored), len(stored) - 1))

    out = {
        "pair_g6": g6s,
        "n": 23,
        "kmax_verified": KMAX,
     "arithmetic": "exact Python big integers; no floating point",
      "walks_equal_through_kmax": agree,
        "first_disagreement_k": first_diff,
        "cospectral": cospectral,
        "charpoly_first_diff_index": cp_diff,
        "charpolys": [[str(c) for c in cp] for cp in cps],
        "common_recurrence_order_bound": order,
        "argument": (
            "Each W-sequence satisfies the order-23 Cayley-Hamilton recurrence "
       "of its own characteristic polynomial, so both satisfy the common "
   "recurrence of order at most %d given by the product. Agreement on "
            "%d consecutive initial values, far more than %d, therefore forces "
      "agreement for every k." % (order, KMAX + 1, order)
        ),
      "walks": {str(k): str(W[0][k]) for k in range(KMAX + 1)},
    }
    dst = ROOT / "results" / "c23_allk_walk_witness.json"
    io.open(dst, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, indent=1) + "\n")
    print("witness ->", dst)


if __name__ == "__main__":
    main()
