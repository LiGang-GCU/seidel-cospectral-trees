"""Computer-assisted derivation: m(T,3) as a polynomial in walk counts.

We KNOW (empirically, walk_census n<=15) that walk-equivalent trees are
cospectral, hence m(T,3) is a function of the walk sequence. Find the
explicit identity by fitting

  m(T,3) = c1*W1^3 + c2*W1*W2 + c3*W3 + c4*W1^2 + c5*W2 + c6*W1 + c7

(exact rationals) on trees n=7..12, then VERIFY on n=13,14 (out of sample).

Derivation guide (double counting):
  m(T,3) = C(m,3) - X(m-2) + P3 + 2*S3
  X  = sum_v C(d_v,2) = (W2 - W1)/2
  P3 = sum_{uv in E} (d_u-1)(d_v-1) = W3/2 - W2 + W1/2
  S3 = sum_v C(d_v,3)   <- needs sum d^3; expected from W4-level terms,
     but weight bookkeeping says m3 has edge-weight 3, so the fit above
     may fail; if it does, extend basis with weight-3 combos from W4, W5
     (e.g. W4 - something, since sum d^3 = W4 - 2Y with Y of weight 3).
"""
import sys
from fractions import Fraction
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))


def tree_data(G):
    n = G.number_of_nodes()
    A = [[int(x) for x in row] for row in nx.to_numpy_array(G)]
    one = [1] * n
    v = list(one)
    W = [Fraction(n)]
    for _ in range(7):
        v = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
        W.append(Fraction(sum(v)))
    # matching numbers m1..m3 by brute force on edges
    E = list(G.edges())
    m = len(E)
    m2 = m3 = 0
    for i in range(m):
        for j in range(i + 1, m):
            if set(E[i]) & set(E[j]):
                continue
            m2 += 1
            for k in range(j + 1, m):
                if not (set(E[i]) & set(E[k]) or set(E[j]) & set(E[k])):
                    m3 += 1
    return W, m2, m3


def solve_rational(M, b):
    """Solve M x = b over Q via fraction Gauss elimination.
    Returns solution or None."""
    n_r, n_c = len(M), len(M[0])
    A = [row[:] + [b[i]] for i, row in enumerate(M)]
    piv = []
    r = 0
    for c in range(n_c):
        p = next((i for i in range(r, n_r) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        for i in range(n_r):
            if i != r and A[i][c] != 0:
                f = A[i][c] / A[r][c]
                A[i] = [A[i][j] - f * A[r][j] for j in range(n_c + 1)]
        piv.append(c)
        r += 1
        if r == n_r:
            break
    for i in range(r, n_r):
        if A[i][n_c] != 0:
            return None
    x = [Fraction(0)] * n_c
    for i, c in enumerate(piv):
        x[c] = A[i][n_c] / A[i][c]
    return x


def monomial_basis(W):
    """All monomials in W1..W5 with index-weight <= 5 (universal candidates)."""
    W1, W2, W3, W4, W5 = W[1], W[2], W[3], W[4], W[5]
    return [
        ("1", Fraction(1)),
        ("W1", W1),
        ("W2", W2), ("W1^2", W1 ** 2),
        ("W3", W3), ("W1*W2", W1 * W2), ("W1^3", W1 ** 3),
        ("W4", W4), ("W1*W3", W1 * W3), ("W2^2", W2 ** 2),
        ("W1^2*W2", W1 ** 2 * W2), ("W1^4", W1 ** 4),
        ("W5", W5), ("W1*W4", W1 * W4), ("W2*W3", W2 * W3),
        ("W1^2*W3", W1 ** 2 * W3), ("W1*W2^2", W1 * W2 ** 2),
        ("W1^3*W2", W1 ** 3 * W2), ("W1^5", W1 ** 5),
    ]


def main():
    train, test = [], []
    for n in range(7, 15):
        for G in nx.nonisomorphic_trees(n):
            W, m2, m3 = tree_data(G)
            feats = [v for _, v in monomial_basis(W)]
            (train if n <= 12 else test).append((feats, Fraction(m3)))
        print(f"n={n} done", flush=True)
    feat_names = [k for k, _ in monomial_basis([Fraction(0)] * 8)]

    x = solve_rational([f for f, _ in train], [t for _, t in train])
    if x is None:
        print("NO exact rational fit with this basis (need wider basis)")
        return
    print("fit coefficients:")
    for name, c in zip(feat_names, x):
        print(f"  {name:6s} {c}")
    bad = sum(1 for f, t in train + test
              if sum(c * fi for c, fi in zip(x, f)) != t)
    print(f"verification: {bad} mismatches out of {len(train)+len(test)} trees "
          f"(train n<=12, test n=13,14)")


if __name__ == "__main__":
    main()
