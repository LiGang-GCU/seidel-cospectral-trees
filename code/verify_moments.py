"""Exact verification of the Seidel moment identities used in Lemma 3.2
(word-expansion rungs) on all trees of order n <= 12.

Checks:
  (i)   C_4 = 2 W_2 - 2(n-1)                       (tree identity)
  (ii)  tr(S^3) - 12 W_2 depends only on n         (so the Seidel spectrum
                                                    determines W_2 on trees)
  (iii) tr(S^4) - 16 C_4 + 32 W_3 depends only on (n, W_2)
        (so the Seidel spectrum determines W_3 on trees, given rung (ii))
"""
from collections import defaultdict

import networkx as nx
import numpy as np


def traces(T):
    n = T.number_of_nodes()
    A = nx.to_numpy_array(T, dtype=object)
    S = np.ones((n, n), dtype=object) - np.eye(n, dtype=object) - 2 * A
    A2 = A @ A
    A3 = A2 @ A
    A4 = A3 @ A
    S2 = S @ S
    S3 = S2 @ S
    S4 = S3 @ S
    tr = lambda M: int(sum(M[i][i] for i in range(n)))
    W = lambda M: int(sum(sum(M[i][j] for j in range(n)) for i in range(n)))
    return n, tr(A2), tr(A4), W(A), W(A2), W(A3), tr(S3), tr(S4)


def main():
    res3 = defaultdict(set)
    res4 = defaultdict(set)
    total = 0
    for n in range(2, 13):
        for T in nx.nonisomorphic_trees(n):
            total += 1
            n_, C2, C4, W1, W2, W3, mS3, mS4 = traces(T)
            assert C4 == 2 * W2 - 2 * (n - 1), (n, "C4 identity failed")
            res3[n].add(mS3 - 12 * W2)
            res4[(n, W2)].add(mS4 - 16 * C4 + 32 * W3)
    bad3 = {n: v for n, v in res3.items() if len(v) != 1}
    bad4 = {k: v for k, v in res4.items() if len(v) != 1}
    print(f"trees checked: {total}")
    print("(i)   C4 = 2 W2 - 2(n-1): OK (asserted for every tree)")
    print("(ii)  tr(S^3) - 12 W2 depends only on n:",
          "OK" if not bad3 else f"FAILED {bad3}")
    print("(iii) tr(S^4) - 16 C4 + 32 W3 depends only on (n, W2):",
          "OK" if not bad4 else f"FAILED {dict(list(bad4.items())[:3])}")
    assert not bad3 and not bad4
    # Additional check: verify the explicit polynomial for tr(S^3).
    # Correct formula (Lemma 3.2(iii)): tr(S^3) = 12*W2 + n^3 - 15*n^2 + 14*n
    bad3_explicit = {}
    for n_val, vals in res3.items():
        expected = n_val**3 - 15*n_val**2 + 14*n_val
        actual_set = vals  # each element is tr(S^3) - 12*W2 for some tree of order n_val
        if not all(v == expected for v in actual_set):
            bad3_explicit[n_val] = (expected, actual_set)
    print("(iv)  tr(S^3) == 12*W2 + n^3 - 15*n^2 + 14*n (explicit formula):",
          "OK" if not bad3_explicit
          else f"FAILED {bad3_explicit}")
    assert not bad3_explicit, f"Explicit tr(S^3) formula failed: {bad3_explicit}"
    print("ALL MOMENT IDENTITY CHECKS PASSED")


if __name__ == "__main__":
    main()
