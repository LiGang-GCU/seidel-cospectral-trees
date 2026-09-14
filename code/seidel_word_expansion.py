"""Symbolic word expansion of Seidel moments m_k(S) = tr((J-I-2A)^k).

Rule (proved): for a word A^{a0} J A^{a1} J ... J A^{aj} with j>=1 J's,
    tr = W_{a0+aj} * prod_{i=1}^{j-1} W_{ai},   (W_0 := n)
and tr(A^a) = C_a (C_0 := n). Letters: J (coef 1), I (coef -1), A (coef -2).
I-letters are deleted (shortening the word).

Outputs m_2, m_3, m_4 as exact polynomials in n, W_j, C_j; then the tree
specializations (C1=C3=0, C2=2(n-1), C4=2W2-2(n-1), W1=2(n-1)) giving
Lemma 3.2(iii),(iv) explicitly. Finally verifies numerically against direct
integer tr(S^k) on all trees up to 12 vertices.
"""
import itertools
import sys
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent
KMAX = 4
n = sp.symbols("n")
W = [n] + list(sp.symbols(f"W1:{KMAX+1}"))      # W0 = n
C = [n] + list(sp.symbols(f"C1:{KMAX+1}"))      # C0 = n


def trace_word(word):
    """word: tuple of letters in {'J','I','A'}. Return sympy expr."""
    coef = sp.Integer(1)
    runs = []          # lengths of A-runs
    jays = 0
    cur = 0
    for ch in word:
        if ch == "I":
            coef *= -1
        elif ch == "A":
            coef *= -2
            cur += 1
        else:  # J
            runs.append(cur)
            cur = 0
            jays += 1
    runs.append(cur)
    if jays == 0:
        return coef * C[runs[0]]
    expr = W[runs[0] + runs[-1]]
    for i in range(1, jays):
        expr *= W[runs[i]]
    return coef * expr


def moment(k):
    return sp.expand(sum(trace_word(w)
                         for w in itertools.product("JIA", repeat=k)))


def tree_subs(expr):
    W1 = W[1]
    return sp.expand(expr.subs({
        C[1]: 0, C[3]: 0,
        C[2]: 2 * (n - 1),
        C[4]: 2 * W[2] - 2 * (n - 1),
        W1: 2 * (n - 1)}))


for k in (2, 3, 4):
    mk = moment(k)
    print(f"m_{k}(S) = {mk}")
    print(f"   on trees: {tree_subs(mk)}\n")

# ---- numeric verification on all trees up to 12 vertices ----
import networkx as nx
import numpy as np

sys.path.insert(0, str(HERE))


def exact_moments(G, k=KMAX):
    nn = G.number_of_nodes()
    A = nx.to_numpy_array(G).astype(object)
    S = np.ones((nn, nn), dtype=object) - np.eye(nn, dtype=object) - 2 * A
    wk = [nn]
    v = np.ones(nn, dtype=object)
    for _ in range(k):
        v = A @ v
        wk.append(int(v.sum()))
    ck = [nn]
    P = np.eye(nn, dtype=object)
    for _ in range(k):
        P = A @ P
        ck.append(int(np.trace(P)))
    mk = [nn]
    Q = np.eye(nn, dtype=object)
    for _ in range(k):
        Q = S @ Q
        mk.append(int(np.trace(Q)))
    return wk, ck, mk


bad = 0
tested = 0
mom = {k: moment(k) for k in (2, 3, 4)}
for nn_ in range(2, 13):
    for G in nx.nonisomorphic_trees(nn_):
        tested += 1
        wk, ck, mk = exact_moments(G)
        env = {n: nn_}
        env.update({W[i]: wk[i] for i in range(1, KMAX + 1)})
        env.update({C[i]: ck[i] for i in range(1, KMAX + 1)})
        for k in (2, 3, 4):
            val = int(mom[k].subs(env))
            if val != mk[k]:
                bad += 1
                print("MISMATCH", nn_, k, val, mk[k])
print(f"numeric check: {tested} trees, mismatches={bad}")
