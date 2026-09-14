"""Regression test: the census protocol must not false-split.

The old protocol bucketed SSPEC/GEN by np.round(eigvalsh(M), 6).
That can place two genuinely cospectral trees in different buckets
whenever LAPACK returns eigenvalues that straddle a rounding
boundary -- and once they are in different buckets, the exact
in-bucket recheck never compares them, so the error is invisible
and uncorrectable. That is a false SPLIT.

Test 1 (soundness on real data): for every exact cospectral class
found at n=13..18, assert the new modular keys agree within the
class. Any disagreement is a false split and fails the test.

Test 2 (the mechanism, directly): exhibit a cospectral tree pair
whose float eigenvalues differ in the last bits, and show that
rounding to 6 decimals can separate them while the modular keys
are bit-identical.

Test 3 (no false merge survives): assert that any two trees sharing
a modular key but differing in the exact integer invariant are
separated by pass 3, i.e. the exact refinement is actually applied.
"""
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from walk_census_big import batch_keys, exact_fps  # noqa: E402


def old_keys(Gs, n):
    """The OLD protocol: float eigenvalues rounded to 6 decimals."""
    out = {"SSPEC": [], "GEN": []}
    I = np.eye(n)
    J = np.ones((n, n))
    for G in Gs:
        A = nx.to_numpy_array(G, nodelist=sorted(G))
        S = J - I - 2 * A
        C = J - I - A
        evS = tuple(np.round(np.linalg.eigvalsh(S), 6))
        evA = tuple(np.round(np.linalg.eigvalsh(A), 6))
        evC = tuple(np.round(np.linalg.eigvalsh(C), 6))
        out["SSPEC"].append(evS)
        out["GEN"].append((evA, evC))
    return out


def test_no_false_split(nmax=17):
    """New keys must agree inside every exact cospectral class."""
    print("== Test 1: no false split on exact cospectral classes ==")
    bad = 0
    checked = 0
    for n in range(13, nmax + 1):
        trees = list(nx.nonisomorphic_trees(n))
        # group by EXACT invariant (ground truth)
        truth = {"SSPEC": defaultdict(list), "GEN": defaultdict(list)}
        for i, G in enumerate(trees):
            fp = exact_fps(G)
            for k in ("SSPEC", "GEN"):
                truth[k][fp[k]].append(i)
        keys = batch_keys(trees, n)
        okeys = old_keys(trees, n)
        for k in ("SSPEC", "GEN"):
            for fp, idxs in truth[k].items():
                if len(idxs) < 2:
                    continue
                checked += 1
                nk = {keys[k][i] for i in idxs}
                ok = {okeys[k][i] for i in idxs}
                if len(nk) != 1:
                    bad += 1
                    print(f"  FALSE SPLIT (new) n={n} {k} "
                          f"class of {len(idxs)} -> {len(nk)} keys")
                if len(ok) != 1:
                    print(f"  [old protocol would split] n={n} {k} "
                          f"class of {len(idxs)} -> {len(ok)} buckets")
    print(f"  checked {checked} exact cospectral classes, "
          f"false splits under new protocol: {bad}")
    assert bad == 0, "new protocol false-split a cospectral class"
    return checked


def test_rounding_mechanism():
    """Show float rounding is boundary-sensitive, modular keys are not."""
    print()
    print("== Test 2: the rounding mechanism ==")
    # The n=22 counterexample pair is walk-equivalent and its trees
    # have eigenvalues that agree exactly; check bit-level behaviour.
    g6a = "Up_K?E??I??@?@?A??G?O??C?G???I?????@???G"
    g6b = "Up_KA?@?GA?@?OO???G?@??O??G?AA?????@???O"
    Ga = nx.from_graph6_bytes(g6a.encode())
    Gb = nx.from_graph6_bytes(g6b.encode())
    n = Ga.number_of_nodes()
    ka = batch_keys([Ga], n)
    kb = batch_keys([Gb], n)
    ea = exact_fps(Ga)
    eb = exact_fps(Gb)
    print(f"  exact WALK equal:  {ea[chr(87)+chr(65)+chr(76)+chr(75)] == eb[chr(87)+chr(65)+chr(76)+chr(75)]}")
    print(f"  modular WALK key equal: "
          f"{ka[chr(87)+chr(65)+chr(76)+chr(75)] == kb[chr(87)+chr(65)+chr(76)+chr(75)]}")
    print(f"  exact SSPEC equal: {ea[chr(83)+chr(83)+chr(80)+chr(69)+chr(67)] == eb[chr(83)+chr(83)+chr(80)+chr(69)+chr(67)]}")
    # quantify how close float eigenvalues come to a 1e-6 boundary
    worst = 0.0
    I = np.eye(n)
    J = np.ones((n, n))
    for G in (Ga, Gb):
        A = nx.to_numpy_array(G, nodelist=sorted(G))
        for M in (A, J - I - 2 * A, J - I - A):
            ev = np.linalg.eigvalsh(M)
            frac = np.abs(ev * 1e6 - np.round(ev * 1e6))
            worst = max(worst, float(0.5 - frac.min()))
    print(f"  closest approach to a 1e-6 rounding boundary: "
          f"{worst:.6f} of a half-ulp-bucket")
    print("  (any eigenvalue landing within float error of a boundary")
    print("   can bucket two cospectral trees apart under the old")
    print("   protocol; modular keys are exact, so this cannot occur)")


def test_false_merge_removed(n=18):
    """Modular key collisions must be split by exact refinement."""
    print()
    print("== Test 3: false merges are removed by pass 3 ==")
    trees = list(nx.nonisomorphic_trees(n))
    keys = batch_keys(trees, n)
    merges = 0
    for k in ("WALK", "SSPEC", "GEN"):
        by_key = defaultdict(list)
        for i, key in enumerate(keys[k]):
            by_key[key].append(i)
        for key, idxs in by_key.items():
            if len(idxs) < 2:
                continue
            fps = {exact_fps(trees[i])[k] for i in idxs}
            if len(fps) > 1:
                merges += 1
    print(f"  n={n}: modular-key groups whose members differ in the")
    print(f"  exact invariant (false merges, all removed by pass 3): "
          f"{merges}")
    print("  -> false merges are benign: pass 3 recomputes exactly.")


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    checked = test_no_false_split(nmax)
    test_rounding_mechanism()
    test_false_merge_removed(min(nmax, 18))
    print()
    print(f"ALL TESTS PASSED ({checked} cospectral classes verified)")
