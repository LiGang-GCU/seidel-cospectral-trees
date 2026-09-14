"""Stage B': DL/DQ spectral descriptors for QSPR — 28 alkanes bp benchmark.

Same protocol as the pencil-descriptor study (qspr_bp.py):
  - 28 alkanes C5-C8 with verified boiling points
  - single-descriptor Pearson r ranking
  - nested-LOO Q2 for 2-descriptor models (selection inside the loop)
  - permutation test (y-label shuffling) for the best single descriptor
    and for nested Q2, 10,000 permutations

Descriptor families compared:
  classic : the 9 classical descriptors (W, M1, M2, R, E_A, LEL, E_Q, Estrada, lam1)
  dldq    : DL/DQ spectral descriptors (eigenvalue moments, distance
            algebraic connectivity, Wiener index, transmission stats)
  any     : union

DL = diag(tr) - D (distance Laplacian), DQ = diag(tr) + D (distance signless).
On trees n<=18 both spectra are blind-pair-free (matrix census, 2026-09-04).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "spectral-curve-research" / "code"))
sys.path.insert(0, str(_HERE))  # local dir takes precedence

from molecular_qspr import build_tree, classic_descriptors, pearson, loo_q2, OCTANES  # noqa: E402
from qspr_bp import EXTRA_ALKANES, OCTANE_BP, ALK_SPEC, CLASSIC  # noqa: E402
from matrix_census import dist_based  # noqa: E402
import networkx as nx  # noqa: E402

RESULTS = _HERE.parent / "results"
RNG = np.random.default_rng(20260904)
NPERM = 10000


def dldq_descriptors(A):
    """DL/DQ spectral descriptors from adjacency A (numpy)."""
    n = A.shape[0]
    G = nx.from_numpy_array(A)
    Dm = dist_based(G, "dist")
    tr = Dm.sum(1)
    DL = np.diag(tr) - Dm
    DQ = np.diag(tr) + Dm
    evL = np.linalg.eigvalsh(DL)
    evQ = np.linalg.eigvalsh(DQ)
    out = {
        # distance-Laplacian family
        "dl_m2": float((evL ** 2).sum()),          # second moment
        "dl_m3": float((evL ** 3).sum()),
        "dl_a2": float(np.sort(evL)[1]),           # distance algebraic connectivity
        "dl_lmax": float(evL.max()),
        "dl_gap": float(evL.max() - np.sort(evL)[1]),
        # distance-signless family
        "dq_m2": float((evQ ** 2).sum()),
        "dq_lmax": float(evQ.max()),               # DQ Perron root
        "dq_m3": float((evQ ** 3).sum()),
        # transmission / Wiener family
        "wiener": float(tr.sum() / 2.0),           # Wiener index
        "tr_max": float(tr.max()),
        "tr_var": float(tr.var()),
    }
    return out


def press_loo_q2(X2, y):
    """Exact LOO Q2 via PRESS (hat matrix), no refitting. X2: (n,k)."""
    n = len(y)
    Xd = np.column_stack([np.ones(n), X2])
    XtX = Xd.T @ Xd
    try:
        H = Xd @ np.linalg.solve(XtX, Xd.T)
    except np.linalg.LinAlgError:
        return -np.inf
    h = np.clip(np.diag(H), 0, 1 - 1e-12)
    resid = y - H @ y
    press = ((resid / (1 - h)) ** 2).sum()
    return 1 - press / ((y - y.mean()) ** 2).sum()


def nested_loo_q2(X, pool, y):
    from itertools import combinations
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, bool)
        mask[i] = False
        best_q, best_pair = -9e9, None
        for a, b in combinations(pool, 2):
            q = press_loo_q2(np.column_stack([X[a][mask], X[b][mask]]), y[mask])
            if q > best_q:
                best_q, best_pair = q, (a, b)
        a, b = best_pair
        Xd = np.column_stack([np.ones(mask.sum()), X[a][mask], X[b][mask]])
        beta, *_ = np.linalg.lstsq(Xd, y[mask], rcond=None)
        preds[i] = beta[0] + beta[1] * X[a][i] + beta[2] * X[b][i]
    return 1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    alk_items = []
    for name, spec, bp in EXTRA_ALKANES:
        alk_items.append((name, build_tree(*spec), bp))
    for name, bp in OCTANE_BP.items():
        alk_items.append((name, build_tree(*ALK_SPEC[name]), bp))
    print(f"dataset: {len(alk_items)} alkanes", flush=True)

    rows = []
    for name, A, bp in alk_items:
        d = classic_descriptors(A)
        d.update(dldq_descriptors(A))
        rows.append({"name": name, "bp": bp, "desc": d})
    X = {k: np.array([r["desc"][k] for r in rows]) for k in rows[0]["desc"]}
    y = np.array([r["bp"] for r in rows])
    DLDQ = [k for k in X if k not in CLASSIC]

    # ---- single-descriptor ranking ----
    tab = sorted(((k, pearson(X[k], y)) for k in X), key=lambda kv: -abs(kv[1]))
    print("\n== single-descriptor |r| with bp (top 10) ==")
    for k, r in tab[:10]:
        print(f"  {k:12s} r={r:+.4f}  [{'DLDQ' if k in DLDQ else 'classic'}]")

    # ---- nested LOO ----
    res = {}
    for grp, pool in (("classic", CLASSIC), ("dldq", DLDQ), ("any", list(X))):
        res[grp] = nested_loo_q2(X, pool, y)
        print(f"[nested-LOO] {grp:7s} Q2={res[grp]:.4f}", flush=True)

    out = {
        "n_compounds": len(rows),
        "single_top10": [(k, float(r)) for k, r in tab[:10]],
        "nested_loo_q2": {k: float(v) for k, v in res.items()},
    }
    dest = RESULTS / "dldq_qspr_bp.json"
    dest.write_text(json.dumps(out, indent=1))  # intermediate save

    # ---- permutation test ----
    NQ2 = 300
    k_best, r_best = tab[0]
    print(f"\n[perm] best single descriptor: {k_best} r={r_best:+.4f}; "
          f"running {NPERM} permutations...", flush=True)
    t0 = time.time()
    cnt_r = cnt_q = 0
    q_any = res["any"]
    for p in range(NPERM):
        yp = RNG.permutation(y)
        rabs = max(abs(pearson(X[k], yp)) for k in X)
        if rabs >= abs(r_best):
            cnt_r += 1
        if p < NQ2:  # Q2 permutation is expensive: subsample
            if nested_loo_q2(X, list(X), yp) >= q_any:
                cnt_q += 1
        if (p + 1) % 2500 == 0:
            print(f"  perm {p+1}/{NPERM} ({time.time()-t0:.0f}s)", flush=True)
    p_r = (1 + cnt_r) / (1 + NPERM)
    p_q = (1 + cnt_q) / (1 + NQ2)
    print(f"[perm] single |r|: p={p_r:.5f}  nested-LOO any Q2: "
          f"p={p_q:.5f} ({NQ2} perms)", flush=True)

    out["perm_test"] = {"best_single": [k_best, float(r_best)],
                        "p_single_r": float(p_r), "n_perm_r": NPERM,
                        "p_nested_q2_any": float(p_q), "n_perm_q2": NQ2}
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
