# Seidel and Generalized Cospectrality on Trees — Code and Data

Exhaustive computational census of three spectral equivalence relations on
unlabeled trees, together with the exact certificates behind the reported
boundary results.

## What is here

This repository contains the enumeration code, the raw result files, and the
verification scripts. Every number that appears in the accompanying write-up
is reproducible from these files.

```
code/       enumeration, census, and verification scripts
results/    raw per-order JSON output (89 files)
litcheck/   literature-search tables used for prior-art checking
figures/    generated figures (reproducible via code/make_figures.py)
docs/       framework notes and the GNN separation experiment log
```

## Main computational results

| Result | Range | Evidence |
|---|---|---|
| Seidel cospectrality and generalized cospectrality induce **identical** partitions of trees | all 9,114,285 unlabeled trees on n ≤ 22 | `results/exact_census_n{2..22}.json` |
| Walk counts determine the adjacency spectrum of a tree | n ≤ 21 | `results/exact_census_n{2..21}.json` |
| Least order of a walk-equivalent, non-cospectral tree pair | n = 22 (unique class, 2 members) | `results/n22_counterexample.json` |
| Walk-determination on alkane skeletons (trees with Δ ≤ 4) | 24,029,256 skeletons, 4 ≤ n ≤ 24 | `results/chem_boundary_n{4..24}.json` |
| Unique chemical failure: one C23H48 pair; determination holds again at n = 24 | n = 23 | `results/chem_boundary_n23.json`, `results/c23_allk_walk_witness.json` |
| Among the 463 nontrivial Seidel collision classes at n = 17, 18, **no** class is switching-equivalent | n = 17, 18 | `results/seidel_mechanism.json` |

## Two-layer computational protocol

All census scripts separate filtering from adjudication, so that no reported
partition ever depends on floating-point arithmetic:

- **Filter layer** — computes fingerprints in exact modular arithmetic, modulo
  two coprime 24-bit integers: the prime `16777213` and the composite
  `16777211 = 11 × 101 × 15101`. This layer only decides *which trees are
  worth comparing exactly*.
- **Verdict layer** — inside every flagged group, recomputes big-integer walk
  tuples and integral characteristic polynomials via Faddeev–LeVerrier, and
  re-forms the blocks from those exact values. This layer alone produces the
  reported counts.

Each result file records its own protocol string under the `protocol` /
`arithmetic` keys, so the guarantee is auditable per file.

## Reproducing the census

```bash
pip install networkx sympy numpy

# Exact census of all three relations, one order at a time
python code/exact_census.py --n 12

# Walk-relation census (big-integer verdict layer)
python code/walk_census_big.py --n 22

# Alkane-skeleton (Δ ≤ 4) boundary census
python code/exp3_chem_boundary.py --n 23
```

Tree generation follows the constant-time free-tree algorithm of
Wright, Richmond, Odlyzko and McKay (*SIAM J. Comput.* **15** (1986) 540–548);
per-order skeleton counts were cross-checked against OEIS
[A000055](https://oeis.org/A000055) (all trees) and
[A000602](https://oeis.org/A000602) (alkane skeletons) as an independent
correctness check on the generator.

## Independent verification scripts

| Script | Checks |
|---|---|
| `verify_c23_allk.py` | walk-count agreement of the C23H48 pair for 0 ≤ k ≤ 200, beyond the order-46 common recurrence bound, hence for all k |
| `verify_moments.py` | walk-moment identities on every nontrivial tree from n = 2 to 12 (986 trees) |
| `verify_word_identities_symbolic.py` | symbolic Seidel word-expansion identities |
| `verify_a000602_total.py` | per-order alkane skeleton counts against OEIS A000602 |
| `audit_cospectral_nonanes.py` | exact big-integer walk counts of the five cospectral nonane pairs |
| `test_census_protocol.py` | filter/verdict layer separation invariants |

## Notes on the data files

- Graphs are stored in **graph6** format (McKay).
- Characteristic polynomials are exact integer coefficient arrays, leading
  coefficient first.
- Walk tuples are exact big integers serialized as JSON strings where they
  exceed 2^53.
- Boiling-point data in `results/nist_cospectral_nonane_pairs.json` is quoted
  from the NIST Chemistry WebBook; each entry carries its own `tboil_note`
  recording whether the value is an average or a single evaluated measurement.

## License

Code and generated data are released under the MIT License (see `LICENSE`).
Literature-search tables in `litcheck/` contain bibliographic metadata about
third-party publications and are provided for prior-art auditing only.
