"""Correct the stale false claim in results/nist_cospectral_nonane_pairs.json.

Round-1 finding W3 established that cospectral mates share every
eigenvalue-determined descriptor but NOT the total walk counts
1^T A^k 1, which also depend on the main-eigenvalue weights. The
manuscript was fixed; this supplemental artifact still carried the old
"ALL spectral/walk descriptors" wording, so a reader of the artifact
alone would be misled. Rewrite the note and the interpretation to the
corrected statement.

The artifact's nonane boiling-range figures disagree with the
manuscript's (390.4/424.1 K with 2,2,3,3-tetramethylpentane here versus
395.4/423.8 K with 2,2,4,4-tetramethylpentane in the text). This machine
has no network access, so neither can be checked against NIST. Rather
than silently picking one, the disagreement is recorded in the artifact
as an explicit open item.
"""
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "results" / "nist_cospectral_nonane_pairs.json"

NOTE = (
    "NIST WebBook phase-change data for four of the five cospectral "
    "alkane-skeleton pairs (all C9H20 nonanes); the fifth pair "
    "(2,3,3-trimethylhexane / 3-ethyl-2,2-dimethylpentane) has no "
    "comparably evaluated NIST data and is omitted here. Within each "
    "pair the adjacency spectrum is identical, hence so is every "
    "descriptor determined by the eigenvalues alone: the characteristic "
    "and matching polynomials, the closed-walk counts tr(A^k), and the "
    "Hosoya index Z. This does NOT extend to the total walk counts "
    "W_k = 1^T A^k 1, which depend in addition on the main-eigenvalue "
    "weights; in fact all five cospectral nonane classes have different "
    "total walk counts, diverging at k=3 or k=4 (see "
    "results/cospectral_nonanes_exact.json). Boiling points nevertheless "
    "differ by up to ~6 K."
)

INTERP = (
    "Pair 2 shows a ~6 K boiling-point difference that no "
    "eigenvalue-determined descriptor of the carbon skeleton can "
    "capture, since both skeletons share the identical characteristic "
    "polynomial; pairs 1/3/4 differ by 1-2 K. Note that the total walk "
    "counts are not eigenvalue-determined and do separate these pairs."
)

OPEN_ITEM = (
    "The nonane_bp_range_K figures below disagree with the manuscript "
    "(paper/sections/6_chemistry.tex), which quotes 395.4 +- 0.7 K for "
    "2,2,4,4-tetramethylpentane to 423.8 +- 0.3 K for n-nonane. Both the "
    "endpoints and the low-boiling compound differ. This was recorded, "
    "not resolved: the machine that produced this file has no network "
    "access, so neither figure could be re-checked against NIST SRD 69. "
    "Verify against the WebBook before submission and make the two agree."
)

d = json.load(io.open(PATH, encoding="utf-8"))
d["note"] = NOTE
d["context"]["interpretation"] = INTERP
d["context"]["open_item_bp_range_disagreement"] = OPEN_ITEM

io.open(PATH, "w", encoding="utf-8", newline="\n").write(
    json.dumps(d, indent=1, ensure_ascii=False) + "\n")

chk = json.load(io.open(PATH, encoding="utf-8"))
assert "ALL spectral/walk descriptors" not in json.dumps(chk)
assert len(chk["pairs"]) == 4, len(chk["pairs"])
print("corrected:", PATH.name)
print("pairs retained:", len(chk["pairs"]))
