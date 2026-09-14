"""Add protocol metadata to the chemistry census artifacts (reviewer W7).

The n<=22 walk-census JSONs already carry a `protocol` block describing the
exact-modular-filter / exact-integer-verdict design; the chem_boundary_n*.json
files did not, so a reader could not tell from the artifact alone that the
reported classes come from exact integer arithmetic rather than the modular
filter. This adds the same block. Numeric content is untouched.
"""
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROTOCOL = {
    "design": "two-layer: exact-modular filter, exact-integer verdict",
    "filter": {
   "quantity": "walk tuple (W_0,...,W_{n-1})",
        "arithmetic": "exact 64-bit integer, reduced modulo the two moduli",
        "moduli": [16777213, 16777211],
        "note": ("Fingerprints can only false-MERGE, never false-SPLIT: equal "
      "integers have equal residues modulo any fixed integer, so "
           "primality of the moduli is irrelevant to correctness. "
         "16777213 is prime; 16777211 = 11 * 101 * 15101 is not."),
    },
    "verdict": {
        "arithmetic": ("Python big integers; integral characteristic "
             "polynomials by Faddeev-LeVerrier"),
        "note": ("Every flagged group is recomputed exactly; the reported "
     "classes are formed from those exact values alone. No "
      "floating-point eigenvalue enters the pipeline."),
    },
}

for path in sorted((ROOT / "results").glob("chem_boundary_n*.json")):
    d = json.load(io.open(path, encoding="utf-8"))
    if d.get("protocol") == PROTOCOL:
        print("unchanged (already tagged):", path.name)
        continue
    keys_before = set(d)
    d["protocol"] = PROTOCOL
    assert set(d) - keys_before == {"protocol"}, "unexpected key change"
    io.open(path, "w", encoding="utf-8", newline="\n").write(
        json.dumps(d, indent=1) + "\n")
    print("tagged:", path.name)
