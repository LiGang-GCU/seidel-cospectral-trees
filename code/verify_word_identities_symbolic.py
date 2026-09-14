"""Word-by-word symbolic verification of the Seidel moment identities (3.3)/(3.4).

For k = 3, 4 this script enumerates all 3^k words over {J, -I, -2A},
reduces each word by the PROVED evaluation rule (eq. (2.3) / eq:evalrule),

    tr(A^{a0} J A^{a1} J ... J A^{aj}) = W_{a0+aj} * prod_{i=1}^{j-1} W_{ai},
    tr(A^a) = C_a,          (W_0 = C_0 = n),

groups words by their reduced {J,A}-skeleton up to cyclic rotation, and
collects the exact polynomial in n, W_j, C_j.  The collected polynomial is
compared term by term against the identities (3.3)/(3.4) as printed in
sections/3_reduction.tex (targets transcribed below, and additionally
cross-checked against the LaTeX source).  A word-class contribution table
is emitted as a LaTeX fragment for the appendix.

Outputs:
  results/word_identities_symbolic.json   (machine-readable audit record)
  ../paper/sections/app_words.tex         (appendix LaTeX fragment)
"""
import itertools
import json
import re
import sys
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
KMAX = 4
n = sp.symbols("n")
W = [n] + list(sp.symbols(f"W1:{KMAX+1}"))      # W0 = n
C = [n] + list(sp.symbols(f"C1:{KMAX+1}"))      # C0 = n


# ---------------------------------------------------------------- reduction
def reduce_word(word):
    """word: tuple over 'JIA'.  Return (skeleton, coef) where skeleton is a
    tuple over 'JA' (I's deleted) and coef = (-1)^{#I} (-2)^{#A}."""
    coef = sp.Integer(1)
    skel = []
    for ch in word:
        if ch == "I":
            coef *= -1
        elif ch == "A":
            coef *= -2
            skel.append("A")
        else:
            skel.append("J")
    return tuple(skel), coef


def canonical(skel):
    """Canonical representative under cyclic rotation."""
    if not skel:
        return skel
    rots = [skel[i:] + skel[:i] for i in range(len(skel))]
    return min(rots)


def eval_skeleton(skel):
    """Apply the proved evaluation rule to a {J,A}-skeleton."""
    if not skel:                       # empty word: tr(I) = n
        return n
    runs, jays, cur = [], 0, 0
    for ch in skel:
        if ch == "A":
            cur += 1
        else:
            runs.append(cur)
            cur = 0
            jays += 1
    runs.append(cur)
    if jays == 0:
        return C[runs[0]]
    expr = W[runs[0] + runs[-1]]
    for i in range(1, jays):
        expr *= W[runs[i]]
    return expr


def orbit_size(skel):
    if not skel:
        return 1
    return len({skel[i:] + skel[:i] for i in range(len(skel))})


def expand_moment(k):
    """Return (poly, classes) with classes a list of dicts for the table."""
    poly = sp.Integer(0)
    groups = {}
    for word in itertools.product("JIA", repeat=k):
        skel, coef = reduce_word(word)
        key = canonical(skel)
        groups.setdefault(key, {"coef_sum": sp.Integer(0), "count": 0})
        groups[key]["coef_sum"] += coef
        groups[key]["count"] += 1
    classes = []
    for key in sorted(groups, key=lambda s: (len(s), s)):
        g = groups[key]
        val = eval_skeleton(key)
        contrib = sp.expand(g["coef_sum"] * val)
        poly += contrib
        classes.append({
            "skeleton": "".join(key) if key else "(empty)",
            "kp": len(key),
            "orbit": orbit_size(key),
            "n_words": g["count"],          # = orbit * C(k, k')
            "coef_sum": int(g["coef_sum"]),  # signed coefficient sum
            "value": str(val),
            "contribution": str(contrib),
        })
    return sp.expand(poly), classes


# ------------------------------------------- targets, transcribed from paper
W1, W2, W3 = W[1], W[2], W[3]
C1, C2, C3, C4 = C[1], C[2], C[3], C[4]
TARGET = {
    2: 4 * C1 + 4 * C2 - 4 * W1 + n**2 - n,
    3: (n**3 - 3 * n**2 + 2 * n + (12 - 6 * n) * W1 + 12 * W2
        - 6 * C1 - 12 * C2 - 8 * C3),
    4: (n**4 - 4 * n**3 + 6 * n**2 - 3 * n - (8 * n**2 - 24 * n + 24) * W1
        + 8 * W1**2 + (16 * n - 48) * W2 - 32 * W3
        + 8 * C1 + 24 * C2 + 32 * C3 + 16 * C4),
}

# tree specializations, Lemma 3.2(iii),(iv)
TREE_TARGET = {
    3: 12 * W2 + n**3 - 15 * n**2 + 14 * n,
    4: -32 * W3 + 16 * (n - 1) * W2 + n**4 - 20 * n**3 + 102 * n**2 - 147 * n + 64,
}


def tree_subs(expr):
    return sp.expand(expr.subs({
        C1: 0, C3: 0,
        C2: 2 * (n - 1),
        C4: 2 * W2 - 2 * (n - 1),
        W1: 2 * (n - 1)}))


# ------------------------------------------------------------------- run
report = {"script": "verify_word_identities_symbolic.py", "checks": []}
all_classes = {}
ok_all = True
for k in (2, 3, 4):
    poly, classes = expand_moment(k)
    all_classes[k] = classes
    diff = sp.expand(poly - TARGET[k])
    ok = (diff == 0)
    ok_all &= ok
    report["checks"].append({
        "k": k, "identity_matches_paper": ok,
        "collected": str(poly), "target": str(sp.expand(TARGET[k])),
        "n_words": sum(c["n_words"] for c in classes),
        "n_classes": len(classes)})
    print(f"k={k}: {sum(c['n_words'] for c in classes)} words, "
          f"{len(classes)} cyclic classes, match={ok}")
    if not ok:
        print("  DIFF:", diff)

for k in (3, 4):
    poly, _ = expand_moment(k)
    t = tree_subs(poly)
    ok = sp.expand(t - TREE_TARGET[k]) == 0
    ok_all &= ok
    report["checks"].append({
        "k": k, "tree_specialization_matches_lemma": ok,
        "tree_form": str(t)})
    print(f"tree k={k}: {t}  match={ok}")

# sanity: total word counts are 3^k
for k in (2, 3, 4):
    assert sum(c["n_words"] for c in all_classes[k]) == 3**k

# cross-check targets against the LaTeX source of section 3
tex = (ROOT / "paper" / "sections" / "3_reduction.tex").read_text(
    encoding="utf-8")
m3 = re.search(r"m_3\(S\)&=(.*?),?\s*\\label\{eq:mthree\}", tex, re.S)
m4 = re.search(r"m_4\(S\)&=(.*?)\\label\{eq:mfour\}", tex, re.S)
latex_ok = True
for match, terms in (
    (m3, ["n^3-3n^2+2n", "(12-6n)\\Wc_1", "12\\Wc_2",
          "-6\\Cc_1", "-12\\Cc_2", "-8\\Cc_3"]),
    (m4, ["n^4-4n^3+6n^2-3n", "(8n^2-24n+24)\\Wc_1", "8\\Wc_1^2",
          "(16n-48)\\Wc_2", "-32\\Wc_3", "8\\Cc_1", "24\\Cc_2",
          "32\\Cc_3", "16\\Cc_4"]),
):
    body = match.group(1).replace(" ", "").replace("\n", "")
    for t in terms:
        if t.replace(" ", "") not in body:
            latex_ok = False
            print("LATEX TERM MISSING:", t)
report["latex_crosscheck_ok"] = latex_ok
ok_all &= latex_ok
print("LaTeX cross-check:", latex_ok)

# ------------------------------------------------------- LaTeX appendix
def skel_latex(s):
    if s == "(empty)":
        return r"$\varnothing$"
    out, i = [], 0
    while i < len(s):
        if s[i] == "J":
            out.append("J")
            i += 1
        else:
            j = i
            while j < len(s) and s[j] == "A":
                j += 1
            a = j - i
            out.append("A" if a == 1 else f"A^{{{a}}}")
            i = j
    return "$" + "".join(out) + "$"


lines = [
    r"\begin{table}[htbp]", r"\centering", r"\small",
    r"\caption{Word-class expansions of $m_3(S)$ and $m_4(S)$.  All $3^k$"
    r" words over $\{J,-I,-2A\}$ are grouped by their reduced"
    r" $\{J,A\}$-skeleton up to cyclic rotation; `words' counts the words in"
    r" the class, `coef' the signed coefficient sum, and `value' the trace"
    r" given by the evaluation rule \eqref{eq:evalrule}.  The contributions"
    r" sum to \eqref{eq:mthree} and \eqref{eq:mfour} respectively. Generated"
    r" by \texttt{verify\_word\_identities\_symbolic.py}.}",
    r"\label{tab:words}",
    r"\begin{tabular}{lrrll}", r"\toprule",
    r"skeleton & words & coef & value & contribution\\",
    r"\midrule",
    r"\multicolumn{5}{l}{\emph{$k=3$}}\\",
    r"\midrule",
]
for k in (3, 4):
    if k == 4:
        lines += [r"\midrule", r"\multicolumn{5}{l}{\emph{$k=4$}}\\",
                  r"\midrule"]
    for c in all_classes[k]:
        contrib = c["contribution"].replace("**", "^").replace("*", "")
        val = c["value"].replace("**", "^").replace("*", "")
        # use \Wc/\Cc notation
        val_tex = re.sub(r"W(\d)", r"\\Wc_{\1}", val)
        val_tex = re.sub(r"C(\d)", r"\\Cc_{\1}", val_tex)
        con_tex = re.sub(r"W(\d)", r"\\Wc_{\1}", contrib)
        con_tex = re.sub(r"C(\d)", r"\\Cc_{\1}", con_tex)
        lines.append(
            f"{skel_latex(c['skeleton'])} & {c['n_words']} & "
            f"{c['coef_sum']} & ${val_tex}$ & ${con_tex}$\\\\")
    if k == 3:
        pass  # separator already added by the k=4 header block
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

app = ROOT / "paper" / "sections" / "app_words.tex"
app.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", app)

(HERE / "results").mkdir(exist_ok=True)
with open(HERE / "results" / "word_identities_symbolic.json", "w",
          encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print("ALL OK" if ok_all else "FAILURES PRESENT")
sys.exit(0 if ok_all else 1)
