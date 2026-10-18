"""Master dataset specification: every tunable constant, the bucket rules, the label generators, and the image decorations.

Pure Python (no images), so the plan and its distribution can be reviewed before anything is rendered.

LABEL CONVENTION: the label is the CHARACTERS ONLY (digits and letters, case kept). Brackets, dots, spaces and a "Q." / "Q" question
prefix are drawn in the image but stripped from the label, exactly like the annotation workbook's `Normalize` column (Q.3) -> 3, 2a) -> 2a).

Buckets (priority order), all defined on the label:
  B1 NNa                    two-digit number + one letter, character-balanced (12a, 47k, 05z)
  B2 single digit / letter  0-9 and a-z (i, v, x count as Roman and are covered by B3)
  B3 legacy floor           NN alone, Roman numerals i-x in both cases, Na
Image styles (what is drawn around the label): plain, trailing mark, wrapped, Q-prefix, interior marks (NN)a, NN.a, NN(a) ...), square brackets.
"""
import random
import re

# ---- Synthetic volume cap (hard constraint, per LABEL) ---------------------------------------------------------------
CAP_MULTIPLIER = 5       # synthetic crops of a label <= CAP_MULTIPLIER * its real count ...
CAP_FLOOR = 20           # ... but never less than this, so labels with no/few real crops can still be seen

# ---- Bucket sizes and per-label targets (synthetic crops are added until real + synthetic reaches the target) -------
NNA_BUDGET = 3000            # B1: plain+decorated NNa crops, balanced by character; with MARKED_NNA_BUDGET the NNa total is ~5,000
LEADING_ZERO_SHARE = 0.10    # share of numbers written 0N (02a); 0.10 makes digit 0 as frequent as 1-9 overall
SINGLE_DIGIT_TARGET = 450    # B2: total crops (real + synthetic) wanted per digit
SINGLE_LETTER_TARGET = 150   # B2: total crops wanted per lowercase letter a-z (except i, v, x)
NN_TARGET = 25               # B3: total crops wanted per two-digit number 10-99
ROMAN_TARGET = 100           # B3: total crops wanted per Roman numeral i-x, each case
NA_BUDGET = 400              # B3: synthetic N+letter crops (7a, 2k), character-balanced
MARKED_NNA_BUDGET = 2000     # extra NNa crops forced to carry interior/trailing marks: NN)a NN.a NNa) NN(a) (NN)a (NNa) ...
SQUARE_BUDGET = 200          # extra NNa/NN crops with [ ] in any position (token presence only)
SPARSE_REAL = 10             # a B1-B3 label with fewer real crops than this is flagged in the report

# ---- Image decoration (probability of each style on an ordinary crop; the rest are plain) ----------------------------
Q_PREFIX_SHARE = 0.30        # "Q.3" "Q16" "Q 4": the question-number prefix seen in the newer documents (was 0.15)
Q_REAL_REPEAT = 5            # each real training crop that has a Q prefix in the workbook is used this many times in total
NEWDOC_REPEAT = 5            # each newly labelled new-document crop (data/roi_labels.csv, holdout=False) is used this many times in total
TRAILING_SHARE = 0.18        # 3)  3.  3.)
WRAPPED_SHARE = 0.08         # (3)
INTERIOR_DOT_SHARE = 0.04    # 2.a  (only for 2+ character labels)
SQUARE_SHARE = 0.02          # [3]

LETTERS = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
ROMAN_CHARS = "ivxIVX"

# Templates for the forced-mark crops. NN = two digits, a = letter. Weight order = priority order.
MARKED_PATTERNS = [("NN)a", 3), ("NN.a", 3), ("NNa)", 3), ("NNa.", 3), ("NN(a)", 2), ("(NN)a", 2), ("(NNa)", 2), ("NN(a", 1), ("(NNa", 1)]
SQUARE_PATTERNS = [("[NN]a", 2), ("NN[a]", 2), ("[NNa]", 2), ("[NNa", 1), ("NNa]", 1), ("NN]a", 1), ("NN[a", 1), ("[NN", 1)]

BUCKETS = ["B1 NNa", "B2 single digit", "B2 single letter", "B3 NN alone", "B3 Roman", "B3 Na", "other"]
STYLES = ["plain", "trailing mark", "wrapped", "Q prefix", "interior mark", "square bracket", "NNa marks (forced)"]


def strip_marks(text):
    return re.sub(r"[^0-9A-Za-z]", "", str(text))


def roman(n):
    out = ""
    for v, r in ((10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while n >= v:
            out, n = out + r, n - v
    return out


def bucket_of(label):
    if re.fullmatch(r"\d\d[A-Za-z]", label):
        return "B1 NNa"
    if re.fullmatch(r"\d", label):
        return "B2 single digit"
    if re.fullmatch(r"[ivxIVX]+", label):
        return "B3 Roman"
    if re.fullmatch(r"[A-Za-z]", label):
        return "B2 single letter"
    if re.fullmatch(r"\d\d", label):
        return "B3 NN alone"
    if re.fullmatch(r"\d[A-Za-z]", label):
        return "B3 Na"
    return "other"


def cap(label, real_counts):
    return max(CAP_MULTIPLIER * real_counts.get(label, 0), CAP_FLOOR)


def balanced(rng, items, n):
    """n draws where every item appears floor(n/len) or ceil(n/len) times, in random order."""
    out = []
    while len(out) < n:
        block = list(items)
        rng.shuffle(block)
        out += block
    out = out[:n]
    rng.shuffle(out)
    return out


def numbers(rng, n):
    """n two-digit numbers (strings). Units uniform 0-9; tens uniform 1-9 except LEADING_ZERO_SHARE written 0N."""
    zeros = round(n * LEADING_ZERO_SHARE)
    tens = ["0"] * zeros + balanced(rng, DIGITS[1:], n - zeros)
    rng.shuffle(tens)
    return [t + u for t, u in zip(tens, balanced(rng, DIGITS, n))]


def templated(rng, n, templates):
    """n strings from templates over character-balanced numbers and letters (marks included)."""
    nums, lets = numbers(rng, n), balanced(rng, LETTERS, n)
    names, weights = zip(*templates)
    return [t.replace("NN", nums[i]).replace("a", lets[i]) for i, t in enumerate(rng.choices(names, weights, k=n))]


def decorate(label, rng):
    """-> (text drawn in the image, style). The label itself is unchanged."""
    r = rng.random()
    if r < Q_PREFIX_SHARE:
        tail = rng.choice(["", "", ")", ".", ".)"]) if rng.random() < 0.4 else ""
        return rng.choice(["Q.", "Q.", "Q", "Q ", "Q-"]) + label + tail, "Q prefix"
    r -= Q_PREFIX_SHARE
    if r < TRAILING_SHARE:
        return label + rng.choice([")", ".", ".)", ")", "."]), "trailing mark"
    r -= TRAILING_SHARE
    if r < WRAPPED_SHARE:
        return f"({label})", "wrapped"
    r -= WRAPPED_SHARE
    if r < INTERIOR_DOT_SHARE and len(label) >= 2:
        i = rng.randint(1, len(label) - 1)
        return label[:i] + "." + label[i:], "interior mark"
    r -= INTERIOR_DOT_SHARE
    if r < SQUARE_SHARE:
        return f"[{label}]", "square bracket"
    return label, "plain"


def plan_synthetic(real_counts, seed=0):
    """-> (rows, notes, truncated). rows = [(label, drawn_text, style)] after the per-label cap; notes = per dense label cap info.

    Dense labels (digits, letters, NN, Roman) are topped up to their TARGET, limited by the cap. Tail buckets (NNa, Na, marked NNa,
    square brackets) are sampled by character-balanced budget, then capped per label. Counts are per LABEL (the characters only).
    """
    rng = random.Random(seed)
    rows, notes = [], []

    def dense(label, target):
        want = max(0, target - real_counts.get(label, 0))
        got = min(want, cap(label, real_counts))
        for _ in range(got):
            rows.append((label, *decorate(label, rng)))
        notes.append({"label": label, "bucket": bucket_of(label), "real": real_counts.get(label, 0), "wanted": want, "granted": got,
                      "capped": got < want})

    for d in DIGITS:
        dense(d, SINGLE_DIGIT_TARGET)
    for l in LETTERS:
        if l not in "ivx":
            dense(l, SINGLE_LETTER_TARGET)
    for n in range(10, 100):
        dense(str(n), NN_TARGET)
    for n in range(1, 11):
        dense(roman(n), ROMAN_TARGET)
        dense(roman(n).upper(), ROMAN_TARGET)

    tail = []                                                 # (label, drawn text, style)
    for s in templated(rng, NNA_BUDGET, [("NNa", 1)]):
        tail.append((s, *decorate(s, rng)))
    na = [d + l for d, l in zip(balanced(rng, DIGITS, NA_BUDGET), balanced(rng, LETTERS, NA_BUDGET))]
    for s in na:
        tail.append((s, *decorate(s, rng)))
    for s in templated(rng, MARKED_NNA_BUDGET, MARKED_PATTERNS):
        tail.append((strip_marks(s), s, "NNa marks (forced)"))
    for s in templated(rng, SQUARE_BUDGET, SQUARE_PATTERNS):
        tail.append((strip_marks(s), s, "square bracket"))
    used = {}
    for row in rows:                                          # dense rows already obey the cap; tail rows share the same per-label budget
        used[row[0]] = used.get(row[0], 0) + 1
    truncated = 0
    for row in tail:
        if used.get(row[0], 0) < cap(row[0], real_counts):
            used[row[0]] = used.get(row[0], 0) + 1
            rows.append(row)
        else:
            truncated += 1
    random.Random(seed + 1).shuffle(rows)
    return rows, notes, truncated
