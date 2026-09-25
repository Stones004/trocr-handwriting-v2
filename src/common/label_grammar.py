"""The closed label grammar this project trains on: a digit run with an optional trailing subpart letter, a
Roman numeral, or a bare subpart letter, optionally wrapped in parens and/or with a trailing '.'/')'/'.)'.

The pattern itself lives in config.yaml (labels.clean_pattern) so it can be changed without editing code; this
module compiles it. It is the single source of truth, shared between:
- prepare_dataset.py, which uses it to route non-conforming training rows to freetext.csv (rather than
  training on them, since they fall outside the vocabulary the model is meant to learn), and
- inference (predict.py, infer_parseq.py), which use it to flag predictions that don't look like a valid
  label for this task -- a real, if rare, failure mode: PARSeq is not literally constrained to only output
  grammar-conforming strings, so a low-quality or out-of-distribution crop can occasionally produce something the
  model was never trained to output. This is a different, complementary signal to confidence -- a prediction can be
  low-confidence and still grammar-conforming, or high-confidence and still grammar-violating (an overconfident
  wrong prediction won't be caught by a confidence threshold, but might be caught by this).
"""
from project_config import label_regex

CLEAN_PATTERN = label_regex("clean_pattern")


def matches_grammar(text):
    return bool(CLEAN_PATTERN.match(str(text).strip()))
