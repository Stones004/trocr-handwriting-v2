"""
Vocabulary-constrained decoding + confidence scoring for TrOCR generate() calls.

Usage:
    from constraints import get_logits_processor, transition_confidence

    logits_processor = get_logits_processor(processor)
    outputs = model.generate(
        pixel_values,
        num_beams=4,
        max_new_tokens=16,
        logits_processor=logits_processor,
        output_scores=True,
        return_dict_in_generate=True,
    )
    prediction = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    conf = transition_confidence(model, outputs)[0]   # per-sample confidence dict
"""

import torch
from transformers import LogitsProcessor, LogitsProcessorList

# Derived from the actual clean-labeled subset of the dataset (train+val+test,
# after stripping Excel artifacts and excluding free-text rows via
# prepare_dataset.py's CLEAN_PATTERN) -- NOT a guess. If you re-derive this
# after future data additions, regenerate from data/processed/{train,val,test}.csv
# rather than hand-editing, since this list is only as good as the sample it
# came from.
DEFAULT_VOCAB_CHARS = list("()0123456789ABIVXabdiv")


def build_allowed_token_ids(tokenizer, vocab_chars=None):
    """
    Character-level fallback: map each character in vocab_chars to its
    token id(s). ONLY reliable if every character in your grammar is
    guaranteed to tokenize as its own token in isolation -- byte-level BPE
    often merges repeated/common character sequences (e.g. "II", "III")
    into a single multi-character token, which this function cannot see
    since it only ever encodes one character at a time. If your labels
    contain repeated-letter sequences (Roman numerals, doubled letters),
    use build_allowed_token_ids_from_labels instead -- it derives the
    allowed set from how your real labels actually tokenize, which is
    correct by construction.
    """
    vocab_chars = vocab_chars or DEFAULT_VOCAB_CHARS
    allowed = set()
    unmapped = []

    for ch in vocab_chars:
        found = False
        for candidate in (ch, "Ġ" + ch, "▁" + ch):
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                allowed.add(ids[0])
                found = True
        if not found:
            unmapped.append(ch)

    if unmapped:
        print(
            "[constraints] WARNING: these vocab chars did not map to a "
            f"single token and are NOT enforced: {unmapped}"
        )

    return _add_special_ids(tokenizer, allowed)


def build_allowed_token_ids_from_labels(tokenizer, labels):
    """
    Derive the allowed token set by tokenizing your ACTUAL labels the same
    way they were tokenized during fine-tuning (dataset.py / train.py both
    call tokenizer(text, ...) directly with no character-by-character
    assumption). Whatever tokens appear across your real label set are
    exactly the tokens the model learned to predict -- this sidesteps the
    single-character/BPE-merge mismatch entirely, by construction.

    labels: an iterable of label strings (e.g. a pandas Series/column).
    """
    allowed = set()
    for text in labels:
        ids = tokenizer(str(text), add_special_tokens=False).input_ids
        allowed.update(ids)
    return _add_special_ids(tokenizer, allowed)


def _add_special_ids(tokenizer, allowed):
    for special_id in (
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
    ):
        if special_id is not None:
            allowed.add(special_id)
    return allowed


class VocabConstrainedLogitsProcessor(LogitsProcessor):
    """
    Flat vocabulary mask: at every decode step, zeroes out (sets to -inf)
    the probability of any token not in allowed_token_ids. This enforces
    "every generated character is in my vocabulary" but NOT sequence-level
    grammar (e.g. it won't stop '-' from being followed by ')' instead of
    '>'). Swap this for a stateful/WFST-driven processor later if you need
    that -- same __call__ signature, just make the mask depend on input_ids.
    """

    def __init__(self, allowed_token_ids):
        self.allowed = torch.tensor(sorted(allowed_token_ids), dtype=torch.long)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        allowed = self.allowed.to(scores.device)
        mask[:, allowed] = scores[:, allowed]
        return mask


def get_logits_processor(processor, vocab_chars=None):
    """Character-guess variant -- see build_allowed_token_ids caveat."""
    allowed = build_allowed_token_ids(processor.tokenizer, vocab_chars)
    return LogitsProcessorList([VocabConstrainedLogitsProcessor(allowed)])


def get_logits_processor_from_csvs(processor, csv_paths, text_column="text"):
    """
    Preferred constructor: builds the mask from real label data.
    csv_paths: e.g. ["data/processed/train.csv", "data/processed/val.csv"]
    """
    import pandas as pd

    labels = []
    for path in csv_paths:
        labels.extend(pd.read_csv(path)[text_column].astype(str).tolist())

    allowed = build_allowed_token_ids_from_labels(processor.tokenizer, labels)
    return LogitsProcessorList([VocabConstrainedLogitsProcessor(allowed)])


def transition_confidence(model, outputs, temperature=1.0):
    """
    Post-mask confidence per sample, correct for any num_beams, using HF's
    own beam-bookkeeping (model.compute_transition_scores). This answers
    "how confident was the model in the tokens it actually produced, among
    the allowed set" -- NOT "was the grammar overriding a strong raw guess"
    (see raw_margin_confidence below for that, greedy-only).

    temperature: divides raw logits before softmax. temperature=1.0 is a
    no-op (unscaled). A fitted temperature > 1.0 (see fit_temperature)
    flattens overconfident softmax outputs so the reported probability
    better tracks actual accuracy -- it does NOT change which token wins
    argmax, so your predictions/exact-match accuracy are unaffected by
    this; only the confidence numbers used for review-flagging change.

    Returns a list of dicts, one per sample in the batch:
        {
            "seq_min_prob": float,   # weakest single-token confidence in the sequence
            "seq_mean_prob": float,  # average per-token confidence
            "token_probs": [float, ...],
        }
    """
    scaled_scores = tuple(s / temperature for s in outputs.scores)

    transition_scores = model.compute_transition_scores(
        outputs.sequences,
        scaled_scores,
        getattr(outputs, "beam_indices", None),
        normalize_logits=True,
    )  # log-probs, shape (batch, generated_len), padded with large negative for finished seqs

    results = []
    for row in transition_scores:
        probs = torch.exp(row)
        probs = probs[probs > 1e-6]  # drop padding positions past EOS
        if len(probs) == 0:
            results.append({"seq_min_prob": 0.0, "seq_mean_prob": 0.0, "token_probs": []})
            continue
        results.append(
            {
                "seq_min_prob": probs.min().item(),
                "seq_mean_prob": probs.mean().item(),
                "token_probs": probs.tolist(),
            }
        )
    return results


def fit_temperature(model, processor, csv_path, device, max_iter=100):
    """
    Fits a single temperature scalar via teacher-forced forward passes on
    a held-out CSV (use val.csv -- never train.csv, temperature fit on
    training data would just re-learn the model's own overconfidence).

    Standard approach (Guo et al. 2017): freeze all model weights, run
    each sample through the model with the TRUE label fed as decoder
    input (teacher forcing -- same computation as a validation-loss
    pass), collect the raw per-token logits and true next-token ids
    across the whole val set, then optimize T (a single number) to
    minimize the negative log-likelihood of the true tokens under
    temperature-scaled softmax. This treats each token position as an
    independent classification instance, which is the standard
    simplification for calibrating seq2seq/token-level models.

    Returns the fitted T (float). Print and hardcode this into your
    evaluate.py / error_analysis.py --temperature flag once fitted --
    this function is meant to be run once, not on every eval.
    """
    import pandas as pd
    from PIL import Image
    from dataset import resize_with_padding, get_target_size

    df = pd.read_csv(csv_path)
    model.eval()
    target_size = get_target_size(processor)

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for _, row in df.iterrows():
            image = Image.open(row["image_path"]).convert("RGB")
            image = resize_with_padding(image, target_size=target_size)
            pixel_values = processor(
                images=image, return_tensors="pt"
            ).pixel_values.to(device)
            labels = processor.tokenizer(
                str(row["text"]), return_tensors="pt"
            ).input_ids.to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            # outputs.logits: (1, seq_len, vocab_size) -- one row per
            # generated position, teacher-forced against the true label.
            all_logits.append(outputs.logits.squeeze(0).cpu())
            all_targets.append(labels.squeeze(0).cpu())

    logits = torch.cat(all_logits, dim=0)    # (total_tokens_across_val, vocab)
    targets = torch.cat(all_targets, dim=0)  # (total_tokens_across_val,)

    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / T, targets)
        loss.backward()
        return loss

    optimizer.step(closure)

    fitted_T = T.item()
    print(f"[fit_temperature] Fitted T = {fitted_T:.3f} "
          f"(on {len(targets)} tokens from {csv_path})")
    print("[fit_temperature] Pass this to --temperature in evaluate.py / "
          "error_analysis.py, e.g.:")
    print(f"  python src/error_analysis.py --temperature {fitted_T:.3f}")

    return fitted_T


class TrieNode:
    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children = {}
        self.is_end = False


def build_label_trie(tokenizer, labels):
    """
    Builds a trie over the token-id sequences of every unique label string
    seen. Since your full label space is a small closed set (tens of
    distinct strings, not open-ended text), this lets decoding be
    restricted to producing EXACTLY one of those known strings -- nothing
    else is reachable, no matter how many decode steps run.
    """
    root = TrieNode()
    for text in sorted(set(str(l) for l in labels)):
        ids = tokenizer(text, add_special_tokens=False).input_ids
        node = root
        for tid in ids:
            node = node.children.setdefault(tid, TrieNode())
        node.is_end = True
    return root


class ClosedVocabularyLogitsProcessor(LogitsProcessor):
    """
    Restricts generation to exactly one of a fixed, closed set of known
    valid label strings (built by build_label_trie from every unique
    label in your train+val data).

    Unlike a flat character/token mask, this also bounds SEQUENCE
    STRUCTURE and LENGTH -- it's what actually stops runaway repetition
    like 'iii' -> 'iiii': once the trie has walked to the end of a known
    string, only EOS is a legal next token. There's no path to a 4th 'i'
    that was never in your recorded label set.

    State is derived FRESH from input_ids on every call (walking the trie
    from the root using each row's own generated-so-far sequence) rather
    than stored in mutable instance state across calls. This makes it
    safe under beam search: each row of input_ids always reflects that
    specific beam's own history, so there's no cross-beam bookkeeping to
    get wrong, unlike a processor that accumulates a flat list of
    per-call side info (see TrackingVocabConstrainedLogitsProcessor's
    beams>1 caveat above -- this design avoids that problem for the mask
    itself, though see TrackingClosedVocabularyLogitsProcessor below for
    the remaining caveat on margin *reporting* under beam search).
    """

    def __init__(self, trie_root, decoder_start_token_id, eos_token_id):
        self.root = trie_root
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id

    def _allowed_for_row(self, generated_ids):
        node = self.root
        for tid in generated_ids[1:]:  # skip the leading decoder-start token
            tid = int(tid)
            if tid == self.eos_token_id:
                break
            child = node.children.get(tid)
            if child is None:
                # Fell off the trie -- shouldn't happen if this mask was
                # applied from step 1, but fail safe rather than crash.
                return {self.eos_token_id}
            node = child
        allowed = set(node.children.keys())
        if node.is_end:
            allowed.add(self.eos_token_id)
        return allowed

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for i in range(input_ids.size(0)):
            allowed = self._allowed_for_row(input_ids[i].tolist())
            allowed_t = torch.tensor(
                list(allowed), dtype=torch.long, device=scores.device
            )
            mask[i, allowed_t] = scores[i, allowed_t]
        return mask


class TrackingClosedVocabularyLogitsProcessor(ClosedVocabularyLogitsProcessor):
    """
    Same trie-based masking, plus records the model's RAW (pre-mask)
    top-2 tokens/probs at every step -- same "which candidate almost
    won" diagnostic as TrackingVocabConstrainedLogitsProcessor.

    Same caveat applies: only trust step_records / raw_margin_confidence
    with num_beams=1. The mask itself is beam-safe (state comes from
    input_ids, not stored history), but attributing "this margin belongs
    to the sequence that ultimately won" requires knowing which beam
    survived at every step, which isn't tracked here. For a beams=1
    review pass this is exact.
    """

    def __init__(self, trie_root, decoder_start_token_id, eos_token_id):
        super().__init__(trie_root, decoder_start_token_id, eos_token_id)
        self.step_records = []

    def reset(self):
        self.step_records = []

    def __call__(self, input_ids, scores):
        raw_probs = torch.softmax(scores, dim=-1)
        top2 = torch.topk(raw_probs[0], k=2)
        top1_id = top2.indices[0].item()
        top1_prob = top2.values[0].item()
        top2_prob = top2.values[1].item()
        allowed = self._allowed_for_row(input_ids[0].tolist())
        was_overridden = top1_id not in allowed
        self.step_records.append((top1_prob, top2_prob, was_overridden))
        return super().__call__(input_ids, scores)

    def raw_margin_confidence(self):
        if not self.step_records:
            return {"min_top1": 0.0, "min_margin": 0.0, "any_overridden": False}
        top1s = [r[0] for r in self.step_records]
        margins = [r[0] - r[1] for r in self.step_records]
        overridden = any(r[2] for r in self.step_records)
        return {
            "min_top1": min(top1s),
            "min_margin": min(margins),
            "any_overridden": overridden,
        }


def get_closed_vocab_processor(processor, model, csv_paths, text_column="text"):
    """
    Preferred constructor for a closed label space: builds the trie from
    every unique label across csv_paths, using model.config for the
    decoder-start/EOS token ids that were actually set during training
    (configure_trocr in train.py), not re-derived independently.

    csv_paths: e.g. ["data/processed/train.csv", "data/processed/val.csv"]
    """
    import pandas as pd

    labels = []
    for path in csv_paths:
        labels.extend(pd.read_csv(path)[text_column].astype(str).tolist())

    trie = build_label_trie(processor.tokenizer, labels)
    return LogitsProcessorList(
        [
            ClosedVocabularyLogitsProcessor(
                trie,
                model.config.decoder_start_token_id,
                model.config.eos_token_id,
            )
        ]
    )


def get_tracking_closed_vocab_processor(processor, model, csv_paths, text_column="text"):
    """Same as get_closed_vocab_processor, but returns the raw tracking
    processor (not wrapped in a list) so you can call .reset() and
    .raw_margin_confidence() on it between samples. Use with beams=1."""
    import pandas as pd

    labels = []
    for path in csv_paths:
        labels.extend(pd.read_csv(path)[text_column].astype(str).tolist())

    trie = build_label_trie(processor.tokenizer, labels)
    return TrackingClosedVocabularyLogitsProcessor(
        trie,
        model.config.decoder_start_token_id,
        model.config.eos_token_id,
    )
    """
    Same masking behavior as VocabConstrainedLogitsProcessor, but also records
    the model's RAW (pre-mask) top-2 tokens/probs at every step it's called.

    Only trust this with num_beams=1 (greedy). With beam search, this processor
    is called once per beam per step and the beams get reordered/pruned by
    generate() afterward, so the recorded steps don't line up 1:1 with the
    beam that ends up winning. Use it as your "flag for review" pass, separate
    from your accuracy-focused beam-search generate() call.
    """

    def __init__(self, allowed_token_ids):
        super().__init__(allowed_token_ids)
        self.step_records = []  # list of (raw_top1_prob, raw_top2_prob, was_overridden)

    def reset(self):
        self.step_records = []

    def __call__(self, input_ids, scores):
        raw_probs = torch.softmax(scores, dim=-1)
        top2 = torch.topk(raw_probs[0], k=2)
        top1_id = top2.indices[0].item()
        top1_prob = top2.values[0].item()
        top2_prob = top2.values[1].item()
        was_overridden = top1_id not in set(self.allowed.tolist())
        self.step_records.append((top1_prob, top2_prob, was_overridden))
        return super().__call__(input_ids, scores)

    def raw_margin_confidence(self):
        """
        Summary over the recorded steps:
            min_top1: lowest raw top-1 confidence seen (low = model was
                      generally unsure, regardless of grammar)
            min_margin: smallest (top1 - top2) gap seen (low = a close call
                        between two candidates, e.g. your 6/b case)
            any_overridden: True if the grammar mask ever discarded the
                             model's actual top pick in favor of a legal
                             but lower-confidence token
        """
        if not self.step_records:
            return {"min_top1": 0.0, "min_margin": 0.0, "any_overridden": False}
        top1s = [r[0] for r in self.step_records]
        margins = [r[0] - r[1] for r in self.step_records]
        overridden = any(r[2] for r in self.step_records)
        return {
            "min_top1": min(top1s),
            "min_margin": min(margins),
            "any_overridden": overridden,
        }