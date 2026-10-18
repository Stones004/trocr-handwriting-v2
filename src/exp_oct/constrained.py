"""Constrained decoding for PARSeq: block illegal characters while the model chooses each character (no retraining).

    modes:  plain    - the model's own greedy decoding (control; reproduces model_io.decode)
            charset  - only 0-9 a-j may be chosen (any order, any length up to max_label_length), then end-of-text
            grammar  - charset + shape \\d{1,2}[a-j]?|[a-j]  (a letter ends the label; at most 2 digits; not empty), max 3 characters

At every autoregressive step the logits of forbidden tokens are set to -inf before the argmax, and the chosen token updates a small state machine.
The refinement pass of PARSeq (1 iteration) is kept: it re-reads the AR output with full context, and its logits are then masked and decoded left to
right with the same state machine, so the final string always obeys the mode. Confidence = probability of the chosen token after masking and
renormalising over the allowed tokens (so it is NOT comparable with the plain confidence).
"""
import torch

DIGITS, LETTERS = "0123456789", "abcdefghij"


def _tables(tok, mode, device, head_size):
    V = head_size                                 # the output head has no bos/pad rows
    d = [tok._stoi[c] for c in DIGITS]
    l = [tok._stoi[c] for c in LETTERS]
    eos = tok.eos_id
    if mode == "charset":
        S = 3                                     # 0 start, 1 running, 2 done
        allowed = torch.zeros(S, V, dtype=torch.bool)
        allowed[0, d + l] = True
        allowed[1, d + l + [eos]] = True
        allowed[2, eos] = True
        nxt = torch.zeros(S, V, dtype=torch.long)
        nxt[0, :] = 1
        nxt[1, :] = 1
        nxt[1, eos] = 2
        nxt[2, :] = 2
    else:                                         # grammar
        S = 5                                     # 0 start, 1 one digit, 2 two digits, 3 after letter, 4 done
        allowed = torch.zeros(S, V, dtype=torch.bool)
        allowed[0, d + l] = True
        allowed[1, d + l + [eos]] = True
        allowed[2, l + [eos]] = True
        allowed[3, eos] = True
        allowed[4, eos] = True
        nxt = torch.full((S, V), 4, dtype=torch.long)
        nxt[0, d], nxt[0, l] = 1, 3
        nxt[1, d], nxt[1, l], nxt[1, eos] = 2, 3, 4
        nxt[2, l], nxt[2, eos] = 3, 4
        nxt[3, :] = 4
        nxt[4, :] = 4
    return allowed.to(device), nxt.to(device)


def _masked_argmax(logits, state, allowed, nxt):
    masked = logits.masked_fill(~allowed[state], float("-inf"))
    tok = masked.argmax(-1)
    return masked, tok, nxt[state, tok]


@torch.no_grad()
def forward(model, images, mode="grammar"):
    """Returns probs (N, T, V) over the allowed tokens, ready for model.tokenizer.decode."""
    tok, m = model.tokenizer, model.model
    allowed, nxt = _tables(tok, mode, images.device, m.head.out_features)
    max_length = 3 if mode == "grammar" else m.max_label_length
    num_steps, bs = max_length + 1, images.shape[0]
    memory = m.encode(images)
    pos_queries = m.pos_queries[:, :num_steps].expand(bs, -1, -1)
    tgt_mask = query_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=images.device), 1)
    tgt_in = torch.full((bs, num_steps), tok.pad_id, dtype=torch.long, device=images.device)
    tgt_in[:, 0] = tok.bos_id
    state = torch.zeros(bs, dtype=torch.long, device=images.device)
    logits = []
    for i in range(num_steps):
        j = i + 1
        out = m.decode(tgt_in[:, :j], memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j], tgt_query_mask=query_mask[i:j, :j])
        p_i = m.head(out)
        masked, t, state = _masked_argmax(p_i.squeeze(1), state, allowed, nxt)
        logits.append(masked.unsqueeze(1))
        if j < num_steps:
            tgt_in[:, j] = t
    logits = torch.cat(logits, dim=1)
    if m.refine_iters:
        query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=images.device), 2)] = 0
        bos = torch.full((bs, 1), tok.bos_id, dtype=torch.long, device=images.device)
        for _ in range(m.refine_iters):
            tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
            pad_mask = (tgt_in == tok.eos_id).int().cumsum(-1) > 0
            out = m.decode(tgt_in, memory, tgt_mask, pad_mask, pos_queries, query_mask[:, : tgt_in.shape[1]])
            raw = m.head(out)
            state = torch.zeros(bs, dtype=torch.long, device=images.device)
            cols = []
            for i in range(num_steps):                 # left-to-right so the final string obeys the state machine
                masked, _, state = _masked_argmax(raw[:, i], state, allowed, nxt)
                cols.append(masked.unsqueeze(1))
            logits = torch.cat(cols, dim=1)
    return logits.softmax(-1)


def decode(model, x, mode="grammar"):
    """Same return as model_io.decode: [(prediction, min_conf, mean_conf)] per image."""
    if mode == "plain":
        from model_io import decode as plain
        return plain(model, x)
    preds, char_probs = model.tokenizer.decode(forward(model, x, mode))
    return [(p.strip(), pr.min().item() if len(pr) else 0.0, pr.mean().item() if len(pr) else 0.0) for p, pr in zip(preds, char_probs)]
