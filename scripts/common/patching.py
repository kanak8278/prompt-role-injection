"""Activation capture and exact replacement via forward hooks (protocol §8, §10).

§8 requires stating exactly which tensor is patched. Here it is the **output residual stream
of a decoder block**, i.e. what `model.model.layers[i]` returns, before it becomes the input
of block i+1. That is deliberately not `out.hidden_states[i]`, which is the block's *input*,
and not `hidden_states[n_layers]`, which is post-final-norm and therefore not a residual at
all.

Requirements this module enforces rather than assumes:
  - `use_cache=False` on every patched forward, so downstream activations are recomputed
    rather than served from a cache built before the intervention (§8);
  - a no-op hook must not change dtype, precision, masking or model state (§8) -- so the
    replacement is cast to the destination dtype and written in place on a clone;
  - hook state is cleared per example (§8);
  - absolute token positions only. §10 warns that a shifted command must not invalidate
    index-based comparisons, so callers pass spans from the aligned renderer.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch


def _blocks(model):
    return model.model.layers


def _split(output):
    """Decoder blocks may return a tensor or a tuple; normalise both directions."""
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _rejoin(hidden, rest):
    return hidden if rest is None else (hidden, *rest)


@contextlib.contextmanager
def capture(model, layers):
    """Capture post-block residuals for `layers`. Yields a dict filled on each forward."""
    store: dict[int, torch.Tensor] = {}
    handles = []

    def mk(i):
        def hook(_m, _i, output):
            h, _ = _split(output)
            store[i] = h.detach().clone()
            return output
        return hook

    blk = _blocks(model)
    for i in layers:
        handles.append(blk[i].register_forward_hook(mk(i)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()
        store.clear()


@contextlib.contextmanager
def patch(model, spec):
    """Replace post-block residuals at given (layer, positions) with donor activations.

    `spec` maps layer index -> (positions, donor) where `donor` is [1, seq, d_model] (the full
    donor sequence) or [len(positions), d_model]. Dtype/device are taken from the destination
    so a patch cannot silently change precision.

    `spec` may be empty, which installs hooks that write nothing -- the no-op control.
    """
    handles = []

    def mk(i, positions, donor):
        idx = torch.as_tensor(positions, dtype=torch.long)

        def hook(_m, _i, output):
            h, rest = _split(output)
            src = donor
            if src.dim() == 3:
                src = src[0]
            src = src.to(device=h.device, dtype=h.dtype)
            if src.shape[0] != len(idx):
                src = src[idx.to(src.device)]
            new = h.clone()
            new[0, idx.to(new.device), :] = src
            return _rejoin(new, rest)
        return hook

    blk = _blocks(model)
    for i, (positions, donor) in spec.items():
        handles.append(blk[i].register_forward_hook(mk(i, positions, donor)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def noop_hooks(model, layers):
    """Hooks that read and return the output untouched. §8's no-op control: must be bitwise
    identical to no hooks at all."""
    handles = []

    def hook(_m, _i, output):
        return output

    blk = _blocks(model)
    for i in layers:
        handles.append(blk[i].register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def steer(model, layers, vector, positions, alpha: float):
    """Additive steering at given positions: h += alpha * vector.

    `alpha=0.0` is the zero-strength control G4 requires and must reproduce baseline exactly.
    """
    handles = []

    def mk(i):
        idx = torch.as_tensor(positions, dtype=torch.long)

        def hook(_m, _i, output):
            h, rest = _split(output)
            if alpha == 0.0:
                return output
            v = vector.to(device=h.device, dtype=h.dtype)
            new = h.clone()
            new[0, idx.to(new.device), :] += alpha * v
            return _rejoin(new, rest)
        return hook

    blk = _blocks(model)
    for i in layers:
        handles.append(blk[i].register_forward_hook(mk(i)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def forward_logprobs(model, input_ids, device="cuda", decision_pos=None):
    """One patched-or-unpatched forward; returns log-probs at the decision position.

    `use_cache=False` is not optional here: with a cache, downstream positions would not be
    recomputed after an intervention (§8).
    """
    ids = torch.as_tensor([list(input_ids)], device=device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   use_cache=False).logits
    pos = ids.shape[1] - 1 if decision_pos is None else decision_pos
    return torch.log_softmax(logits[0, pos].float(), dim=-1).cpu()


def margin_from_logprobs(logprobs, lab_ids, authorized: str, target: str) -> float:
    """§9's margin: log P(attacker target) - log P(authorized answer).

    Each label's log-prob is the logsumexp over its bare and leading-space spellings.
    """
    lt = torch.logsumexp(logprobs[lab_ids[target]], dim=0).item()
    la = torch.logsumexp(logprobs[lab_ids[authorized]], dim=0).item()
    return lt - la
