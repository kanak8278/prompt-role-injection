"""Minimal, auditable GCG (Greedy Coordinate Gradient, Zou et al. 2023) for gpt-oss-20b + Harmony.

Why our own (not nanoGCG): we need two things a stock GCG doesn't give — (1) place the optimizable
adversarial tokens inside the Harmony *analysis* channel (the span an injection attacker controls),
and (2) run the loss/gradient forward pass with the refusal-direction STEERING hooks active (the
"attacker moves second vs the mechanism defense" test). Full control also lets us verify each step.

Algorithm per step:
  1. one-hot gradient of the target-completion CE loss w.r.t. the adv token positions (embedding trick)
  2. per position, take top-k tokens by -grad (candidates that most reduce the loss)
  3. sample B candidates = current adv with ONE random position swapped to one of its top-k
  4. evaluate the TRUE loss of all B candidates (batched forward, no grad); keep the best
Special/added (Harmony control) tokens are BANNED from the adv so GCG can't cheat by emitting
<|end|>/<|channel|> etc.; candidates that don't retokenize consistently are dropped (transfer safety).

forward_hooks: optional list of (module, hook_fn) installed during BOTH the grad and eval passes —
this is how the steering defense is made active while the attacker optimizes against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def get_embeds(model):
    return model.get_input_embeddings().weight        # [V, d]


def banned_ids(tok):
    """All added/special (Harmony control) token ids — never selectable as adversarial tokens."""
    ids = set(tok.all_special_ids)
    ids.update(tok.get_added_vocab().values())
    return torch.tensor(sorted(ids), dtype=torch.long)


def _install(hooks):
    return [m.register_forward_hook(fn) for (m, fn) in (hooks or [])]


def _remove(handles):
    for h in handles:
        h.remove()


def onehot_grad(model, embW, ids, adv_slice, target_slice, hooks=None):
    """grad of the target CE loss w.r.t. a one-hot over adv positions. ids: 1D LongTensor on device."""
    V, d = embW.shape
    n = adv_slice.stop - adv_slice.start
    oh = torch.zeros(n, V, device=embW.device, dtype=embW.dtype)
    oh[torch.arange(n, device=embW.device), ids[adv_slice]] = 1.0
    oh.requires_grad_(True)

    pre = embW[ids[:adv_slice.start]].detach()
    post = embW[ids[adv_slice.stop:]].detach()
    adv = oh @ embW                                    # [n, d], carries grad to oh only
    embs = torch.cat([pre, adv, post], dim=0).unsqueeze(0)   # [1, T, d]

    handles = _install(hooks)
    try:
        logits = model(inputs_embeds=embs).logits[0]   # [T, V]
    finally:
        _remove(handles)
    ls, le = target_slice.start, target_slice.stop
    loss = F.cross_entropy(logits[ls - 1:le - 1].float(), ids[ls:le])
    loss.backward()
    g = oh.grad.detach().clone()
    del logits, loss, oh, embs, adv
    return g, None


@torch.no_grad()
def eval_candidates(model, ids, adv_slice, target_slice, cand, microbatch, hooks=None):
    """CE loss for each candidate adv (cand: [B, n] LongTensor). Returns [B] on cpu."""
    B, T = cand.shape[0], ids.shape[0]
    full = ids.unsqueeze(0).repeat(B, 1)
    full[:, adv_slice] = cand
    ls, le = target_slice.start, target_slice.stop
    tgt = ids[ls:le]
    out = torch.empty(B)
    for i in range(0, B, microbatch):
        chunk = full[i:i + microbatch]
        handles = _install(hooks)
        try:
            logits = model(input_ids=chunk).logits            # [b, T, V]
        finally:
            _remove(handles)
        lg = logits[:, ls - 1:le - 1, :].float()               # [b, tgt, V]
        t = tgt.unsqueeze(0).expand(chunk.shape[0], -1)
        l = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), t.reshape(-1), reduction="none")
        out[i:i + microbatch] = l.view(chunk.shape[0], -1).mean(1).cpu()
        del logits, lg, l
    return out


def sample_candidates(adv_ids, grad, topk, batch_size, banned, tok, device):
    """Top-k by -grad per position; sample single-swap candidates; drop retokenization-inconsistent."""
    g = grad.clone()
    g[:, banned.to(g.device)] = float("inf")               # never pick banned tokens
    top = (-g).topk(topk, dim=1).indices                    # [n, topk]
    n = adv_ids.shape[0]
    pos = torch.randint(0, n, (batch_size,), device=device)
    pick = torch.randint(0, topk, (batch_size,), device=device)
    newtok = top[pos, pick]
    cand = adv_ids.unsqueeze(0).repeat(batch_size, 1).to(device)
    cand[torch.arange(batch_size, device=device), pos] = newtok
    # retokenization filter: decode->encode of the adv span must round-trip (transfer safety)
    keep = []
    for r in range(batch_size):
        toks = cand[r].tolist()
        if tok.encode(tok.decode(toks), add_special_tokens=False) == toks:
            keep.append(r)
    if not keep:                                            # fallback: keep all if filter nukes batch
        return cand
    return cand[torch.tensor(keep, device=device)]


@dataclass
class GCGConfig:
    n_steps: int = 250
    topk: int = 256
    batch_size: int = 256
    microbatch: int = 64
    early_stop_loss: float = 0.05
    seed: int = 0
    verbose_every: int = 25


def run_gcg(model, tok, ids, adv_slice, target_slice, cfg: GCGConfig, hooks=None, banned=None):
    """Optimize ids[adv_slice] to minimize the target CE loss. Returns (best_adv_ids, best_loss, hist)."""
    torch.manual_seed(cfg.seed)
    device = model.device
    embW = get_embeds(model)
    if banned is None:
        banned = banned_ids(tok)
    ids = ids.clone().to(device)
    adv_ids = ids[adv_slice].clone()
    best_adv, best_loss, hist = adv_ids.clone(), float("inf"), []

    for step in range(cfg.n_steps):
        grad, _ = onehot_grad(model, embW, ids, adv_slice, target_slice, hooks=hooks)
        cand = sample_candidates(adv_ids, grad, cfg.topk, cfg.batch_size, banned, tok, device)
        losses = eval_candidates(model, ids, adv_slice, target_slice, cand, cfg.microbatch, hooks=hooks)
        b = int(losses.argmin())
        adv_ids = cand[b].clone()
        ids[adv_slice] = adv_ids
        cur = float(losses[b])
        hist.append(cur)
        if cur < best_loss:
            best_adv, best_loss = adv_ids.clone(), cur
        if cfg.verbose_every and (step % cfg.verbose_every == 0 or step == cfg.n_steps - 1):
            print(f"  [gcg] step {step:4d}  loss {cur:.4f}  best {best_loss:.4f}  "
                  f"cand {cand.shape[0]}", flush=True)
        del grad
        if best_loss <= cfg.early_stop_loss:
            print(f"  [gcg] early stop at step {step} (loss {best_loss:.4f})", flush=True)
            break
    return best_adv, best_loss, hist
