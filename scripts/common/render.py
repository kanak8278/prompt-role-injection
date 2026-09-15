"""Canonical message rendering, span alignment, and payload safety assertions (§6, §8).

Two facts measured in G0 drive the design here (see notes/04-rendering-findings.md):

1. **Llama-3.1 JSON-encodes tool content.** A newline inside a tool message becomes a literal
   backslash-n and a quote becomes \\". Qwen2.5 passes it through raw. So char offsets into the
   *document* do not map to the rendered prompt, and all spans must be located in the
   **rendered string**. Cue and command strings are kept single-line and quote-free so they
   survive rendering verbatim on both models.

2. **Qwen2.5 renders tool results inside a `user` block** (`<|im_start|>user` +
   `<tool_response>`), while Llama-3.1 uses a distinct `ipython` role. Therefore
   `rendered_source_role` is recorded per model and is NOT assumed to equal the actual source.

Payload safety is enforced on decoded token ids, not just text: `<|im_start|>`/`<|im_end|>` on
Qwen and `<|start_header_id|>`/`<|eot_id|>` on Llama parse to real special ids from plain text,
so untrusted content could otherwise forge a genuine chat turn (§4 puts that out of scope).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

# Pin Llama-3.1's auto-inserted date so the rendered prompt is reproducible (§8: "Fix any
# automatically inserted date or system text"). The template's own default is this value; we
# pass it explicitly so a future template revision cannot silently change our prompts.
LLAMA_DATE_STRING = "26 Jul 2024"

# Kept for reference and for the deviation note in `scaffold`: this is the marker Llama-3.1's
# model card names as the precondition for its `ipython` role. We deliberately do NOT emit it
# -- see `scaffold` for the measurements behind that choice.
LLAMA_IPYTHON_MARKER = "Environment: ipython"

TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "fetch_document",
        "description": "Retrieve a stored document by query key.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Document key."}},
            "required": ["query"],
        },
    },
}]

# Spellings that must never appear in untrusted content, per model family. Verified in G0 to
# parse as single special ids from plain text on the corresponding model.
NATIVE_SPECIALS = {
    "llama": ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
              "<|eom_id|>", "<|python_tag|>", "<|finetune_right_pad_id|>"],
    "qwen": ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_call>", "</tool_call>"],
}


def scaffold(messages: list[dict], fam: str) -> tuple[list[dict], dict[str, Any]]:
    """Per-model tool scaffold. Measured, not assumed -- the two models need different setups.

    **Qwen2.5**: pass `tools=`. Its template then emits the `# Tools` / `<tools>` system block
    that makes a `<tool_response>` body in-distribution, and the model answers normally.

    **Llama-3.1**: pass NEITHER `tools=` nor the `Environment: ipython` marker. Both were
    measured and both are harmful here:

      - `tools=` injects the JSON function signature into the first user turn, priming the
        model so strongly that it emits another `fetch_document` call instead of answering --
        100% invalid-output rate, unchanged across every system-policy wording tried.
      - `Environment: ipython` declares a Python interpreter environment, and the model
        responds by emitting Python code instead of an answer.

    Its model card names the marker as the precondition for the `ipython` role, so omitting it
    is a documented deviation. The justification is that our tool returns a *document*, not
    interpreter output, so declaring a Python environment misdescribes the setting; and
    empirically the model reads the `ipython`-wrapped document correctly without it (98% clean
    accuracy on record lookup). Recorded in notes/05-design-decisions.md.

    This asymmetry is a scaffold difference between models and is recorded per rendering, so
    §13's "present effects by model before any pooled average" can be honoured.
    """
    if fam == "qwen":
        return messages, {"tools": TOOL_SCHEMA}
    return list(messages), {"date_string": LLAMA_DATE_STRING}


def model_family(model_name: str) -> str:
    n = model_name.lower()
    if "llama" in n:
        return "llama"
    if "qwen" in n:
        return "qwen"
    raise ValueError(f"unknown model family for {model_name!r}; add its native specials first")


class RenderError(Exception):
    pass


@dataclass
class Rendered:
    model: str
    condition: str
    scenario_id: str
    rendered: str
    input_ids: list[int]
    offsets: list[tuple[int, int]]
    spans: dict[str, tuple[int, int]]      # name -> [tok_start, tok_end) half-open
    char_spans: dict[str, tuple[int, int]]
    decision_pos: int                       # index of the last prompt token
    rendered_source_role: str | None        # role header the tool body actually sits under
    n_tokens: int
    truncated: bool
    template_sha: str
    warnings: list[str] = field(default_factory=list)


def _tok_range(offsets: list[tuple[int, int]], lo: int, hi: int) -> tuple[int, int]:
    """Half-open token range covering char range [lo, hi).

    Special tokens report (0, 0) in the offset mapping on both tokenizers, so they are skipped
    rather than being treated as covering char 0.
    """
    idx = [i for i, (a, b) in enumerate(offsets)
           if not (a == 0 and b == 0) and a < hi and b > lo]
    if not idx:
        raise RenderError(f"no tokens cover char range [{lo},{hi})")
    return idx[0], idx[-1] + 1


def _find_unique(hay: str, needle: str, what: str) -> tuple[int, int]:
    n = hay.count(needle)
    if n == 0:
        raise RenderError(f"{what} not found verbatim in rendered prompt: {needle!r}")
    if n > 1:
        raise RenderError(f"{what} appears {n} times; span would be ambiguous: {needle!r}")
    i = hay.index(needle)
    return i, i + len(needle)


def render(tokenizer, model_name: str, rc, base_scenario,
           max_tokens: int = 1024) -> Rendered:
    """Render one condition and align every semantic span.

    `rc` is a scenarios.RenderedCondition, `base_scenario` a scenarios.BaseScenario.
    """
    fam = model_family(model_name)
    messages, kwargs = scaffold(rc.messages, fam)

    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs)
    # §8 reproducibility: the template must be a pure function of the messages. Rendering
    # twice catches a template that injects a live date (Llama 3.2/3.3 use strftime_now).
    if tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs) != rendered:
        raise RenderError("chat template is nondeterministic across calls")

    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    input_ids, offsets = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]

    warnings: list[str] = []

    # --- payload safety: untrusted content must not forge a chat boundary (§4, §6) --------
    for sp in NATIVE_SPECIALS[fam]:
        if sp in rc.document:
            raise RenderError(
                f"document contains {fam} native special spelling {sp!r}; literal delimiter "
                "injection is a separate threat model (§4)")
    # Structural check on decoded ids: the number of real turn-delimiter tokens must match the
    # message count exactly. Catches an escape that the substring scan above would miss.
    _assert_turn_count(tokenizer, fam, input_ids, messages)

    # --- semantic spans, located in the RENDERED string ----------------------------------
    char_spans: dict[str, tuple[int, int]] = {}
    spans: dict[str, tuple[int, int]] = {}

    # the tool body as it appears after any template escaping
    body_needle = _escaped_body(rc.document, fam)
    char_spans["tool_body"] = _find_unique(rendered, body_needle, "tool body")

    if rc.cue is not None:
        char_spans["cue"] = _find_unique(rendered, rc.cue, "cue")
    # The full inserted region, whatever it is: cue+command for P/S, the bare command for B,
    # the length-matched neutral/mention text for C/M. This is the span that C, M and B share
    # by construction and that the aligned C/M/B analysis screens.
    if rc.insert is not None:
        char_spans["insert"] = _find_unique(rendered, rc.insert, "insert")
    if rc.command is not None:
        # In U the command lives in the later user turn, not the document; in P/S/Q it lives
        # in the document. Either way it must occur exactly once in the whole prompt.
        char_spans["command"] = _find_unique(rendered, rc.command, "command")

    # the task text, so we can hold "reading the request" apart from "selecting an instruction"
    char_spans["task"] = _find_unique(rendered, rc.messages[1]["content"], "task text")

    for k, (lo, hi) in char_spans.items():
        spans[k] = _tok_range(offsets, lo, hi)

    # decision position: last prompt token, where the first answer token is predicted
    decision_pos = len(input_ids) - 1
    spans["decision"] = (decision_pos, decision_pos + 1)

    truncated = len(input_ids) > max_tokens
    if truncated:
        raise RenderError(
            f"prompt is {len(input_ids)} tokens, over the {max_tokens} ceiling; never "
            "silently truncate an attack or its evidence (§6)")

    return Rendered(
        model=model_name,
        condition=rc.condition,
        scenario_id=rc.scenario_id,
        rendered=rendered,
        input_ids=input_ids,
        offsets=offsets,
        spans=spans,
        char_spans=char_spans,
        decision_pos=decision_pos,
        rendered_source_role=_rendered_source_role(rendered, char_spans["tool_body"][0], fam),
        n_tokens=len(input_ids),
        truncated=truncated,
        template_sha=hashlib.sha256((tokenizer.chat_template or "").encode()).hexdigest()[:16],
        warnings=warnings,
    )


def _escaped_body(document: str, fam: str) -> str:
    """How `document` will appear inside the rendered prompt.

    Llama's template JSON-encodes tool content. Our documents contain no newlines or quotes by
    construction, so the escaped form equals the raw form -- but we compute it rather than
    assume it, so that a future document format change fails loudly instead of misaligning
    spans silently.
    """
    if fam == "llama":
        import json
        enc = json.dumps(document)[1:-1]   # strip the surrounding quotes json adds
        return enc
    return document


def _assert_turn_count(tokenizer, fam: str, input_ids: list[int], messages: list[dict]) -> None:
    """Assert the prompt contains exactly the turn delimiters a legitimate conversation needs.

    An extra delimiter means untrusted text opened its own turn -- the escape measured on
    Qwen2.5 in G0. Counting decoded ids catches it even if the text scan is evaded.
    """
    n_msgs = len(messages)
    if fam == "qwen":
        start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        # one <|im_start|> per message plus one for the generation prompt
        expect = n_msgs + 1
        got = input_ids.count(start)
        if got != expect:
            raise RenderError(
                f"qwen turn-delimiter count {got} != expected {expect}; untrusted content "
                "appears to have forged a chat boundary")
    else:
        hdr = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
        expect = n_msgs + 1
        got = input_ids.count(hdr)
        if got != expect:
            raise RenderError(
                f"llama header count {got} != expected {expect}; untrusted content appears "
                "to have forged a chat boundary")


def _rendered_source_role(rendered: str, body_char_start: int, fam: str) -> str | None:
    """Which role header the tool body actually sits under in the rendered prompt.

    Recorded because it is NOT the actual source: on Qwen2.5 a genuine tool result renders
    under a `user` header, so this field and the true source disagree by design (§8).
    """
    prefix = rendered[:body_char_start]
    best_role, best_pos = None, -1
    roles = ("system", "user", "assistant", "tool", "ipython")
    pats = ([f"<|start_header_id|>{r}<|end_header_id|>" for r in roles] if fam == "llama"
            else [f"<|im_start|>{r}" for r in roles])
    for r, pat in zip(roles, pats):
        p = prefix.rfind(pat)
        if p > best_pos:
            best_role, best_pos = r, p
    return best_role if best_pos >= 0 else None


def check_span_alignment(tokenizer, model_name: str, p: Rendered, s: Rendered,
                         span: str = "cue") -> dict:
    """Generalisation of check_cue_alignment to any named span.

    Aligned means identical token count, the span at the same index with the same length, and
    identical token ids everywhere outside it. For a C/M/B contrast the span is `insert`.
    """
    res = {"scenario_id": p.scenario_id, "model": model_name, "span": span,
           "aligned": False, "reasons": []}
    if p.n_tokens != s.n_tokens:
        res["reasons"].append(f"token counts differ ({p.n_tokens} vs {s.n_tokens})")
    if p.spans.get(span) != s.spans.get(span):
        res["reasons"].append(
            f"{span} spans differ ({p.spans.get(span)} vs {s.spans.get(span)})")
    if not res["reasons"]:
        lo, hi = p.spans[span]
        if p.input_ids[:lo] + p.input_ids[hi:] != s.input_ids[:lo] + s.input_ids[hi:]:
            n_diff = sum(1 for a, b in zip(p.input_ids[:lo] + p.input_ids[hi:],
                                           s.input_ids[:lo] + s.input_ids[hi:]) if a != b)
            res["reasons"].append(f"{n_diff} token ids differ outside the {span} span")
    res["aligned"] = not res["reasons"]
    res["span_range"] = p.spans.get(span)
    res["decision_pos"] = p.decision_pos
    return res


def check_cue_alignment(tokenizer, model_name: str, p: Rendered, s: Rendered) -> dict:
    """Is the P/S pair positionally aligned for exact activation patching (§4, §10)?

    Aligned means: same token count, identical ids outside the cue span, and cue spans at the
    same index with the same length. If this fails the pair is excluded from the aligned
    causal analysis but retained for behavioral evaluation, and the exclusion is reported.
    """
    res = {"scenario_id": p.scenario_id, "model": model_name, "aligned": False,
           "reasons": []}
    if p.n_tokens != s.n_tokens:
        res["reasons"].append(f"token counts differ ({p.n_tokens} vs {s.n_tokens})")
    if p.spans.get("cue") != s.spans.get("cue"):
        res["reasons"].append(f"cue spans differ ({p.spans.get('cue')} vs {s.spans.get('cue')})")
    if p.spans.get("command") != s.spans.get("command"):
        res["reasons"].append(
            f"command spans differ ({p.spans.get('command')} vs {s.spans.get('command')})")
    if not res["reasons"]:
        cs, ce = p.spans["cue"]
        outside_p = p.input_ids[:cs] + p.input_ids[ce:]
        outside_s = s.input_ids[:cs] + s.input_ids[ce:]
        if outside_p != outside_s:
            n_diff = sum(1 for a, b in zip(outside_p, outside_s) if a != b)
            res["reasons"].append(f"{n_diff} token ids differ outside the cue span")
    res["aligned"] = not res["reasons"]
    res["cue_span"] = p.spans.get("cue")
    res["command_span"] = p.spans.get("command")
    res["decision_pos"] = p.decision_pos
    return res
