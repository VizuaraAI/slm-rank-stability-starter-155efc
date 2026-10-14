"""The fixed evaluation harness.

Two scorers:

  mcqa        length-normalised log-likelihood over the answer options. Robust
              to a uniform logit shift, which is exactly why we expect these
              axes to keep their ranking under quantization.
  generative  greedy decode, then match the final numeric answer. One wrong
              token discards the item, which is why we expect these axes to
              reorder.

The important output is not the score. It is per_item, the per-question
correctness record. Scores can be recomputed from it, but a flip rate cannot be
recovered from a score, so the item-level record is what gets written to disk.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Sequence

from .config import AxisSpec, EvaluationSpec
from .prompts import (
    GenerativeItem,
    MCQAItem,
    expand_mcqa_variants,
    get_generative_template,
    get_mcqa_template,
)

_FINAL_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass
class AxisResult:
    """Scores plus the item-level record for one (model, arm, axis)."""

    model_id: str
    arm_name: str
    axis_name: str
    accuracy: float
    per_template_accuracy: Dict[str, float] = field(default_factory=dict)
    per_item: Dict[str, int] = field(default_factory=dict)
    n_items: int = 0
    wall_time_s: float = 0.0
    tokens_per_s: float = 0.0

    @property
    def template_spread(self) -> float:
        """Max minus min accuracy across prompt templates. This is the noise
        floor a quantization effect has to exceed to count as real."""
        if len(self.per_template_accuracy) < 2:
            return 0.0
        vals = list(self.per_template_accuracy.values())
        return max(vals) - min(vals)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["template_spread"] = self.template_spread
        return out


def _batched(seq: Sequence, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _score_continuations(model, tokenizer, context: str, continuations: List[str],
                         max_seq_len: int) -> List[float]:
    """Length-normalised log-likelihood of each continuation given context."""
    import torch

    scores: List[float] = []
    device = getattr(model, "device", "cpu")

    for cont in continuations:
        ctx_ids = tokenizer(context, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(context + cont, add_special_tokens=True)["input_ids"]

        # Degenerate tokenization (continuation adds nothing) scores as -inf so
        # it never wins by accident.
        if len(full_ids) <= len(ctx_ids):
            scores.append(float("-inf"))
            continue

        full_ids = full_ids[-max_seq_len:]
        n_cont = len(full_ids) - len(ctx_ids)

        input_ids = torch.tensor([full_ids], device=device)
        with torch.no_grad():
            logits = model(input_ids).logits.float()

        logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
        targets = input_ids[0, 1:]
        token_lp = logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
        cont_lp = token_lp[-n_cont:].sum().item()
        scores.append(cont_lp / n_cont)

    return scores


def evaluate_mcqa(model, tokenizer, items: Sequence[MCQAItem], axis: AxisSpec,
                  ev: EvaluationSpec, model_id: str, arm_name: str) -> AxisResult:
    start = time.time()
    per_template: Dict[str, float] = {}
    # item_id -> list of correctness values across templates and option orders
    votes: Dict[str, List[int]] = {it.item_id: [] for it in items}

    for template_name in ev.prompt_templates:
        template = get_mcqa_template(template_name)
        variants = expand_mcqa_variants(items, ev.shuffle_options, ev.shuffle_seeds)
        correct = 0

        for _seed, item in variants:
            context, continuations = template(item)
            scores = _score_continuations(
                model, tokenizer, context, continuations, ev.max_seq_len
            )
            predicted = max(range(len(scores)), key=lambda i: scores[i])
            is_correct = int(predicted == item.answer_index)
            correct += is_correct
            votes[item.item_id].append(is_correct)

        per_template[template_name] = correct / max(1, len(variants))
        print(f"[eval] {model_id}/{arm_name}/{axis.name}/{template_name}: "
              f"{per_template[template_name]:.4f}")

    # An item counts as correct if the majority of its variants are correct.
    # This keeps the item-level record (and so the flip rate) well defined even
    # though each item is scored several times.
    per_item = {
        item_id: int(sum(v) * 2 > len(v)) for item_id, v in votes.items() if v
    }
    accuracy = sum(per_item.values()) / max(1, len(per_item))
    elapsed = time.time() - start

    return AxisResult(
        model_id=model_id,
        arm_name=arm_name,
        axis_name=axis.name,
        accuracy=accuracy,
        per_template_accuracy=per_template,
        per_item=per_item,
        n_items=len(per_item),
        wall_time_s=round(elapsed, 2),
    )


def _extract_answer(text: str) -> str:
    """Pull the final answer out of a generation.

    Prefers an explicit 'Answer: X' marker, falls back to the last number in
    the text, which is the standard GSM8K convention.
    """
    marked = re.findall(r"[Aa]nswer\s*:\s*(-?[\d,]*\.?\d+)", text)
    if marked:
        return marked[-1].replace(",", "")
    numbers = _FINAL_NUMBER.findall(text)
    return numbers[-1].replace(",", "") if numbers else ""


def _answers_match(predicted: str, reference: str) -> bool:
    if not predicted:
        return False
    try:
        return abs(float(predicted) - float(reference)) < 1e-4
    except ValueError:
        return predicted.strip() == reference.strip()


def evaluate_generative(model, tokenizer, items: Sequence[GenerativeItem],
                        axis: AxisSpec, ev: EvaluationSpec, model_id: str,
                        arm_name: str) -> AxisResult:
    import torch

    start = time.time()
    per_template: Dict[str, float] = {}
    votes: Dict[str, List[int]] = {it.item_id: [] for it in items}
    total_new_tokens = 0

    for template_name in ev.prompt_templates:
        template = get_generative_template(template_name)
        correct = 0

        for batch in _batched(list(items), ev.batch_size):
            prompts = [template(it) for it in batch]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=ev.max_seq_len,
            ).to(getattr(model, "device", "cpu"))

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=axis.max_new_tokens,
                    do_sample=not ev.greedy,
                    pad_token_id=tokenizer.pad_token_id,
                )

            prompt_len = enc["input_ids"].shape[1]
            for item, row in zip(batch, out):
                new_tokens = row[prompt_len:]
                total_new_tokens += int(new_tokens.shape[0])
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                is_correct = int(_answers_match(_extract_answer(text), item.reference))
                correct += is_correct
                votes[item.item_id].append(is_correct)

        per_template[template_name] = correct / max(1, len(items))
        print(f"[eval] {model_id}/{arm_name}/{axis.name}/{template_name}: "
              f"{per_template[template_name]:.4f}")

    per_item = {
        item_id: int(sum(v) * 2 > len(v)) for item_id, v in votes.items() if v
    }
    accuracy = sum(per_item.values()) / max(1, len(per_item))
    elapsed = time.time() - start

    return AxisResult(
        model_id=model_id,
        arm_name=arm_name,
        axis_name=axis.name,
        accuracy=accuracy,
        per_template_accuracy=per_template,
        per_item=per_item,
        n_items=len(per_item),
        wall_time_s=round(elapsed, 2),
        tokens_per_s=round(total_new_tokens / elapsed, 2) if elapsed > 0 else 0.0,
    )


def evaluate_axis(model, tokenizer, items: Sequence, axis: AxisSpec,
                  ev: EvaluationSpec, model_id: str, arm_name: str) -> AxisResult:
    """Dispatch to the right scorer for this axis."""
    if axis.kind == "mcqa":
        return evaluate_mcqa(model, tokenizer, items, axis, ev, model_id, arm_name)
    return evaluate_generative(model, tokenizer, items, axis, ev, model_id, arm_name)
