"""Prompt templates and option-order controls.

These exist for one reason: so that a rank change can be blamed on
quantization rather than on prompt formatting. Formatting alone is known to
move accuracy enough to reorder models, so every model is scored under several
semantically equivalent templates and the spread across templates becomes the
noise floor the quantization effect has to clear.

Add a template by writing a function and registering it in MCQA_TEMPLATES or
GENERATIVE_TEMPLATES. Never edit an existing template once results are logged
against it; add a new one instead.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

LETTERS = "ABCDEFGHIJ"


def stable_hash(text: str) -> int:
    """Deterministic across processes, unlike the builtin hash(), which is
    salted by PYTHONHASHSEED and would silently break reproducibility."""
    return zlib.crc32(text.encode("utf-8"))


@dataclass(frozen=True)
class MCQAItem:
    """One multiple-choice question with a known answer index."""

    item_id: str
    question: str
    options: List[str]
    answer_index: int

    def shuffled(self, seed: int) -> "MCQAItem":
        """Return the same item with options permuted and answer_index moved
        to follow the correct option."""
        order = list(range(len(self.options)))
        random.Random(seed).shuffle(order)
        return MCQAItem(
            item_id=self.item_id,
            question=self.question,
            options=[self.options[i] for i in order],
            answer_index=order.index(self.answer_index),
        )


@dataclass(frozen=True)
class GenerativeItem:
    """One free-generation question with a reference answer string."""

    item_id: str
    question: str
    reference: str


# --------------------------------------------------------------------------
# MCQA templates. Each returns (context, continuations). The scorer picks the
# continuation with the highest length-normalised log-likelihood.
# --------------------------------------------------------------------------

def mcqa_plain(item: MCQAItem) -> Tuple[str, List[str]]:
    context = f"{item.question.strip()}\nAnswer:"
    return context, [f" {opt.strip()}" for opt in item.options]


def mcqa_lettered(item: MCQAItem) -> Tuple[str, List[str]]:
    lines = [item.question.strip(), ""]
    for letter, opt in zip(LETTERS, item.options):
        lines.append(f"{letter}. {opt.strip()}")
    lines.append("")
    lines.append("The correct answer is:")
    context = "\n".join(lines)
    return context, [f" {letter}" for letter in LETTERS[: len(item.options)]]


def mcqa_qa_prefix(item: MCQAItem) -> Tuple[str, List[str]]:
    context = (
        "The following is a multiple choice question. Answer it correctly.\n\n"
        f"Question: {item.question.strip()}\nAnswer:"
    )
    return context, [f" {opt.strip()}" for opt in item.options]


# --------------------------------------------------------------------------
# Generative templates. Each returns a single prompt string.
# --------------------------------------------------------------------------

def gen_plain(item: GenerativeItem) -> str:
    return f"{item.question.strip()}\n"


def gen_lettered(item: GenerativeItem) -> str:
    return (
        f"Question: {item.question.strip()}\n"
        "Work through the problem step by step, then give the final answer "
        "on a new line in the form 'Answer: <value>'.\n"
    )


def gen_qa_prefix(item: GenerativeItem) -> str:
    return (
        "Solve the problem. End your response with 'Answer: <value>'.\n\n"
        f"Problem: {item.question.strip()}\nSolution:"
    )


MCQA_TEMPLATES: Dict[str, Callable[[MCQAItem], Tuple[str, List[str]]]] = {
    "plain": mcqa_plain,
    "lettered": mcqa_lettered,
    "qa_prefix": mcqa_qa_prefix,
}

GENERATIVE_TEMPLATES: Dict[str, Callable[[GenerativeItem], str]] = {
    "plain": gen_plain,
    "lettered": gen_lettered,
    "qa_prefix": gen_qa_prefix,
}


def get_mcqa_template(name: str) -> Callable[[MCQAItem], Tuple[str, List[str]]]:
    if name not in MCQA_TEMPLATES:
        raise KeyError(f"unknown mcqa template {name!r}; known: {sorted(MCQA_TEMPLATES)}")
    return MCQA_TEMPLATES[name]


def get_generative_template(name: str) -> Callable[[GenerativeItem], str]:
    if name not in GENERATIVE_TEMPLATES:
        raise KeyError(
            f"unknown generative template {name!r}; known: {sorted(GENERATIVE_TEMPLATES)}"
        )
    return GENERATIVE_TEMPLATES[name]


def expand_mcqa_variants(
    items: Sequence[MCQAItem],
    shuffle: bool,
    shuffle_seeds: Sequence[int],
) -> List[Tuple[int, MCQAItem]]:
    """Expand each item into its option-order variants.

    Returns (seed, item) pairs. Seed -1 means the original, unshuffled order,
    which is always included so results stay comparable to published numbers.
    """
    out: List[Tuple[int, MCQAItem]] = [(-1, it) for it in items]
    if shuffle:
        for seed in shuffle_seeds:
            out.extend(
                (seed, it.shuffled(seed + stable_hash(it.item_id) % 9973)) for it in items
            )
    return out
