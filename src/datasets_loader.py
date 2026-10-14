"""Benchmark loading.

Every axis is downloaded from the HuggingFace Hub at run time. No datasets are
committed to this repository. Each loader returns a list of MCQAItem or
GenerativeItem with a STABLE item_id, which matters because the whole study is
paired at the item level: to compute a flip rate you need to know that item
"arc_challenge/0042" is the same question in the BF16 run and in the 4-bit run.
"""

from __future__ import annotations

import re
from typing import List, Sequence

from .config import AxisSpec
from .prompts import GenerativeItem, MCQAItem

_GSM8K_ANSWER = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def _load_hf_split(axis: AxisSpec):
    """Import datasets lazily so that the statistics modules stay importable
    without the heavy dependency stack installed."""
    from datasets import load_dataset

    if axis.hf_config:
        return load_dataset(axis.hf_dataset, axis.hf_config, split=axis.split)
    return load_dataset(axis.hf_dataset, split=axis.split)


def _truncate(items: List, max_items: int) -> List:
    """Take a deterministic prefix. A prefix (not a random sample) keeps item
    ids identical between a 100-item pilot and a 500-item full run, so pilot
    results stay directly comparable."""
    if max_items and max_items > 0:
        return items[:max_items]
    return items


def load_arc(axis: AxisSpec) -> List[MCQAItem]:
    rows = _load_hf_split(axis)
    items: List[MCQAItem] = []
    for i, row in enumerate(rows):
        choices = row["choices"]
        texts = list(choices["text"])
        labels = list(choices["label"])
        key = row["answerKey"]
        if key not in labels:
            continue
        items.append(
            MCQAItem(
                item_id=f"{axis.name}/{i:05d}",
                question=row["question"],
                options=texts,
                answer_index=labels.index(key),
            )
        )
    return _truncate(items, axis.max_items)


def load_mmlu(axis: AxisSpec) -> List[MCQAItem]:
    rows = _load_hf_split(axis)
    items: List[MCQAItem] = []
    for i, row in enumerate(rows):
        items.append(
            MCQAItem(
                item_id=f"{axis.name}/{i:05d}",
                question=row["question"],
                options=list(row["choices"]),
                answer_index=int(row["answer"]),
            )
        )
    return _truncate(items, axis.max_items)


def load_gsm8k(axis: AxisSpec) -> List[GenerativeItem]:
    rows = _load_hf_split(axis)
    items: List[GenerativeItem] = []
    for i, row in enumerate(rows):
        match = _GSM8K_ANSWER.search(row["answer"])
        if not match:
            continue
        items.append(
            GenerativeItem(
                item_id=f"{axis.name}/{i:05d}",
                question=row["question"],
                reference=match.group(1).replace(",", ""),
            )
        )
    return _truncate(items, axis.max_items)


# Register a new axis by adding its loader here. The key must match the axis
# "name" in the config file.
LOADERS = {
    "arc_challenge": load_arc,
    "mmlu": load_mmlu,
    "gsm8k": load_gsm8k,
}


def load_axis(axis: AxisSpec) -> Sequence:
    """Load one capability axis into memory."""
    if axis.name not in LOADERS:
        raise KeyError(
            f"no loader for axis {axis.name!r}. Add one to src/datasets_loader.py "
            f"LOADERS. Known axes: {sorted(LOADERS)}"
        )
    items = LOADERS[axis.name](axis)
    if not items:
        raise RuntimeError(f"axis {axis.name!r} loaded zero items; check split and config")
    print(f"[data] {axis.name}: {len(items)} items ({axis.kind})")
    return items


def load_calibration_texts(spec) -> List[str]:
    """Load the shared calibration corpus used by the GPTQ and AWQ arms.

    Holding this fixed across arms is deliberate: calibration data is a
    confound, and varying it is its own ablation rather than something that
    should drift between arms by accident.
    """
    from datasets import load_dataset

    stream = load_dataset(spec.dataset, spec.subset, split="train", streaming=True)
    stream = stream.shuffle(seed=spec.seed, buffer_size=10_000)

    texts: List[str] = []
    for row in stream:
        text = row.get("text", "")
        if len(text.split()) < 64:
            continue
        texts.append(text)
        if len(texts) >= spec.n_samples:
            break

    print(f"[data] calibration: {len(texts)} sequences from {spec.dataset}")
    return texts
