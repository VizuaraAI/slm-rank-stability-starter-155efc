# Instructions for the coding agent

This file is for the AI coding agent working in this repository with Karnan.
Read it before making changes.

## Project goal

Measure whether small language model leaderboard rankings survive 4-bit
quantization. Twelve instruction-tuned SLMs are evaluated at BF16 and under four
4-bit arms (GPTQ, AWQ, bitsandbytes NF4, GGUF Q4_K_M) with one fixed harness,
then the two leaderboards are compared with **rank statistics** rather than mean
score deltas.

Target: a workshop paper at an ICLR 2027 workshop, roughly eight weeks of work.
The paper skeleton lives alongside this repo in `paper/main.tex`.

## The one invariant that must never break

**Results are paired at the item level.** `item_id` for a given benchmark
question must be byte-identical across every model and every arm.

If item ids drift between arms, the flip rate and the paired bootstrap silently
become meaningless while still producing plausible-looking numbers. That is the
worst failure mode in this project, because nothing crashes.

Consequences for you:

- Never make item ids depend on `hash()` (salted per process), wall-clock time,
  dict ordering, or a random sample. `src/prompts.py` has `stable_hash` for this.
- `_truncate` in `datasets_loader.py` takes a deterministic PREFIX, not a random
  sample, so a 100-item pilot and a 500-item run share ids. Keep it that way.
- Never aggregate away `per_item` to save space. A score cannot reconstruct a
  flip rate.

## Layout

| Path | Role |
|---|---|
| `configs/default.yaml` | every knob; prefer config changes over code changes |
| `src/config.py` | YAML into frozen dataclasses |
| `src/datasets_loader.py` | benchmark loading, stable item ids |
| `src/prompts.py` | prompt templates, option shuffling |
| `src/quantize.py` | the five precision arms |
| `src/evaluate.py` | mcqa and generative scorers |
| `src/rank_stats.py` | tau-b, top-k overlap, flip rate, paired bootstrap |
| `src/build_variants.py` | entry point: build variants |
| `src/run_eval.py` | entry point: evaluate |
| `src/report.py` | results into paper tables |

## Conventions

- **Lazy heavy imports.** `torch`, `transformers`, `datasets`, `gptqmodel` and
  `awq` are imported INSIDE functions, never at module top level. This keeps
  `src.rank_stats` and `src.config` importable on a laptop with no GPU stack, so
  the statistics stay testable while a sweep runs elsewhere. Preserve this.
- **Arms degrade, they do not crash.** An unbuildable arm raises `ArmUnavailable`
  and the runner records it as skipped. A partial leaderboard is still
  analysable. Do not convert these into hard failures.
- **Both entry points are resumable.** `build_variants` reuses cached variants,
  `run_eval` skips existing result files unless `--overwrite`. Any new long-
  running step must be resumable too.
- Type hints on function signatures. `from __future__ import annotations` at top.
- Comments explain WHY, not what. Several existing comments record a decision and
  its rationale; do not strip them as noise.
- No em dashes in code comments or docs. Use commas, colons or parentheses.
- Default to small: new config knobs get a cheap default so the pilot stays fast.

## Statistical rules you must not quietly violate

1. **The noise floor gates every claim.** A quantization effect counts only if it
   exceeds the prompt-template-only tau for that axis (`noise_floor.csv`) and the
   paired bootstrap CI excludes it. If asked to report a finding that fails this,
   say so rather than reporting it.
2. **Bootstrap resamples ITEMS, not models.** Models are the population of
   interest; benchmark questions are the noise source. `paired_bootstrap_tau`
   does this correctly, and both arms are resampled on the same indices to keep
   the pairing. Do not "simplify" it into an unpaired or model-level bootstrap.
3. **Kendall tau-b needs at least 3 models**, and is coarse below about 8. The
   code returns NaN below 3 rather than a misleading number. Keep that guard.
4. **A null result is a real result.** If rankings turn out stable, that is the
   finding. Never tune the protocol until a reordering appears.

## When adding things

- **New benchmark axis:** add a loader to `LOADERS` in `datasets_loader.py`
  returning `MCQAItem` or `GenerativeItem` with stable ids, then add the axis to
  the config. Nothing else should need to change.
- **New quantization arm:** add a `build_*` function in `quantize.py`, dispatch it
  in `build_variant`, raise `ArmUnavailable` when the environment cannot support
  it, and add the arm to the config.
- **New prompt template:** add a function and register it in `MCQA_TEMPLATES` or
  `GENERATIVE_TEMPLATES`. **Never edit an existing template** once results are
  logged against it; results stop being comparable. Add a new one instead.

## Current state and next steps

Working: config, loaders, prompts, all five arms (GGUF needs
`scripts/build_gguf.sh` run first), both scorers, the rank statistics, and the
reporting pipeline.

Next, in order:

1. Run `scripts/run_pilot.sh` and confirm the loop closes end to end.
2. Measure the noise floor on BF16 only, across all axes, before any
   quantization. This is ablation A2 and it gates everything else.
3. Build the full BF16 reference leaderboard.
4. One arm (NF4 is the easiest to install) end to end across all axes.
5. Remaining arms, then ablations A1 and A3 to A6.
6. Add MMLU-Pro, IFEval, HumanEval and TruthfulQA loaders. The paper's tables
   assume six capability axes; the config and `LOADERS` currently cover three
   (ARC-Challenge, MMLU, GSM8K), so three remain to be added.

## What to ask Karnan about rather than deciding alone

- Dropping an arm or an axis from the study.
- Any change to how an item id is constructed.
- Reporting a result that does not clear the noise floor.
- Changing `max_items`, which trades statistical power for wall time.
