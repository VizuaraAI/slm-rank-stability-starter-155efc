# slm-rank-stability

**Do SLM rankings survive the deployment pipeline?**

Starter repository for Karnan Seevara, Vizuara SLM Bootcamp research track.

## The question

Small language model leaderboards are read as deployment advice: pick the
top-ranked model that fits your memory budget and ship it. But nobody ships the
model that was ranked. The model that reaches a phone has been quantized to 4
bits, and quantization damages capabilities unevenly rather than uniformly.

So: **does the full-precision ranking still hold after 4-bit quantization?**

That question is ordinal, which means the right statistic is a rank correlation
and not a score delta. Published quantization studies report "average accuracy
drops 1.2 points". That number is compatible with the leaderboard staying
identical and with it being completely reshuffled. Nobody has reported which.

This repo measures the difference.

## What makes this publishable

Three things, and it is worth being clear about them before you write any code:

1. **The statistic is new for this question.** Everyone reports mean score
   deltas. Kendall tau-b between the BF16 and 4-bit leaderboards, per capability
   axis, is the quantity practitioners actually act on, and it is not in the
   literature.
2. **The noise floor is built in.** Prompt formatting alone can reorder models.
   Measuring that floor first (ablation A2) is what separates a real finding from
   a harness artifact, and it is what a reviewer will ask about first.
3. **A negative result still publishes.** If rankings turn out to be stable, that
   is a useful, citable reassurance for every practitioner shipping on-device.
   You cannot lose this experiment, which is a rare and very good property for an
   eight-week project.

## Setup

```bash
git clone <your-repo-url> slm-rank-stability
cd slm-rank-stability
bash scripts/setup.sh
source .venv/bin/activate
```

Needs one GPU with at least 16 GB for the 3B models. Smaller models run on 8 GB.
No datasets are committed; loaders download public benchmarks on first use.

If `gptqmodel` or `autoawq` will not install on your GPU, install everything else
and delete that arm from `configs/default.yaml`. The study works with fewer arms.

## First three things to run

```bash
# 1. Smoke test the statistics with no GPU and no model downloads
python -c "
from src.rank_stats import kendall_tau_b, flip_rate
ref  = {'a': 0.80, 'b': 0.75, 'c': 0.70, 'd': 0.65}
cand = {'a': 0.70, 'b': 0.72, 'c': 0.60, 'd': 0.63}   # a and b swapped
print('tau_b:', kendall_tau_b(ref, cand))
print('flips:', flip_rate({'i1':1,'i2':0,'i3':1}, {'i1':1,'i2':1,'i3':0}))
"

# 2. Full pipeline on three small models, two arms, one axis (~20 min)
bash scripts/run_pilot.sh

# 3. Look at what came out
cat artifacts/tables/rank_agreement.csv
cat artifacts/tables/noise_floor.csv
```

Then scale up with `bash scripts/run_full_sweep.sh`.

## Layout

```
configs/default.yaml        every knob: models, arms, axes, statistics
src/config.py               YAML into dataclasses
src/datasets_loader.py      benchmark loading, stable item ids
src/prompts.py              prompt templates and option-order controls
src/quantize.py             the five precision arms
src/evaluate.py             the fixed harness (mcqa and generative scorers)
src/rank_stats.py           tau-b, top-k overlap, flip rate, paired bootstrap
src/build_variants.py       entry point 1: build quantized variants
src/run_eval.py             entry point 2: evaluate and log per item
src/report.py               results into the paper's tables
scripts/                    setup, pilot, full sweep, GGUF build
experiments/log_template.md one entry per run, columns map to paper tables
```

## The one design decision to understand

`src/evaluate.py` writes **per-item correctness**, not just a score. This is the
whole architecture in one sentence.

A score can be recomputed from per-item records, but a flip rate can never be
recovered from a score. Two models can both score 61.4% while disagreeing on a
third of the questions. That gap between score stability and behavioural
stability is the paper's central finding, and you only get to see it because
every result file carries the item-level record.

So: never replace those records with an aggregate to save disk space.

## Working order (8 weeks)

1. Pilot loop closes, statistics smoke-tested.
2. Noise floor measured on BF16 only. Do this before any quantization.
3. Full BF16 leaderboard: the reference everything is compared against.
4. One quantization arm end to end.
5. Remaining arms, then the ablations.
6. Figures and paper.

Measure the noise floor before you measure the effect. If you do it the other
way round you will spend a week excited about a reordering that was prompt
formatting all along.

## Honesty checklist before any number enters the paper

- [ ] The effect is larger than the prompt-template noise floor for that axis.
- [ ] The paired bootstrap CI excludes the noise-floor tau.
- [ ] Item ids are identical across the arms being compared (pairing intact).
- [ ] The run is reproducible from a logged config hash and commit.
- [ ] A surprising result has been actively checked for bugs, not just believed.
