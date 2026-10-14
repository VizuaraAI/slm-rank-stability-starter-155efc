# Experiment log

One entry per run. Copy the template at the bottom, fill it in, never delete an
old entry. Runs that failed or produced a boring result are the ones you will
most want to reread in week 7, and "I already tried that" is only useful if you
wrote down what happened.

The columns map onto the paper's tables, so a complete log fills the paper in
for you:

| Log field | Lands in |
|---|---|
| Axis, Arm, Kendall tau-b, Mean score delta | Table 1 (rank agreement per axis) |
| Flip rate, Rank swaps, Top-k overlap, Peak memory | Table 2 (composite arm summary) |
| Rank displacement by family | Ablation A1, Figure 2(b) |
| Template spread (noise floor) | Ablation A2 |
| Calibration corpus | Ablation A3 |
| Bit width | Ablation A4 |
| Flip stratification, outlier correlation | Ablation A5 |
| Accuracy per GB, accuracy per token/s | Ablation A6 |

## Rules that keep the log honest

1. Record the config hash and git commit for every run. A number you cannot
   regenerate is not a result.
2. Record the noise floor alongside the effect, every single time. A tau of 0.62
   means nothing until you know whether prompt templates alone produce 0.95 or
   0.60 on that axis.
3. Write the "what I expected" line BEFORE you look at the output. This is the
   cheapest protection against talking yourself into a story afterwards.
4. If a result surprises you, the first hypothesis is a bug, not a discovery.
   Note what you did to rule that out.

---

## Run template

### Run ID: `RUN-000`

| Field | Value |
|---|---|
| Date | YYYY-MM-DD |
| Git commit | `abc1234` |
| Config file | `configs/default.yaml` |
| Models (n) | |
| Arms | |
| Axes | |
| Bit width | 4 |
| Calibration corpus | |
| Prompt templates | |
| Option shuffling | yes / no |
| Items per axis | |
| Hardware | |
| Wall time | |

**What I expected before running:**

> One or two sentences. Write this first.

**Headline numbers**

| Axis | Arm | Kendall tau-b | 95% CI | Mean score delta | Flip rate | Rank swaps | Top-3 overlap |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

**Noise floor for comparison (ablation A2)**

| Axis | Mean pairwise tau between prompt templates | Is the effect above the floor? |
|---|---|---|
| | | |

**Rank displacement by family (ablation A1)**

| Family | n | Mean signed | Mean absolute | Max absolute |
|---|---|---|---|---|
| | | | | |

**Deployment cost**

| Arm | Peak memory (GB) | Tokens/sec | Accuracy per GB |
|---|---|---|---|
| | | | |

**What actually happened:**

>

**Did it match my expectation? If not, what is the most likely bug?**

>

**Decision: what I do next because of this run**

>

---

## Run index

| Run ID | Date | One-line summary | Used in paper? |
|---|---|---|---|
| RUN-000 | | template | no |
