"""Rank statistics: the actual contribution of this project.

Everyone else reports score deltas. The decision a leaderboard drives is
ordinal ("which model do I ship?"), so the statistic has to be ordinal too.
This module is intentionally dependency-light (numpy and scipy only) so you can
develop and test it on a laptop while the GPU sweep runs elsewhere.

Vocabulary:
  tau_b         Kendall rank correlation between the BF16 and 4-bit orderings.
                1.0 means the leaderboard survived deployment intact.
  topk_overlap  what a practitioner picking from a shortlist actually feels.
  flip_rate     fraction of items whose correctness changed. Can be large even
                when the score barely moves, which is the whole point.
  displacement  signed change in leaderboard position, per model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau


@dataclass(frozen=True)
class RankComparison:
    axis_name: str
    arm_name: str
    tau_b: float
    tau_p_value: float
    tau_ci_low: float
    tau_ci_high: float
    topk_overlap: Dict[int, float]
    mean_abs_score_delta: float
    flip_rate: float
    n_swaps: int
    displacement: Dict[str, int]

    def survived(self, threshold: float = 0.9) -> bool:
        """A leaderboard 'survives' if rank agreement is high AND the
        confidence interval does not reach down into scrambled territory."""
        return self.tau_b >= threshold and self.tau_ci_low >= 0.6


def leaderboard(scores: Mapping[str, float]) -> List[str]:
    """Model ids ordered best to worst. Ties break on model id so that the
    ordering is deterministic across runs."""
    return sorted(scores, key=lambda m: (-scores[m], m))


def ranks(scores: Mapping[str, float]) -> Dict[str, int]:
    """Model id -> 1-based leaderboard position."""
    return {model_id: i + 1 for i, model_id in enumerate(leaderboard(scores))}


def kendall_tau_b(
    reference: Mapping[str, float], candidate: Mapping[str, float]
) -> Tuple[float, float]:
    """Kendall tau-b between two score dicts, over their shared models."""
    shared = sorted(set(reference) & set(candidate))
    if len(shared) < 3:
        return float("nan"), float("nan")
    a = [reference[m] for m in shared]
    b = [candidate[m] for m in shared]
    result = kendalltau(a, b, variant="b")
    return float(result.statistic), float(result.pvalue)


def topk_overlap(
    reference: Mapping[str, float], candidate: Mapping[str, float], k: int
) -> float:
    """Fraction of the reference top-k that is still in the candidate top-k."""
    ref_top = set(leaderboard(reference)[:k])
    cand_top = set(leaderboard(candidate)[:k])
    if not ref_top:
        return float("nan")
    return len(ref_top & cand_top) / len(ref_top)


def displacement(
    reference: Mapping[str, float], candidate: Mapping[str, float]
) -> Dict[str, int]:
    """Signed rank change per model. Positive means the model moved UP the
    leaderboard after quantization (a smaller rank number)."""
    ref_ranks, cand_ranks = ranks(reference), ranks(candidate)
    return {
        m: ref_ranks[m] - cand_ranks[m]
        for m in sorted(set(ref_ranks) & set(cand_ranks))
    }


def count_swaps(
    reference: Mapping[str, float], candidate: Mapping[str, float], min_positions: int = 2
) -> int:
    """How many models moved by at least min_positions."""
    return sum(1 for d in displacement(reference, candidate).values() if abs(d) >= min_positions)


def flip_rate(
    reference_items: Mapping[str, int], candidate_items: Mapping[str, int]
) -> float:
    """Fraction of shared items whose correctness changed.

    This is the statistic from 'Accuracy is Not All You Need': a model can hold
    its score while changing a large share of its individual answers, so a
    stable number does not mean stable behaviour.
    """
    shared = set(reference_items) & set(candidate_items)
    if not shared:
        return float("nan")
    changed = sum(1 for i in shared if reference_items[i] != candidate_items[i])
    return changed / len(shared)


def paired_bootstrap_tau(
    reference_items: Mapping[str, Mapping[str, int]],
    candidate_items: Mapping[str, Mapping[str, int]],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """Confidence interval on tau_b by resampling ITEMS, not models.

    Resampling items is the right move here: the models are the population we
    care about, while the benchmark questions are the sample introducing noise.
    Both arms are resampled on the same item indices, which preserves the
    pairing and is what makes this interval tighter than an unpaired one.

    Args:
        reference_items: model_id -> {item_id: 0/1} at full precision.
        candidate_items: model_id -> {item_id: 0/1} at 4 bits.

    Returns:
        (ci_low, ci_high) for Kendall tau_b.
    """
    shared_models = sorted(set(reference_items) & set(candidate_items))
    if len(shared_models) < 3:
        return float("nan"), float("nan")

    shared_items = sorted(
        set.intersection(*[set(reference_items[m]) for m in shared_models])
        & set.intersection(*[set(candidate_items[m]) for m in shared_models])
    )
    if len(shared_items) < 10:
        return float("nan"), float("nan")

    ref_matrix = np.array(
        [[reference_items[m][i] for i in shared_items] for m in shared_models], dtype=float
    )
    cand_matrix = np.array(
        [[candidate_items[m][i] for i in shared_items] for m in shared_models], dtype=float
    )

    rng = np.random.default_rng(seed)
    n_items = len(shared_items)
    taus = np.empty(n_resamples, dtype=float)

    for b in range(n_resamples):
        idx = rng.integers(0, n_items, size=n_items)
        ref_scores = ref_matrix[:, idx].mean(axis=1)
        cand_scores = cand_matrix[:, idx].mean(axis=1)
        taus[b] = kendalltau(ref_scores, cand_scores, variant="b").statistic

    taus = taus[~np.isnan(taus)]
    if taus.size == 0:
        return float("nan"), float("nan")

    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(taus, alpha)), float(np.quantile(taus, 1.0 - alpha))


def compare_arm(
    axis_name: str,
    arm_name: str,
    reference_scores: Mapping[str, float],
    candidate_scores: Mapping[str, float],
    reference_items: Mapping[str, Mapping[str, int]],
    candidate_items: Mapping[str, Mapping[str, int]],
    topk: Sequence[int] = (1, 3, 5),
    n_resamples: int = 10000,
    confidence: float = 0.95,
) -> RankComparison:
    """Full rank comparison for one axis and one 4-bit arm. This produces one
    row of the paper's main results table."""
    tau, p_value = kendall_tau_b(reference_scores, candidate_scores)
    ci_low, ci_high = paired_bootstrap_tau(
        reference_items, candidate_items, n_resamples=n_resamples, confidence=confidence
    )

    shared_models = sorted(set(reference_scores) & set(candidate_scores))
    mean_delta = (
        float(np.mean([abs(candidate_scores[m] - reference_scores[m]) for m in shared_models]))
        if shared_models
        else float("nan")
    )
    flips = [
        flip_rate(reference_items[m], candidate_items[m])
        for m in shared_models
        if m in reference_items and m in candidate_items
    ]
    flips = [f for f in flips if not np.isnan(f)]

    return RankComparison(
        axis_name=axis_name,
        arm_name=arm_name,
        tau_b=tau,
        tau_p_value=p_value,
        tau_ci_low=ci_low,
        tau_ci_high=ci_high,
        topk_overlap={k: topk_overlap(reference_scores, candidate_scores, k) for k in topk},
        mean_abs_score_delta=mean_delta,
        flip_rate=float(np.mean(flips)) if flips else float("nan"),
        n_swaps=count_swaps(reference_scores, candidate_scores),
        displacement=displacement(reference_scores, candidate_scores),
    )


def noise_floor_tau(per_template_scores: Mapping[str, Mapping[str, float]]) -> float:
    """Rank instability caused by prompt formatting ALONE, at full precision.

    Pass {template_name: {model_id: accuracy}} from the BF16 arm. Returns the
    mean pairwise tau_b between templates. Any quantization effect you report
    must be worse than this, otherwise you have measured your own harness
    instead of quantization.
    """
    names = sorted(per_template_scores)
    taus: List[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            tau, _ = kendall_tau_b(per_template_scores[names[i]], per_template_scores[names[j]])
            if not np.isnan(tau):
                taus.append(tau)
    return float(np.mean(taus)) if taus else float("nan")


def displacement_by_family(
    disp: Mapping[str, int], families: Mapping[str, str]
) -> Dict[str, Dict[str, float]]:
    """Group rank displacement by architecture family (ablation A1).

    If family explains the movement, rank transfer across precision can be
    trusted within a family and not across families.
    """
    grouped: Dict[str, List[int]] = {}
    for model_id, d in disp.items():
        grouped.setdefault(families.get(model_id, "unknown"), []).append(d)

    return {
        fam: {
            "n": len(vals),
            "mean_signed": round(float(np.mean(vals)), 3),
            "mean_abs": round(float(np.mean(np.abs(vals))), 3),
            "max_abs": int(np.max(np.abs(vals))),
        }
        for fam, vals in sorted(grouped.items())
    }
