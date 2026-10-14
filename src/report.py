"""Turn the per-item result files into the paper's tables.

    python -m src.report --config configs/default.yaml

Produces, under paths.tables_dir:
  rank_agreement.csv   one row per (axis, arm): this IS the paper's Table 1
  arm_summary.csv      one row per arm on the composite axis: Table 2
  family_analysis.csv  rank displacement grouped by architecture family: A1
  noise_floor.csv      prompt-template-only rank instability: A2

Read noise_floor.csv before you believe anything in rank_agreement.csv. If the
quantization effect is not clearly larger than the noise floor, you have
measured your harness rather than quantization.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import pandas as pd

from .config import load_config
from .rank_stats import compare_arm, displacement_by_family, noise_floor_tau


def load_results(results_dir: Path) -> Dict[str, dict]:
    """Read every result JSON into {(model, arm, axis) -> payload}."""
    out: Dict[str, dict] = {}
    for path in sorted(results_dir.glob("*__*__*.json")):
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        key = (payload["model_id"], payload["arm_name"], payload["axis_name"])
        out[key] = payload
    if not out:
        raise SystemExit(f"no result files found in {results_dir}; run src.run_eval first")
    print(f"[report] loaded {len(out)} result files")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the paper's tables")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.paths.mkdirs()
    results = load_results(cfg.paths.results_dir)

    ref_arm = cfg.reference_arm.name
    quant_arms = [a.name for a in cfg.arms if not a.is_reference]
    families = {m.id: m.family for m in cfg.models}

    # ---------------------------------------------------- Table 1 and Table 2
    rows, family_rows = [], []
    composite_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    composite_items: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(dict)

    for axis in cfg.axes:
        ref_scores, ref_items = {}, {}
        for model in cfg.models:
            payload = results.get((model.id, ref_arm, axis.name))
            if payload:
                ref_scores[model.id] = payload["accuracy"]
                ref_items[model.id] = payload["per_item"]
        if len(ref_scores) < 3:
            print(f"[report] skipping {axis.name}: need 3+ reference models, "
                  f"have {len(ref_scores)}")
            continue

        for arm in quant_arms:
            cand_scores, cand_items = {}, {}
            for model in cfg.models:
                payload = results.get((model.id, arm, axis.name))
                if payload:
                    cand_scores[model.id] = payload["accuracy"]
                    cand_items[model.id] = payload["per_item"]
            shared = set(ref_scores) & set(cand_scores)
            if len(shared) < 3:
                continue

            cmp_ = compare_arm(
                axis.name, arm,
                {m: ref_scores[m] for m in shared},
                {m: cand_scores[m] for m in shared},
                {m: ref_items[m] for m in shared},
                {m: cand_items[m] for m in shared},
                topk=cfg.statistics.topk,
                n_resamples=cfg.statistics.bootstrap_resamples,
                confidence=cfg.statistics.confidence,
            )
            rows.append({
                "axis": axis.name,
                "arm": arm,
                "n_models": len(shared),
                "kendall_tau_b": round(cmp_.tau_b, 4),
                "tau_ci_low": round(cmp_.tau_ci_low, 4),
                "tau_ci_high": round(cmp_.tau_ci_high, 4),
                "mean_abs_score_delta": round(cmp_.mean_abs_score_delta, 4),
                "flip_rate": round(cmp_.flip_rate, 4),
                "rank_swaps": cmp_.n_swaps,
                "survived": cmp_.survived(),
                **{f"top{k}_overlap": round(v, 3) for k, v in cmp_.topk_overlap.items()},
            })
            for fam, stats in displacement_by_family(cmp_.displacement, families).items():
                family_rows.append({"axis": axis.name, "arm": arm, "family": fam, **stats})

        # Accumulate the composite leaderboard: mean accuracy over axes, and
        # item ids namespaced by axis so they stay unique when pooled.
        for arm_name in [ref_arm] + quant_arms:
            for model in cfg.models:
                payload = results.get((model.id, arm_name, axis.name))
                if not payload:
                    continue
                composite_scores[arm_name].setdefault(model.id, 0.0)
                composite_scores[arm_name][model.id] += payload["accuracy"] / len(cfg.axes)
                composite_items[arm_name].setdefault(model.id, {}).update(
                    {f"{axis.name}/{k}": v for k, v in payload["per_item"].items()}
                )

    rank_df = pd.DataFrame(rows)
    rank_path = cfg.paths.tables_dir / "rank_agreement.csv"
    rank_df.to_csv(rank_path, index=False)
    print(f"[report] wrote {rank_path} ({len(rank_df)} rows)")

    # ------------------------------------------------- composite arm summary
    arm_rows = []
    for arm in quant_arms:
        shared = set(composite_scores[ref_arm]) & set(composite_scores.get(arm, {}))
        if len(shared) < 3:
            continue
        cmp_ = compare_arm(
            "composite", arm,
            {m: composite_scores[ref_arm][m] for m in shared},
            {m: composite_scores[arm][m] for m in shared},
            {m: composite_items[ref_arm][m] for m in shared},
            {m: composite_items[arm][m] for m in shared},
            topk=cfg.statistics.topk,
            n_resamples=cfg.statistics.bootstrap_resamples,
            confidence=cfg.statistics.confidence,
        )
        arm_rows.append({
            "arm": arm,
            "mean_abs_score_delta": round(cmp_.mean_abs_score_delta, 4),
            "flip_rate": round(cmp_.flip_rate, 4),
            "rank_swaps": cmp_.n_swaps,
            "kendall_tau_b": round(cmp_.tau_b, 4),
            "tau_ci_low": round(cmp_.tau_ci_low, 4),
            "tau_ci_high": round(cmp_.tau_ci_high, 4),
            **{f"top{k}_overlap": round(v, 3) for k, v in cmp_.topk_overlap.items()},
        })

    arm_path = cfg.paths.tables_dir / "arm_summary.csv"
    pd.DataFrame(arm_rows).to_csv(arm_path, index=False)
    print(f"[report] wrote {arm_path} ({len(arm_rows)} rows)")

    fam_path = cfg.paths.tables_dir / "family_analysis.csv"
    pd.DataFrame(family_rows).to_csv(fam_path, index=False)
    print(f"[report] wrote {fam_path} ({len(family_rows)} rows)")

    # --------------------------------------------------------- A2 noise floor
    noise_rows = []
    for axis in cfg.axes:
        per_template: Dict[str, Dict[str, float]] = defaultdict(dict)
        for model in cfg.models:
            payload = results.get((model.id, ref_arm, axis.name))
            if not payload:
                continue
            for template, acc in payload.get("per_template_accuracy", {}).items():
                per_template[template][model.id] = acc
        usable = {t: s for t, s in per_template.items() if len(s) >= 3}
        if len(usable) >= 2:
            noise_rows.append({
                "axis": axis.name,
                "n_templates": len(usable),
                "mean_pairwise_tau_between_templates": round(noise_floor_tau(usable), 4),
            })

    noise_path = cfg.paths.tables_dir / "noise_floor.csv"
    pd.DataFrame(noise_rows).to_csv(noise_path, index=False)
    print(f"[report] wrote {noise_path} ({len(noise_rows)} rows)")

    if not rank_df.empty:
        print("\n[report] rank agreement (paper Table 1):")
        print(rank_df[["axis", "arm", "kendall_tau_b", "flip_rate", "survived"]]
              .to_string(index=False))
    if noise_rows:
        print("\n[report] noise floor (A2). Compare against the taus above:")
        print(pd.DataFrame(noise_rows).to_string(index=False))


if __name__ == "__main__":
    main()
