"""Entry point 2 of 2: evaluate every variant and write per-item results.

    python -m src.run_eval --config configs/default.yaml
    python -m src.run_eval --config configs/default.yaml --axes arc_challenge --models qwen2.5-0.5b-i

Writes one JSON file per (model, arm, axis) into paths.results_dir. Each file
carries the per-item correctness record, which is what makes flip rates and
paired bootstrap intervals possible later. Existing files are skipped unless
--overwrite is passed, so an interrupted sweep resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .datasets_loader import load_axis, load_calibration_texts
from .evaluate import evaluate_axis
from .quantize import ArmUnavailable, build_variant


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate all variants on all axes")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--axes", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.paths.mkdirs()

    models = [m for m in cfg.models if args.models is None or m.id in args.models]
    arms = [a for a in cfg.arms if args.arms is None or a.name in args.arms]
    axes = [a for a in cfg.axes if args.axes is None or a.name in args.axes]
    if not (models and arms and axes):
        raise SystemExit("nothing to evaluate: check --models / --arms / --axes")

    # Load every axis once up front. Item ids must be identical across variants,
    # otherwise the pairing that the whole analysis depends on silently breaks.
    axis_items = {ax.name: load_axis(ax) for ax in axes}

    calib_texts = []
    if any(a.needs_calibration for a in arms):
        calib_texts = load_calibration_texts(cfg.calibration)

    for model in models:
        for arm in arms:
            pending = []
            for ax in axes:
                out_path = cfg.paths.results_dir / f"{model.id}__{arm.name}__{ax.name}.json"
                if out_path.exists() and not args.overwrite:
                    print(f"[eval] skip (exists) {out_path.name}")
                    continue
                pending.append((ax, out_path))
            if not pending:
                continue

            # Build once per (model, arm), then evaluate every pending axis, so
            # a 3B model is not reloaded six times.
            try:
                print(f"\n[eval] loading {model.id}/{arm.name}")
                built, tokenizer = build_variant(
                    model, arm, cfg.paths.variants_dir, cfg.calibration, calib_texts
                )
            except ArmUnavailable as exc:
                print(f"[eval] SKIP {model.id}/{arm.name}: {exc}")
                continue

            for ax, out_path in pending:
                result = evaluate_axis(
                    built, tokenizer, axis_items[ax.name], ax,
                    cfg.evaluation, model.id, arm.name,
                )
                with out_path.open("w", encoding="utf-8") as fh:
                    json.dump(result.to_dict(), fh, indent=2)
                print(f"[eval] wrote {out_path.name}  acc={result.accuracy:.4f} "
                      f"spread={result.template_spread:.4f}")

            del built

    print(f"\n[eval] done. Results in {cfg.paths.results_dir}")
    print("[eval] next: python -m src.report --config " + args.config)


if __name__ == "__main__":
    main()
