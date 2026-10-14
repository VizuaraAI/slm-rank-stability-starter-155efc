"""Entry point 1 of 2: build the quantized model variants.

This is the slow, GPU-bound half of the project (the analogue of a training
run). It materialises every model-by-arm variant named in the config and caches
it on disk so that evaluation can be rerun cheaply.

    python -m src.build_variants --config configs/default.yaml
    python -m src.build_variants --config configs/default.yaml --models qwen2.5-0.5b-i --arms gptq-w4

Arms that cannot be built in this environment (missing wheel, no GGUF artifact)
are reported as skipped rather than crashing the sweep: a partial leaderboard is
still analysable, you just report fewer arms.
"""

from __future__ import annotations

import argparse
import json
from typing import List

from .config import load_config
from .datasets_loader import load_calibration_texts
from .quantize import ArmUnavailable, build_variant, measure_footprint, variant_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 4-bit deployment variants")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--models", nargs="*", default=None, help="subset of model ids")
    p.add_argument("--arms", nargs="*", default=None, help="subset of arm names")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.paths.mkdirs()

    models = [m for m in cfg.models if args.models is None or m.id in args.models]
    arms = [a for a in cfg.arms if args.arms is None or a.name in args.arms]
    if not models or not arms:
        raise SystemExit("nothing to build: check --models / --arms against your config")

    # Calibration data is loaded once and shared, so every calibrated arm sees
    # exactly the same bytes. Varying it is ablation A3, not an accident.
    calib_texts: List[str] = []
    if any(a.needs_calibration for a in arms):
        calib_texts = load_calibration_texts(cfg.calibration)

    report = []
    for model in models:
        for arm in arms:
            tag = f"{model.id}/{arm.name}"
            try:
                print(f"\n[build] {tag}")
                built, _tokenizer = build_variant(
                    model, arm, cfg.paths.variants_dir, cfg.calibration, calib_texts
                )
                footprint = measure_footprint(built)
                report.append({"model": model.id, "arm": arm.name, "status": "ok", **footprint})
                print(f"[build] {tag} ok  {footprint}")
                del built
            except ArmUnavailable as exc:
                report.append({"model": model.id, "arm": arm.name,
                               "status": "skipped", "reason": str(exc)})
                print(f"[build] {tag} SKIPPED: {exc}")
            except Exception as exc:                       # noqa: BLE001
                report.append({"model": model.id, "arm": arm.name,
                               "status": "failed", "reason": repr(exc)})
                print(f"[build] {tag} FAILED: {exc!r}")

    out = cfg.paths.variants_dir / "build_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    ok = sum(1 for r in report if r["status"] == "ok")
    print(f"\n[build] {ok}/{len(report)} variants built. Report: {out}")
    print(f"[build] variants cached under {variant_dir(cfg.paths.variants_dir, models[0], arms[0]).parent.parent}")


if __name__ == "__main__":
    main()
