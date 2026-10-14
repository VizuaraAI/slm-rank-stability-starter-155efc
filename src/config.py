"""Configuration loading.

The whole study is driven by one YAML file (see configs/default.yaml). This
module turns it into lightweight dataclasses so that the rest of the code gets
attribute access and a single place to look for defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class ModelSpec:
    id: str
    hf_name: str
    family: str
    params_b: float


@dataclass(frozen=True)
class ArmSpec:
    name: str
    method: str
    bits: Optional[int] = None
    group_size: Optional[int] = None
    double_quant: bool = False
    quant_type: Optional[str] = None

    @property
    def is_reference(self) -> bool:
        return self.method == "none"

    @property
    def needs_calibration(self) -> bool:
        return self.method in {"gptq", "awq"}


@dataclass(frozen=True)
class AxisSpec:
    name: str
    kind: str                      # "mcqa" or "generative"
    hf_dataset: str
    split: str = "test"
    hf_config: Optional[str] = None
    max_items: int = 300
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        if self.kind not in {"mcqa", "generative"}:
            raise ValueError(
                f"axis {self.name!r}: kind must be 'mcqa' or 'generative', got {self.kind!r}"
            )


@dataclass(frozen=True)
class CalibrationSpec:
    dataset: str = "allenai/c4"
    subset: str = "en"
    n_samples: int = 128
    seq_len: int = 2048
    seed: int = 1234


@dataclass(frozen=True)
class EvaluationSpec:
    prompt_templates: List[str] = field(default_factory=lambda: ["plain"])
    shuffle_options: bool = False
    shuffle_seeds: List[int] = field(default_factory=lambda: [0])
    batch_size: int = 8
    max_seq_len: int = 2048
    greedy: bool = True
    seed: int = 0


@dataclass(frozen=True)
class StatisticsSpec:
    bootstrap_resamples: int = 10000
    topk: List[int] = field(default_factory=lambda: [1, 3, 5])
    confidence: float = 0.95


@dataclass(frozen=True)
class Paths:
    variants_dir: Path = Path("artifacts/variants")
    results_dir: Path = Path("artifacts/results")
    tables_dir: Path = Path("artifacts/tables")

    def mkdirs(self) -> None:
        for p in (self.variants_dir, self.results_dir, self.tables_dir):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    run_name: str
    models: List[ModelSpec]
    arms: List[ArmSpec]
    axes: List[AxisSpec]
    calibration: CalibrationSpec
    evaluation: EvaluationSpec
    statistics: StatisticsSpec
    paths: Paths

    def model(self, model_id: str) -> ModelSpec:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"unknown model id {model_id!r}; known: {[m.id for m in self.models]}")

    def arm(self, arm_name: str) -> ArmSpec:
        for a in self.arms:
            if a.name == arm_name:
                return a
        raise KeyError(f"unknown arm {arm_name!r}; known: {[a.name for a in self.arms]}")

    def axis(self, axis_name: str) -> AxisSpec:
        for ax in self.axes:
            if ax.name == axis_name:
                return ax
        raise KeyError(f"unknown axis {axis_name!r}; known: {[a.name for a in self.axes]}")

    @property
    def reference_arm(self) -> ArmSpec:
        for a in self.arms:
            if a.is_reference:
                return a
        raise ValueError("config defines no reference arm (method: none)")


def _filter_kwargs(cls: type, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only keys the dataclass actually declares, so a stray YAML key
    produces a clear warning instead of a TypeError deep in the call stack."""
    allowed = set(getattr(cls, "__dataclass_fields__", {}))
    unknown = set(raw) - allowed
    if unknown:
        print(f"[config] ignoring unknown keys for {cls.__name__}: {sorted(unknown)}")
    return {k: v for k, v in raw.items() if k in allowed}


def load_config(path: str | Path) -> Config:
    """Read a YAML config file into a Config object."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    paths_raw = raw.get("paths", {}) or {}
    paths = Paths(
        variants_dir=Path(paths_raw.get("variants_dir", "artifacts/variants")),
        results_dir=Path(paths_raw.get("results_dir", "artifacts/results")),
        tables_dir=Path(paths_raw.get("tables_dir", "artifacts/tables")),
    )

    return Config(
        run_name=raw.get("run_name", path.stem),
        models=[ModelSpec(**_filter_kwargs(ModelSpec, m)) for m in raw.get("models", [])],
        arms=[ArmSpec(**_filter_kwargs(ArmSpec, a)) for a in raw.get("arms", [])],
        axes=[AxisSpec(**_filter_kwargs(AxisSpec, a)) for a in raw.get("axes", [])],
        calibration=CalibrationSpec(**_filter_kwargs(CalibrationSpec, raw.get("calibration", {}) or {})),
        evaluation=EvaluationSpec(**_filter_kwargs(EvaluationSpec, raw.get("evaluation", {}) or {})),
        statistics=StatisticsSpec(**_filter_kwargs(StatisticsSpec, raw.get("statistics", {}) or {})),
        paths=paths,
    )
