"""Build the 4-bit deployment variants.

One function per arm. Each returns a loaded model ready for evaluation, and,
where the arm produces an on-disk artifact, writes it under
paths.variants_dir/<model_id>/<arm_name> so it can be reloaded without
re-quantizing (quantization is the slow part of this project).

Design note: all heavy imports are local to the functions that need them. That
way a missing autoawq wheel only breaks the AWQ arm instead of making the whole
package unimportable, and the rank statistics stay testable on a laptop.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .config import ArmSpec, CalibrationSpec, ModelSpec


class ArmUnavailable(RuntimeError):
    """Raised when an arm cannot be built in this environment. The runner
    catches this and records the arm as skipped rather than crashing the sweep,
    because a partial leaderboard is still analysable."""


def variant_dir(variants_dir: Path, model: ModelSpec, arm: ArmSpec) -> Path:
    return Path(variants_dir) / model.id / arm.name


def load_tokenizer(model: ModelSpec):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.hf_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"      # required for batched generation
    return tok


def build_bf16(model: ModelSpec):
    """The reference arm: no quantization, just load in bfloat16."""
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model.hf_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


def build_bnb_nf4(model: ModelSpec, arm: ArmSpec):
    """NF4 with double quantization, the bitsandbytes format that ships
    through the HuggingFace stack."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=bool(arm.double_quant),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return AutoModelForCausalLM.from_pretrained(
        model.hf_name,
        quantization_config=quant_config,
        device_map="auto",
    )


def build_gptq(
    model: ModelSpec,
    arm: ArmSpec,
    calib_texts: List[str],
    calib: CalibrationSpec,
    out_dir: Path,
):
    """GPTQ W4A16 via gptqmodel (the maintained successor to auto-gptq)."""
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
    except ImportError as exc:
        raise ArmUnavailable(f"gptqmodel not installed: {exc}") from exc

    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        print(f"[quant] reusing cached GPTQ variant at {out_dir}")
        return GPTQModel.load(str(out_dir), device_map="auto")

    quant_config = QuantizeConfig(bits=arm.bits or 4, group_size=arm.group_size or 128)
    gptq_model = GPTQModel.load(model.hf_name, quant_config)
    gptq_model.quantize([{"text": t[: calib.seq_len * 4]} for t in calib_texts])

    out_dir.mkdir(parents=True, exist_ok=True)
    gptq_model.save(str(out_dir))
    print(f"[quant] wrote GPTQ variant to {out_dir}")
    return GPTQModel.load(str(out_dir), device_map="auto")


def build_awq(
    model: ModelSpec,
    arm: ArmSpec,
    calib_texts: List[str],
    out_dir: Path,
):
    """AWQ W4A16 via autoawq."""
    try:
        from awq import AutoAWQForCausalLM
    except ImportError as exc:
        raise ArmUnavailable(f"autoawq not installed: {exc}") from exc

    from transformers import AutoModelForCausalLM

    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        print(f"[quant] reusing cached AWQ variant at {out_dir}")
        return AutoModelForCausalLM.from_pretrained(str(out_dir), device_map="auto")

    awq_model = AutoAWQForCausalLM.from_pretrained(model.hf_name)
    tokenizer = load_tokenizer(model)
    awq_model.quantize(
        tokenizer,
        quant_config={
            "zero_point": True,
            "q_group_size": arm.group_size or 128,
            "w_bit": arm.bits or 4,
            "version": "GEMM",
        },
        calib_data=calib_texts,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    awq_model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[quant] wrote AWQ variant to {out_dir}")
    return AutoModelForCausalLM.from_pretrained(str(out_dir), device_map="auto")


def build_gguf(model: ModelSpec, arm: ArmSpec, out_dir: Path):
    """GGUF k-quants are produced by llama.cpp's convert and quantize tools,
    not by a Python library, so this arm is handled out of band.

    Run scripts/build_gguf.sh to produce the .gguf file, then evaluate it
    through llama-cpp-python. Until that artifact exists this raises
    ArmUnavailable so the sweep skips the arm cleanly.

    This arm matters despite the extra friction: Q4_K_M is what consumer
    on-device runtimes actually load, so excluding it would mean studying a
    deployment path nobody uses.
    """
    out_dir = Path(out_dir)
    quant_type = arm.quant_type or "Q4_K_M"
    gguf_path = out_dir / f"{model.id}.{quant_type}.gguf"
    if not gguf_path.exists():
        raise ArmUnavailable(
            f"no GGUF artifact at {gguf_path}. Build it with "
            f"scripts/build_gguf.sh {model.hf_name} {quant_type}"
        )

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ArmUnavailable(f"llama-cpp-python not installed: {exc}") from exc

    return Llama(model_path=str(gguf_path), logits_all=True, verbose=False)


def build_variant(
    model: ModelSpec,
    arm: ArmSpec,
    variants_dir: Path,
    calib: CalibrationSpec,
    calib_texts: Optional[List[str]] = None,
) -> Tuple[object, object]:
    """Build (or load from cache) one model-by-arm variant.

    Returns (model, tokenizer). Raises ArmUnavailable if this environment
    cannot build the arm.
    """
    out_dir = variant_dir(variants_dir, model, arm)
    tokenizer = load_tokenizer(model)

    if arm.method == "none":
        return build_bf16(model), tokenizer
    if arm.method == "bnb_nf4":
        return build_bnb_nf4(model, arm), tokenizer
    if arm.method == "gptq":
        if not calib_texts:
            raise ValueError("GPTQ arm requires calibration texts")
        return build_gptq(model, arm, calib_texts, calib, out_dir), tokenizer
    if arm.method == "awq":
        if not calib_texts:
            raise ValueError("AWQ arm requires calibration texts")
        return build_awq(model, arm, calib_texts, out_dir), tokenizer
    if arm.method == "gguf":
        return build_gguf(model, arm, out_dir), tokenizer

    raise ValueError(f"unknown quantization method {arm.method!r} for arm {arm.name!r}")


def measure_footprint(model) -> dict:
    """Peak memory and parameter footprint for the deployment-cost columns.

    Rank changes are only interesting next to the efficiency they purchase, so
    this gets logged for every variant.
    """
    import torch

    stats = {"peak_mem_gb": None, "param_bytes_gb": None}
    try:
        total_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        stats["param_bytes_gb"] = round(total_bytes / 1024**3, 3)
    except (AttributeError, TypeError):
        pass      # llama.cpp models do not expose .parameters()

    if torch.cuda.is_available():
        stats["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    return stats
