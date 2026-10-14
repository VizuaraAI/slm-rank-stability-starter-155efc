#!/usr/bin/env bash
# Build the GGUF Q4_K_M artifact for one model using llama.cpp.
#
# GGUF is the format consumer on-device runtimes actually load, which is why it
# is worth the extra friction of a non-Python toolchain. It is produced by
# llama.cpp's conversion and quantization tools rather than by a pip package.
#
# Usage:
#   bash scripts/build_gguf.sh Qwen/Qwen2.5-0.5B-Instruct qwen2.5-0.5b-i Q4_K_M
set -euo pipefail

HF_NAME="${1:?usage: build_gguf.sh <hf_name> <model_id> [quant_type]}"
MODEL_ID="${2:?usage: build_gguf.sh <hf_name> <model_id> [quant_type]}"
QUANT="${3:-Q4_K_M}"

LLAMA_DIR="${LLAMA_CPP_DIR:-third_party/llama.cpp}"
OUT_DIR="artifacts/variants/${MODEL_ID}/gguf-q4km"

if [ ! -d "$LLAMA_DIR" ]; then
  echo "==> cloning llama.cpp into $LLAMA_DIR"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR"
  cmake --build "$LLAMA_DIR/build" --config Release -j
fi

mkdir -p "$OUT_DIR" "artifacts/hf/${MODEL_ID}"

echo "==> downloading $HF_NAME"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('${HF_NAME}', local_dir='artifacts/hf/${MODEL_ID}')
"

echo "==> converting to f16 GGUF"
python "$LLAMA_DIR/convert_hf_to_gguf.py" "artifacts/hf/${MODEL_ID}" \
  --outfile "$OUT_DIR/${MODEL_ID}.f16.gguf" --outtype f16

echo "==> quantizing to $QUANT"
"$LLAMA_DIR/build/bin/llama-quantize" \
  "$OUT_DIR/${MODEL_ID}.f16.gguf" \
  "$OUT_DIR/${MODEL_ID}.${QUANT}.gguf" \
  "$QUANT"

rm -f "$OUT_DIR/${MODEL_ID}.f16.gguf"
echo "==> wrote $OUT_DIR/${MODEL_ID}.${QUANT}.gguf"
echo "    src/quantize.py will now find this arm automatically."
