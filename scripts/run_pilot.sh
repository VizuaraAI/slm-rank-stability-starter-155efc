#!/usr/bin/env bash
# The smallest run that exercises the ENTIRE pipeline: build, evaluate, report.
#
# Three small models, two arms, one axis. The rank statistics will be noisy and
# meaningless at this size, and that is fine: the point is to prove the loop
# closes before you spend GPU hours on the full sweep.
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
MODELS="qwen2.5-0.5b-i smollm2-360m-i tinyllama-1.1b-chat"
ARMS="bf16 nf4"
AXES="arc_challenge"

echo "==> [1/3] building variants: $ARMS"
python -m src.build_variants --config "$CONFIG" --models $MODELS --arms $ARMS

echo "==> [2/3] evaluating on: $AXES"
python -m src.run_eval --config "$CONFIG" --models $MODELS --arms $ARMS --axes $AXES

echo "==> [3/3] building tables"
python -m src.report --config "$CONFIG"

cat <<'EOF'

Pilot complete. Look at artifacts/tables/rank_agreement.csv.

Three models cannot produce a trustworthy Kendall tau: with n=3 the only
possible values are coarse. Treat this purely as a smoke test, then scale up
with scripts/run_full_sweep.sh.
EOF
