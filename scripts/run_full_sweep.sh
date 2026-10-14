#!/usr/bin/env bash
# The full study: every model, every arm, every axis.
#
# Expect many GPU hours. Both stages are resumable, so interrupting this is
# safe: build_variants reuses cached variants and run_eval skips result files
# that already exist. Run it under tmux or nohup.
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"

echo "==> building all variants (slow, cached on disk)"
python -m src.build_variants --config "$CONFIG"

echo "==> evaluating all variants on all axes (resumable)"
python -m src.run_eval --config "$CONFIG"

echo "==> building tables"
python -m src.report --config "$CONFIG"

cat <<'EOF'

Full sweep complete.

  artifacts/tables/rank_agreement.csv   paper Table 1
  artifacts/tables/arm_summary.csv      paper Table 2
  artifacts/tables/family_analysis.csv  ablation A1
  artifacts/tables/noise_floor.csv      ablation A2, read this FIRST

Before writing any result into the paper, check that the quantization effect in
rank_agreement.csv is clearly larger than the corresponding noise floor. If it
is not, the honest finding is "rankings are stable to within measurement
noise", which is still a publishable result and still worth reporting.
EOF
