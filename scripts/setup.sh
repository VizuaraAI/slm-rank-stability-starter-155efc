#!/usr/bin/env bash
# One-time environment setup.
#
# The 4-bit libraries are the fragile part. If gptqmodel or autoawq refuse to
# build on your GPU, install the rest and drop that arm from configs/default.yaml
# for now: the study still works with fewer arms, and you can add it back later.
set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "==> creating virtual environment in .venv"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> upgrading pip"
pip install --upgrade pip wheel setuptools

echo "==> installing requirements"
pip install -r requirements.txt

echo "==> verifying the import-light modules (no GPU needed)"
python -c "from src.rank_stats import kendall_tau_b, flip_rate; print('rank_stats ok')"
python -c "from src.config import load_config; c = load_config('configs/default.yaml'); print(f'config ok: {len(c.models)} models, {len(c.arms)} arms, {len(c.axes)} axes')"

cat <<'EOF'

Setup done. Activate with:  source .venv/bin/activate

Next:
  bash scripts/run_pilot.sh     # smallest end-to-end loop, ~20 minutes on one GPU
EOF
