#!/usr/bin/env bash
# Single entrypoint for running the full modular pipeline on a fresh Linux GPU box (e.g. a
# RunPod pod): sets up the uv-managed environment if needed, then runs prepare_data.py,
# train_model.py, evaluate_model.py in order. See docs/CLAUDE.md / docs/MODULAR_PIPELINE_PLAN.md
# (local-only, not tracked in git -- see .gitignore) for what each stage does and why.
#
# Usage: ./run_pipeline.sh
#
# Requires data/raw/data.pkl to already be present (upload it to the pod/volume yourself --
# this script does not fetch it). Everything else (venv, deps, the three pipeline stages) is
# handled here.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f data/raw/data.pkl ]; then
    echo "ERROR: data/raw/data.pkl not found. Upload it to this machine before running." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "== Installing uv =="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d .venv ]; then
    echo "== Creating virtual environment (Python 3.11) =="
    uv venv --python 3.11 .venv
fi

echo "== Installing dependencies =="
# UV_HTTP_TIMEOUT raised because the pinned torch wheel is ~2.7GB; uv's 30s default can time
# out on it depending on the pod's network.
UV_HTTP_TIMEOUT=600 uv pip install -r requirements.txt -p .venv

PYTHON=.venv/bin/python

echo "== nvidia-smi =="
nvidia-smi || echo "(nvidia-smi not available -- continuing, but training will be very slow/impossible without a GPU)"

echo "== Stage 1/3: prepare_data.py =="
"$PYTHON" prepare_data.py

echo "== Stage 2/3: train_model.py =="
"$PYTHON" train_model.py

echo "== Stage 3/3: evaluate_model.py =="
"$PYTHON" evaluate_model.py

echo "== Done. Final model: models/finetuned/  |  checkpoints: models/checkpoints/ =="
