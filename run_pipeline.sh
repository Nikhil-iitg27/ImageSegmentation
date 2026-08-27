#!/usr/bin/env bash
# Single entrypoint for running the full modular pipeline on a fresh Linux GPU box (e.g. a
# RunPod pod): sets up the uv-managed environment if needed, fetches data/raw/data.pkl from
# Kaggle if it isn't already present, then runs prepare_data.py, train_model.py,
# evaluate_model.py in order. See docs/CLAUDE.md / docs/MODULAR_PIPELINE_PLAN.md (local-only,
# not tracked in git -- see .gitignore) for what each stage does and why.
#
# Usage: ./run_pipeline.sh
#
# data/raw/data.pkl: if not already present, this script tries to download it from the Kaggle
# dataset "shayakbhattacharya/finetune" (confirmed to be the same file as the data_s.pkl the
# notebook itself produces -- same size). This requires a Kaggle API token already configured
# on this machine (however Kaggle's own site told you to set it up -- a kaggle.json/access_token
# file under ~/.kaggle/, or KAGGLE_USERNAME/KAGGLE_KEY env vars; the kaggle CLI resolves whichever
# is present on its own, this script doesn't need to know which). If no token is configured, or
# the download fails for any reason, this falls back to telling you to transfer the file
# manually instead of failing with a confusing Kaggle stack trace.
set -euo pipefail

cd "$(dirname "$0")"

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

if [ ! -f data/raw/data.pkl ]; then
    echo "== data/raw/data.pkl not found -- attempting to fetch from Kaggle =="
    uv pip install kaggle -p .venv
    mkdir -p data/raw

    if .venv/bin/kaggle datasets download -d shayakbhattacharya/finetune -p data/raw --unzip; then
        # The file's name inside the Kaggle dataset isn't guaranteed to be data.pkl (the
        # notebook that produced it saves it as data_s.pkl) -- find whatever .pkl landed and
        # move it to the exact path prepare_data.py expects.
        if [ ! -f data/raw/data.pkl ]; then
            found="$(find data/raw -maxdepth 3 -iname '*.pkl' ! -name 'data.pkl' | head -n 1)"
            if [ -n "$found" ]; then
                echo "Found $found -- moving to data/raw/data.pkl"
                mv "$found" data/raw/data.pkl
            fi
        fi
    fi

    if [ ! -f data/raw/data.pkl ]; then
        echo "ERROR: data/raw/data.pkl still not found after attempting Kaggle download." >&2
        echo "Either configure a Kaggle API token on this machine and re-run, or transfer" >&2
        echo "data.pkl here manually (see docs/RUNPOD_GUIDE.md step 4)." >&2
        exit 1
    fi
    echo "== data/raw/data.pkl ready (from Kaggle) =="
fi

echo "== nvidia-smi =="
nvidia-smi || echo "(nvidia-smi not available -- continuing, but training will be very slow/impossible without a GPU)"

echo "== Stage 1/3: prepare_data.py =="
"$PYTHON" prepare_data.py

echo "== Stage 2/3: train_model.py =="
"$PYTHON" train_model.py

echo "== Stage 3/3: evaluate_model.py =="
"$PYTHON" evaluate_model.py

echo "== Done. Final model: models/finetuned/  |  checkpoints: models/checkpoints/ =="
