#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${1:-/workspace/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Creating virtual environment: ${VENV_PATH}"
"${PYTHON_BIN}" -m venv "${VENV_PATH}"

echo "Upgrading pip"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip

echo "Installing requirements"
"${VENV_PATH}/bin/pip" install -r /workspace/Disertatie/requirements.txt

echo "Running environment preflight checks"
"${VENV_PATH}/bin/python" /workspace/Disertatie/scripts/check_env.py

echo "Bootstrap complete."
