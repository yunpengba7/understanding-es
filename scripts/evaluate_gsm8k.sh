#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 MODEL LABEL OUTPUT_DIR" >&2
  exit 2
fi

uv run es-evaluate --model "$1" --label "$2" --output-dir "$3"
