#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 MODEL OUTPUT_DIR" >&2
  exit 2
fi

uv run es-train --model "$1" --output-dir "$2" --mode two-epochs
