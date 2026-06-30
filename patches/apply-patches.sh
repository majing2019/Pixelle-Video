#!/bin/bash
# Apply comfykit patches to support param_mappings in RunningHub workflows.
# Run this after `pip install comfykit` or after recreating the venv.

set -e

VENV_DIR="$(cd "$(dirname "$0")/.." && pwd)/.venv"
COMFYKIT_DIR="$VENV_DIR/lib/python3.12/site-packages/comfykit"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$COMFYKIT_DIR" ]; then
    echo "Error: comfykit not found at $COMFYKIT_DIR"
    echo "Make sure comfykit is installed and the venv is at .venv"
    exit 1
fi

echo "Applying comfykit patches..."

patch -p1 -d "$COMFYKIT_DIR/.." < "$SCRIPT_DIR/comfykit-executor.patch"
echo "  ✓ comfykit-executor.patch applied"

patch -p1 -d "$COMFYKIT_DIR/.." < "$SCRIPT_DIR/comfykit-runninghub-executor.patch"
echo "  ✓ comfykit-runninghub-executor.patch applied"

echo "All patches applied successfully."
