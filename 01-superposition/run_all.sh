#!/bin/sh
# Runs every experiment in this folder in order, in the uv environment defined by the repo-root pyproject.toml.
# 3, 7 and 8 reuse models saved by 2 and 6. About 8 minutes on a laptop CPU.
cd "$(dirname "$0")" && for f in [0-9][0-9]_*.py; do echo "=== $f"; uv run python "$f" || exit 1; done
