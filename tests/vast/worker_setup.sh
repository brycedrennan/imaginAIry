#!/bin/bash
set -euo pipefail
set -o xtrace
cd ~/project

T_TOTAL=$SECONDS
T_UV_INSTALL=0

if ! command -v uv &>/dev/null; then
  t=$SECONDS
  curl -LsSf https://astral.sh/uv/install.sh | sh
  T_UV_INSTALL=$((SECONDS - t))
fi
source "$HOME/.local/bin/env" 2>/dev/null || true

# OpenCV system libs + pip deps — run apt-get in background while uv installs
apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 libxcb1 &
APT_PID=$!

t=$SECONDS
uv pip install --system -e . --group dev
T_UV_INSTALL_DEPS=$((SECONDS - t))

# Wait for apt-get to finish (probably already done)
wait $APT_PID

mkdir -p ./tests/test_output

t=$SECONDS
pytest --co -q
T_PYTEST_COLLECT=$((SECONDS - t))

# Quick smoke test — verify CLI can generate on GPU (2 steps, tiny image)
# HD benchmark removed — benchmark_instance already validated GPU TFLOPS
t=$SECONDS
imagine "pizza" --steps 2
T_SMOKE=$((SECONDS - t))

T_TOTAL=$((SECONDS - T_TOTAL))

set +o xtrace
echo ""
echo "============================================"
echo "  Worker Setup Performance Summary"
echo "============================================"
echo "  uv install:           ${T_UV_INSTALL}s"
echo "  uv pip install deps:  ${T_UV_INSTALL_DEPS}s"
echo "  pytest collect:       ${T_PYTEST_COLLECT}s"
echo "  smoke (pizza 2-step): ${T_SMOKE}s"
echo "  ---"
echo "  total:                ${T_TOTAL}s"
echo "============================================"
