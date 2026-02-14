#!/usr/bin/env bash
# Compare composition with and without the details controlnet
set -euo pipefail

AIMG=".venv/bin/aimg"
OUTDIR="outputs/compose-controlnet-comparison"
SIZE="1920x1080"
SEED=42
STEPS=40

PROMPTS=(
    "a majestic mountain landscape at golden hour, dramatic clouds, alpine lake reflection"
    "a cyberpunk city street at night, neon signs, rain-slicked pavement, detailed architecture"
    "a detailed portrait of an elderly fisherman, weathered face, ocean background, dramatic lighting"
)

LABELS=(
    "mountain-landscape"
    "cyberpunk-city"
    "fisherman-portrait"
)

mkdir -p "$OUTDIR"

for i in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$i]}"
    label="${LABELS[$i]}"
    echo ""
    echo "=== Generating: $label ==="

    echo "  [1/2] WITH controlnet..."
    $AIMG imagine "$prompt" \
        --seed "$SEED" --steps "$STEPS" --size "$SIZE" \
        --compose-phase-controlnet \
        --outdir "$OUTDIR/${label}_with_controlnet" 2>&1 | tail -3

    echo "  [2/2] WITHOUT controlnet..."
    $AIMG imagine "$prompt" \
        --seed "$SEED" --steps "$STEPS" --size "$SIZE" \
        --no-compose-phase-controlnet \
        --outdir "$OUTDIR/${label}_without_controlnet" 2>&1 | tail -3

    echo "  Done: $label"
done

echo ""
echo "All images saved to $OUTDIR"
ls -lh "$OUTDIR"
