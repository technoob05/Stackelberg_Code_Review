#!/usr/bin/env bash
# reproduce.sh  –  One-command reproduction of all results from
# "Strategic Attention in AI-Generated Code Review:
#  A Resource-Constrained Stackelberg Game for Small Language Models"
#
# Usage:
#   bash reproduce.sh
#
# Requirements: Python 3.9+ on PATH
# Results (CSV + PNG) will be written to ./results/

set -euo pipefail

echo "============================================================"
echo " Stackelberg Code Review – Reproduction Script"
echo "============================================================"

# ── 1. Install dependencies ──────────────────────────────────────
echo ""
echo "[1/3] Installing Python dependencies …"
pip install -r requirements.txt --quiet

# ── 2. Run full experiment pipeline ─────────────────────────────
echo ""
echo "[2/3] Running full 4-stage experiment pipeline …"
echo "       Stage 1 – single-budget eval  (budget=40%)"
echo "       Stage 2 – budget sweep        (10% … 80%)"
echo "       Stage 3 – N=30 repeated runs  (bootstrap CI + Wilcoxon)"
echo "       Stage 4 – chunk-size ablation (6 sizes × 10 seeds)"
echo ""
python main.py

# ── 3. Report output location ────────────────────────────────────
echo ""
echo "[3/3] Done. All artefacts written to ./results/"
echo ""
echo "  Key files:"
echo "    results/comparison_vdr.png        – Stage 1 VDR bar chart"
echo "    results/budget_sweep_vdr.png      – Stage 2 budget sweep"
echo "    results/ci_bars_VDR.png           – Stage 3 CI bars (VDR)"
echo "    results/ci_bars_F1.png            – Stage 3 CI bars (F1)"
echo "    results/ablation_chunk_size.png   – Stage 4 ablation"
echo "    results/radar_chart.png           – Multi-metric radar chart"
echo ""
echo "  Expected headline result:"
echo "    SSG VDR ≈ 42 %  vs  Sequential ≈ 20 %  vs  Random ≈ 14 %"
echo "    (at 40 % token budget, Devign-100 benchmark)"
echo "============================================================"
