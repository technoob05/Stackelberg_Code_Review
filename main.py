"""
main.py
Entry point for the Stackelberg Code Review project.
Runs the full evaluation pipeline and generates publication-ready plots.

Plots generated
---------------
  vdr_comparison.png       – Bar chart: VDR per strategy at default budget
  latency_comparison.png   – Bar chart: latency per strategy
  f1_comparison.png        – Bar chart: F1-score per strategy
  budget_sweep_vdr.png     – Line chart: VDR vs. budget ratio (all strategies)
  budget_sweep_f1.png      – Line chart: F1  vs. budget ratio (all strategies)
  budget_sweep_efficiency.png – Line chart: detection efficiency vs. budget
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for Kaggle/CI)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from src.evaluate import run_experiment, run_budget_sweep
from src.config import RESULTS_FILE, RESULTS_DIR

# ─── Plot styling ─────────────────────────────────────────────────────────────

STRATEGY_COLORS = {
    "Sequential": "#6c757d",
    "Random":     "#0d6efd",
    "SSG":        "#198754",
}
LINEWIDTH = 2.2
MARKER    = "o"
FONTSIZE_TITLE = 13
FONTSIZE_AXIS  = 11
FONTSIZE_LABEL = 9.5


def _style_ax(ax, title: str, ylabel: str, xlabel: str = "Strategy"):
    ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_AXIS)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_AXIS)
    ax.tick_params(labelsize=FONTSIZE_LABEL)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


# ─── Bar charts (single budget run) ──────────────────────────────────────────

def plot_bar_results(df: pd.DataFrame) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── VDR ──
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [STRATEGY_COLORS.get(s, "#555") for s in df["Strategy"]]
    bars   = ax.bar(df["Strategy"], df["VDR"] * 100, color=colors, width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.8,
            f"{h:.1f}%", ha="center", va="bottom",
            fontsize=FONTSIZE_LABEL, fontweight="bold",
        )
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    _style_ax(ax, "Vulnerability Detection Rate (VDR) by Strategy",
              "VDR (%)")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "vdr_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # ── F1 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df["Strategy"], df["F1"], color=colors, width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.005,
            f"{h:.3f}", ha="center", va="bottom",
            fontsize=FONTSIZE_LABEL, fontweight="bold",
        )
    ax.set_ylim(0, 1.05)
    _style_ax(ax, "F1-Score by Strategy", "F1")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "f1_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Latency ──
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df["Strategy"], df["Latency_s"],
                  color=["#fd7e14", "#dc3545", "#6f42c1"], width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.001,
            f"{h:.3f}s", ha="center", va="bottom",
            fontsize=FONTSIZE_LABEL, fontweight="bold",
        )
    _style_ax(ax, "Execution Latency by Strategy", "Latency (s)")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "latency_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── Budget sweep line charts ─────────────────────────────────────────────────

def plot_budget_sweep(sweep_df: pd.DataFrame) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    strategies = sweep_df["Strategy"].unique()

    def _sweep_line(metric: str, ylabel: str, title: str, filename: str,
                    as_pct: bool = False):
        fig, ax = plt.subplots(figsize=(9, 5))
        for s in strategies:
            sub = sweep_df[sweep_df["Strategy"] == s].sort_values("Budget_Ratio")
            y   = sub[metric] * 100 if as_pct else sub[metric]
            ax.plot(
                sub["Budget_Ratio"] * 100, y,
                label=s, color=STRATEGY_COLORS.get(s, "#555"),
                linewidth=LINEWIDTH, marker=MARKER, markersize=5,
            )
        ax.set_xlabel("Token Budget (% of total PR tokens)", fontsize=FONTSIZE_AXIS)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE_AXIS)
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight="bold", pad=8)
        ax.tick_params(labelsize=FONTSIZE_LABEL)
        ax.grid(linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=FONTSIZE_LABEL, framealpha=0.7)
        if as_pct:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        fig.tight_layout()
        path = os.path.join(RESULTS_DIR, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")

    _sweep_line("VDR",        "VDR (%)",
                "Vulnerability Detection Rate vs. Token Budget",
                "budget_sweep_vdr.png", as_pct=True)
    _sweep_line("F1",         "F1-Score",
                "F1-Score vs. Token Budget",
                "budget_sweep_f1.png")
    _sweep_line("Efficiency", "Detections per 1K Tokens",
                "Token Efficiency vs. Token Budget",
                "budget_sweep_efficiency.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Stackelberg Security Games — AI Code Review Audit")
    print("=" * 60)

    # ── 1. Single-budget experiment ──
    print("\n[1/2]  Single-budget evaluation …")
    results_df = run_experiment(num_samples=100)

    print("\n" + "=" * 60)
    print("RESULTS TABLE (default budget)")
    print("=" * 60)
    print(
        results_df[["Strategy", "VDR", "F1", "Precision", "Recall",
                    "Tokens_Used", "Efficiency", "Latency_s"]]
        .to_string(index=False)
    )
    print("=" * 60)

    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved → {RESULTS_FILE}")

    print("\n  Generating bar charts …")
    plot_bar_results(results_df)

    # ── 2. Budget sweep ──
    print("\n[2/2]  Budget sweep (10 % → 80 %) …")
    sweep_df = run_budget_sweep(num_samples=100)

    sweep_file = os.path.join(RESULTS_DIR, "budget_sweep.csv")
    sweep_df.to_csv(sweep_file, index=False)
    print(f"Saved → {sweep_file}")

    print("\n  Generating budget-sweep line charts …")
    plot_budget_sweep(sweep_df)

    print("\n✓ All done — results in the 'results/' directory.")


if __name__ == "__main__":
    main()
