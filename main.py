"""
main.py
Entry point for the Stackelberg Code Review project.
Runs the full evaluation pipeline and generates publication-ready plots.

Plots generated
---------------
  vdr_comparison.png          – Bar chart: VDR per strategy at default budget
  f1_comparison.png           – Bar chart: F1-score per strategy
  latency_comparison.png      – Bar chart: latency per strategy
  budget_sweep_vdr.png        – Line chart: VDR vs. budget ratio (all strategies)
  budget_sweep_f1.png         – Line chart: F1  vs. budget ratio (all strategies)
  budget_sweep_efficiency.png – Line chart: detection efficiency vs. budget
  vdr_ci_bars.png             – Bar chart with 95% bootstrap CI error bars (N=30 runs)
  ablation_chunk_size.png     – VDR vs. chunk-token-size (ablation study)
  radar_chart.png             – Radar chart: multi-metric comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for Kaggle/CI)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from math import pi

from src.evaluate import (
    run_experiment,
    run_budget_sweep,
    run_experiment_repeated,
    run_chunk_size_ablation,
)
from src.significance import summarise_runs, wilcoxon_test, cohens_d, effect_label
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


# ─── CI error-bar chart (N=30 runs) ──────────────────────────────────────────

def plot_ci_bars(repeated_df: pd.DataFrame, metric: str = "VDR") -> None:
    """Bar chart with 95 % bootstrap CI error bars across N repeated runs."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stats   = summarise_runs(list(repeated_df.to_dict("records")), metric=metric)
    strats  = ["Sequential", "Random", "SSG"]
    means   = [stats[s]["mean"] for s in strats]
    ci_lo   = [stats[s]["mean"] - stats[s]["ci_lo"] for s in strats]
    ci_hi   = [stats[s]["ci_hi"] - stats[s]["mean"] for s in strats]
    colors  = [STRATEGY_COLORS[s] for s in strats]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(strats))
    bars = ax.bar(x, [m * 100 for m in means], color=colors, width=0.5,
                  yerr=[[v * 100 for v in ci_lo], [v * 100 for v in ci_hi]],
                  capsize=6, error_kw={"elinewidth": 1.8, "ecolor": "#333"})
    for bar, m in zip(bars, means):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{m*100:.1f}%", ha="center", va="bottom",
                fontsize=FONTSIZE_LABEL, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(strats)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    n = repeated_df["Run"].nunique()
    _style_ax(ax, f"{metric} with 95 % Bootstrap CI  (N={n} runs)", f"{metric} (%)")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{metric.lower()}_ci_bars.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # Print significance table
    ssg_vals = repeated_df[repeated_df.Strategy == "SSG"][metric].tolist()
    for base in ["Sequential", "Random"]:
        base_vals = repeated_df[repeated_df.Strategy == base][metric].tolist()
        _, pval = wilcoxon_test(ssg_vals, base_vals)
        d       = cohens_d(ssg_vals, base_vals)
        sig     = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"  SSG vs {base}: Wilcoxon p={pval:.4f} {sig}  Cohen's d={d:.2f} ({effect_label(d)})")


# ─── Ablation: VDR vs chunk-token-size ───────────────────────────────────────

def plot_ablation_chunk_size(ablation_df: pd.DataFrame) -> None:
    """Line chart: mean VDR ± 95% CI for each strategy vs chunk token size."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    strategies = ["Sequential", "Random", "SSG"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for s in strategies:
        sub   = ablation_df[ablation_df.Strategy == s]
        grp   = sub.groupby("ChunkSize")["VDR"]
        means = grp.mean() * 100
        stds  = grp.std() * 100
        xs    = means.index.values
        ax.plot(xs, means.values, label=s,
                color=STRATEGY_COLORS[s], linewidth=LINEWIDTH, marker=MARKER, markersize=5)
        ax.fill_between(xs,
                        means.values - stds.values,
                        means.values + stds.values,
                        color=STRATEGY_COLORS[s], alpha=0.15)
    ax.set_xlabel("Chunk Token Size", fontsize=FONTSIZE_AXIS)
    ax.set_ylabel("VDR (%)", fontsize=FONTSIZE_AXIS)
    ax.set_title("Ablation: VDR vs. Chunk Token Size (40 % budget, mean ± 1σ)",
                 fontsize=FONTSIZE_TITLE, fontweight="bold", pad=8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.tick_params(labelsize=FONTSIZE_LABEL)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FONTSIZE_LABEL, framealpha=0.7)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "ablation_chunk_size.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── Radar chart (multi-metric comparison) ───────────────────────────────────

def plot_radar_chart(df: pd.DataFrame) -> None:
    """Radar (spider) chart comparing Sequential / Random / SSG on 5 metrics."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics      = ["VDR", "F1", "Precision", "Recall", "Efficiency"]
    metric_labels= ["VDR", "F1", "Precision", "Recall", "Efficiency\n(norm)"]
    strategies   = ["Sequential", "Random", "SSG"]

    # Normalise Efficiency to [0,1] for radar scale
    df = df.copy()
    max_eff = df["Efficiency"].max() or 1.0
    df["Efficiency"] = df["Efficiency"] / max_eff

    N     = len(metrics)
    angles= [n / float(N) * 2 * pi for n in range(N)]
    angles+= angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=FONTSIZE_LABEL)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.grid(linestyle="--", alpha=0.5)

    for s in strategies:
        row    = df[df.Strategy == s].iloc[0]
        values = [float(row[m]) for m in metrics] + [float(row[metrics[0]])]
        color  = STRATEGY_COLORS[s]
        ax.plot(angles, values, color=color, linewidth=LINEWIDTH, label=s)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_title("Multi-Metric Comparison (40 % budget)",
                 fontsize=FONTSIZE_TITLE, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              fontsize=FONTSIZE_LABEL, framealpha=0.8)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "radar_chart.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Stackelberg Security Games — AI Code Review Audit")
    print("=" * 60)

    # ── 1. Single-budget experiment ──
    print("\n[1/4]  Single-budget evaluation …")
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
    plot_radar_chart(results_df)

    # ── 2. Budget sweep ──
    print("\n[2/4]  Budget sweep (10 % → 80 %) …")
    sweep_df = run_budget_sweep(num_samples=100)
    sweep_file = os.path.join(RESULTS_DIR, "budget_sweep.csv")
    sweep_df.to_csv(sweep_file, index=False)
    print(f"Saved → {sweep_file}")
    print("\n  Generating budget-sweep line charts …")
    plot_budget_sweep(sweep_df)

    # ── 3. Repeated runs (CI + significance) ──
    print("\n[3/4]  Repeated-seed evaluation (N=30) for CI & significance …")
    repeated_df = run_experiment_repeated(num_samples=100, n_runs=30)
    rep_file    = os.path.join(RESULTS_DIR, "repeated_runs.csv")
    repeated_df.to_csv(rep_file, index=False)
    print(f"Saved → {rep_file}")
    print("\n  Generating CI bar chart + significance …")
    plot_ci_bars(repeated_df, metric="VDR")
    plot_ci_bars(repeated_df, metric="F1")

    # ── 4. Ablation: chunk size ──
    print("\n[4/4]  Chunk-size ablation …")
    ablation_df = run_chunk_size_ablation(num_samples=100, n_runs=10)
    abl_file    = os.path.join(RESULTS_DIR, "ablation_chunk_size.csv")
    ablation_df.to_csv(abl_file, index=False)
    print(f"Saved → {abl_file}")
    print("\n  Generating ablation chart …")
    plot_ablation_chunk_size(ablation_df)

    print("\n✓ All done — results in the 'results/' directory.")


if __name__ == "__main__":
    main()
