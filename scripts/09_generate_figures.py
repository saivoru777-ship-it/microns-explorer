#!/usr/bin/env python3
"""Generate publication figures for the MICrONS replication paper.

Figure structure:
  Fig 1: L/λ distribution + threshold sensitivity sweep
  Fig 2: Continuous L/λ vs spatial metrics (scatterplots, the objectivity check)
  Fig 3: Regime-stratified spatial organization (violin/box by regime)
  Fig 4: Subtype breakdown with Cohen's d effect sizes
  Fig 5: Partner compactness by regime (mechanistic result)
  Fig S1: Pairwise compactness with non-monotonicity discussion
  Fig S2: Excitatory vs inhibitory continuous comparison
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "replication_full"
FIGURES_DIR = PROJECT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── Wong 2011 colorblind-safe palette ──
COLORS = {
    "summation-like": "#0072B2",
    "nonlinear-prone": "#E69F00",
    "compartmentalized": "#D55E00",
    "exc": "#009E73",
    "inh": "#CC79A7",
    "bpc": "#F0E442",
    "ngc": "#56B4E9",  # sky blue
    "null": "#999999",
}

SUBTYPE_COLORS = {
    "excitatory": "#009E73",
    "BC": "#CC79A7",
    "MC": "#882255",  # dark pink
    "BPC": "#F0E442",
    "NGC": "#56B4E9",
}

REGIME_ORDER = ["summation-like", "nonlinear-prone", "compartmentalized"]
REGIME_COLORS = [COLORS["summation-like"], COLORS["nonlinear-prone"], COLORS["compartmentalized"]]


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# ═══════════════════════════════════════════════════════════
# FIGURE 1: L/λ distribution + sensitivity sweep
# ═══════════════════════════════════════════════════════════

def fig1_electrotonic_and_sensitivity(df, save_path=None):
    """Panel A: L/λ histogram by cell type with regime thresholds.
    Panel B: Sensitivity sweep showing coefficient stability across thresholds."""
    setup_style()
    fig, (ax_hist, ax_sweep) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                             gridspec_kw={"width_ratios": [1.2, 1]})

    # ── Panel A: L/λ distribution ──
    valid = df[df["electrotonic_length"].notna() & np.isfinite(df["electrotonic_length"])]
    e_vals = valid["electrotonic_length"].values

    # Get thresholds
    from src.structural_regimes import compute_tertile_thresholds
    t_lo, t_hi = compute_tertile_thresholds(e_vals)

    for ct, color, label in [
        ("excitatory", COLORS["exc"], "Excitatory"),
        ("inhibitory", COLORS["inh"], "Inhibitory"),
        ("inhibitory_BPC", COLORS["bpc"], "BPC"),
        ("inhibitory_NGC", COLORS["ngc"], "NGC"),
    ]:
        subset = valid[valid["cell_type"] == ct]["electrotonic_length"]
        if len(subset) > 0:
            ax_hist.hist(subset, bins=np.logspace(np.log10(0.005), np.log10(5), 50),
                        alpha=0.5, color=color, label=label, density=True)

    ax_hist.axvline(t_lo, color="k", ls="--", lw=1, alpha=0.7, label=f"Tertile ({t_lo:.3f})")
    ax_hist.axvline(t_hi, color="k", ls="--", lw=1, alpha=0.7, label=f"Tertile ({t_hi:.3f})")
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("Electrotonic length (L/λ)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("(A) Electrotonic length distribution")
    ax_hist.legend(frameon=False, fontsize=6)

    # ── Panel B: Sensitivity sweep ──
    metrics_sweep = {}
    for metric_name, label, color in [
        ("clark_evans_z", "Clark-Evans", "#0072B2"),
        ("interval_cv_z", "Interval CV", "#D55E00"),
    ]:
        # Try both naming conventions
        sweep_path = RESULTS_DIR / f"sensitivity_{metric_name}.csv"
        if not sweep_path.exists():
            sweep_path = RESULTS_DIR / f"sensitivity_fixed_{metric_name}.csv"
        if sweep_path.exists():
            sweep = pd.read_csv(sweep_path)
            metrics_sweep[metric_name] = sweep
            converged = sweep[sweep["converged"]]
            if len(converged) > 0:
                ax_sweep.plot(range(len(converged)), converged["max_abs_coefficient"].values,
                             "o-", color=color, ms=4, lw=1.2, label=label)

    ax_sweep.set_xlabel("Threshold variant")
    ax_sweep.set_ylabel("|Regime 2 coefficient|")
    ax_sweep.set_title("(B) Effect size stability")
    ax_sweep.legend(frameon=False)

    # Label x-ticks
    if metrics_sweep:
        first_sweep = list(metrics_sweep.values())[0]
        converged = first_sweep[first_sweep["converged"]]
        labels = []
        for _, row in converged.iterrows():
            if row["source"] == "primary_tertile":
                labels.append("Tertile")
            elif row["source"] == "biophysical":
                labels.append("Biophys.")
            else:
                labels.append(f"Q{len(labels)-1}")
        ax_sweep.set_xticks(range(len(labels)))
        ax_sweep.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════
# FIGURE 2: Continuous L/λ vs spatial metrics (THE KEY FIGURE)
# ═══════════════════════════════════════════════════════════

def fig2_continuous_scatterplots(df, save_path=None):
    """Scatterplots of L/λ vs Clark-Evans z and Interval CV z.
    Split by cell type (exc vs inh). Regression lines overlaid.
    This is the objectivity figure — no arbitrary thresholds."""
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))

    metrics = [
        ("clark_evans_z", "Clark-Evans (z)"),
        ("interval_cv_z", "Interval CV (z)"),
    ]
    cell_groups = [
        ("Excitatory", df[df["cell_type"] == "excitatory"], COLORS["exc"]),
        ("Inhibitory", df[df["cell_type"].str.startswith("inhibitory")], COLORS["inh"]),
    ]

    for col, (metric, ylabel) in enumerate(metrics):
        for row, (ct_label, ct_df, color) in enumerate(cell_groups):
            ax = axes[row, col]
            work = ct_df.dropna(subset=[metric, "electrotonic_length"]).copy()
            work = work[np.isfinite(work["electrotonic_length"])]

            if len(work) < 50:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                       transform=ax.transAxes)
                continue

            # Clip metric outliers at 1st/99th percentile for visualization
            lo, hi = np.percentile(work[metric], [1, 99])
            plot_work = work[(work[metric] >= lo) & (work[metric] <= hi)]

            # Subsample for visual clarity (plot max 3000 points)
            rng = np.random.default_rng(42)
            if len(plot_work) > 3000:
                idx = rng.choice(len(plot_work), 3000, replace=False)
                plot_df = plot_work.iloc[idx]
            else:
                plot_df = plot_work

            ax.scatter(plot_df["electrotonic_length"], plot_df[metric],
                      s=3, alpha=0.15, color=color, edgecolors="none", rasterized=True)

            # Binned medians for trend visibility (use median, more robust than mean)
            # Only bin within the 5th-95th percentile of L/λ to avoid sparse-tail artifacts
            el_vals = work["electrotonic_length"].values
            el_lo, el_hi = np.percentile(el_vals, [5, 95])
            core = work[(el_vals >= el_lo) & (el_vals <= el_hi)]
            log_el = np.log10(core["electrotonic_length"].values)
            bins = np.linspace(log_el.min(), log_el.max(), 15)
            bin_idx = np.digitize(log_el, bins)
            bin_means_x = []
            bin_means_y = []
            bin_sems = []
            for b in range(1, len(bins)):
                mask = bin_idx == b
                if mask.sum() >= 20:
                    bin_means_x.append(10**((bins[b-1] + bins[b]) / 2))
                    bin_means_y.append(np.median(core[metric].values[mask]))
                    bin_sems.append(core[metric].values[mask].std() / np.sqrt(mask.sum()))

            if bin_means_x:
                ax.errorbar(bin_means_x, bin_means_y, yerr=bin_sems,
                           fmt="o-", color="black", ms=4, lw=1.5, capsize=2,
                           zorder=10, label="Binned median ± SEM")

            # Spearman correlation (on full data, not clipped)
            rho, p = stats.spearmanr(work["electrotonic_length"], work[metric])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(0.03, 0.97, f"ρ = {rho:.3f} {sig}\nn = {len(work):,}",
                   transform=ax.transAxes, va="top", ha="left", fontsize=7,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            ax.set_xscale("log")
            ax.axhline(0, color="k", ls=":", lw=0.5, alpha=0.5)
            ax.set_ylim(lo - 0.1 * abs(hi - lo), hi + 0.1 * abs(hi - lo))

            if row == 1:
                ax.set_xlabel("Electrotonic length (L/λ)")
            ax.set_ylabel(ylabel)

            if row == 0:
                ax.set_title(ylabel)

            # Cell type label
            ax.text(0.97, 0.97, ct_label, transform=ax.transAxes, va="top", ha="right",
                   fontsize=8, fontweight="bold", color=color)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════
# FIGURE 3: Regime-stratified spatial organization
# ═══════════════════════════════════════════════════════════

def fig3_regime_violin(df, save_path=None):
    """Violin/box of Clark-Evans z and Interval CV z by regime.
    Primary metrics only (compactness → supplement)."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

    metrics = [
        ("clark_evans_z", "Clark-Evans (z)"),
        ("interval_cv_z", "Interval CV (z)"),
    ]

    valid = df[df["regime"].isin([0, 1, 2])].copy()

    for ax, (metric, ylabel) in zip(axes, metrics):
        data_by_regime = []
        positions = []
        colors = []
        for j, (regime_idx, regime_name) in enumerate(zip([0, 1, 2], REGIME_ORDER)):
            subset = valid[valid["regime"] == regime_idx][metric].dropna()
            if len(subset) > 0:
                data_by_regime.append(subset.values)
                positions.append(j)
                colors.append(REGIME_COLORS[j])

        if data_by_regime:
            parts = ax.violinplot(data_by_regime, positions=positions, showextrema=False)
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.3)

            bp = ax.boxplot(data_by_regime, positions=positions, widths=0.3,
                           patch_artist=True, showfliers=False)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        # Set y-limits based on 1st/99th percentile to avoid outlier compression
        all_vals = np.concatenate(data_by_regime)
        lo, hi = np.percentile(all_vals, [1, 99])
        margin = 0.15 * abs(hi - lo)
        ax.set_ylim(lo - margin, hi + margin)

        ax.set_xticks(range(3))
        ax.set_xticklabels(["Summation\n-like", "Nonlinear\n-prone", "Compart-\nmentalized"],
                          fontsize=6)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="k", ls=":", lw=0.5, alpha=0.5)

        # Add median labels
        for j, data in enumerate(data_by_regime):
            med = np.median(data)
            ax.text(j, hi + margin * 0.3, f"med={med:.2f}",
                   ha="center", fontsize=6, color=colors[j])

    n_neurons = df["neuron_label"].nunique()
    fig.suptitle(f"Spatial organization by electrotonic regime (n={n_neurons:,} neurons)", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════
# FIGURE 4: Subtype breakdown with effect sizes
# ═══════════════════════════════════════════════════════════

def fig4_subtype_effect_sizes(results_path, save_path=None):
    """Bar chart of Cohen's d by subtype for each metric.
    Shows effect size magnitude, not just p-values."""
    setup_style()

    with open(results_path) as f:
        results = json.load(f)

    # Handle both old format (nested under "subtype_breakdown") and new (flat)
    subtype_data = results.get("subtype_breakdown", results)
    subtypes_order = ["excitatory", "BC", "MC", "BPC", "NGC"]
    metrics = ["clark_evans_z", "interval_cv_z"]
    metric_labels = {"clark_evans_z": "Clark-Evans", "interval_cv_z": "Interval CV"}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

    bar_width = 0.15
    x = np.arange(len(subtypes_order))

    for ax, metric in zip(axes, metrics):
        # Mixed-model regime 2 coefficients
        coefs = []
        colors = []
        n_neurons = []
        for st in subtypes_order:
            if st in subtype_data and metric in subtype_data[st].get("metrics", {}):
                m = subtype_data[st]["metrics"][metric]
                coefs.append(abs(m["regime_2_coef"]))
                n_neurons.append(subtype_data[st]["n_neurons"])
            else:
                coefs.append(0)
                n_neurons.append(0)
            colors.append(SUBTYPE_COLORS.get(st, "#999999"))

        bars = ax.bar(x, coefs, width=0.6, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)

        # Add n labels
        for i, (bar, n) in enumerate(zip(bars, n_neurons)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"n={n}", ha="center", va="bottom", fontsize=6)

        # Add significance stars
        for i, st in enumerate(subtypes_order):
            if st in subtype_data and metric in subtype_data[st].get("metrics", {}):
                p = subtype_data[st]["metrics"][metric]["regime_2_p"]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                if sig:
                    ax.text(i, coefs[i] * 0.5, sig, ha="center", va="center",
                           fontsize=8, fontweight="bold", color="white")

        ax.set_xticks(x)
        ax.set_xticklabels(subtypes_order, rotation=0, fontsize=7)
        ax.set_ylabel("|Regime 2 coefficient|")
        ax.set_title(metric_labels[metric])
        ax.axhline(0, color="k", ls=":", lw=0.5)

    fig.suptitle("Regime coupling strength by cell subtype", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════
# FIGURE 5: Partner compactness by regime
# ═══════════════════════════════════════════════════════════

def fig5_partner_compactness(partner_path, save_path=None):
    """Partner spatial spread (mean pairwise distance) by dominant regime.
    Split by postsynaptic cell type."""
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))

    partner_df = pd.read_csv(partner_path)
    partner_df = partner_df.dropna(subset=["mean_pairwise_dist_nm"])
    partner_df = partner_df[partner_df["dominant_regime"].isin([0, 1, 2])]

    subsets = [
        ("All postsynaptic", partner_df),
        ("Excitatory post", partner_df[partner_df["cell_type"] == "excitatory"]),
        ("Inhibitory post", partner_df[partner_df["cell_type"].str.startswith("inhibitory")]),
    ]

    for ax, (title, subset) in zip(axes, subsets):
        data_by_regime = []
        colors = []
        ns = []
        for regime in [0, 1, 2]:
            vals = subset[subset["dominant_regime"] == regime]["mean_pairwise_dist_nm"].values / 1000  # → μm
            data_by_regime.append(vals)
            colors.append(REGIME_COLORS[regime])
            ns.append(len(vals))

        if all(len(d) > 0 for d in data_by_regime):
            parts = ax.violinplot(data_by_regime, positions=[0, 1, 2], showextrema=False)
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.3)

            bp = ax.boxplot(data_by_regime, positions=[0, 1, 2], widths=0.3,
                           patch_artist=True, showfliers=False)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        # Median ratio annotation
        if len(data_by_regime[0]) > 0 and len(data_by_regime[2]) > 0:
            med0 = np.median(data_by_regime[0])
            med2 = np.median(data_by_regime[2])
            ratio = med2 / med0 if med0 > 0 else np.nan
            H, p = stats.kruskal(*[d for d in data_by_regime if len(d) > 0])
            ax.text(0.5, 0.97, f"R2/R0 = {ratio:.2f}×\nKW p = {p:.1e}",
                   transform=ax.transAxes, va="top", ha="center", fontsize=7,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Summ.", "Nonlin.", "Compart."], fontsize=6)
        ax.set_title(f"{title}\n(n={sum(ns):,} partners)")
        if ax == axes[0]:
            ax.set_ylabel("Partner synapse spread (μm)")

    fig.suptitle("Presynaptic partner spatial spread by postsynaptic branch regime", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════
# SUPPLEMENTARY: Pairwise compactness with non-monotonicity
# ═══════════════════════════════════════════════════════════

def figS1_compactness_nonmonotonic(df, save_path=None):
    """Pairwise compactness by regime — showing the non-monotonic pattern.
    Panel A: Violin by regime (all subtypes pooled).
    Panel B: Median compactness per subtype × regime (showing regime 1 peak)."""
    setup_style()
    fig, (ax_violin, ax_lines) = plt.subplots(1, 2, figsize=(7.0, 3.5))

    valid = df[df["regime"].isin([0, 1, 2])].copy()
    metric = "pairwise_compactness_z"

    # Panel A: Violin
    data_by_regime = []
    for regime in [0, 1, 2]:
        data_by_regime.append(valid[valid["regime"] == regime][metric].dropna().values)

    parts = ax_violin.violinplot(data_by_regime, positions=[0, 1, 2], showextrema=False)
    for pc, color in zip(parts["bodies"], REGIME_COLORS):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)

    bp = ax_violin.boxplot(data_by_regime, positions=[0, 1, 2], widths=0.3,
                          patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], REGIME_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax_violin.set_xticks([0, 1, 2])
    ax_violin.set_xticklabels(["Summ.", "Nonlin.", "Compart."], fontsize=6)
    ax_violin.set_ylabel("Pairwise compactness (z)")
    ax_violin.axhline(0, color="k", ls=":", lw=0.5)
    ax_violin.set_title("(A) Compactness by regime")

    # Panel B: Per-subtype median trend
    subtypes = ["excitatory", "BC", "MC", "BPC", "NGC"]
    for st in subtypes:
        if st == "excitatory":
            st_df = valid[valid["cell_type"] == "excitatory"]
        else:
            st_df = valid[valid["subtype"] == st]

        if len(st_df) < 20:
            continue

        medians = []
        for regime in [0, 1, 2]:
            vals = st_df[st_df["regime"] == regime][metric].dropna()
            medians.append(vals.median() if len(vals) > 0 else np.nan)

        color = SUBTYPE_COLORS.get(st, "#999999")
        ax_lines.plot([0, 1, 2], medians, "o-", color=color, ms=5, lw=1.5, label=st)

    ax_lines.set_xticks([0, 1, 2])
    ax_lines.set_xticklabels(["Summation\n-like", "Nonlinear\n-prone", "Compart-\nmentalized"], fontsize=6)
    ax_lines.set_ylabel("Median compactness (z)")
    ax_lines.axhline(0, color="k", ls=":", lw=0.5)
    ax_lines.legend(frameon=False, fontsize=6)
    ax_lines.set_title("(B) Non-monotonic pattern by subtype")

    fig.suptitle("Pairwise compactness: regime 1 as transitional integration zone", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════
# SUPPLEMENTARY: Exc vs Inh continuous comparison
# ═══════════════════════════════════════════════════════════

def figS2_exc_inh_continuous(df, save_path=None):
    """Side-by-side continuous L/λ scatterplots for exc and inh,
    showing that inhibitory coupling is more continuous/graded."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

    metric = "clark_evans_z"
    ylabel = "Clark-Evans (z)"

    for ax, (ct_label, ct_filter, color) in zip(axes, [
        ("Excitatory (ρ borderline)", df["cell_type"] == "excitatory", COLORS["exc"]),
        ("Inhibitory (ρ strong)", df["cell_type"].str.startswith("inhibitory"), COLORS["inh"]),
    ]):
        work = df[ct_filter].dropna(subset=[metric, "electrotonic_length"]).copy()
        work = work[np.isfinite(work["electrotonic_length"])]

        if len(work) < 50:
            continue

        # Subsample
        rng = np.random.default_rng(42)
        if len(work) > 2000:
            idx = rng.choice(len(work), 2000, replace=False)
            plot_df = work.iloc[idx]
        else:
            plot_df = work

        ax.scatter(plot_df["electrotonic_length"], plot_df[metric],
                  s=3, alpha=0.15, color=color, edgecolors="none", rasterized=True)

        # Binned medians (core 5th-95th percentile to avoid tail artifacts)
        el_vals = work["electrotonic_length"].values
        el_lo, el_hi = np.percentile(el_vals, [5, 95])
        core = work[(el_vals >= el_lo) & (el_vals <= el_hi)]
        log_el = np.log10(core["electrotonic_length"].values)
        bins = np.linspace(log_el.min(), log_el.max(), 15)
        bin_idx = np.digitize(log_el, bins)
        bin_x, bin_y, bin_sem = [], [], []
        for b in range(1, len(bins)):
            mask = bin_idx == b
            if mask.sum() >= 20:
                bin_x.append(10**((bins[b-1] + bins[b]) / 2))
                bin_y.append(np.median(core[metric].values[mask]))
                bin_sem.append(core[metric].values[mask].std() / np.sqrt(mask.sum()))

        ax.errorbar(bin_x, bin_y, yerr=bin_sem, fmt="o-", color="black",
                   ms=4, lw=1.5, capsize=2, zorder=10)

        rho, p = stats.spearmanr(work["electrotonic_length"], work[metric])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(0.03, 0.97, f"ρ = {rho:.3f} {sig}\nn = {len(work):,}",
               transform=ax.transAxes, va="top", fontsize=7,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Clip y-axis to 1st/99th percentile
        lo, hi = np.percentile(work[metric].dropna(), [1, 99])
        margin = 0.15 * abs(hi - lo)
        ax.set_ylim(lo - margin, hi + margin)

        ax.set_xscale("log")
        ax.set_xlabel("Electrotonic length (L/λ)")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="k", ls=":", lw=0.5)
        ax.set_title(ct_label)

    fig.suptitle("Continuous L/λ coupling: inhibitory is more graded than excitatory", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    sys.path.insert(0, str(Path.home() / "research" / "dendritic-regime-coupling"))

    print("Loading data...")
    df = pd.read_csv(RESULTS_DIR / "all_branch_features_with_spatial.csv")
    print(f"  {len(df)} branches, {df['neuron_label'].nunique()} neurons")

    print("\nGenerating figures...")

    # Fig 1
    fig1_electrotonic_and_sensitivity(df, FIGURES_DIR / "fig1_electrotonic_sensitivity.pdf")
    fig1_electrotonic_and_sensitivity(df, FIGURES_DIR / "fig1_electrotonic_sensitivity.png")

    # Fig 2
    fig2_continuous_scatterplots(df, FIGURES_DIR / "fig2_continuous_scatterplots.pdf")
    fig2_continuous_scatterplots(df, FIGURES_DIR / "fig2_continuous_scatterplots.png")

    # Fig 3
    fig3_regime_violin(df, FIGURES_DIR / "fig3_regime_violin.pdf")
    fig3_regime_violin(df, FIGURES_DIR / "fig3_regime_violin.png")

    # Fig 4
    results_path = RESULTS_DIR / "subtype_breakdown.json"
    fig4_subtype_effect_sizes(results_path, FIGURES_DIR / "fig4_subtype_effects.pdf")
    fig4_subtype_effect_sizes(results_path, FIGURES_DIR / "fig4_subtype_effects.png")

    # Fig 5
    partner_path = RESULTS_DIR / "partner_regime_mapping.csv"
    if partner_path.exists():
        fig5_partner_compactness(partner_path, FIGURES_DIR / "fig5_partner_compactness.pdf")
        fig5_partner_compactness(partner_path, FIGURES_DIR / "fig5_partner_compactness.png")

    # Supplementary
    figS1_compactness_nonmonotonic(df, FIGURES_DIR / "figS1_compactness_nonmonotonic.pdf")
    figS1_compactness_nonmonotonic(df, FIGURES_DIR / "figS1_compactness_nonmonotonic.png")

    figS2_exc_inh_continuous(df, FIGURES_DIR / "figS2_exc_inh_continuous.pdf")
    figS2_exc_inh_continuous(df, FIGURES_DIR / "figS2_exc_inh_continuous.png")

    print("\nAll figures saved to:", FIGURES_DIR)
    plt.close("all")


if __name__ == "__main__":
    main()
