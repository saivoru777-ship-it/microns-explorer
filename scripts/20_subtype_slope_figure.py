#!/usr/bin/env python3
"""Generate subtype-stratified ICV slope distribution figure.

Replaces Figure 6 (excitatory vs inhibitory blobs) with a per-subtype
breakdown that makes the bimodal inhibitory distribution visible.

Key findings to show:
- BC: +1.808 (inverted, outside preferred territory)
- MC: -0.804 (intermediate)
- BPC: -2.755, NGC: -3.248 (steeper than excitatory average)
- 6PCT: -3.039 (steepest excitatory, not L5-ET)
- Inhibitory pooled mean ≈ -0.07 is an averaging artifact

Output: figures/fig6_icv_slope_by_subtype.pdf / .png
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "replication_full"
FIGURES_DIR = PROJECT_DIR / "figures"

sys.path.insert(0, str(Path.home() / "research" / "neurostat"))
sys.path.insert(0, str(Path.home() / "research" / "neurostat-input-clustering" / "src"))
sys.path.insert(0, str(Path.home() / "research" / "dendritic-regime-coupling"))

import statsmodels.formula.api as smf


def compute_slopes_with_subtype(branch_df):
    """Fit random-slope LME on ALL neurons; return per-neuron slopes with subtype."""
    df = branch_df.dropna(subset=["interval_cv_z", "electrotonic_length",
                                   "synapse_count", "total_length_nm",
                                   "neuron_label", "cell_type", "subtype"]).copy()
    df = df[df["synapse_count"] >= 3]

    print(f"  Fitting on {len(df):,} branches, {df['neuron_label'].nunique()} neurons...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = smf.mixedlm(
            "interval_cv_z ~ electrotonic_length + synapse_count + total_length_nm",
            df,
            groups=df["neuron_label"],
            re_formula="~electrotonic_length"
        ).fit(reml=True, method="lbfgs")

    fixed_slope = float(mod.params.get("electrotonic_length", 0))

    neuron_meta = df.drop_duplicates("neuron_label").set_index("neuron_label")[
        ["cell_type", "subtype"]
    ]

    slopes = []
    for nl, re_dict in mod.random_effects.items():
        keys = list(re_dict.keys())
        random_slope = float(re_dict[keys[1]]) if len(keys) >= 2 else \
                       float(re_dict.get("electrotonic_length", 0))
        total = fixed_slope + random_slope
        ctype = neuron_meta.loc[nl, "cell_type"] if nl in neuron_meta.index else "unknown"
        sub = neuron_meta.loc[nl, "subtype"] if nl in neuron_meta.index else "unknown"
        slopes.append({"neuron_label": nl, "total_slope": total,
                        "cell_type": ctype, "subtype": sub})

    return pd.DataFrame(slopes)


def main():
    print("Generating subtype-stratified ICV slope figure...")
    branch_df = pd.read_csv(RESULTS_DIR / "all_branch_features_with_spatial.csv")
    slopes_df = compute_slopes_with_subtype(branch_df)

    # Clip extreme values for visualization (1-99th percentile per subtype)
    def clip_group(arr):
        lo, hi = np.percentile(arr, [1, 99])
        return np.clip(arr, lo, hi)

    # Define subtype groups with display labels, ordered by median slope
    # Inhibitory subtypes first (sorted by slope desc), then excitatory (sorted)
    subtype_config = [
        # (subtype_key, display_label, color, group)
        ("BC",   "BC\n(PV-like)",      "#e31a1c", "inhibitory"),
        ("MC",   "MC\n(SST-like)",     "#fd8d3c", "inhibitory"),
        ("BPC",  "BPC\n(VIP-like)",    "#9e9ac8", "inhibitory"),
        ("NGC",  "NGC",                "#6a3d9a", "inhibitory"),
        ("5PET", "L5-ET",              "#a8ddb5", "excitatory"),
        ("5PIT", "L5-IT",              "#7bccc4", "excitatory"),
        ("5PNP", "L5-NP",              "#43a2ca", "excitatory"),
        ("23P",  "L2/3",               "#2b8cbe", "excitatory"),
        ("4P",   "L4",                 "#0868ac", "excitatory"),
        ("6PIT", "L6-IT",              "#084081", "excitatory"),
        ("6PCT", "L6-CT",              "#001f3f", "excitatory"),
    ]

    # Build plot data
    plot_data = []
    for sub_key, label, color, group in subtype_config:
        if group == "inhibitory":
            mask = slopes_df["subtype"] == sub_key
        else:
            mask = (slopes_df["cell_type"] == "excitatory") & (slopes_df["subtype"] == sub_key)
        arr = slopes_df[mask]["total_slope"].values
        if len(arr) >= 5:
            plot_data.append((sub_key, label, color, group, arr, clip_group(arr)))

    n_groups = len(plot_data)
    positions = list(range(n_groups))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                              gridspec_kw={"width_ratios": [3, 1]})

    # ── Left panel: violin by subtype ─────────────────────────────────────
    ax = axes[0]

    # Separator between inhibitory and excitatory
    n_inh = sum(1 for _, _, _, g, _, _ in plot_data if g == "inhibitory")
    ax.axvspan(-0.5, n_inh - 0.5, alpha=0.06, color="#e31a1c", zorder=0)
    ax.axvspan(n_inh - 0.5, n_groups - 0.5, alpha=0.06, color="#2171b5", zorder=0)

    # Reference line at 0 and at excitatory median
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.4, zorder=1)
    ax.axhline(-2.23, color="#2171b5", lw=1.5, ls=":", alpha=0.6, zorder=1,
               label="Excitatory population median (−2.23)")

    for pos, (sub_key, label, color, group, arr, arr_clip) in enumerate(plot_data):
        # Violin
        parts = ax.violinplot([arr_clip], positions=[pos], showmedians=False,
                               showextrema=False, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
            pc.set_edgecolor("none")

        # Median dot + IQR bar
        med = np.median(arr)
        q25, q75 = np.percentile(arr, [25, 75])
        ax.plot([pos], [med], "o", color=color, ms=7, zorder=5,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.plot([pos, pos], [q25, q75], "-", color=color, lw=2.5, zorder=4, alpha=0.9)
        ax.text(pos, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 2,
                f"{med:+.2f}", ha="center", va="bottom", fontsize=7.5,
                color=color, fontweight="bold")

    # Print median values on plot
    y_annot = max(arr_clip.max() for _, _, _, _, arr, arr_clip in plot_data) + 0.5
    for pos, (sub_key, label, color, group, arr, arr_clip) in enumerate(plot_data):
        med = np.median(arr)
        ax.text(pos, y_annot, f"{med:+.2f}", ha="center", va="bottom",
                fontsize=7.5, color=color, fontweight="bold", rotation=0)

    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label, _, _, _, _ in plot_data], fontsize=9)
    ax.set_ylabel("Per-neuron ICV coupling slope\n(L/λ → ICV z-score)", fontsize=11)
    ax.set_title("ICV coupling slopes by cell subtype\n"
                 "Median ± IQR shown; inhibitory (red zone) vs excitatory (blue zone)",
                 fontsize=10, fontweight="bold")

    # Group labels
    ax.text((n_inh - 1) / 2, ax.get_ylim()[0] + 0.3, "INHIBITORY", ha="center",
            fontsize=9, color="#c0392b", fontweight="bold", alpha=0.7)
    ax.text(n_inh + (n_groups - n_inh - 1) / 2, ax.get_ylim()[0] + 0.3,
            "EXCITATORY", ha="center", fontsize=9, color="#2171b5",
            fontweight="bold", alpha=0.7)

    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Right panel: summary bar chart of medians ─────────────────────────
    ax2 = axes[1]
    medians = [(label, np.median(arr), color)
               for _, label, color, _, arr, _ in plot_data]
    labels_r, meds_r, colors_r = zip(*medians)
    y_pos = list(range(len(labels_r)))[::-1]  # flip for readability

    for y, (label, med, color) in zip(y_pos, zip(labels_r, meds_r, colors_r)):
        ax2.barh(y, med, color=color, alpha=0.75, height=0.7)

    ax2.axvline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax2.axvline(-2.23, color="#2171b5", lw=1.5, ls=":", alpha=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_r, fontsize=9)
    ax2.set_xlabel("Median slope", fontsize=10)
    ax2.set_title("Medians\n(sorted by position)", fontsize=10, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "The inhibitory near-zero slope masks opposing subtype behaviors:\n"
        "BC inverted (+1.81), BPC/NGC steeper than excitatory average (−2.76, −3.25)",
        fontsize=10, fontstyle="italic"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    for fmt in ["pdf", "png"]:
        out = FIGURES_DIR / f"fig6_icv_slope_by_subtype.{fmt}"
        plt.savefig(out, dpi=150 if fmt == "png" else None, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close()

    # Print summary stats for paper text
    print("\n  Summary for paper text:")
    for sub_key, label, color, group, arr, _ in plot_data:
        med = np.median(arr)
        n = len(arr)
        print(f"    {sub_key:6s} ({group[:3]}): n={n:4d}  median={med:+.3f}")


if __name__ == "__main__":
    main()
