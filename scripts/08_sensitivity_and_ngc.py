#!/usr/bin/env python3
"""Sensitivity sweep (with fixed NaN handling) + NGC neuron analysis.

1. Re-run sensitivity sweep on existing 350-neuron replication data
   (now that fit_mixed_model properly drops NaN covariates)
2. Process all 24 NGC neurons through the full pipeline and add to analysis
3. Run complete subtype breakdown with NGC included
4. Effect size robustness: Cohen's d and rank-biserial r alongside p-values
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
MICRONS_DATA = PROJECT_DIR / "data" / "neurons"
RESULTS_DIR = PROJECT_DIR / "results" / "replication"

sys.path.insert(0, str(Path.home() / "research" / "neurostat"))
sys.path.insert(0, str(Path.home() / "research" / "neurostat-input-clustering" / "src"))
sys.path.insert(0, str(Path.home() / "research" / "dendritic-regime-coupling"))

from neurostat.io.swc import NeuronSkeleton, SnapResult
from soma_distance import precompute_branch_endpoint_distances

from src.branch_morphometry import (
    compute_branch_order, compute_mean_diameter, compute_electrotonic_length,
    compute_soma_distance_per_branch, count_synapses_per_branch,
    count_exc_inh_per_branch, RM, RI,
)
from src.structural_regimes import (
    classify_branches, classify_by_electrotonic_length,
    compute_tertile_thresholds, BIOPHYSICAL_THRESHOLD_LOW, BIOPHYSICAL_THRESHOLD_HIGH,
)
from src.branch_null import compute_branch_z_scores
from src.coupling_tests import (
    fit_mixed_model, sensitivity_sweep, quantile_threshold_sweep,
)

from scipy.spatial import cKDTree
import statsmodels.formula.api as smf


# ── Reuse loading functions from 06 ──

def load_neuron(stem, data_dir):
    swc_path = data_dir / f"{stem}.swc"
    syn_path = data_dir / f"{stem}_synapses.csv"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skeleton_raw = NeuronSkeleton.from_swc_file(str(swc_path), scale_factor=1000.0)
    dendrite_skel = skeleton_raw.filter_by_type([1, 3, 4])
    if len(dendrite_skel.branches) < 3:
        dendrite_skel = skeleton_raw
    syn_df = pd.read_csv(syn_path)
    syn_coords_nm = syn_df[["x_um", "y_um", "z_um"]].values * 1000.0
    snap = dendrite_skel.snap_points(syn_coords_nm, d_max=50000.0)
    valid = snap.valid
    n_valid = valid.sum()
    if n_valid < 20:
        return None
    snap_valid = SnapResult(
        branch_ids=snap.branch_ids[valid],
        branch_positions=snap.branch_positions[valid],
        distances=snap.distances[valid],
        valid=np.ones(n_valid, dtype=bool),
    )
    return dendrite_skel, snap_valid


def load_partner_types_direct(stem, snap_result, data_dir):
    partner_path = data_dir / f"{stem}_presynaptic.csv"
    syn_path = data_dir / f"{stem}_synapses.csv"
    if not partner_path.exists():
        return None
    partner_df = pd.read_csv(partner_path)
    partner_coords = partner_df[["x_nm", "y_nm", "z_nm"]].values.astype(float)
    syn_df = pd.read_csv(syn_path)
    syn_coords_nm = syn_df[["x_um", "y_um", "z_um"]].values * 1000.0
    tree = cKDTree(partner_coords)
    _, indices = tree.query(syn_coords_nm)
    return partner_df["pre_cell_type_broad"].values[indices]


def compute_branch_features_for_neuron(label, root_id, stem, data_dir):
    result = load_neuron(stem, data_dir)
    if result is None:
        return None
    skeleton, snap_valid = result
    n_branches = len(skeleton.branches)
    orders = compute_branch_order(skeleton)
    diameters = np.array([compute_mean_diameter(skeleton, i) for i in range(n_branches)])
    lengths = np.array([br.total_length for br in skeleton.branches])
    e_lengths = np.array([
        compute_electrotonic_length(lengths[i], diameters[i])
        for i in range(n_branches)
    ])
    endpoint_nodes, node_to_idx, endpoint_dists = precompute_branch_endpoint_distances(skeleton)
    soma_dists = compute_soma_distance_per_branch(skeleton, endpoint_nodes, node_to_idx, endpoint_dists)
    syn_counts = count_synapses_per_branch(snap_valid, n_branches)

    syn_path = data_dir / f"{stem}_synapses.csv"
    all_types = load_partner_types_direct(stem, snap_valid, data_dir)
    if all_types is not None:
        syn_df = pd.read_csv(syn_path)
        syn_coords_nm = syn_df[["x_um", "y_um", "z_um"]].values * 1000.0
        snap_full = skeleton.snap_points(syn_coords_nm, d_max=50000.0)
        valid_mask = snap_full.valid
        valid_types = all_types[valid_mask]
        exc_counts, inh_counts = count_exc_inh_per_branch(snap_valid, valid_types, n_branches)
    else:
        exc_counts = np.full(n_branches, np.nan)
        inh_counts = np.full(n_branches, np.nan)

    if label.startswith("exc"):
        cell_type = "excitatory"
    elif "BPC" in label:
        cell_type = "inhibitory_BPC"
    elif "NGC" in label:
        cell_type = "inhibitory_NGC"
    else:
        cell_type = "inhibitory"

    parts = label.split("_")
    subtype = parts[1] if len(parts) > 1 else "unknown"

    df = pd.DataFrame({
        "neuron_label": f"{label}_{root_id}",
        "root_id": root_id,
        "cell_type": cell_type,
        "subtype": subtype,
        "branch_idx": np.arange(n_branches),
        "branch_order": orders,
        "mean_diameter_nm": diameters,
        "total_length_nm": lengths,
        "electrotonic_length": e_lengths,
        "soma_distance_nm": soma_dists,
        "synapse_count": syn_counts,
        "exc_count": exc_counts,
        "inh_count": inh_counts,
    })
    total = df["exc_count"] + df["inh_count"]
    df["exc_fraction"] = np.where(total > 0, df["exc_count"] / total, np.nan)
    return df, skeleton, snap_valid


# ── Part 1: Process NGC neurons ──

def process_ngc_neurons():
    """Process all NGC neurons and add to the replication dataset."""
    print("=" * 60)
    print("PART 1: Processing NGC neurons")
    print("=" * 60)

    # Find NGC neurons
    ngc_neurons = []
    for swc in sorted(MICRONS_DATA.glob("inh_NGC_*.swc")):
        stem = swc.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            label = parts[0]
            root_id = parts[1]
            syn_path = MICRONS_DATA / f"{stem}_synapses.csv"
            partner_path = MICRONS_DATA / f"{stem}_presynaptic.csv"
            if syn_path.exists() and partner_path.exists():
                ngc_neurons.append((label, root_id, stem))

    print(f"  Found {len(ngc_neurons)} NGC neurons")

    # Process each
    all_dfs = []
    skeletons = {}
    snap_results = {}
    t0 = time.time()

    for i, (label, root_id, stem) in enumerate(ngc_neurons):
        result = compute_branch_features_for_neuron(label, root_id, stem, MICRONS_DATA)
        if result is not None:
            df, skel, snap = result
            all_dfs.append(df)
            nl = f"{label}_{root_id}"
            skeletons[nl] = skel
            snap_results[nl] = snap
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(ngc_neurons)} neurons ({time.time()-t0:.0f}s)")

    if not all_dfs:
        print("  No NGC data produced")
        return None

    ngc_features = pd.concat(all_dfs, ignore_index=True)
    print(f"  NGC features: {len(ngc_features)} branches across {len(all_dfs)} neurons")

    # Classify regimes using EXISTING thresholds from the 350-neuron analysis
    existing_df = pd.read_csv(RESULTS_DIR / "all_branch_features_with_spatial.csv")
    e_valid = existing_df["electrotonic_length"].dropna()
    e_valid = e_valid[~np.isinf(e_valid)]
    t_lo, t_hi = compute_tertile_thresholds(e_valid.values)
    print(f"  Using existing tertile thresholds: {t_lo:.4f} | {t_hi:.4f}")

    ngc_features["regime"] = classify_by_electrotonic_length(
        ngc_features["electrotonic_length"].values, t_lo, t_hi
    )

    # Compute spatial z-scores for NGC
    print("  Computing spatial z-scores for NGC neurons...")
    all_metrics = []
    neuron_labels = ngc_features["neuron_label"].unique()
    for i, nl in enumerate(neuron_labels):
        if nl not in skeletons:
            continue
        skel = skeletons[nl]
        snap = snap_results[nl]
        z_scores = compute_branch_z_scores(snap, skel.branches, n_draws=1000, seed=42)
        neuron_df = ngc_features[ngc_features["neuron_label"] == nl].copy()
        for metric in ["clark_evans_raw", "interval_cv_raw", "pairwise_compactness_raw",
                        "clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]:
            vals = z_scores[metric]
            neuron_df[metric] = vals[:len(neuron_df)]
        all_metrics.append(neuron_df)

    ngc_spatial = pd.concat(all_metrics, ignore_index=True)
    print(f"  NGC with spatial: {len(ngc_spatial)} branches, {ngc_spatial['neuron_label'].nunique()} neurons")

    # Save NGC standalone
    ngc_spatial.to_csv(RESULTS_DIR / "ngc_branch_features.csv", index=False)

    return ngc_spatial


# ── Part 2: Sensitivity sweep with fixed NaN handling ──

def run_sensitivity_sweep(df):
    """Re-run sensitivity sweep now that fit_mixed_model drops NaN covariates."""
    print("\n" + "=" * 60)
    print("PART 2: Sensitivity sweep (fixed NaN handling)")
    print("=" * 60)

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    covariates = ["synapse_count", "total_length_nm", "exc_fraction"]

    all_sweep = {}
    for metric in metrics:
        print(f"\n  {metric}:")
        sweep_df = sensitivity_sweep(df, metric, covariates=covariates)

        print(f"    Thresholds tested: {len(sweep_df)}")
        converged = sweep_df[sweep_df["converged"]]
        print(f"    Converged: {len(converged)}/{len(sweep_df)}")

        if len(converged) > 0:
            print(f"    Max abs coef range: [{converged['max_abs_coefficient'].min():.3f}, {converged['max_abs_coefficient'].max():.3f}]")
            # Show each
            for _, row in sweep_df.iterrows():
                status = "OK" if row["converged"] else "FAIL"
                coef = f"{row['max_abs_coefficient']:.3f}" if row["converged"] else "---"
                print(f"    {row['source']:20s} [{row['threshold_low']:.4f}, {row['threshold_high']:.4f}] "
                      f"n=[{int(row['n_regime_0'])},{int(row['n_regime_1'])},{int(row['n_regime_2'])}] "
                      f"coef={coef} {status}")

        sweep_df.to_csv(RESULTS_DIR / f"sensitivity_fixed_{metric}.csv", index=False)
        all_sweep[metric] = sweep_df

    return all_sweep


# ── Part 3: Complete subtype breakdown with effect sizes ──

def cohens_d(group1, group2):
    """Compute Cohen's d (pooled)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def rank_biserial_r(U, n1, n2):
    """Rank-biserial r from Mann-Whitney U."""
    return 1 - 2 * U / (n1 * n2)


def subtype_breakdown(df):
    """Full subtype breakdown with effect sizes and confidence intervals."""
    print("\n" + "=" * 60)
    print("PART 3: Complete subtype breakdown with effect sizes")
    print("=" * 60)

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    covariates = ["synapse_count", "total_length_nm", "exc_fraction"]

    # Define subtypes
    subtypes = {
        "excitatory": df[df["cell_type"] == "excitatory"],
        "BC": df[df["subtype"] == "BC"],
        "MC": df[df["subtype"] == "MC"],
        "BPC": df[df["subtype"] == "BPC"],
        "NGC": df[df["subtype"] == "NGC"],
    }

    results = {}
    for stype, sdf in subtypes.items():
        n_neurons = sdf["neuron_label"].nunique()
        n_branches = len(sdf)
        print(f"\n  {stype}: {n_neurons} neurons, {n_branches} branches")

        if n_neurons < 3:
            print(f"    SKIPPED (too few neurons)")
            continue

        results[stype] = {"n_neurons": n_neurons, "n_branches": n_branches, "metrics": {}}

        for metric in metrics:
            # Mixed model
            cols = [metric, "regime", "neuron_label"] + covariates
            work = sdf.dropna(subset=cols).copy()
            work = work[work["regime"] >= 0]

            if len(work) < 20 or work["neuron_label"].nunique() < 3:
                print(f"    {metric}: insufficient data after NaN drop")
                continue

            try:
                formula = f"{metric} ~ C(regime) + synapse_count + total_length_nm + exc_fraction"
                model = smf.mixedlm(formula, work, groups=work["neuron_label"])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = model.fit(reml=True, maxiter=500)

                coef_2 = res.params.get("C(regime)[T.2]", np.nan)
                p_2 = res.pvalues.get("C(regime)[T.2]", np.nan)
                coef_1 = res.params.get("C(regime)[T.1]", np.nan)
                p_1 = res.pvalues.get("C(regime)[T.1]", np.nan)

                # Effect sizes: regime 0 vs regime 2 (raw metric)
                r0 = work[work["regime"] == 0][metric].values
                r2 = work[work["regime"] == 2][metric].values

                d = cohens_d(r2, r0)

                if len(r0) >= 3 and len(r2) >= 3:
                    U, mw_p = stats.mannwhitneyu(r0, r2, alternative="two-sided")
                    r_rb = rank_biserial_r(U, len(r0), len(r2))
                else:
                    mw_p = np.nan
                    r_rb = np.nan

                sig = "***" if p_2 < 0.001 else "**" if p_2 < 0.01 else "*" if p_2 < 0.05 else "ns"
                print(f"    {metric}: coef={coef_2:.3f} (p={p_2:.2e}) {sig}  |  Cohen's d={d:.2f}, r_rb={r_rb:.3f}")

                results[stype]["metrics"][metric] = {
                    "regime_2_coef": float(coef_2),
                    "regime_2_p": float(p_2),
                    "regime_1_coef": float(coef_1),
                    "regime_1_p": float(p_1),
                    "cohens_d_0v2": float(d),
                    "rank_biserial_r": float(r_rb),
                    "mw_p_0v2": float(mw_p),
                    "n_regime_0": int((work["regime"] == 0).sum()),
                    "n_regime_2": int((work["regime"] == 2).sum()),
                    "n_obs": len(work),
                }

            except Exception as e:
                print(f"    {metric}: {str(e)[:60]}")

    return results


# ── Part 4: Monotonicity and dose-response check ──

def dose_response_check(df):
    """Check that regime effect is monotonic (regime 0 → 1 → 2).

    This is the visual objectivity check: if patterns aren't monotonic,
    the tertile classification is arbitrary.
    """
    print("\n" + "=" * 60)
    print("PART 4: Dose-response monotonicity check")
    print("=" * 60)

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    subtypes = {
        "all": df,
        "excitatory": df[df["cell_type"] == "excitatory"],
        "BC": df[df["subtype"] == "BC"],
        "MC": df[df["subtype"] == "MC"],
        "BPC": df[df["subtype"] == "BPC"],
        "NGC": df[df["subtype"] == "NGC"],
    }

    results = {}
    for stype, sdf in subtypes.items():
        work = sdf[sdf["regime"].isin([0, 1, 2])].copy()
        if len(work) < 20:
            continue

        results[stype] = {}
        print(f"\n  {stype} (n={work['neuron_label'].nunique()} neurons):")

        for metric in metrics:
            valid = work.dropna(subset=[metric])
            medians = valid.groupby("regime")[metric].median()

            if len(medians) < 3:
                continue

            m0 = medians.get(0, np.nan)
            m1 = medians.get(1, np.nan)
            m2 = medians.get(2, np.nan)

            # Expected direction: clark_evans goes up, interval_cv goes down, compactness goes up
            if "interval_cv" in metric:
                monotonic = m0 > m1 > m2  # decreasing = more regular
                direction = "decreasing"
            else:
                monotonic = m0 < m1 < m2  # increasing = more clustered
                direction = "increasing"

            # Jonckheere-Terpstra trend test
            groups = [valid[valid["regime"] == r][metric].values for r in [0, 1, 2] if r in valid["regime"].values]
            # Simple trend: Spearman correlation between regime and metric
            rho, p_trend = stats.spearmanr(valid["regime"].values, valid[metric].values)

            status = "MONOTONIC" if monotonic else "NON-MONOTONIC"
            print(f"    {metric}: median [R0={m0:.2f}, R1={m1:.2f}, R2={m2:.2f}] "
                  f"{direction} {status} | Spearman ρ={rho:.3f}, p={p_trend:.2e}")

            results[stype][metric] = {
                "median_r0": float(m0), "median_r1": float(m1), "median_r2": float(m2),
                "monotonic": bool(monotonic),
                "spearman_rho": float(rho), "spearman_p": float(p_trend),
            }

    return results


# ── Part 5: Continuous electrotonic length analysis ──

def continuous_electrotonic_check(df):
    """Test coupling using continuous L/λ instead of discrete regimes.

    If the pattern is real, a continuous predictor should work as well or better
    than the 3-regime discretization. This proves the result isn't an artifact
    of arbitrary threshold choice.
    """
    print("\n" + "=" * 60)
    print("PART 5: Continuous electrotonic length analysis")
    print("=" * 60)

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]

    results = {}
    subtypes = {
        "all": df,
        "excitatory": df[df["cell_type"] == "excitatory"],
        "inhibitory": df[df["cell_type"].str.startswith("inhibitory")],
    }

    for stype, sdf in subtypes.items():
        print(f"\n  {stype} ({sdf['neuron_label'].nunique()} neurons):")
        results[stype] = {}

        for metric in metrics:
            cols = [metric, "electrotonic_length", "neuron_label", "synapse_count", "total_length_nm", "exc_fraction"]
            work = sdf.dropna(subset=cols).copy()
            work = work[~np.isinf(work["electrotonic_length"])]

            if len(work) < 50:
                continue

            try:
                formula = f"{metric} ~ electrotonic_length + synapse_count + total_length_nm + exc_fraction"
                model = smf.mixedlm(formula, work, groups=work["neuron_label"])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = model.fit(reml=True, maxiter=500)

                el_coef = res.params.get("electrotonic_length", np.nan)
                el_p = res.pvalues.get("electrotonic_length", np.nan)

                # Compare R² (marginal) between continuous and regime models
                # Just report the continuous coefficient
                sig = "***" if el_p < 0.001 else "**" if el_p < 0.01 else "*" if el_p < 0.05 else "ns"
                print(f"    {metric}: L/λ coef={el_coef:.4f} (p={el_p:.2e}) {sig}")

                results[stype][metric] = {
                    "el_coefficient": float(el_coef),
                    "el_p": float(el_p),
                    "n_obs": len(work),
                }

            except Exception as e:
                print(f"    {metric}: {str(e)[:60]}")

    return results


def main():
    # 1. Process NGC neurons
    ngc_df = process_ngc_neurons()

    # 2. Load existing 350-neuron data
    existing_df = pd.read_csv(RESULTS_DIR / "all_branch_features_with_spatial.csv")
    print(f"\nExisting dataset: {len(existing_df)} branches, {existing_df['neuron_label'].nunique()} neurons")

    # 3. Merge NGC into combined dataset
    if ngc_df is not None:
        combined = pd.concat([existing_df, ngc_df], ignore_index=True)
        combined.to_csv(RESULTS_DIR / "all_branch_features_with_spatial_plus_ngc.csv", index=False)
        print(f"Combined dataset: {len(combined)} branches, {combined['neuron_label'].nunique()} neurons")
    else:
        combined = existing_df

    # 4. Sensitivity sweep on the combined data
    sweep_results = run_sensitivity_sweep(combined)

    # 5. Subtype breakdown with effect sizes (now including NGC)
    subtype_results = subtype_breakdown(combined)

    # 6. Dose-response monotonicity check
    dose_results = dose_response_check(combined)

    # 7. Continuous electrotonic length analysis
    continuous_results = continuous_electrotonic_check(combined)

    # Save everything
    all_results = {
        "sensitivity_sweep": {
            metric: {
                "n_converged": int(df["converged"].sum()),
                "n_total": len(df),
                "coef_range": [float(df[df["converged"]]["max_abs_coefficient"].min()),
                               float(df[df["converged"]]["max_abs_coefficient"].max())]
                if df["converged"].any() else None,
            }
            for metric, df in sweep_results.items()
        },
        "subtype_breakdown": subtype_results,
        "dose_response": dose_results,
        "continuous_electrotonic": continuous_results,
    }

    with open(RESULTS_DIR / "sensitivity_ngc_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved to {RESULTS_DIR / 'sensitivity_ngc_results.json'}")


if __name__ == "__main__":
    main()
