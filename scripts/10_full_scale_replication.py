#!/usr/bin/env python3
"""Full-scale replication on ALL 2,138+ analysis-ready neurons.

Uses multiprocessing to parallelize the z-score bottleneck (Step 3).
Then runs: regime classification, coupling tests, subtype breakdown,
partner mediation, and sensitivity sweep — all on the full dataset.

Output saved to results/replication_full/
"""

import argparse
import json
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

PROJECT_DIR = Path(__file__).resolve().parent.parent
MICRONS_DATA = PROJECT_DIR / "data" / "neurons"
RESULTS_DIR = PROJECT_DIR / "results" / "replication_full"

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
from src.structural_regimes import classify_branches, classify_by_electrotonic_length, compute_tertile_thresholds
from src.branch_null import compute_branch_z_scores
from src.coupling_tests import fit_mixed_model, sensitivity_sweep, kruskal_wallis_per_neuron

import statsmodels.formula.api as smf


# ═══════════════════════════════════════════════════════════
# NEURON PROCESSING (Step 1 & 3 combined for parallelization)
# ═══════════════════════════════════════════════════════════

def discover_neurons(data_dir):
    """Discover all neurons with SWC + synapse + partner CSV triplets."""
    neurons = []
    for swc in sorted(data_dir.glob("*.swc")):
        stem = swc.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        label, root_id = parts
        syn_path = data_dir / f"{stem}_synapses.csv"
        partner_path = data_dir / f"{stem}_presynaptic.csv"
        if syn_path.exists() and partner_path.exists():
            neurons.append((label, root_id, stem))
    return neurons


def process_single_neuron(args):
    """Process one neuron: load → branch features → z-scores.
    Designed for multiprocessing.Pool.map().
    Returns a DataFrame or None.
    """
    label, root_id, stem, data_dir_str = args
    data_dir = Path(data_dir_str)

    try:
        # Load skeleton
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

        skeleton = dendrite_skel
        n_branches = len(skeleton.branches)

        # Branch features
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

        # Partner types
        partner_path = data_dir / f"{stem}_presynaptic.csv"
        if partner_path.exists():
            partner_df = pd.read_csv(partner_path)
            partner_coords = partner_df[["x_nm", "y_nm", "z_nm"]].values.astype(float)
            tree = cKDTree(partner_coords)
            _, indices = tree.query(syn_coords_nm)
            all_types = partner_df["pre_cell_type_broad"].values[indices]
            valid_types = all_types[valid]
            exc_counts, inh_counts = count_exc_inh_per_branch(snap_valid, valid_types, n_branches)
        else:
            exc_counts = np.full(n_branches, np.nan)
            inh_counts = np.full(n_branches, np.nan)

        # Cell type classification
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

        # Z-scores (the bottleneck — 1000 null draws per branch)
        z_scores = compute_branch_z_scores(snap_valid, skeleton.branches, n_draws=1000, seed=42)

        # Build DataFrame
        neuron_label = f"{label}_{root_id}"
        df = pd.DataFrame({
            "neuron_label": neuron_label,
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

        # Add z-scores
        for metric in ["clark_evans_raw", "interval_cv_raw", "pairwise_compactness_raw",
                        "clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]:
            vals = z_scores[metric]
            df[metric] = vals[:n_branches]

        return df

    except Exception as e:
        return None


# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

def step1_parallel_processing(neurons, data_dir, n_workers=None):
    """Process all neurons in parallel: features + z-scores."""
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print(f"\n{'='*60}")
    print(f"STEP 1+3: Processing {len(neurons)} neurons ({n_workers} workers)")
    print(f"{'='*60}")

    # Prepare args for pool.map
    args_list = [(label, root_id, stem, str(data_dir)) for label, root_id, stem in neurons]

    t0 = time.time()
    all_dfs = []

    # Process in chunks to show progress
    chunk_size = 50
    for chunk_start in range(0, len(args_list), chunk_size):
        chunk = args_list[chunk_start:chunk_start + chunk_size]
        with Pool(n_workers) as pool:
            results = pool.map(process_single_neuron, chunk)

        for r in results:
            if r is not None:
                all_dfs.append(r)

        elapsed = time.time() - t0
        done = chunk_start + len(chunk)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(args_list) - done) / rate if rate > 0 else 0
        print(f"  {done}/{len(args_list)} neurons processed | "
              f"{len(all_dfs)} successful | "
              f"{elapsed:.0f}s elapsed | ETA: {eta:.0f}s")

    if not all_dfs:
        print("ERROR: No data produced")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)

    print(f"\n  Total: {len(combined)} branches across {combined['neuron_label'].nunique()} neurons")
    print(f"  Branches with synapses: {(combined['synapse_count'] > 0).sum()}")
    print(f"  Total synapses: {int(combined['synapse_count'].sum())}")
    print(f"  Time: {time.time()-t0:.0f}s")

    return combined


def step2_classify_regimes(df, output_dir):
    """Classify structural regimes using electrotonic tertiles."""
    print(f"\n{'='*60}")
    print("STEP 2: Classifying structural regimes")
    print(f"{'='*60}")

    classified = classify_branches(df)
    classified.to_csv(output_dir / "all_branch_features_with_spatial.csv", index=False)

    t_lo = classified.attrs.get("threshold_low", "?")
    t_hi = classified.attrs.get("threshold_high", "?")
    print(f"  Tertile thresholds: L/λ < {t_lo:.4f} | {t_hi:.4f}")

    for regime in [0, 1, 2]:
        n = (classified["regime"] == regime).sum()
        print(f"  Regime {regime}: {n} branches ({n/len(classified):.1%})")

    return classified


def step3_coupling_tests(df, output_dir):
    """Run coupling tests: mixed models, KW, sensitivity sweep."""
    print(f"\n{'='*60}")
    print("STEP 3: Coupling tests")
    print(f"{'='*60}")

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    covariates = ["synapse_count", "total_length_nm", "exc_fraction"]
    all_results = {}

    for metric in metrics:
        print(f"\n  {metric}:")

        # Mixed model
        mm = fit_mixed_model(df, metric, covariates=covariates)
        if mm.get("converged"):
            for k, v in mm.get("regime_coefficients", {}).items():
                pv = mm["regime_pvalues"].get(k, np.nan)
                print(f"    {k}: coef={v:.4f}, p={pv:.4g}")
        else:
            print(f"    Mixed model: FAILED TO CONVERGE")

        # Sensitivity sweep
        sweep = sensitivity_sweep(df, metric, covariates=covariates)
        n_conv = sweep["converged"].sum()
        print(f"    Sensitivity: {n_conv}/{len(sweep)} converged")
        if n_conv > 0:
            coef_range = sweep[sweep["converged"]]["max_abs_coefficient"]
            print(f"    Coef range: [{coef_range.min():.3f}, {coef_range.max():.3f}]")
        sweep.to_csv(output_dir / f"sensitivity_{metric}.csv", index=False)

        # KW per neuron
        kw = kruskal_wallis_per_neuron(df, metric)
        kw.to_csv(output_dir / f"kw_{metric}.csv", index=False)
        n_sig = (kw["p_bh"] < 0.05).sum() if "p_bh" in kw.columns else 0
        print(f"    Per-neuron KW: {n_sig}/{len(kw)} significant (BH q<0.05)")

        all_results[metric] = {
            "mixed_model": {k: v for k, v in mm.items() if k != "model"},
            "n_sig_neurons": int(n_sig),
            "n_total_neurons": len(kw),
        }

    with open(output_dir / "coupling_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


def step4_subtype_breakdown(df, output_dir):
    """Full subtype breakdown with effect sizes."""
    print(f"\n{'='*60}")
    print("STEP 4: Subtype breakdown")
    print(f"{'='*60}")

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    covariates = ["synapse_count", "total_length_nm", "exc_fraction"]

    subtypes = {
        "excitatory": df[df["cell_type"] == "excitatory"],
        "inhibitory_all": df[df["cell_type"].str.startswith("inhibitory")],
        "BC": df[df["subtype"] == "BC"],
        "MC": df[df["subtype"] == "MC"],
        "BPC": df[df["subtype"] == "BPC"],
        "NGC": df[df["subtype"] == "NGC"],
    }

    # Also add excitatory subtypes
    for st in df[df["cell_type"] == "excitatory"]["subtype"].unique():
        st_df = df[df["subtype"] == st]
        if st_df["neuron_label"].nunique() >= 10:
            subtypes[f"exc_{st}"] = st_df

    results = {}
    for stype, sdf in subtypes.items():
        n_neurons = sdf["neuron_label"].nunique()
        n_branches = len(sdf)
        print(f"\n  {stype}: {n_neurons} neurons, {n_branches} branches")

        if n_neurons < 3:
            print(f"    SKIPPED")
            continue

        results[stype] = {"n_neurons": n_neurons, "n_branches": n_branches, "metrics": {}}

        for metric in metrics:
            cols = [metric, "regime", "neuron_label"] + covariates
            work = sdf.dropna(subset=cols).copy()
            work = work[work["regime"] >= 0]

            if len(work) < 20 or work["neuron_label"].nunique() < 3:
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

                # Effect sizes
                r0 = work[work["regime"] == 0][metric].values
                r2 = work[work["regime"] == 2][metric].values
                d = np.nan
                if len(r0) >= 2 and len(r2) >= 2:
                    n1, n2 = len(r0), len(r2)
                    v1, v2 = np.var(r0, ddof=1), np.var(r2, ddof=1)
                    pooled = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
                    d = (np.mean(r2) - np.mean(r0)) / pooled if pooled > 0 else np.nan

                sig = "***" if p_2 < 0.001 else "**" if p_2 < 0.01 else "*" if p_2 < 0.05 else "ns"
                print(f"    {metric}: coef={coef_2:.3f} (p={p_2:.2e}) {sig}  d={d:.2f}")

                results[stype]["metrics"][metric] = {
                    "regime_2_coef": float(coef_2),
                    "regime_2_p": float(p_2),
                    "regime_1_coef": float(coef_1),
                    "regime_1_p": float(p_1),
                    "cohens_d_0v2": float(d),
                    "n_obs": len(work),
                }

            except Exception as e:
                print(f"    {metric}: {str(e)[:60]}")

    with open(output_dir / "subtype_breakdown.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def step5_continuous_analysis(df, output_dir):
    """Continuous L/λ analysis (no regime discretization)."""
    print(f"\n{'='*60}")
    print("STEP 5: Continuous electrotonic analysis")
    print(f"{'='*60}")

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    groups = {
        "all": df,
        "excitatory": df[df["cell_type"] == "excitatory"],
        "inhibitory": df[df["cell_type"].str.startswith("inhibitory")],
    }

    results = {}
    for gname, gdf in groups.items():
        print(f"\n  {gname} ({gdf['neuron_label'].nunique()} neurons):")
        results[gname] = {}

        for metric in metrics:
            cols = [metric, "electrotonic_length", "neuron_label", "synapse_count", "total_length_nm", "exc_fraction"]
            work = gdf.dropna(subset=cols).copy()
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
                sig = "***" if el_p < 0.001 else "**" if el_p < 0.01 else "*" if el_p < 0.05 else "ns"
                print(f"    {metric}: L/λ coef={el_coef:.4f} (p={el_p:.2e}) {sig}")

                results[gname][metric] = {
                    "el_coefficient": float(el_coef),
                    "el_p": float(el_p),
                    "n_obs": len(work),
                }
            except Exception as e:
                print(f"    {metric}: {str(e)[:60]}")

    with open(output_dir / "continuous_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def step6_dose_response(df, output_dir):
    """Monotonicity check: median by regime, Spearman trend."""
    print(f"\n{'='*60}")
    print("STEP 6: Dose-response monotonicity")
    print(f"{'='*60}")

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    subtypes = {
        "all": df,
        "excitatory": df[df["cell_type"] == "excitatory"],
        "inhibitory": df[df["cell_type"].str.startswith("inhibitory")],
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
        print(f"\n  {stype} (n={work['neuron_label'].nunique()}):")
        for metric in metrics:
            valid = work.dropna(subset=[metric])
            medians = valid.groupby("regime")[metric].median()
            m0, m1, m2 = medians.get(0, np.nan), medians.get(1, np.nan), medians.get(2, np.nan)
            if "interval_cv" in metric:
                monotonic = m0 > m1 > m2
            else:
                monotonic = m0 < m1 < m2
            rho, p = stats.spearmanr(valid["regime"].values, valid[metric].values)
            status = "MONO" if monotonic else "NON-MONO"
            print(f"    {metric}: [{m0:.2f}, {m1:.2f}, {m2:.2f}] {status} ρ={rho:.3f}")
            results[stype][metric] = {
                "median_r0": float(m0), "median_r1": float(m1), "median_r2": float(m2),
                "monotonic": bool(monotonic),
                "spearman_rho": float(rho), "spearman_p": float(p),
            }

    with open(output_dir / "dose_response.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def step7_mediation(df, output_dir):
    """Mediation analysis: does exc_fraction mediate regime → spatial?"""
    print(f"\n{'='*60}")
    print("STEP 7: Mediation by E/I partner balance")
    print(f"{'='*60}")

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    results = {}

    for subset_name, subset_df in [
        ("all", df),
        ("excitatory", df[df["cell_type"] == "excitatory"]),
        ("inhibitory", df[df["cell_type"].str.startswith("inhibitory")]),
    ]:
        print(f"\n  {subset_name}:")
        results[subset_name] = {}
        for metric in metrics:
            base_cols = [metric, "regime", "neuron_label", "synapse_count", "total_length_nm"]
            full_cols = base_cols + ["exc_fraction"]
            work_red = subset_df.dropna(subset=base_cols).copy()
            work_red = work_red[work_red["regime"] >= 0]
            work_full = subset_df.dropna(subset=full_cols).copy()
            work_full = work_full[work_full["regime"] >= 0]

            if len(work_full) < 50 or work_full["neuron_label"].nunique() < 5:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    formula_red = f"{metric} ~ C(regime) + synapse_count + total_length_nm"
                    res_red = smf.mixedlm(formula_red, work_red, groups=work_red["neuron_label"]).fit(reml=True, maxiter=500)
                    formula_full = f"{metric} ~ C(regime) + synapse_count + total_length_nm + exc_fraction"
                    res_full = smf.mixedlm(formula_full, work_full, groups=work_full["neuron_label"]).fit(reml=True, maxiter=500)

                coef_red = res_red.params.get("C(regime)[T.2]", np.nan)
                coef_full = res_full.params.get("C(regime)[T.2]", np.nan)
                mediation_pct = (1 - abs(coef_full) / abs(coef_red)) * 100 if abs(coef_red) > 1e-10 else np.nan
                exc_coef = res_full.params.get("exc_fraction", np.nan)
                exc_p = res_full.pvalues.get("exc_fraction", np.nan)

                print(f"    {metric}: mediation={mediation_pct:.1f}%, exc_frac p={exc_p:.2e}")
                results[subset_name][metric] = {
                    "coef_reduced": float(coef_red),
                    "coef_full": float(coef_full),
                    "mediation_pct": float(mediation_pct),
                    "exc_fraction_coef": float(exc_coef),
                    "exc_fraction_p": float(exc_p),
                }
            except Exception as e:
                print(f"    {metric}: {str(e)[:60]}")

    with open(output_dir / "mediation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    parser = argparse.ArgumentParser(description="Full-scale replication")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--data-dir", type=str, default=str(MICRONS_DATA))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Discover all neurons
    all_neurons = discover_neurons(data_dir)
    print(f"Discovered {len(all_neurons)} neurons with complete data")

    type_counts = {}
    for label, _, _ in all_neurons:
        parts = label.split("_")
        key = f"{parts[0]}_{parts[1]}" if len(parts) > 1 else label
        type_counts[key] = type_counts.get(key, 0) + 1
    print("By type:")
    for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # Step 1+3: Parallel processing (features + z-scores)
    df = step1_parallel_processing(all_neurons, data_dir, n_workers=args.workers)
    if df is None:
        sys.exit(1)

    # Save raw features
    df.to_csv(RESULTS_DIR / "all_branch_features_raw.csv", index=False)

    # Step 2: Classify regimes
    df = step2_classify_regimes(df, RESULTS_DIR)

    # Step 3: Coupling tests + sensitivity
    coupling = step3_coupling_tests(df, RESULTS_DIR)

    # Step 4: Subtype breakdown
    subtypes = step4_subtype_breakdown(df, RESULTS_DIR)

    # Step 5: Continuous analysis
    continuous = step5_continuous_analysis(df, RESULTS_DIR)

    # Step 6: Dose-response
    dose = step6_dose_response(df, RESULTS_DIR)

    # Step 7: Mediation
    mediation = step7_mediation(df, RESULTS_DIR)

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"COMPLETE: {df['neuron_label'].nunique()} neurons, {len(df)} branches")
    print(f"Total time: {elapsed:.0f}s ({elapsed/3600:.1f} hours)")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'='*60}")

    # Save summary
    summary = {
        "n_neurons": int(df["neuron_label"].nunique()),
        "n_branches": len(df),
        "total_time_s": elapsed,
        "type_counts": type_counts,
    }
    with open(RESULTS_DIR / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
