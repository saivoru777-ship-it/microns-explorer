#!/usr/bin/env python3
"""Phase 3: Staged replication of regime-coupling analysis on expanded dataset.

Runs the full analysis pipeline on a stratified subset:
- 100 excitatory neurons (mix of 23P, 4P, 5PIT, 5PET, 6PCT, 6PIT)
- 100 basket cells (BC)
- 100 Martinotti cells (MC)
- 50 bipolar cells (BPC)

Steps:
1. Compute branch features (morphometry + electrotonic length + partner types)
2. Classify structural regimes (electrotonic tertiles)
3. Compute spatial organization metrics + z-scores
4. Run coupling tests (mixed-effects, KW, permutation, sensitivity)
5. Report exc vs inh split

Usage:
    python scripts/06_staged_replication.py [--max-exc 100] [--max-bc 100] [--max-mc 100] [--max-bpc 50] [--full]
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MICRONS_DATA = PROJECT_DIR / "data" / "neurons"
RESULTS_DIR = PROJECT_DIR / "results" / "replication"

# External dependencies
sys.path.insert(0, str(Path.home() / "research" / "neurostat"))
sys.path.insert(0, str(Path.home() / "research" / "neurostat-input-clustering" / "src"))
sys.path.insert(0, str(Path.home() / "research" / "dendritic-regime-coupling"))

from neurostat.io.swc import NeuronSkeleton, SnapResult
from soma_distance import precompute_branch_endpoint_distances

# Reuse existing analysis modules
from src.branch_morphometry import (
    compute_branch_order, compute_mean_diameter, compute_electrotonic_length,
    compute_soma_distance_per_branch, count_synapses_per_branch,
    count_exc_inh_per_branch, RM, RI,
)
from src.structural_regimes import classify_branches, regime_size_table, check_regime_agreement
from src.branch_null import compute_branch_z_scores
from src.coupling_tests import run_all_coupling_tests, fit_mixed_model, kruskal_wallis_per_neuron

from scipy.spatial import cKDTree


def discover_neurons(data_dir):
    """Discover all neurons with SWC + synapse + partner CSV triplets."""
    neurons = []
    for swc in sorted(data_dir.glob("*.swc")):
        stem = swc.stem  # e.g., "exc_23P_864691135848859998"
        parts = stem.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        label = parts[0]
        root_id = parts[1]
        syn_path = data_dir / f"{stem}_synapses.csv"
        partner_path = data_dir / f"{stem}_presynaptic.csv"
        if syn_path.exists() and partner_path.exists():
            neurons.append((label, root_id, stem))
    return neurons


def select_stratified_subset(neurons, max_exc=100, max_bc=100, max_mc=100, max_bpc=50):
    """Select stratified random subset."""
    rng = np.random.default_rng(42)

    by_type = {}
    for label, root_id, stem in neurons:
        parts = label.split("_")
        broad = parts[0]  # exc or inh
        fine = parts[1] if len(parts) > 1 else "unknown"
        key = f"{broad}_{fine}"
        if key not in by_type:
            by_type[key] = []
        by_type[key].append((label, root_id, stem))

    selected = []

    # Excitatory: sample across subtypes
    exc_pool = []
    for k, v in by_type.items():
        if k.startswith("exc_"):
            exc_pool.extend(v)
    if len(exc_pool) > max_exc:
        idx = rng.choice(len(exc_pool), max_exc, replace=False)
        exc_pool = [exc_pool[i] for i in idx]
    selected.extend(exc_pool)

    # BC
    bc_pool = by_type.get("inh_BC", [])
    if len(bc_pool) > max_bc:
        idx = rng.choice(len(bc_pool), max_bc, replace=False)
        bc_pool = [bc_pool[i] for i in idx]
    selected.extend(bc_pool)

    # MC
    mc_pool = by_type.get("inh_MC", [])
    if len(mc_pool) > max_mc:
        idx = rng.choice(len(mc_pool), max_mc, replace=False)
        mc_pool = [mc_pool[i] for i in idx]
    selected.extend(mc_pool)

    # BPC
    bpc_pool = by_type.get("inh_BPC", [])
    if len(bpc_pool) > max_bpc:
        idx = rng.choice(len(bpc_pool), max_bpc, replace=False)
        bpc_pool = [bpc_pool[i] for i in idx]
    selected.extend(bpc_pool)

    return selected


def load_neuron(stem, data_dir):
    """Load skeleton and snap synapses. Returns (skeleton, snap_valid) or None."""
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
    """Load partner CSV and match to snapped synapses via coordinate KDTree."""
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

    all_types = partner_df["pre_cell_type_broad"].values[indices]
    return all_types


def compute_branch_features_for_neuron(label, root_id, stem, data_dir):
    """Compute all branch features for a single neuron."""
    result = load_neuron(stem, data_dir)
    if result is None:
        return None

    skeleton, snap_valid = result
    n_branches = len(skeleton.branches)

    # Branch order
    orders = compute_branch_order(skeleton)

    # Diameter and electrotonic length
    diameters = np.array([compute_mean_diameter(skeleton, i) for i in range(n_branches)])
    lengths = np.array([br.total_length for br in skeleton.branches])
    e_lengths = np.array([
        compute_electrotonic_length(lengths[i], diameters[i])
        for i in range(n_branches)
    ])

    # Soma distance
    endpoint_nodes, node_to_idx, endpoint_dists = precompute_branch_endpoint_distances(skeleton)
    soma_dists = compute_soma_distance_per_branch(skeleton, endpoint_nodes, node_to_idx, endpoint_dists)

    # Synapse counts
    syn_counts = count_synapses_per_branch(snap_valid, n_branches)

    # Exc/inh counts from partner CSV
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

    # Cell type
    if label.startswith("exc"):
        cell_type = "excitatory"
    elif "BPC" in label:
        cell_type = "inhibitory_BPC"
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


def step1_branch_features(neurons, data_dir, output_dir):
    """Step 1: Compute branch features for all neurons."""
    print(f"\n{'='*60}")
    print("STEP 1: Computing branch morphometry features")
    print(f"{'='*60}")

    all_dfs = []
    skeletons = {}
    snap_results = {}
    t0 = time.time()

    for i, (label, root_id, stem) in enumerate(neurons):
        result = compute_branch_features_for_neuron(label, root_id, stem, data_dir)
        if result is not None:
            df, skel, snap = result
            all_dfs.append(df)
            skeletons[f"{label}_{root_id}"] = skel
            snap_results[f"{label}_{root_id}"] = snap

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(neurons)} neurons processed ({time.time()-t0:.0f}s)")

    if not all_dfs:
        print("ERROR: No data produced")
        return None, None, None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output_dir / "all_branch_features.csv", index=False)

    print(f"\n  Total: {len(combined)} branches across {len(all_dfs)} neurons")
    print(f"  Branches with synapses: {(combined['synapse_count'] > 0).sum()}")
    print(f"  Total synapses: {combined['synapse_count'].sum()}")
    print(f"  Electrotonic length range: [{combined['electrotonic_length'].min():.4f}, {combined['electrotonic_length'].max():.2f}]")
    print(f"  Time: {time.time()-t0:.0f}s")

    return combined, skeletons, snap_results


def step2_classify_regimes(features_df, output_dir):
    """Step 2: Classify structural regimes."""
    print(f"\n{'='*60}")
    print("STEP 2: Classifying structural regimes")
    print(f"{'='*60}")

    df = classify_branches(features_df)
    df.to_csv(output_dir / "all_branch_features_with_regimes.csv", index=False)

    # Report
    t_lo = df.attrs.get("threshold_low", "?")
    t_hi = df.attrs.get("threshold_high", "?")
    print(f"  Tertile thresholds: L/λ < {t_lo:.4f} | {t_hi:.4f}")

    size_table = regime_size_table(df)
    print(f"\n  Regime sizes:")
    for name, row in size_table.iterrows():
        print(f"    {name}: {int(row['count'])} ({row['fraction']:.1%})")

    agreement = check_regime_agreement(df["regime"].values, df["order_regime"].values)
    print(f"\n  Electrotonic vs order agreement:")
    print(f"    Cohen's κ = {agreement['kappa']:.3f}")
    print(f"    Agreement fraction = {agreement['agreement_fraction']:.3f}")

    return df


def step3_spatial_metrics(features_df, skeletons, snap_results, output_dir):
    """Step 3: Compute spatial organization metrics + z-scores."""
    print(f"\n{'='*60}")
    print("STEP 3: Computing spatial organization metrics + z-scores")
    print(f"{'='*60}")

    t0 = time.time()
    all_metrics = []

    neuron_labels = features_df["neuron_label"].unique()
    for i, nl in enumerate(neuron_labels):
        if nl not in skeletons:
            continue

        skel = skeletons[nl]
        snap = snap_results[nl]

        z_scores = compute_branch_z_scores(snap, skel.branches, n_draws=1000, seed=42)

        neuron_df = features_df[features_df["neuron_label"] == nl].copy()
        n_branches = len(skel.branches)

        for metric in ["clark_evans_raw", "interval_cv_raw", "pairwise_compactness_raw",
                        "clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]:
            vals = z_scores[metric]
            neuron_df[metric] = vals[:len(neuron_df)]

        all_metrics.append(neuron_df)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(neuron_labels)} neurons ({time.time()-t0:.0f}s)")

    df = pd.concat(all_metrics, ignore_index=True)
    df.to_csv(output_dir / "all_branch_features_with_spatial.csv", index=False)

    print(f"\n  Computed z-scores for {len(neuron_labels)} neurons")
    print(f"  Time: {time.time()-t0:.0f}s")

    # Quick summary
    for metric in ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]:
        valid = df[metric].dropna()
        print(f"  {metric}: mean={valid.mean():.3f}, std={valid.std():.3f}, n={len(valid)}")

    return df


def step4_coupling_tests(features_df, output_dir):
    """Step 4: Run coupling tests."""
    print(f"\n{'='*60}")
    print("STEP 4: Running coupling tests")
    print(f"{'='*60}")

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    covariates = ["synapse_count", "total_length_nm", "exc_fraction"]

    all_results = {}
    for metric in metrics:
        print(f"\n  Testing {metric}...")
        results = run_all_coupling_tests(
            features_df, metric, covariates=covariates, n_perms=10000, seed=42
        )

        # Print key results
        mm = results["mixed_model"]
        if mm.get("converged"):
            print(f"    Mixed model (n={mm['n_obs']}):")
            for k, v in mm.get("regime_coefficients", {}).items():
                pv = mm["regime_pvalues"].get(k, np.nan)
                print(f"      {k}: coef={v:.4f}, p={pv:.4g}")

        perm = results["permutation"]
        print(f"    Permutation test: stat={perm.get('observed_stat', np.nan):.4f}, p={perm.get('p_value', np.nan):.4g}")

        resid = results["residual"]
        if resid.get("converged"):
            print(f"    Residual analysis: order R²={resid['order_r2']:.3f}, residual KW p={resid['residual_p']:.4g}")

        order_shuf = results["order_shuffle"]
        print(f"    Order-shuffle: stat={order_shuf.get('observed_stat', np.nan):.4f}, p={order_shuf.get('p_value', np.nan):.4g}")

        # Convert results to serializable format
        serializable = {}
        for k, v in results.items():
            if k == "kruskal_wallis":
                v.to_csv(output_dir / f"coupling_kw_{metric}.csv", index=False)
                serializable[k] = "saved to CSV"
            elif k == "sensitivity":
                v.to_csv(output_dir / f"coupling_sensitivity_{metric}.csv", index=False)
                serializable[k] = "saved to CSV"
            elif k == "mixed_model":
                serializable[k] = {kk: vv for kk, vv in v.items() if kk != "model"}
            elif isinstance(v, dict):
                serializable[k] = {
                    kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                    for kk, vv in v.items()
                    if not isinstance(vv, np.ndarray)
                }
            else:
                serializable[k] = str(v)

        all_results[metric] = serializable

    # Save results
    with open(output_dir / "coupling_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


def step5_exc_inh_split(features_df, output_dir):
    """Step 5: Excitatory vs inhibitory split analysis."""
    print(f"\n{'='*60}")
    print("STEP 5: Excitatory vs Inhibitory split")
    print(f"{'='*60}")

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    covariates = ["synapse_count", "total_length_nm", "exc_fraction"]

    exc_df = features_df[features_df["cell_type"] == "excitatory"].copy()
    inh_df = features_df[features_df["cell_type"].str.startswith("inhibitory")].copy()

    print(f"  Excitatory: {exc_df['neuron_label'].nunique()} neurons, {len(exc_df)} branches")
    print(f"  Inhibitory: {inh_df['neuron_label'].nunique()} neurons, {len(inh_df)} branches")

    split_results = {}
    for metric in metrics:
        print(f"\n  {metric}:")
        for name, subset in [("excitatory", exc_df), ("inhibitory", inh_df)]:
            if len(subset) < 50:
                print(f"    {name}: too few branches, skipping")
                continue

            mm = fit_mixed_model(subset, metric, covariates=covariates)
            if mm.get("converged"):
                coefs = mm.get("regime_coefficients", {})
                max_coef = max(abs(v) for v in coefs.values()) if coefs else np.nan
                print(f"    {name}: max|coef|={max_coef:.4f}, n={mm['n_obs']}")
                for k, v in coefs.items():
                    pv = mm["regime_pvalues"].get(k, np.nan)
                    print(f"      {k}: coef={v:.4f}, p={pv:.4g}")
            else:
                print(f"    {name}: model did not converge")

            # KW per neuron
            kw = kruskal_wallis_per_neuron(subset, metric)
            n_sig = (kw["p_bh"] < 0.05).sum() if "p_bh" in kw.columns else 0
            n_total = len(kw)
            print(f"    {name}: {n_sig}/{n_total} neurons significant (BH q<0.05)")

            split_results[f"{name}_{metric}"] = {
                "n_neurons": int(subset["neuron_label"].nunique()),
                "n_branches": len(subset),
                "n_sig_neurons": int(n_sig),
                "n_total_neurons": n_total,
                "frac_sig": float(n_sig / n_total) if n_total > 0 else 0,
            }

    with open(output_dir / "exc_inh_split_results.json", "w") as f:
        json.dump(split_results, f, indent=2)

    return split_results


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Staged replication")
    parser.add_argument("--max-exc", type=int, default=100)
    parser.add_argument("--max-bc", type=int, default=100)
    parser.add_argument("--max-mc", type=int, default=100)
    parser.add_argument("--max-bpc", type=int, default=50)
    parser.add_argument("--full", action="store_true", help="Use all available neurons")
    parser.add_argument("--data-dir", type=str, default=str(MICRONS_DATA))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Discover neurons
    all_neurons = discover_neurons(data_dir)
    print(f"Discovered {len(all_neurons)} neurons with complete data")

    # Select subset
    if args.full:
        selected = all_neurons
        print(f"Using ALL {len(selected)} neurons")
    else:
        selected = select_stratified_subset(
            all_neurons, args.max_exc, args.max_bc, args.max_mc, args.max_bpc
        )
        print(f"Selected stratified subset: {len(selected)} neurons")

    # Count by type
    type_counts = {}
    for label, _, _ in selected:
        parts = label.split("_")
        key = f"{parts[0]}_{parts[1]}"
        type_counts[key] = type_counts.get(key, 0) + 1
    print("  By type:")
    for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v}")

    # Step 1: Branch features
    features_df, skeletons, snap_results = step1_branch_features(
        selected, data_dir, RESULTS_DIR
    )
    if features_df is None:
        sys.exit(1)

    # Step 2: Classify regimes
    regime_df = step2_classify_regimes(features_df, RESULTS_DIR)

    # Step 3: Spatial metrics
    spatial_df = step3_spatial_metrics(regime_df, skeletons, snap_results, RESULTS_DIR)

    # Step 4: Coupling tests
    coupling_results = step4_coupling_tests(spatial_df, RESULTS_DIR)

    # Step 5: Exc vs Inh split
    split_results = step5_exc_inh_split(spatial_df, RESULTS_DIR)

    # Final summary
    print(f"\n{'='*60}")
    print("REPLICATION COMPLETE")
    print(f"{'='*60}")
    print(f"Neurons: {len(selected)}")
    print(f"Branches: {len(spatial_df)}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
