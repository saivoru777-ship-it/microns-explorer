#!/usr/bin/env python3
"""Partner-level mediation analysis for the regime-coupling replication.

Two analyses:
1. Formal mediation test: does exc_fraction mediate regime → spatial coupling?
   Compare mixed-effects models with and without exc_fraction.

2. Partner compactness by regime: for each presynaptic partner with ≥3 synapses,
   compute the geodesic spread of their synapses, then test whether partners
   targeting compartmentalized branches are more compact than those targeting
   summation-like branches.

Both stratified by excitatory vs inhibitory postsynaptic neurons.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

PROJECT_DIR = Path(__file__).resolve().parent.parent
MICRONS_DATA = PROJECT_DIR / "data" / "neurons"
RESULTS_DIR = PROJECT_DIR / "results" / "replication"

sys.path.insert(0, str(Path.home() / "research" / "neurostat"))
sys.path.insert(0, str(Path.home() / "research" / "neurostat-input-clustering" / "src"))
sys.path.insert(0, str(Path.home() / "research" / "dendritic-regime-coupling"))

from neurostat.io.swc import NeuronSkeleton, SnapResult
from src.branch_morphometry import RM, RI

import statsmodels.formula.api as smf


def mediation_analysis(spatial_df):
    """Test whether exc_fraction mediates the regime → spatial coupling.

    Compare:
    - Full model: spatial ~ C(regime) + synapse_count + total_length_nm + exc_fraction + (1|neuron)
    - Reduced model: spatial ~ C(regime) + synapse_count + total_length_nm + (1|neuron)

    If the regime coefficient shrinks substantially when exc_fraction is added,
    exc_fraction mediates the coupling.
    """
    print("=" * 60)
    print("MEDIATION ANALYSIS: Does exc_fraction mediate regime → spatial?")
    print("=" * 60)

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    results = {}

    for subset_name, subset_df in [
        ("all", spatial_df),
        ("excitatory", spatial_df[spatial_df["cell_type"] == "excitatory"]),
        ("inhibitory", spatial_df[spatial_df["cell_type"].str.startswith("inhibitory")]),
    ]:
        print(f"\n--- {subset_name} ---")
        results[subset_name] = {}

        for metric in metrics:
            base_cols = [metric, "regime", "neuron_label", "synapse_count", "total_length_nm"]
            full_cols = base_cols + ["exc_fraction"]

            # Reduced model (without exc_fraction)
            work_red = subset_df.dropna(subset=base_cols).copy()
            work_red = work_red[work_red["regime"] >= 0]

            # Full model (with exc_fraction)
            work_full = subset_df.dropna(subset=full_cols).copy()
            work_full = work_full[work_full["regime"] >= 0]

            if len(work_full) < 50 or work_full["neuron_label"].nunique() < 5:
                print(f"  {metric}: insufficient data")
                continue

            try:
                # Reduced
                formula_red = f"{metric} ~ C(regime) + synapse_count + total_length_nm"
                model_red = smf.mixedlm(formula_red, work_red, groups=work_red["neuron_label"])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res_red = model_red.fit(reml=True, maxiter=500)

                # Full
                formula_full = f"{metric} ~ C(regime) + synapse_count + total_length_nm + exc_fraction"
                model_full = smf.mixedlm(formula_full, work_full, groups=work_full["neuron_label"])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res_full = model_full.fit(reml=True, maxiter=500)

                # Extract regime 2 coefficient
                coef_red = res_red.params.get("C(regime)[T.2]", np.nan)
                coef_full = res_full.params.get("C(regime)[T.2]", np.nan)

                # Mediation = fractional reduction in coefficient
                if abs(coef_red) > 1e-10:
                    mediation_pct = (1 - abs(coef_full) / abs(coef_red)) * 100
                else:
                    mediation_pct = np.nan

                # exc_fraction coefficient in full model
                exc_coef = res_full.params.get("exc_fraction", np.nan)
                exc_p = res_full.pvalues.get("exc_fraction", np.nan)

                print(f"  {metric}:")
                print(f"    Regime 2 coef (without exc_frac): {coef_red:.4f}")
                print(f"    Regime 2 coef (with exc_frac):    {coef_full:.4f}")
                print(f"    Mediation: {mediation_pct:.1f}%")
                print(f"    exc_fraction coef: {exc_coef:.4f}, p={exc_p:.2e}")

                results[subset_name][metric] = {
                    "coef_reduced": float(coef_red),
                    "coef_full": float(coef_full),
                    "mediation_pct": float(mediation_pct),
                    "exc_fraction_coef": float(exc_coef),
                    "exc_fraction_p": float(exc_p),
                    "n_obs_reduced": len(work_red),
                    "n_obs_full": len(work_full),
                }

            except Exception as e:
                print(f"  {metric}: {str(e)[:60]}")

    return results


def compute_partner_compactness(snap_valid, partner_ids, branch_ids_per_synapse):
    """Compute per-partner synapse compactness on the dendrite.

    For each partner with ≥3 synapses:
    - Mean pairwise geodesic distance between their synapse positions
    - Normalized by the mean pairwise distance of ALL synapses on those branches

    Returns DataFrame with partner_id, n_synapses, mean_pairwise_dist, normalized_compactness.
    """
    unique_partners, inverse = np.unique(partner_ids, return_inverse=True)
    positions = snap_valid.branch_positions  # 1D position along branch
    branch_ids = snap_valid.branch_ids

    records = []
    for i, pid in enumerate(unique_partners):
        mask = inverse == i
        n_syn = mask.sum()
        if n_syn < 3:
            continue

        # Get 3D positions from branch_positions
        # Since we're comparing within the same neuron, use branch_positions directly
        # But these are 1D positions within different branches — we need the original 3D coords
        # Actually, for compactness we want the actual spatial spread
        # Use the snap distances as a proxy — but better to use raw synapse coordinates

        records.append({
            "partner_id": pid,
            "n_synapses": n_syn,
        })

    return pd.DataFrame(records) if records else None


def partner_regime_analysis(spatial_df, data_dir):
    """Map presynaptic partners to branch regimes and compute compactness.

    For each neuron:
    1. Load partner CSV
    2. Match partner synapses to snapped synapses → branch assignments
    3. Determine each partner's dominant regime
    4. Compute partner spatial spread (mean pairwise distance of their synapses in 3D)
    5. Test: does partner compactness differ by regime?
    """
    print("\n" + "=" * 60)
    print("PARTNER COMPACTNESS BY REGIME")
    print("=" * 60)

    neuron_labels = spatial_df["neuron_label"].unique()
    all_partner_records = []
    t0 = time.time()

    for i, nl in enumerate(neuron_labels):
        parts = nl.rsplit("_", 1)
        if len(parts) != 2:
            continue
        label, root_id = parts[0], parts[1]
        stem = nl

        swc_path = data_dir / f"{stem}.swc"
        syn_path = data_dir / f"{stem}_synapses.csv"
        partner_path = data_dir / f"{stem}_presynaptic.csv"

        if not all(p.exists() for p in [swc_path, syn_path, partner_path]):
            continue

        # Load skeleton and snap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skeleton = NeuronSkeleton.from_swc_file(str(swc_path), scale_factor=1000.0)
        dendrite_skel = skeleton.filter_by_type([1, 3, 4])
        if len(dendrite_skel.branches) < 3:
            dendrite_skel = skeleton

        syn_df = pd.read_csv(syn_path)
        syn_coords_nm = syn_df[["x_um", "y_um", "z_um"]].values * 1000.0

        snap = dendrite_skel.snap_points(syn_coords_nm, d_max=50000.0)
        valid_mask = snap.valid

        if valid_mask.sum() < 20:
            continue

        # Load partner data
        partner_df = pd.read_csv(partner_path)
        partner_coords = partner_df[["x_nm", "y_nm", "z_nm"]].values.astype(float)

        # Match synapses to partners
        tree = cKDTree(partner_coords)
        _, p_indices = tree.query(syn_coords_nm)
        all_partner_ids = partner_df["pre_root_id"].values[p_indices]
        all_partner_types = partner_df["pre_cell_type_broad"].values[p_indices]

        # Filter to valid synapses
        valid_partner_ids = all_partner_ids[valid_mask]
        valid_partner_types = all_partner_types[valid_mask]
        valid_branch_ids = snap.branch_ids[valid_mask]
        valid_coords = syn_coords_nm[valid_mask]

        # Get regime for this neuron's branches
        neuron_regime = spatial_df[spatial_df["neuron_label"] == nl]
        branch_to_regime = dict(zip(
            neuron_regime["branch_idx"].astype(int),
            neuron_regime["regime"].astype(int)
        ))

        # Map each synapse to regime
        syn_regimes = np.array([branch_to_regime.get(bid, -1) for bid in valid_branch_ids])

        # For each partner with ≥3 synapses
        unique_pids = np.unique(valid_partner_ids)
        for pid in unique_pids:
            mask = valid_partner_ids == pid
            n_syn = mask.sum()
            if n_syn < 3:
                continue

            p_coords = valid_coords[mask]
            p_regimes = syn_regimes[mask]
            p_types = valid_partner_types[mask]

            valid_regime_mask = p_regimes >= 0
            if valid_regime_mask.sum() < 3:
                continue

            # Dominant regime
            regime_counts = np.bincount(p_regimes[valid_regime_mask], minlength=3)
            dominant_regime = np.argmax(regime_counts)

            # Compute 3D mean pairwise distance (compactness)
            if len(p_coords) >= 2:
                dists = np.sqrt(((p_coords[:, None] - p_coords[None, :]) ** 2).sum(axis=2))
                triu = np.triu_indices(len(p_coords), k=1)
                mean_pw_dist = np.mean(dists[triu])
            else:
                mean_pw_dist = np.nan

            # Partner type (most common)
            partner_type = p_types[0] if len(p_types) > 0 else "unknown"

            # Spans regimes?
            unique_regimes = set(p_regimes[valid_regime_mask])
            spans = len(unique_regimes) > 1

            all_partner_records.append({
                "neuron_label": nl,
                "cell_type": neuron_regime["cell_type"].iloc[0] if len(neuron_regime) > 0 else "unknown",
                "partner_id": pid,
                "partner_type": partner_type,
                "n_synapses": n_syn,
                "dominant_regime": int(dominant_regime),
                "regime_0_count": int(regime_counts[0]),
                "regime_1_count": int(regime_counts[1]),
                "regime_2_count": int(regime_counts[2]),
                "spans_regimes": spans,
                "mean_pairwise_dist_nm": mean_pw_dist,
            })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(neuron_labels)} neurons ({time.time()-t0:.0f}s)")

    if not all_partner_records:
        print("  No partner data collected")
        return None

    partner_regime_df = pd.DataFrame(all_partner_records)
    print(f"\n  Total partners (k≥3): {len(partner_regime_df)}")
    print(f"  Unique neurons: {partner_regime_df['neuron_label'].nunique()}")
    print(f"  Time: {time.time()-t0:.0f}s")

    # Save
    partner_regime_df.to_csv(RESULTS_DIR / "partner_regime_mapping.csv", index=False)

    # Test: does compactness differ by dominant regime?
    test_results = {}

    for subset_name, subset in [
        ("all", partner_regime_df),
        ("excitatory_post", partner_regime_df[partner_regime_df["cell_type"] == "excitatory"]),
        ("inhibitory_post", partner_regime_df[partner_regime_df["cell_type"].str.startswith("inhibitory")]),
    ]:
        work = subset.dropna(subset=["mean_pairwise_dist_nm"])
        work = work[work["dominant_regime"].isin([0, 1, 2])]

        if len(work) < 10:
            print(f"\n  {subset_name}: insufficient partners")
            continue

        groups = {r: g["mean_pairwise_dist_nm"].values
                  for r, g in work.groupby("dominant_regime")}

        print(f"\n  {subset_name}:")
        print(f"    N partners: {len(work)}")
        for r in [0, 1, 2]:
            g = groups.get(r, np.array([]))
            if len(g) > 0:
                print(f"    Regime {r}: n={len(g)}, median={np.median(g):.0f} nm, mean={np.mean(g):.0f} nm")

        # KW test
        valid_groups = [g for g in groups.values() if len(g) >= 3]
        if len(valid_groups) >= 2:
            H, p = stats.kruskal(*valid_groups)
            print(f"    Kruskal-Wallis: H={H:.2f}, p={p:.2e}")

            test_results[subset_name] = {
                "kw_H": float(H), "kw_p": float(p),
                "n_partners": len(work),
            }

            # Mann-Whitney regime 0 vs 2
            g0 = groups.get(0, np.array([]))
            g2 = groups.get(2, np.array([]))
            if len(g0) >= 3 and len(g2) >= 3:
                U, p_mw = stats.mannwhitneyu(g0, g2, alternative="two-sided")
                print(f"    Mann-Whitney (0 vs 2): U={U:.0f}, p={p_mw:.2e}")

                # Effect size: ratio of medians
                ratio = np.median(g2) / np.median(g0) if np.median(g0) > 0 else np.nan
                print(f"    Median ratio (regime 2 / regime 0): {ratio:.3f}")
                # <1 means more compact on compartmentalized branches

                test_results[subset_name]["mw_U"] = float(U)
                test_results[subset_name]["mw_p"] = float(p_mw)
                test_results[subset_name]["median_ratio_2v0"] = float(ratio)

        # Also test by partner type (exc vs inh partners)
        for ptype in ["excitatory", "inhibitory"]:
            ptype_work = work[work["partner_type"] == ptype]
            if len(ptype_work) < 10:
                continue

            ptype_groups = {r: g["mean_pairwise_dist_nm"].values
                           for r, g in ptype_work.groupby("dominant_regime")}
            valid_ptype_groups = [g for g in ptype_groups.values() if len(g) >= 3]

            if len(valid_ptype_groups) >= 2:
                H, p = stats.kruskal(*valid_ptype_groups)
                g0 = ptype_groups.get(0, np.array([]))
                g2 = ptype_groups.get(2, np.array([]))
                ratio = np.median(g2) / np.median(g0) if len(g0) > 0 and np.median(g0) > 0 else np.nan
                print(f"    {ptype} partners: KW p={p:.2e}, median ratio={ratio:.3f}, n={len(ptype_work)}")

    return test_results


def main():
    spatial_df = pd.read_csv(RESULTS_DIR / "all_branch_features_with_spatial.csv")
    print(f"Loaded {len(spatial_df)} branches, {spatial_df['neuron_label'].nunique()} neurons\n")

    # 1. Mediation analysis
    mediation_results = mediation_analysis(spatial_df)

    # 2. Partner compactness by regime
    partner_results = partner_regime_analysis(spatial_df, MICRONS_DATA)

    # Save all results
    all_results = {
        "mediation": mediation_results,
        "partner_compactness": partner_results,
    }

    with open(RESULTS_DIR / "partner_mediation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved to {RESULTS_DIR / 'partner_mediation_results.json'}")


if __name__ == "__main__":
    main()
