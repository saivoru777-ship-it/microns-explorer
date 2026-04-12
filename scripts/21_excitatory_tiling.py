#!/usr/bin/env python3
"""Excitatory presynaptic tiling analysis.

Extends the B2 test (script 17) from inhibitory-only to:
1. Excitatory neurons — does same-axon >> cross-axon hold?
2. Both cell types side by side for direct comparison
3. Tiling lattice constant characterization
4. Regime-dependent tiling adaptation
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "neurons"
RESULTS_DIR = PROJECT_DIR / "results" / "replication_full"

sys.path.insert(0, str(PROJECT_DIR.parent / "dendritic-regime-coupling"))
sys.path.insert(0, str(PROJECT_DIR.parent / "neurostat"))
sys.path.insert(0, str(PROJECT_DIR.parent / "neurostat-input-clustering" / "src"))


def _import_script12():
    sys.path.insert(0, str(SCRIPT_DIR))
    import importlib
    spec = importlib.util.spec_from_file_location(
        "s12", SCRIPT_DIR / "12_inhibitory_residual.py")
    s12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s12)
    return s12


def get_synapse_arclengths(neuron_label, s12_module):
    """Get synapse arc-length positions with presynaptic partner IDs.

    Uses the same API as script 17: s12._load_neuron_data + s12._match_partners.
    """
    try:
        result = s12_module._load_neuron_data(neuron_label, DATA_DIR)
        if result is None:
            return None
        dendrite_skel, syn_coords_nm, snap, partner_df = result
        matched = s12_module._match_partners(syn_coords_nm, snap, partner_df)

        arc_positions = snap.branch_positions

        rows = []
        for _, row in matched.iterrows():
            sidx = int(row["synapse_idx"])
            rows.append({
                "branch_idx": int(row["branch_idx"]),
                "arc_pos_nm": float(arc_positions[sidx]),
                "pre_root_id": int(row["pre_root_id"]),
                "pre_subtype": row.get("subtype", None),
                "pre_broad": row.get("broad_class", "unknown"),
            })

        return pd.DataFrame(rows) if rows else None
    except Exception as e:
        return None


def compute_nn_distances_on_branch(positions_by_axon):
    """Same-axon and cross-axon nearest-neighbour distances on one branch."""
    all_pos = []
    for axon_id, positions in positions_by_axon.items():
        for p in positions:
            all_pos.append((p, axon_id))

    if len(all_pos) < 2:
        return [], []

    all_pos.sort(key=lambda x: x[0])

    same_nn = []
    cross_nn = []

    for i, (pos_i, axon_i) in enumerate(all_pos):
        nearest_same = np.inf
        nearest_cross = np.inf

        for j, (pos_j, axon_j) in enumerate(all_pos):
            if i == j:
                continue
            d = abs(pos_i - pos_j) / 1000.0  # nm → µm
            if axon_i == axon_j:
                nearest_same = min(nearest_same, d)
            else:
                nearest_cross = min(nearest_cross, d)

        if nearest_same < np.inf:
            same_nn.append(nearest_same)
        if nearest_cross < np.inf:
            cross_nn.append(nearest_cross)

    return same_nn, cross_nn


def compute_lattice_constant(positions_by_axon):
    """Compute mean inter-bouton spacing per axon on this branch.

    Returns list of (axon_id, k, mean_spacing_um) for axons with k≥2.
    """
    results = []
    for axon_id, positions in positions_by_axon.items():
        if len(positions) < 2:
            continue
        positions_sorted = sorted(positions)
        spacings = np.diff(positions_sorted) / 1000.0  # nm → µm
        results.append((axon_id, len(positions), float(np.mean(spacings))))
    return results


def run_b2_for_population(population_name, neuron_labels, bf, s12, syn_type_filter=None):
    """Run the B2 tiling analysis for a given neuron population.

    syn_type_filter: if 'excitatory' or 'inhibitory', only use synapses of that type.
                     if None, use ALL synapses (matching original script 17 for inhibitory).
    """
    print(f"\n{'='*60}")
    print(f"B2 TILING: {population_name} ({len(neuron_labels)} neurons)")
    print(f"{'='*60}")

    same_nn_all = {"all": [], "regime0": [], "regime1": [], "regime2": []}
    cross_nn_all = {"all": [], "regime0": [], "regime1": [], "regime2": []}
    lattice_data = []  # (regime, k, mean_spacing)
    branch_lengths = {"regime0": [], "regime1": [], "regime2": []}

    neurons_processed = 0
    branches_used = 0

    for nl in neuron_labels:
        syn_df = get_synapse_arclengths(nl, s12)
        if syn_df is None or len(syn_df) < 5:
            continue

        # Get branch features for this neuron
        bf_nl = bf[bf["neuron_label"] == nl][["branch_idx", "regime", "total_length_nm"]].copy()
        syn_df = syn_df.merge(bf_nl, on="branch_idx", how="left")

        # Filter by synapse type if requested
        if syn_type_filter == "inhibitory":
            syn_df = syn_df[syn_df["pre_broad"] == "inhibitory"]
        elif syn_type_filter == "excitatory":
            syn_df = syn_df[syn_df["pre_broad"] == "excitatory"]
        # else: use all synapses

        if len(syn_df) < 3:
            continue

        neurons_processed += 1

        for (bid, regime), bgrp in syn_df.groupby(["branch_idx", "regime"]):
            if len(bgrp) < 2 or pd.isna(regime):
                continue

            positions_by_axon = defaultdict(list)
            for _, row in bgrp.iterrows():
                rid = row["pre_root_id"]
                positions_by_axon[rid].append(row["arc_pos_nm"])

            if len(positions_by_axon) < 2:
                continue

            same_nn, cross_nn = compute_nn_distances_on_branch(positions_by_axon)

            regime_key = f"regime{int(regime)}"
            if same_nn:
                same_nn_all["all"].extend(same_nn)
                same_nn_all[regime_key].extend(same_nn)
            if cross_nn:
                cross_nn_all["all"].extend(cross_nn)
                cross_nn_all[regime_key].extend(cross_nn)

            # Lattice constant
            lc = compute_lattice_constant(positions_by_axon)
            for axon_id, k, spacing in lc:
                lattice_data.append((int(regime), k, spacing))

            # Branch length
            bl = bgrp["total_length_nm"].iloc[0] / 1000.0  # µm
            branch_lengths[regime_key].append(bl)

            branches_used += 1

    print(f"  Neurons processed: {neurons_processed}")
    print(f"  Branches used: {branches_used:,}")
    print(f"  Same-axon NN pairs: {len(same_nn_all['all']):,}")
    print(f"  Cross-axon NN pairs: {len(cross_nn_all['all']):,}")

    if len(same_nn_all['all']) < 10 or len(cross_nn_all['all']) < 10:
        print("  INSUFFICIENT DATA")
        return None

    same = np.array(same_nn_all['all'])
    cross = np.array(cross_nn_all['all'])

    # Overall statistics
    print(f"\n  Same-axon NN:  median={np.median(same):.2f} µm  (n={len(same):,})")
    print(f"  Cross-axon NN: median={np.median(cross):.2f} µm  (n={len(cross):,})")

    ratio = np.median(cross) / np.median(same)
    u_stat, u_p = stats.mannwhitneyu(same, cross, alternative='greater')
    print(f"  Ratio (cross/same): {ratio:.3f}")
    print(f"  Mann-Whitney (same > cross): p={u_p:.2e}")

    # Verdict
    if ratio < 0.4:
        verdict = "B2 (presynaptic tiling)"
    elif ratio > 0.7:
        verdict = "B1 (postsynaptic scaffold)"
    else:
        verdict = "MIXED"
    print(f"  VERDICT: {verdict}")

    # By regime
    print(f"\n  By regime:")
    regime_results = {}
    for reg in ["regime0", "regime1", "regime2"]:
        s_r = np.array(same_nn_all[reg])
        c_r = np.array(cross_nn_all[reg])
        if len(s_r) > 10 and len(c_r) > 10:
            r = np.median(c_r) / np.median(s_r)
            print(f"    {reg}: same={np.median(s_r):.2f}µm  cross={np.median(c_r):.2f}µm  "
                  f"ratio={r:.3f}  (n_same={len(s_r)}, n_cross={len(c_r)})")
            regime_results[reg] = {
                "same_median_um": float(np.median(s_r)),
                "cross_median_um": float(np.median(c_r)),
                "ratio": float(r),
                "n_same": len(s_r),
                "n_cross": len(c_r),
            }

    # Lattice constant analysis
    if lattice_data:
        lc_df = pd.DataFrame(lattice_data, columns=["regime", "k", "spacing_um"])
        print(f"\n  Lattice constant (mean inter-bouton spacing):")
        for reg in [0, 1, 2]:
            reg_data = lc_df[lc_df["regime"] == reg]["spacing_um"]
            if len(reg_data) > 10:
                print(f"    regime {reg}: median={np.median(reg_data):.2f} µm  "
                      f"(n={len(reg_data)}, IQR=[{np.percentile(reg_data, 25):.2f}, "
                      f"{np.percentile(reg_data, 75):.2f}])")

        # Does lattice constant scale with branch length?
        if branch_lengths["regime0"] and branch_lengths["regime2"]:
            all_bl = []
            all_lc = []
            for _, row in lc_df.iterrows():
                all_lc.append(row["spacing_um"])
            # Correlation between branch length and lattice constant
            # (can only compute per-regime since we don't have per-branch-pair mapping)

    # Build results
    result = {
        "population": population_name,
        "syn_type_filter": syn_type_filter,
        "neurons_processed": neurons_processed,
        "branches_used": branches_used,
        "same_axon_nn": {
            "n": len(same),
            "median_um": float(np.median(same)),
            "mean_um": float(np.mean(same)),
            "p25_um": float(np.percentile(same, 25)),
            "p75_um": float(np.percentile(same, 75)),
        },
        "cross_axon_nn": {
            "n": len(cross),
            "median_um": float(np.median(cross)),
            "mean_um": float(np.mean(cross)),
            "p25_um": float(np.percentile(cross, 25)),
            "p75_um": float(np.percentile(cross, 75)),
        },
        "ratio_cross_same": float(ratio),
        "mann_whitney_p": float(u_p),
        "verdict": verdict,
        "by_regime": regime_results,
    }

    if lattice_data:
        lc_df = pd.DataFrame(lattice_data, columns=["regime", "k", "spacing_um"])
        lc_summary = {}
        for reg in [0, 1, 2]:
            rd = lc_df[lc_df["regime"] == reg]["spacing_um"]
            if len(rd) > 10:
                lc_summary[f"regime{reg}"] = {
                    "median_um": float(np.median(rd)),
                    "mean_um": float(np.mean(rd)),
                    "n": len(rd),
                }
        result["lattice_constant"] = lc_summary

    return result


def main():
    print("=" * 60)
    print("EXCITATORY PRESYNAPTIC TILING ANALYSIS")
    print("=" * 60)

    bf = pd.read_csv(RESULTS_DIR / "all_branch_features_with_spatial.csv")
    s12 = _import_script12()

    results = {}

    # 1. Excitatory neurons — ALL presynaptic synapses
    exc_neurons = bf[bf["cell_type"] == "excitatory"]["neuron_label"].unique()
    r = run_b2_for_population("Excitatory (all synapses)", exc_neurons, bf, s12,
                              syn_type_filter=None)
    if r:
        results["excitatory_all_synapses"] = r

    # 2. Excitatory neurons — excitatory presynaptic only
    r = run_b2_for_population("Excitatory (exc presynaptic only)", exc_neurons, bf, s12,
                              syn_type_filter="excitatory")
    if r:
        results["excitatory_exc_presynaptic"] = r

    # 3. Excitatory neurons — inhibitory presynaptic only
    r = run_b2_for_population("Excitatory (inh presynaptic only)", exc_neurons, bf, s12,
                              syn_type_filter="inhibitory")
    if r:
        results["excitatory_inh_presynaptic"] = r

    # 4. Inhibitory neurons — ALL synapses (replicate script 17)
    inh_neurons = bf[bf["cell_type"] == "inhibitory"]["neuron_label"].unique()
    r = run_b2_for_population("Inhibitory (all synapses)", inh_neurons, bf, s12,
                              syn_type_filter=None)
    if r:
        results["inhibitory_all_synapses"] = r

    # 5. Inhibitory neurons — inhibitory presynaptic only (original B2)
    r = run_b2_for_population("Inhibitory (inh presynaptic only)", inh_neurons, bf, s12,
                              syn_type_filter="inhibitory")
    if r:
        results["inhibitory_inh_presynaptic"] = r

    # Summary comparison
    print("\n" + "=" * 60)
    print("CROSS-POPULATION COMPARISON")
    print("=" * 60)

    for key, r in results.items():
        print(f"  {r['population']}:")
        print(f"    same={r['same_axon_nn']['median_um']:.2f}µm  "
              f"cross={r['cross_axon_nn']['median_um']:.2f}µm  "
              f"ratio={r['ratio_cross_same']:.3f}  "
              f"verdict={r['verdict']}")

    # Save
    out_path = RESULTS_DIR / "excitatory_tiling.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
