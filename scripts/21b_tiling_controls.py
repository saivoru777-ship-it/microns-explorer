#!/usr/bin/env python3
"""Tiling controls and diagnostics.

Implements the critical controls needed before the tiling paper:
1. K-preserving shuffle null — distinguishes real tiling from multiplicity artifact
2. Inter-bouton interval CV — validates the lattice assumption
3. Random-partner negative control — confirms tiling is partner-specific
4. Branch-length normalization — tests whether regime adaptation is geometric
5. k≥2 data fraction — reports subset coverage

Loads the same data as script 21 using the script 12 pipeline.
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
    """Get synapse arc-length positions with presynaptic partner IDs."""
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
                "pre_broad": row.get("broad_class", "unknown"),
            })

        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


def compute_nn_distances(positions_by_axon):
    """Same-axon and cross-axon NN distances on one branch."""
    all_pos = []
    for axon_id, positions in positions_by_axon.items():
        for p in positions:
            all_pos.append((p, axon_id))

    if len(all_pos) < 2:
        return [], []

    all_pos.sort(key=lambda x: x[0])
    same_nn, cross_nn = [], []

    for i, (pos_i, axon_i) in enumerate(all_pos):
        nearest_same = np.inf
        nearest_cross = np.inf
        for j, (pos_j, axon_j) in enumerate(all_pos):
            if i == j:
                continue
            d = abs(pos_i - pos_j) / 1000.0
            if axon_i == axon_j:
                nearest_same = min(nearest_same, d)
            else:
                nearest_cross = min(nearest_cross, d)
        if nearest_same < np.inf:
            same_nn.append(nearest_same)
        if nearest_cross < np.inf:
            cross_nn.append(nearest_cross)

    return same_nn, cross_nn


# ── Control 1: K-preserving shuffle null ──────────────────────────────────

def k_preserving_shuffle(positions_by_axon, n_shuffles=200, rng=None):
    """Shuffle synapse POSITIONS while preserving how many each axon makes.

    For each shuffle: randomly reassign the observed positions to axons,
    keeping the number of positions per axon fixed. Then compute same-axon NN.

    If the observed same-axon NN matches this null → tiling is a k-artifact.
    If observed same-axon NN >> null → tiling is real self-avoidance.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    all_positions = []
    axon_labels = []
    for axon_id, positions in positions_by_axon.items():
        for p in positions:
            all_positions.append(p)
            axon_labels.append(axon_id)

    all_positions = np.array(all_positions)
    axon_labels = np.array(axon_labels)
    n = len(all_positions)

    if n < 3:
        return []

    shuffle_same_nn_medians = []

    for _ in range(n_shuffles):
        # Shuffle positions, keep axon assignment fixed
        shuffled_pos = rng.permutation(all_positions)

        # Rebuild positions_by_axon with shuffled positions
        shuffled_by_axon = defaultdict(list)
        for pos, axon in zip(shuffled_pos, axon_labels):
            shuffled_by_axon[axon].append(pos)

        # Compute same-axon NN under shuffle
        same_nn_shuffle = []
        for axon_id, positions in shuffled_by_axon.items():
            if len(positions) < 2:
                continue
            positions_sorted = sorted(positions)
            for i, p in enumerate(positions_sorted):
                nearest = np.inf
                for j, q in enumerate(positions_sorted):
                    if i != j:
                        nearest = min(nearest, abs(p - q) / 1000.0)
                if nearest < np.inf:
                    same_nn_shuffle.append(nearest)

        if same_nn_shuffle:
            shuffle_same_nn_medians.append(np.median(same_nn_shuffle))

    return shuffle_same_nn_medians


# ── Control 2: Inter-bouton interval CV ──────────────────────────────────

def compute_interval_cv(positions_by_axon):
    """For each axon with k≥3 boutons, compute CV of consecutive inter-bouton intervals.

    Returns list of (axon_id, k, cv).
    CV < 0.5 → regular (lattice-like)
    CV ≈ 1.0 → random (exponential)
    CV > 1.0 → clustered
    """
    results = []
    for axon_id, positions in positions_by_axon.items():
        if len(positions) < 3:
            continue
        sorted_pos = sorted(positions)
        intervals = np.diff(sorted_pos) / 1000.0  # nm → µm
        if len(intervals) < 2:
            continue
        mean_int = np.mean(intervals)
        if mean_int < 1e-6:
            continue
        cv = np.std(intervals) / mean_int
        results.append((axon_id, len(positions), float(cv)))
    return results


# ── Control 3: Random-partner negative control ───────────────────────────

def random_partner_b2(positions_by_axon, n_shuffles=100, rng=None):
    """Assign each synapse to a random 'fake partner' preserving k distribution.

    If tiling disappears → partner-specific (real tiling).
    If tiling persists → generic spacing, not partner-related.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    all_positions = []
    axon_labels = []
    for axon_id, positions in positions_by_axon.items():
        for p in positions:
            all_positions.append(p)
            axon_labels.append(axon_id)

    axon_labels = np.array(axon_labels)
    n = len(all_positions)

    if n < 3:
        return []

    shuffle_ratios = []

    for _ in range(n_shuffles):
        # Shuffle AXON LABELS, keep positions fixed
        shuffled_labels = rng.permutation(axon_labels)

        shuffled_by_axon = defaultdict(list)
        for pos, axon in zip(all_positions, shuffled_labels):
            shuffled_by_axon[axon].append(pos)

        same_nn, cross_nn = compute_nn_distances(shuffled_by_axon)
        if same_nn and cross_nn:
            ratio = np.median(cross_nn) / np.median(same_nn)
            shuffle_ratios.append(ratio)

    return shuffle_ratios


# ── Main analysis ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TILING CONTROLS AND DIAGNOSTICS")
    print("=" * 60)

    bf = pd.read_csv(RESULTS_DIR / "all_branch_features_with_spatial.csv")
    s12 = _import_script12()

    rng = np.random.default_rng(42)

    # Process both populations
    results = {}

    for pop_name, cell_type_filter in [("excitatory", "excitatory"), ("inhibitory", "inhibitory")]:
        print(f"\n{'='*60}")
        print(f"POPULATION: {pop_name}")
        print(f"{'='*60}")

        neuron_labels = bf[bf["cell_type"] == cell_type_filter]["neuron_label"].unique()
        print(f"Neurons: {len(neuron_labels)}")

        # Collect per-branch data
        all_same_nn = []
        all_cross_nn = []
        all_interval_cvs = []          # Control 2
        all_lattice_constants = []     # (regime, spacing, branch_length)
        shuffle_same_nn_medians = []   # Control 1
        random_partner_ratios = []     # Control 3
        k_distribution = defaultdict(int)  # Control 5

        total_branches = 0
        branches_with_multi = 0
        total_synapses = 0
        synapses_in_multi = 0
        neurons_processed = 0

        n_shuffle_branches = 0
        max_shuffle_branches = 500  # Limit shuffle computation

        for nl in neuron_labels:
            syn_df = get_synapse_arclengths(nl, s12)
            if syn_df is None or len(syn_df) < 5:
                continue

            bf_nl = bf[bf["neuron_label"] == nl][["branch_idx", "regime", "total_length_nm"]].copy()
            syn_df = syn_df.merge(bf_nl, on="branch_idx", how="left")
            neurons_processed += 1

            for (bid, regime), bgrp in syn_df.groupby(["branch_idx", "regime"]):
                if len(bgrp) < 2 or pd.isna(regime):
                    continue

                total_branches += 1
                total_synapses += len(bgrp)
                branch_length_um = bgrp["total_length_nm"].iloc[0] / 1000.0

                positions_by_axon = defaultdict(list)
                for _, row in bgrp.iterrows():
                    positions_by_axon[row["pre_root_id"]].append(row["arc_pos_nm"])

                # Control 5: k distribution
                has_multi = False
                for axon_id, positions in positions_by_axon.items():
                    k_distribution[len(positions)] += 1
                    if len(positions) >= 2:
                        has_multi = True
                        synapses_in_multi += len(positions)

                if has_multi:
                    branches_with_multi += 1

                if len(positions_by_axon) < 2:
                    continue

                # B2 test
                same_nn, cross_nn = compute_nn_distances(positions_by_axon)
                all_same_nn.extend(same_nn)
                all_cross_nn.extend(cross_nn)

                # Control 2: interval CV for axons with k≥3
                cvs = compute_interval_cv(positions_by_axon)
                all_interval_cvs.extend(cvs)

                # Lattice constant with branch length (Control 4)
                for axon_id, positions in positions_by_axon.items():
                    if len(positions) >= 2:
                        sorted_pos = sorted(positions)
                        spacings = np.diff(sorted_pos) / 1000.0
                        mean_spacing = np.mean(spacings)
                        all_lattice_constants.append((
                            int(regime), mean_spacing, branch_length_um,
                            len(positions)
                        ))

                # Controls 1 and 3 (subsample to keep runtime bounded)
                if n_shuffle_branches < max_shuffle_branches and len(positions_by_axon) >= 2:
                    # Control 1: k-preserving shuffle
                    shuf = k_preserving_shuffle(positions_by_axon, n_shuffles=50, rng=rng)
                    shuffle_same_nn_medians.extend(shuf)

                    # Control 3: random-partner
                    rp = random_partner_b2(positions_by_axon, n_shuffles=50, rng=rng)
                    random_partner_ratios.extend(rp)

                    n_shuffle_branches += 1

        # ── Report results ──

        same = np.array(all_same_nn)
        cross = np.array(all_cross_nn)

        if len(same) < 10 or len(cross) < 10:
            print("INSUFFICIENT DATA")
            continue

        observed_ratio = np.median(cross) / np.median(same)
        print(f"\nObserved B2 ratio: {observed_ratio:.3f}")
        print(f"  Same-axon NN median: {np.median(same):.2f} µm (n={len(same):,})")
        print(f"  Cross-axon NN median: {np.median(cross):.2f} µm (n={len(cross):,})")

        # ── Control 1: K-preserving shuffle ──
        print(f"\n--- CONTROL 1: K-Preserving Shuffle Null ---")
        if shuffle_same_nn_medians:
            shuffle_arr = np.array(shuffle_same_nn_medians)
            observed_same_median = np.median(same)
            shuffle_median = np.median(shuffle_arr)
            # How many shuffles have same-axon NN >= observed?
            p_val = np.mean(shuffle_arr >= observed_same_median)
            print(f"  Observed same-axon NN median: {observed_same_median:.2f} µm")
            print(f"  Shuffle null same-axon NN median: {shuffle_median:.2f} µm")
            print(f"  Ratio (observed/shuffle): {observed_same_median / shuffle_median:.2f}")
            print(f"  p-value (shuffle >= observed): {p_val:.4f}")
            if observed_same_median > shuffle_median * 1.2:
                print(f"  VERDICT: Tiling is REAL — same-axon NN is {observed_same_median/shuffle_median:.1f}x "
                      f"larger than expected from k-artifact alone")
            else:
                print(f"  VERDICT: Tiling may be a k-ARTIFACT — same-axon NN matches shuffle null")
        else:
            print("  No shuffle data computed")

        # ── Control 2: Inter-bouton interval CV ──
        print(f"\n--- CONTROL 2: Inter-Bouton Interval CV ---")
        if all_interval_cvs:
            cvs = np.array([cv for _, _, cv in all_interval_cvs])
            ks = np.array([k for _, k, _ in all_interval_cvs])
            print(f"  Axon-branch pairs with k≥3: {len(cvs):,}")
            print(f"  CV median: {np.median(cvs):.3f}")
            print(f"  CV mean: {np.mean(cvs):.3f}")
            print(f"  CV < 0.5 (regular): {100 * np.mean(cvs < 0.5):.1f}%")
            print(f"  CV 0.5-1.0 (moderate): {100 * np.mean((cvs >= 0.5) & (cvs < 1.0)):.1f}%")
            print(f"  CV ≥ 1.0 (random/clustered): {100 * np.mean(cvs >= 1.0):.1f}%")
            if np.median(cvs) < 0.5:
                print(f"  VERDICT: REGULAR spacing — lattice framing is justified")
            elif np.median(cvs) < 0.8:
                print(f"  VERDICT: MODERATELY regular — self-avoidance framing appropriate")
            else:
                print(f"  VERDICT: IRREGULAR — lattice framing may not be justified")

            # CV by k
            for k_min in [3, 4, 5, 10]:
                mask = ks >= k_min
                if mask.sum() > 50:
                    print(f"  CV for k≥{k_min}: median={np.median(cvs[mask]):.3f} (n={mask.sum():,})")
        else:
            print("  No k≥3 axon-branch pairs found")

        # ── Control 3: Random-partner negative control ──
        print(f"\n--- CONTROL 3: Random-Partner Negative Control ---")
        if random_partner_ratios:
            rp_arr = np.array(random_partner_ratios)
            print(f"  Observed B2 ratio: {observed_ratio:.3f}")
            print(f"  Random-partner ratio median: {np.median(rp_arr):.3f}")
            print(f"  Random-partner ratio mean: {np.mean(rp_arr):.3f}")
            if observed_ratio < np.percentile(rp_arr, 5):
                print(f"  VERDICT: Tiling is PARTNER-SPECIFIC — observed ratio is below "
                      f"5th percentile of random-partner null ({np.percentile(rp_arr, 5):.3f})")
            else:
                print(f"  VERDICT: Tiling is NOT partner-specific — could be generic spacing")
        else:
            print("  No random-partner data computed")

        # ── Control 4: Branch-length normalization ──
        print(f"\n--- CONTROL 4: Branch-Length Normalization ---")
        if all_lattice_constants:
            lc_df = pd.DataFrame(all_lattice_constants,
                                 columns=["regime", "spacing_um", "branch_length_um", "k"])
            lc_df["normalized"] = lc_df["spacing_um"] / lc_df["branch_length_um"]

            print(f"  Axon-branch pairs: {len(lc_df):,}")
            print(f"  Normalized spacing (spacing / branch_length) by regime:")
            for reg in [0, 1, 2]:
                rd = lc_df[lc_df["regime"] == reg]
                if len(rd) > 10:
                    print(f"    Regime {reg}: raw={np.median(rd['spacing_um']):.2f} µm, "
                          f"normalized={np.median(rd['normalized']):.4f}, "
                          f"branch_length={np.median(rd['branch_length_um']):.1f} µm "
                          f"(n={len(rd):,})")

            # KW test on normalized spacing across regimes
            groups = [lc_df[lc_df["regime"] == r]["normalized"].values
                      for r in [0, 1, 2] if len(lc_df[lc_df["regime"] == r]) > 10]
            if len(groups) >= 2:
                h, p = stats.kruskal(*groups)
                print(f"  Kruskal-Wallis (normalized spacing ~ regime): H={h:.1f}, p={p:.2e}")
                if p < 0.01:
                    print(f"  VERDICT: Adaptation is BIOLOGICAL — normalized spacing varies with regime")
                else:
                    print(f"  VERDICT: Adaptation is GEOMETRIC — normalized spacing is constant")
        else:
            print("  No lattice constant data")

        # ── Control 5: k≥2 data fraction ──
        print(f"\n--- CONTROL 5: k≥2 Data Fraction ---")
        print(f"  Neurons processed: {neurons_processed}")
        print(f"  Total branches (with ≥2 synapses): {total_branches:,}")
        print(f"  Branches with any k≥2 partner: {branches_with_multi:,} "
              f"({100*branches_with_multi/total_branches:.1f}%)" if total_branches > 0 else "")
        print(f"  Synapses in multi-contact partners: {synapses_in_multi:,} "
              f"({100*synapses_in_multi/total_synapses:.1f}%)" if total_synapses > 0 else "")
        print(f"  k distribution:")
        for k in sorted(k_distribution.keys())[:10]:
            print(f"    k={k}: {k_distribution[k]:,} axon-branch pairs")

        # Save population results
        pop_result = {
            "population": pop_name,
            "neurons_processed": neurons_processed,
            "observed_ratio": float(observed_ratio),
            "same_nn_median": float(np.median(same)),
            "cross_nn_median": float(np.median(cross)),
            "n_same_nn": len(same),
            "n_cross_nn": len(cross),
        }

        if shuffle_same_nn_medians:
            pop_result["control1_k_shuffle"] = {
                "observed_same_nn_median": float(np.median(same)),
                "shuffle_same_nn_median": float(np.median(shuffle_same_nn_medians)),
                "ratio_observed_over_shuffle": float(np.median(same) / np.median(shuffle_same_nn_medians)),
                "p_value": float(np.mean(np.array(shuffle_same_nn_medians) >= np.median(same))),
            }

        if all_interval_cvs:
            cvs = np.array([cv for _, _, cv in all_interval_cvs])
            pop_result["control2_interval_cv"] = {
                "median_cv": float(np.median(cvs)),
                "mean_cv": float(np.mean(cvs)),
                "frac_regular": float(np.mean(cvs < 0.5)),
                "frac_random": float(np.mean(cvs >= 1.0)),
                "n_pairs": len(cvs),
            }

        if random_partner_ratios:
            pop_result["control3_random_partner"] = {
                "observed_ratio": float(observed_ratio),
                "null_median_ratio": float(np.median(random_partner_ratios)),
                "p_below_5th": float(observed_ratio < np.percentile(random_partner_ratios, 5)),
            }

        if all_lattice_constants:
            lc_df = pd.DataFrame(all_lattice_constants,
                                 columns=["regime", "spacing_um", "branch_length_um", "k"])
            lc_df["normalized"] = lc_df["spacing_um"] / lc_df["branch_length_um"]
            pop_result["control4_normalization"] = {}
            for reg in [0, 1, 2]:
                rd = lc_df[lc_df["regime"] == reg]
                if len(rd) > 10:
                    pop_result["control4_normalization"][f"regime{reg}"] = {
                        "raw_median_um": float(np.median(rd["spacing_um"])),
                        "normalized_median": float(np.median(rd["normalized"])),
                        "n": len(rd),
                    }

        pop_result["control5_data_fraction"] = {
            "total_branches": total_branches,
            "branches_with_multi_k2": branches_with_multi,
            "frac_branches": float(branches_with_multi / total_branches) if total_branches > 0 else 0,
            "total_synapses": total_synapses,
            "synapses_in_multi": synapses_in_multi,
            "frac_synapses": float(synapses_in_multi / total_synapses) if total_synapses > 0 else 0,
        }

        results[pop_name] = pop_result

    # Save all results
    out_path = RESULTS_DIR / "tiling_controls.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"All results saved to {out_path}")


if __name__ == '__main__':
    main()
