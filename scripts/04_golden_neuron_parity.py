#!/usr/bin/env python3
"""Golden neuron parity test.

Re-fetches 3 of the original 12 neurons through the new pipeline
and compares to existing data. This is the most important validation step.

Usage:
    python scripts/04_golden_neuron_parity.py
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path.home() / "research" / "neurostat"))

from neurostat.io.swc import NeuronSkeleton

# Golden neurons: one per cell type category
GOLDEN_NEURONS = [
    # (old_label, new_broad, new_fine, root_id)
    ("exc_23P", "excitatory", "23P", 864691135848859998),
    ("inh_BC", "inhibitory", "BC", 864691135293026230),
    ("inh_MC", "inhibitory", "MC", 864691135273485073),
]

# Existing data locations
OLD_DATA_DIR = Path.home() / "research" / "neurostat-input-clustering" / "data" / "microns"
OLD_PARTNER_DIR = Path.home() / "research" / "neurostat-input-clustering" / "results"

# New data location
NEW_DATA_DIR = PROJECT_DIR / "data" / "neurons"

# Tolerances
TOLERANCES = {
    "branch_count": 0.10,      # ±10%
    "total_length": 0.10,      # ±10%
    "synapse_count": 0.05,     # ±5%
    "snap_rate": 0.05,         # ±5%
    "partner_count_k3": 0.10,  # ±10%
    "mean_branch_length": 0.10,  # ±10%
}

SCIENTIFIC_TOLERANCES = {
    "electrotonic_length_median": 0.15,  # ±15%
    "regime_counts": 0.20,               # ±20%
}


def load_skeleton_metrics(swc_path, scale_factor=1000.0):
    """Load skeleton and compute key metrics."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skeleton = NeuronSkeleton.from_swc_file(str(swc_path), scale_factor=scale_factor)

    dendrite_skel = skeleton.filter_by_type([1, 3, 4])
    if len(dendrite_skel.branches) < 3:
        dendrite_skel = skeleton

    n_branches = len(dendrite_skel.branches)
    lengths = [br.total_length for br in dendrite_skel.branches]
    total_length = sum(lengths)
    mean_branch_length = np.mean(lengths) if lengths else 0

    return {
        "skeleton": dendrite_skel,
        "n_branches": n_branches,
        "total_length_nm": total_length,
        "mean_branch_length_nm": mean_branch_length,
    }


def load_synapse_metrics(syn_path, skeleton, col_prefix="x_um"):
    """Load synapses and compute snap metrics."""
    syn_df = pd.read_csv(syn_path)

    if f"{col_prefix}" in syn_df.columns:
        # New format: x_um, y_um, z_um
        syn_coords_nm = syn_df[["x_um", "y_um", "z_um"]].values * 1000.0
    elif "x" in syn_df.columns:
        # Old format: x, y, z (in nm)
        syn_coords_nm = syn_df[["x", "y", "z"]].values
    else:
        raise ValueError(f"Unknown synapse format: {syn_df.columns.tolist()}")

    snap = skeleton.snap_points(syn_coords_nm, d_max=50000.0)
    n_valid = snap.valid.sum()
    snap_rate = n_valid / len(syn_df) if len(syn_df) > 0 else 0

    return {
        "synapse_count": len(syn_df),
        "n_valid": n_valid,
        "snap_rate": snap_rate,
        "snap": snap,
    }


def load_partner_metrics(partner_path, snap_result, syn_coords_nm):
    """Count partners with ≥3 synapses."""
    if not Path(partner_path).exists():
        return {"partner_count_k3": 0}

    partner_df = pd.read_csv(partner_path)
    if "pre_root_id" not in partner_df.columns:
        return {"partner_count_k3": 0}

    # Count per unique partner
    partner_counts = partner_df["pre_root_id"].value_counts()
    k3 = (partner_counts >= 3).sum()

    return {"partner_count_k3": k3}


def compute_parity(old_metrics, new_metrics, metric_name, tolerance):
    """Check if two values are within tolerance."""
    old_val = old_metrics.get(metric_name, 0)
    new_val = new_metrics.get(metric_name, 0)

    if old_val == 0 and new_val == 0:
        return True, 0.0, old_val, new_val

    if old_val == 0:
        return False, float("inf"), old_val, new_val

    rel_diff = abs(new_val - old_val) / abs(old_val)
    passed = rel_diff <= tolerance

    return passed, rel_diff, old_val, new_val


def run_parity_test(old_label, new_broad, new_fine, root_id):
    """Run parity test for one golden neuron."""
    print(f"\n{'='*60}")
    print(f"Golden Neuron: {old_label} (root_id={root_id})")
    print(f"{'='*60}")

    new_label = f"{'exc' if new_broad == 'excitatory' else 'inh'}_{new_fine}_{root_id}"

    # File paths
    old_swc = OLD_DATA_DIR / f"{old_label}_{root_id}.swc"
    new_swc = NEW_DATA_DIR / f"{new_label}.swc"
    old_syn = OLD_DATA_DIR / f"{old_label}_{root_id}_synapses.csv"
    new_syn = NEW_DATA_DIR / f"{new_label}_synapses.csv"
    old_partner = OLD_PARTNER_DIR / f"{old_label}_presynaptic.csv"
    new_partner = NEW_DATA_DIR / f"{new_label}_presynaptic.csv"

    # Check file existence
    missing = []
    for path, desc in [
        (old_swc, "old SWC"),
        (new_swc, "new SWC"),
        (old_syn, "old synapses"),
        (new_syn, "new synapses"),
    ]:
        if not path.exists():
            missing.append(f"{desc}: {path}")

    if missing:
        print("MISSING FILES:")
        for m in missing:
            print(f"  {m}")
        return False

    # Load metrics
    # Old SWC is in nm (scale_factor=1000.0 means file is in μm)
    # Check which format the old file is in
    old_skel_metrics = load_skeleton_metrics(old_swc, scale_factor=1000.0)
    new_skel_metrics = load_skeleton_metrics(new_swc, scale_factor=1000.0)

    old_syn_metrics = load_synapse_metrics(old_syn, old_skel_metrics["skeleton"])
    new_syn_metrics = load_synapse_metrics(new_syn, new_skel_metrics["skeleton"])

    old_partner_metrics = load_partner_metrics(
        old_partner, old_syn_metrics["snap"], None
    )
    new_partner_metrics = load_partner_metrics(
        new_partner, new_syn_metrics["snap"], None
    )

    # Combine
    old_all = {**old_skel_metrics, **old_syn_metrics, **old_partner_metrics}
    new_all = {**new_skel_metrics, **new_syn_metrics, **new_partner_metrics}

    # Run parity checks
    all_passed = True
    print(f"\n{'Metric':<25} {'Old':>12} {'New':>12} {'Diff':>8} {'Tol':>6} {'Result':>8}")
    print("-" * 75)

    for metric, tol in TOLERANCES.items():
        passed, diff, old_val, new_val = compute_parity(old_all, new_all, metric, tol)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(
            f"  {metric:<23} {old_val:>12.1f} {new_val:>12.1f} "
            f"{diff:>7.1%} {tol:>5.0%} {status:>8}"
        )

    # Overall
    print(f"\n{'OVERALL':>25}: {'PASS' if all_passed else 'FAIL'}")

    if not all_passed:
        # Print version metadata for debugging
        metadata_dir = PROJECT_DIR / "metadata"
        metadata_files = sorted(metadata_dir.glob("batch_*.json"))
        if metadata_files:
            with open(metadata_files[-1]) as f:
                meta = json.load(f)
            print(f"\nVersion metadata:")
            for k, v in meta.items():
                print(f"  {k}: {v}")
        print("\nDiagnostic hints:")
        for metric, tol in TOLERANCES.items():
            passed, diff, old_val, new_val = compute_parity(old_all, new_all, metric, tol)
            if not passed:
                if metric == "branch_count":
                    print("  → branch_count: skeleton source or resolution changed")
                elif metric == "synapse_count":
                    print("  → synapse_count: check synapse table version")
                elif metric == "snap_rate":
                    print("  → snap_rate: likely coordinate unit error")
                elif metric == "total_length":
                    print("  → total_length: coordinate scaling issue")

    return all_passed


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print("=" * 60)
    print("GOLDEN NEURON PARITY TEST")
    print("=" * 60)

    all_results = []
    for old_label, new_broad, new_fine, root_id in GOLDEN_NEURONS:
        passed = run_parity_test(old_label, new_broad, new_fine, root_id)
        all_results.append((old_label, passed))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for label, passed in all_results:
        print(f"  {label}: {'PASS' if passed else 'FAIL'}")

    n_passed = sum(1 for _, p in all_results if p)
    n_total = len(all_results)
    print(f"\n{n_passed}/{n_total} golden neurons passed parity")

    if n_passed < n_total:
        print("\nWARNING: Parity failures detected.")
        print("STOP and investigate before scaling to batch downloads.")
        sys.exit(1)
    else:
        print("\nAll golden neurons pass parity. Safe to proceed to batch.")


if __name__ == "__main__":
    main()
