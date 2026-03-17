"""Quality validation for downloaded neuron data.

Checks SWC integrity, synapse counts, snap rates, partner coverage,
and coordinate sanity.
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Add neurostat to path
sys.path.insert(0, str(Path.home() / "research" / "neurostat"))


def validate_neuron(label, root_id, data_dir):
    """Run QC checks on downloaded neuron data.

    Parameters
    ----------
    label : str
        Neuron label.
    root_id : int or str
    data_dir : str or Path
        Directory containing the neuron files.

    Returns
    -------
    dict with check results:
        - label, root_id
        - swc_ok: bool
        - n_branches: int
        - synapse_count: int
        - snap_rate: float
        - median_snap_distance_um: float
        - partner_coverage: float
        - coord_sanity: bool
        - passed: bool (all checks pass)
        - failure_reasons: list of str
    """
    data_dir = Path(data_dir)
    swc_path = data_dir / f"{label}.swc"
    syn_path = data_dir / f"{label}_synapses.csv"
    partner_path = data_dir / f"{label}_presynaptic.csv"

    result = {
        "label": label,
        "root_id": str(root_id),
        "swc_ok": False,
        "n_branches": 0,
        "synapse_count": 0,
        "snap_rate": 0.0,
        "median_snap_distance_um": np.nan,
        "partner_coverage": 0.0,
        "coord_sanity": False,
        "passed": False,
        "failure_reasons": [],
    }

    # 1. SWC integrity
    try:
        from neurostat.io.swc import NeuronSkeleton
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skeleton = NeuronSkeleton.from_swc_file(str(swc_path), scale_factor=1000.0)

        dendrite_skel = skeleton.filter_by_type([1, 3, 4])
        if len(dendrite_skel.branches) < 3:
            dendrite_skel = skeleton

        n_branches = len(dendrite_skel.branches)
        result["swc_ok"] = True
        result["n_branches"] = n_branches

        if n_branches == 0:
            result["failure_reasons"].append("no_dendrite_branches")
    except Exception as e:
        result["failure_reasons"].append(f"swc_parse_error: {e}")
        result["passed"] = False
        return result

    # 2. Synapse count
    try:
        syn_df = pd.read_csv(syn_path)
        result["synapse_count"] = len(syn_df)
        if len(syn_df) < 20:
            result["failure_reasons"].append(f"too_few_synapses ({len(syn_df)})")
    except Exception as e:
        result["failure_reasons"].append(f"synapse_csv_error: {e}")
        result["passed"] = False
        return result

    # 3. Snap rate
    try:
        syn_coords_nm = syn_df[["x_um", "y_um", "z_um"]].values * 1000.0
        snap = dendrite_skel.snap_points(syn_coords_nm, d_max=50000.0)
        n_valid = snap.valid.sum()
        snap_rate = n_valid / len(syn_df) if len(syn_df) > 0 else 0
        result["snap_rate"] = snap_rate

        if snap_rate < 0.80:
            result["failure_reasons"].append(f"low_snap_rate ({snap_rate:.2%})")

        # Median snap distance
        if n_valid > 0:
            median_dist_nm = np.median(snap.distances[snap.valid])
            result["median_snap_distance_um"] = median_dist_nm / 1000.0
            if median_dist_nm / 1000.0 > 10.0:  # > 10 μm
                result["failure_reasons"].append(
                    f"high_median_snap_distance ({median_dist_nm/1000:.1f} μm)"
                )
    except Exception as e:
        result["failure_reasons"].append(f"snap_error: {e}")

    # 4. Partner coverage
    try:
        partner_df = pd.read_csv(partner_path)
        n_total = partner_df["pre_root_id"].nunique()
        n_typed = partner_df[
            partner_df["pre_cell_type"] != "unknown"
        ]["pre_root_id"].nunique()
        coverage = n_typed / n_total if n_total > 0 else 0
        result["partner_coverage"] = coverage

        # Note: CAVEclient cell type table covers ~91K neurons while there are
        # hundreds of thousands of unique presynaptic fragments in the volume.
        # Coverage of 5-15% by unique partner is typical and acceptable.
        # The original HDF5 had a complete vertex table giving ~100% coverage.
        if coverage < 0.01:
            result["failure_reasons"].append(f"low_partner_coverage ({coverage:.2%})")
    except Exception as e:
        result["failure_reasons"].append(f"partner_csv_error: {e}")

    # 5. Coordinate sanity
    try:
        # Check synapse coords fall within skeleton bounding box ± 100 μm
        skel_coords_nm = np.array([
            [n.x, n.y, n.z] for n in dendrite_skel.nodes.values()
        ])
        skel_min = skel_coords_nm.min(axis=0) / 1000.0 - 100.0  # μm
        skel_max = skel_coords_nm.max(axis=0) / 1000.0 + 100.0  # μm

        syn_um = syn_df[["x_um", "y_um", "z_um"]].values
        in_bounds = np.all(
            (syn_um >= skel_min) & (syn_um <= skel_max), axis=1
        )
        frac_in_bounds = in_bounds.mean()
        result["coord_sanity"] = frac_in_bounds > 0.90

        if frac_in_bounds <= 0.90:
            result["failure_reasons"].append(
                f"coord_out_of_bounds ({(1-frac_in_bounds):.1%} outside)"
            )
    except Exception as e:
        result["failure_reasons"].append(f"coord_check_error: {e}")

    # Overall pass/fail
    result["passed"] = len(result["failure_reasons"]) == 0
    return result


def validate_batch(labels_and_ids, data_dir):
    """Run validation on multiple neurons.

    Parameters
    ----------
    labels_and_ids : list of (label, root_id) tuples
    data_dir : str or Path

    Returns
    -------
    pd.DataFrame with validation results, one row per neuron.
    """
    results = []
    for label, root_id in labels_and_ids:
        logger.info(f"Validating {label}...")
        r = validate_neuron(label, root_id, data_dir)
        results.append(r)

    df = pd.DataFrame(results)
    # Convert failure_reasons list to string for CSV compatibility
    df["failure_reasons"] = df["failure_reasons"].apply(lambda x: "; ".join(x) if x else "")
    return df
