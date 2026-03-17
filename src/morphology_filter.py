"""Morphology quality filter for analysis-ready neurons.

"Proofread" does not guarantee analysis-readiness. This module filters
for neurons suitable for dendritic regime analysis based on morphological
criteria beyond basic QC.
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path.home() / "research" / "neurostat"))

# Exclusion thresholds
MIN_BRANCH_COUNT = 10
MIN_TOTAL_LENGTH_UM = 500.0
MIN_SNAPPED_SYNAPSES = 50
MAX_STUB_FRACTION = 0.50  # <50% single-node branches
MIN_EXTENT_UM = 100.0     # must span >100 μm in at least 2 axes
MIN_BRANCHES_WITH_SYNAPSES = 3  # branches with ≥3 synapses


def filter_analysis_ready(qc_results, data_dir):
    """Filter to neurons suitable for dendritic regime analysis.

    Parameters
    ----------
    qc_results : pd.DataFrame
        Output from validate.validate_batch(). Must have 'passed' == True
        for neurons to be considered.
    data_dir : str or Path
        Directory containing neuron files.

    Returns
    -------
    pd.DataFrame with analysis compatibility metrics for passing neurons.
    """
    data_dir = Path(data_dir)

    # Start with QC-passing neurons only
    candidates = qc_results[qc_results["passed"]].copy()
    if len(candidates) == 0:
        logger.warning("No neurons passed QC")
        return pd.DataFrame()

    results = []
    for _, row in candidates.iterrows():
        label = row["label"]
        root_id = row["root_id"]
        metrics = _compute_morphology_metrics(label, root_id, data_dir)
        if metrics is not None:
            results.append(metrics)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Apply exclusion criteria
    df["passes_branch_count"] = df["n_branches"] >= MIN_BRANCH_COUNT
    df["passes_total_length"] = df["total_length_um"] >= MIN_TOTAL_LENGTH_UM
    df["passes_snapped_synapses"] = df["n_snapped_synapses"] >= MIN_SNAPPED_SYNAPSES
    df["passes_stub_fraction"] = df["stub_fraction"] < MAX_STUB_FRACTION
    df["passes_extent"] = df["n_axes_above_100um"] >= 2
    df["analysis_ready"] = (
        df["passes_branch_count"]
        & df["passes_total_length"]
        & df["passes_snapped_synapses"]
        & df["passes_stub_fraction"]
        & df["passes_extent"]
    )

    n_ready = df["analysis_ready"].sum()
    logger.info(
        f"Analysis-ready: {n_ready}/{len(df)} neurons "
        f"({n_ready/len(df):.0%})"
    )

    # Log exclusion reasons
    for col in ["passes_branch_count", "passes_total_length",
                "passes_snapped_synapses", "passes_stub_fraction", "passes_extent"]:
        n_fail = (~df[col]).sum()
        if n_fail > 0:
            logger.info(f"  Excluded by {col}: {n_fail}")

    return df


def _compute_morphology_metrics(label, root_id, data_dir):
    """Compute analysis compatibility metrics for one neuron.

    Returns dict of metrics or None on failure.
    """
    from neurostat.io.swc import NeuronSkeleton

    swc_path = data_dir / f"{label}.swc"
    syn_path = data_dir / f"{label}_synapses.csv"

    if not swc_path.exists() or not syn_path.exists():
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skeleton = NeuronSkeleton.from_swc_file(str(swc_path), scale_factor=1000.0)

        dendrite_skel = skeleton.filter_by_type([1, 3, 4])
        if len(dendrite_skel.branches) < 3:
            dendrite_skel = skeleton

        n_branches = len(dendrite_skel.branches)

        # Total dendritic length in μm
        total_length_nm = sum(br.total_length for br in dendrite_skel.branches)
        total_length_um = total_length_nm / 1000.0

        # Stub fraction (single-node branches)
        n_stubs = sum(1 for br in dendrite_skel.branches if len(br.node_ids) <= 1)
        stub_fraction = n_stubs / n_branches if n_branches > 0 else 1.0

        # Bounding box extent
        coords = np.array([[n.x, n.y, n.z] for n in dendrite_skel.nodes.values()])
        extent_nm = coords.max(axis=0) - coords.min(axis=0)
        extent_um = extent_nm / 1000.0
        n_axes_above = int(np.sum(extent_um > MIN_EXTENT_UM))

        # Snap synapses
        syn_df = pd.read_csv(syn_path)
        syn_coords_nm = syn_df[["x_um", "y_um", "z_um"]].values * 1000.0
        snap = dendrite_skel.snap_points(syn_coords_nm, d_max=50000.0)
        n_snapped = int(snap.valid.sum())

        # Branches with ≥3 synapses
        if n_snapped > 0:
            branch_counts = np.bincount(
                snap.branch_ids[snap.valid], minlength=n_branches
            )
            n_branches_with_synapses = int(np.sum(branch_counts >= 3))
        else:
            branch_counts = np.zeros(n_branches)
            n_branches_with_synapses = 0

        # Electrotonic length distribution (if radii available)
        has_variable_radius = False
        e_length_median = np.nan
        e_length_iqr = np.nan
        try:
            radii = []
            for br in dendrite_skel.branches:
                br_radii = [
                    dendrite_skel.nodes[nid].radius
                    for nid in br.node_ids
                    if nid in dendrite_skel.nodes
                ]
                if br_radii and np.std(br_radii) > 0:
                    has_variable_radius = True

            if has_variable_radius:
                from src.format import _classify_broad  # noqa: avoid circular
                # Compute electrotonic lengths
                RM = 20_000
                RI = 150
                e_lengths = []
                for br in dendrite_skel.branches:
                    br_radii = [
                        dendrite_skel.nodes[nid].radius
                        for nid in br.node_ids
                        if nid in dendrite_skel.nodes
                    ]
                    if br_radii:
                        d_nm = 2.0 * np.mean(br_radii)
                        d_cm = d_nm * 1e-7
                        L_cm = br.total_length * 1e-7
                        lam = np.sqrt(RM * d_cm / (4.0 * RI))
                        if lam > 0:
                            e_lengths.append(L_cm / lam)
                if e_lengths:
                    e_length_median = float(np.median(e_lengths))
                    e_length_iqr = float(np.subtract(*np.percentile(e_lengths, [75, 25])))
        except Exception:
            pass

        return {
            "label": label,
            "root_id": str(root_id),
            "n_branches": n_branches,
            "total_length_um": total_length_um,
            "n_snapped_synapses": n_snapped,
            "n_branches_with_3plus_synapses": n_branches_with_synapses,
            "stub_fraction": stub_fraction,
            "extent_x_um": extent_um[0],
            "extent_y_um": extent_um[1],
            "extent_z_um": extent_um[2],
            "n_axes_above_100um": n_axes_above,
            "has_variable_radius": has_variable_radius,
            "electrotonic_length_median": e_length_median,
            "electrotonic_length_iqr": e_length_iqr,
        }

    except Exception as e:
        logger.error(f"Morphology metrics failed for {label}: {e}")
        return None
