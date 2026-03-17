#!/usr/bin/env python3
"""Coordinate round-trip test.

Fetches one neuron, writes files, reloads through neurostat,
and verifies coordinate sanity. Run before any batch download.

Usage:
    python tests/test_coordinate_roundtrip.py [--root-id ROOT_ID]
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path.home() / "research" / "neurostat"))


def test_coordinate_roundtrip(root_id=None):
    """Fetch one neuron, write files, reload through neurostat, verify sanity.

    Steps:
    1. Fetch one neuron via fetch.py + format.py
    2. Write SWC (μm), synapse CSV (μm), partner CSV (nm)
    3. Load SWC with NeuronSkeleton.from_swc_file(path, scale_factor=1000.0)
    4. Load synapse CSV, multiply by 1000 → nm
    5. Run snap_points(coords_nm, d_max=50000.0)
    6. Assert: snap rate ≥ 80%
    7. Assert: median snap distance < 10 μm (10000 nm)
    8. Assert: no axis is systematically offset (check centroid alignment)
    """
    from caveclient import CAVEclient
    from neurostat.io.swc import NeuronSkeleton

    from src.fetch import fetch_neuron
    from src.format import make_label, save_neuron_files

    print("=" * 60)
    print("COORDINATE ROUND-TRIP TEST")
    print("=" * 60)

    # Connect
    print("\n1. Connecting to MICrONS...")
    client = CAVEclient("minnie65_public")

    # Use a known golden neuron if no root_id specified
    if root_id is None:
        root_id = 864691135848859998  # exc_23P
    print(f"   Root ID: {root_id}")

    # Fetch
    print("\n2. Fetching neuron...")
    fetch_result = fetch_neuron(client, root_id)
    if fetch_result is None:
        print("   FAIL: Could not fetch neuron")
        return False

    print(f"   Skeleton: {fetch_result['skeleton']['n_vertices']} vertices")
    print(f"   Synapses: {fetch_result['n_synapses']}")
    print(f"   Partners: {fetch_result['n_unique_partners']}")

    # Check radius (Phase 1 go/no-go gate)
    print("\n3. Skeleton radius check...")
    if fetch_result["skeleton"]["has_variable_radius"]:
        radii = fetch_result["skeleton"]["radius"]
        print(f"   PASS: Variable radii (mean={np.mean(radii):.1f}, "
              f"std={np.std(radii):.1f})")
    else:
        print("   WARNING: Uniform or missing radii")
        print("   → Electrotonic length may not be meaningful")
        print("   → Consider branch order as alternative regime axis")

    # Save files
    print("\n4. Saving files...")
    test_dir = PROJECT_DIR / "data" / "test_roundtrip"
    label = make_label("excitatory", "23P", root_id)
    paths = save_neuron_files(label, root_id, fetch_result, test_dir)
    print(f"   SWC: {paths['swc_path']}")
    print(f"   Synapses: {paths['synapse_path']}")
    print(f"   Partners: {paths['partner_path']}")

    # Reload through neurostat
    print("\n5. Reloading through neurostat...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skeleton = NeuronSkeleton.from_swc_file(
            paths["swc_path"], scale_factor=1000.0
        )

    dendrite_skel = skeleton.filter_by_type([1, 3, 4])
    if len(dendrite_skel.branches) < 3:
        dendrite_skel = skeleton

    print(f"   Loaded: {len(dendrite_skel.nodes)} nodes, "
          f"{len(dendrite_skel.branches)} branches")

    # Load synapses and convert to nm
    syn_df = pd.read_csv(paths["synapse_path"])
    syn_coords_nm = syn_df[["x_um", "y_um", "z_um"]].values * 1000.0
    print(f"   Synapses: {len(syn_df)} loaded")

    # Snap
    print("\n6. Snapping synapses...")
    snap = dendrite_skel.snap_points(syn_coords_nm, d_max=50000.0)
    n_valid = snap.valid.sum()
    snap_rate = n_valid / len(syn_df) if len(syn_df) > 0 else 0
    print(f"   Snap rate: {snap_rate:.1%} ({n_valid}/{len(syn_df)})")

    # Median snap distance
    if n_valid > 0:
        median_dist_nm = np.median(snap.distances[snap.valid])
        median_dist_um = median_dist_nm / 1000.0
        print(f"   Median snap distance: {median_dist_um:.2f} μm "
              f"({median_dist_nm:.0f} nm)")
    else:
        median_dist_um = float("inf")
        print("   No valid snaps!")

    # Centroid alignment check
    print("\n7. Centroid alignment check...")
    skel_coords = np.array([[n.x, n.y, n.z] for n in dendrite_skel.nodes.values()])
    skel_centroid = skel_coords.mean(axis=0)
    syn_centroid = syn_coords_nm.mean(axis=0)
    offset = np.abs(syn_centroid - skel_centroid)
    offset_um = offset / 1000.0
    print(f"   Skeleton centroid (nm): {skel_centroid}")
    print(f"   Synapse centroid (nm):  {syn_centroid}")
    print(f"   Offset (μm): x={offset_um[0]:.1f}, y={offset_um[1]:.1f}, z={offset_um[2]:.1f}")

    # No axis should be offset by more than 100 μm
    max_offset_um = offset_um.max()

    # Load partner CSV and verify format
    print("\n8. Partner CSV format check...")
    partner_df = pd.read_csv(paths["partner_path"])
    expected_cols = ["x_nm", "y_nm", "z_nm", "pre_root_id", "pre_cell_type",
                     "pre_vertex_idx", "pre_cell_type_broad"]
    has_all_cols = all(c in partner_df.columns for c in expected_cols)
    print(f"   Columns: {partner_df.columns.tolist()}")
    print(f"   All expected columns present: {has_all_cols}")

    # Verify partner coords are in nm (should be ~1000x synapse μm values)
    if len(partner_df) > 0:
        partner_mean = partner_df[["x_nm", "y_nm", "z_nm"]].mean()
        syn_mean_nm = syn_coords_nm.mean(axis=0)
        coord_ratio = partner_mean.values / syn_mean_nm
        print(f"   Partner/synapse coord ratio: {coord_ratio}")
        coords_consistent = np.allclose(coord_ratio, 1.0, atol=0.1)
        print(f"   Coordinates consistent: {coords_consistent}")

    # Assertions
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    tests = [
        ("Snap rate ≥ 80%", snap_rate >= 0.80),
        ("Median snap distance < 10 μm", median_dist_um < 10.0),
        ("Max centroid offset < 100 μm", max_offset_um < 100.0),
        ("Partner CSV has all columns", has_all_cols),
    ]

    all_passed = True
    for name, passed in tests:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    if all_passed:
        print("\nCoordinate round-trip verified. Safe to proceed with batch downloads.")
    else:
        print("\nWARNING: Coordinate issues detected. Fix before batch downloading.")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-id", type=int, default=None)
    args = parser.parse_args()

    success = test_coordinate_roundtrip(args.root_id)
    sys.exit(0 if success else 1)
