#!/usr/bin/env python3
"""Apply combined CAVE + HDF5 partner cell type classification.

Uses two independent sources:
1. CAVE cell type table (91K neurons, fine types like 23P, BC)
2. HDF5 connectome vertex table (72K neurons, exc/inh only) via coordinate matching

Prefers CAVE when available (has fine types), falls back to HDF5.
Adds 'type_source' column: 'cave', 'hdf5', or 'none'.

Usage:
    python scripts/05_combined_partner_types.py [--data-dir PATH]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path.home() / "research" / "neurostat-input-clustering" / "src"))

from src.format import _classify_broad

HDF5_PATH = Path.home() / "research" / "neurostat-input-clustering" / "data" / "microns" / "microns_mm3_connectome_v1181.h5"
MATCH_THRESHOLD_NM = 1000  # 1 μm


def load_hdf5_synapse_index(hdf5_path):
    """Load HDF5 synapse coordinates and pre-vertex types, build KDTree."""
    import h5py

    print("Loading HDF5 connectome...")
    t0 = time.time()
    with h5py.File(str(hdf5_path), "r") as f:
        coords = f["connectivity"]["full"]["edges"]["block0_values"][:, 2:5]
        pre_vtx = f["connectivity"]["full"]["edge_indices"]["block0_values"][:, 0]
        vp = f["connectivity"]["full"]["vertex_properties"]["table"]
        types_raw = vp["values_block_2"][:, 1]
        vtx_types = np.array([
            ct.decode("utf-8") if isinstance(ct, bytes) else str(ct)
            for ct in types_raw
        ])

    print(f"  {len(coords):,} synapses, {len(vtx_types):,} vertices ({time.time()-t0:.1f}s)")

    print("Building KDTree...")
    t0 = time.time()
    tree = cKDTree(coords)
    print(f"  Done in {time.time()-t0:.1f}s")

    return tree, coords, pre_vtx, vtx_types


def classify_partners_combined(partner_df, cave_map, tree, pre_vtx, vtx_types):
    """Classify partners using combined CAVE + HDF5 approach.

    Returns updated DataFrame with pre_cell_type, pre_cell_type_broad, type_source.
    """
    n = len(partner_df)

    # Source 1: CAVE table
    cave_types = partner_df["pre_root_id"].map(
        lambda rid: cave_map.get(int(rid), None)
    )
    cave_broad = cave_types.apply(
        lambda x: _classify_broad(x) if x is not None else None
    )

    # Source 2: HDF5 coordinate match
    query_coords = partner_df[["x_nm", "y_nm", "z_nm"]].values.astype(np.float64)
    distances, indices = tree.query(query_coords)
    matched_vtx = pre_vtx[indices]
    matched_raw = vtx_types[matched_vtx]
    good = distances < MATCH_THRESHOLD_NM

    hdf5_broad = np.full(n, "unknown", dtype=object)
    for i in range(n):
        if good[i]:
            if matched_raw[i] == "excitatory_neuron":
                hdf5_broad[i] = "excitatory"
            elif matched_raw[i] == "inhibitory_neuron":
                hdf5_broad[i] = "inhibitory"

    # Combine: prefer CAVE (fine type), fall back to HDF5
    combined_fine = partner_df["pre_cell_type"].values.copy()
    combined_broad = np.full(n, "unknown", dtype=object)
    source = np.full(n, "none", dtype=object)

    for i in range(n):
        if cave_broad.iloc[i] in ("excitatory", "inhibitory"):
            combined_broad[i] = cave_broad.iloc[i]
            source[i] = "cave"
        elif hdf5_broad[i] in ("excitatory", "inhibitory"):
            combined_broad[i] = hdf5_broad[i]
            combined_fine[i] = matched_raw[i] if good[i] else "unknown"
            source[i] = "hdf5"

    partner_df = partner_df.copy()
    partner_df["pre_cell_type"] = combined_fine
    partner_df["pre_cell_type_broad"] = combined_broad
    partner_df["type_source"] = source
    return partner_df


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default=str(PROJECT_DIR / "data" / "neurons"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not HDF5_PATH.exists():
        print(f"ERROR: HDF5 not found at {HDF5_PATH}")
        sys.exit(1)

    # Load resources
    tree, coords, pre_vtx, vtx_types = load_hdf5_synapse_index(HDF5_PATH)

    cave_cat = pd.read_csv(PROJECT_DIR / "catalog.csv")
    cave_map = dict(zip(
        cave_cat["root_id"].astype(np.int64),
        cave_cat["cell_type_original"],
    ))
    print(f"CAVE table: {len(cave_map):,} entries")

    # Process partner CSVs — skip those already classified
    partner_files = sorted(data_dir.glob("*_presynaptic.csv"))
    needs_classification = []
    already_done = []
    for pf in partner_files:
        with open(pf) as f:
            header = f.readline().strip()
        if "type_source" in header:
            already_done.append(pf)
        else:
            needs_classification.append(pf)

    print(f"\nTotal partner CSVs: {len(partner_files)}")
    print(f"Already classified: {len(already_done)}")
    print(f"Need classification: {len(needs_classification)}")

    if not needs_classification:
        print("Nothing new to classify.")
        # Still rebuild stats from all files
        needs_classification = []

    stats = []
    # Process new files (skip corrupted ones being written concurrently)
    skipped = []
    for i, pf in enumerate(needs_classification):
        try:
            df = pd.read_csv(pf)
        except Exception as e:
            skipped.append((pf.name, str(e)[:80]))
            continue
        df = classify_partners_combined(df, cave_map, tree, pre_vtx, vtx_types)
        df.to_csv(pf, index=False)

        n_cave = (df["type_source"] == "cave").sum()
        n_hdf5 = (df["type_source"] == "hdf5").sum()
        n_typed = n_cave + n_hdf5
        n_exc = (df["pre_cell_type_broad"] == "excitatory").sum()
        n_inh = (df["pre_cell_type_broad"] == "inhibitory").sum()

        stats.append({
            "label": pf.stem.replace("_presynaptic", ""),
            "n_synapses": len(df),
            "n_cave": n_cave,
            "n_hdf5": n_hdf5,
            "n_typed": n_typed,
            "coverage": n_typed / len(df) if len(df) > 0 else 0,
            "n_exc": n_exc,
            "n_inh": n_inh,
            "exc_frac": n_exc / (n_exc + n_inh) if (n_exc + n_inh) > 0 else float("nan"),
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(needs_classification)}...")

    if skipped:
        print(f"\nSkipped {len(skipped)} corrupted files (likely still being written):")
        for name, err in skipped[:5]:
            print(f"  {name}: {err}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped)-5} more")

    # Collect stats from already-classified files too
    for pf in already_done:
        try:
            df = pd.read_csv(pf)
        except Exception:
            continue
        n_cave = (df["type_source"] == "cave").sum()
        n_hdf5 = (df["type_source"] == "hdf5").sum()
        n_typed = n_cave + n_hdf5
        n_exc = (df["pre_cell_type_broad"] == "excitatory").sum()
        n_inh = (df["pre_cell_type_broad"] == "inhibitory").sum()

        stats.append({
            "label": pf.stem.replace("_presynaptic", ""),
            "n_synapses": len(df),
            "n_cave": n_cave,
            "n_hdf5": n_hdf5,
            "n_typed": n_typed,
            "coverage": n_typed / len(df) if len(df) > 0 else 0,
            "n_exc": n_exc,
            "n_inh": n_inh,
            "exc_frac": n_exc / (n_exc + n_inh) if (n_exc + n_inh) > 0 else float("nan"),
        })

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(data_dir.parent / "partner_coverage_summary.csv", index=False)

    print(f"\n{'='*60}")
    print("COMBINED COVERAGE SUMMARY")
    print(f"{'='*60}")
    print(f"Neurons: {len(stats_df)}")
    print(f"Mean coverage: {stats_df['coverage'].mean():.1%}")
    print(f"Mean typed synapses/neuron: {stats_df['n_typed'].mean():.0f}")
    print(f"  from CAVE: {stats_df['n_cave'].mean():.0f}")
    print(f"  from HDF5: {stats_df['n_hdf5'].mean():.0f}")
    print(f"Mean exc fraction: {stats_df['exc_frac'].mean():.1%}")
    print(f"\nSaved to {data_dir.parent / 'partner_coverage_summary.csv'}")


if __name__ == "__main__":
    main()
