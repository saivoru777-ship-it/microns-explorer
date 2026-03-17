#!/usr/bin/env python3
"""Run QC and morphology filter on downloaded neurons.

Usage:
    python scripts/03_validate.py [--data-dir PATH]
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
from src.format import make_label
from src.validate import validate_batch
from src.morphology_filter import filter_analysis_ready


def main():
    parser = argparse.ArgumentParser(description="Validate downloaded neurons")
    parser.add_argument("--data-dir", type=str,
                        default=str(PROJECT_DIR / "data" / "neurons"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist")
        sys.exit(1)

    # Find all neurons by looking for .swc files
    swc_files = sorted(data_dir.glob("*.swc"))
    if not swc_files:
        print("No SWC files found")
        sys.exit(1)

    # Extract (label, root_id) from filenames
    # Label format: {prefix}_{fine}_{root_id}.swc
    labels_and_ids = []
    for swc in swc_files:
        stem = swc.stem  # e.g., "exc_23P_864691135848859998"
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            label = stem
            root_id = parts[1]
            labels_and_ids.append((label, root_id))
        else:
            print(f"  Skipping {swc.name}: can't parse label/root_id")

    print(f"Found {len(labels_and_ids)} neurons to validate")

    # Run QC
    print("\n=== Running QC ===")
    qc_df = validate_batch(labels_and_ids, data_dir)

    n_passed = qc_df["passed"].sum()
    print(f"\nQC Results: {n_passed}/{len(qc_df)} passed")

    # Print failures
    failed = qc_df[~qc_df["passed"]]
    if len(failed) > 0:
        print("\nFailed neurons:")
        for _, row in failed.iterrows():
            print(f"  {row['label']}: {row['failure_reasons']}")

    # Save QC results
    qc_path = data_dir.parent / "qc_results.csv"
    qc_df.to_csv(qc_path, index=False)
    print(f"\nQC results saved to {qc_path}")

    # Run morphology filter
    print("\n=== Running Morphology Filter ===")
    compat_df = filter_analysis_ready(qc_df, data_dir)

    if len(compat_df) > 0:
        n_ready = compat_df["analysis_ready"].sum()
        print(f"\nAnalysis-ready: {n_ready}/{len(compat_df)}")

        # Save compatibility report
        compat_path = data_dir.parent / "analysis_compatibility.csv"
        compat_df.to_csv(compat_path, index=False)
        print(f"Compatibility report saved to {compat_path}")

        # Summary by exclusion criterion
        print("\nExclusion breakdown:")
        for col in ["passes_branch_count", "passes_total_length",
                     "passes_snapped_synapses", "passes_stub_fraction",
                     "passes_extent"]:
            n_fail = (~compat_df[col]).sum()
            print(f"  {col}: {n_fail} excluded")

        # Ready neurons summary
        ready = compat_df[compat_df["analysis_ready"]]
        if len(ready) > 0:
            print(f"\nReady neurons:")
            for _, row in ready.iterrows():
                print(
                    f"  {row['label']}: {row['n_branches']} branches, "
                    f"{row['n_snapped_synapses']} synapses, "
                    f"{row['n_branches_with_3plus_synapses']} usable branches"
                )
    else:
        print("No neurons passed morphology filter")


if __name__ == "__main__":
    main()
