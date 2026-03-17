#!/usr/bin/env python3
"""Fetch neurons from catalog with filtering options.

Downloads skeleton + synapses + partner types for neurons matching
the specified criteria. Supports resume after interruption.

Usage:
    python scripts/02_fetch_neurons.py [options]

Options:
    --cell-type TYPE     Filter by fine cell type (e.g., BC, MC, 23P)
    --broad-type TYPE    Filter by broad type (excitatory, inhibitory)
    --proofread-only     Only proofread neurons (default: True)
    --max N              Maximum neurons to download
    --root-ids ID,...    Specific root IDs to fetch (comma-separated)
    --no-resume          Don't resume from previous progress
    --no-validate        Skip per-neuron QC
    --radius-check       After first neuron, report skeleton radius status and pause
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
from src.batch import fetch_batch


def main():
    parser = argparse.ArgumentParser(description="Fetch MICrONS neurons")
    parser.add_argument("--cell-type", type=str, help="Fine cell type filter")
    parser.add_argument("--broad-type", type=str, help="Broad type filter")
    parser.add_argument("--proofread-only", action="store_true", default=True)
    parser.add_argument("--no-proofread-filter", action="store_true",
                        help="Include non-proofread neurons")
    parser.add_argument("--max", type=int, default=None, help="Max neurons")
    parser.add_argument("--root-ids", type=str, help="Comma-separated root IDs")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--radius-check", action="store_true",
                        help="Check radius on first neuron and pause")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from caveclient import CAVEclient

    # Load catalog
    catalog_path = PROJECT_DIR / "catalog.csv"
    if not catalog_path.exists():
        print("ERROR: catalog.csv not found. Run 01_build_catalog.py first.")
        sys.exit(1)

    catalog = pd.read_csv(catalog_path)
    print(f"Loaded catalog: {len(catalog)} neurons")

    # Build filters
    filters = {}
    if not args.no_proofread_filter:
        filters["proofread_only"] = True

    if args.cell_type:
        filters["cell_type_fine"] = args.cell_type

    if args.broad_type:
        filters["cell_type_broad"] = args.broad_type

    if args.root_ids:
        ids = [int(x.strip()) for x in args.root_ids.split(",")]
        filters["root_ids"] = ids

    # Connect
    print("Connecting to MICrONS...")
    client = CAVEclient("minnie65_public")

    # Radius check mode: fetch one neuron first
    if args.radius_check:
        from src.fetch import fetch_skeleton
        # Pick first matching neuron from catalog
        test_df = catalog.copy()
        if "proofread_only" in filters and filters["proofread_only"]:
            test_df = test_df[test_df["is_proofread"]]
        if len(test_df) == 0:
            print("No neurons match filters for radius check")
            sys.exit(1)

        test_rid = int(test_df.iloc[0]["root_id"])
        print(f"\n=== RADIUS CHECK on root_id={test_rid} ===")
        sk = fetch_skeleton(client, test_rid)
        if sk is None:
            print("Skeleton fetch failed for radius check")
            sys.exit(1)

        if sk["has_variable_radius"]:
            print("PASS: Skeleton has variable radii")
            print(f"  Mean radius: {sk['radius'].mean():.1f}")
            print(f"  Std radius: {sk['radius'].std():.1f}")
            print("  → Electrotonic length is viable as regime axis")
        else:
            print("WARNING: Skeleton radii are UNIFORM or MISSING")
            if sk["radius"] is not None:
                print(f"  All radii: {sk['radius'][:5]}...")
            print("  → Electrotonic length may be meaningless")
            print("  → Consider: branch order, mesh-based radius estimation")
            print("\nPausing. Review and decide before proceeding.")
            response = input("Continue anyway? [y/N]: ")
            if response.lower() != "y":
                sys.exit(0)

    # Fetch
    output_dir = PROJECT_DIR / "data" / "neurons"
    results = fetch_batch(
        client=client,
        catalog_df=catalog,
        output_dir=output_dir,
        filters=filters,
        max_neurons=args.max,
        resume=not args.no_resume,
        validate=not args.no_validate,
    )

    # Summary
    print(f"\n=== Results ===")
    print(f"Completed: {len(results['completed'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Skipped: {len(results['skipped'])}")

    if results["qc_results"]:
        qc_df = pd.DataFrame(results["qc_results"])
        n_passed = qc_df["passed"].sum()
        print(f"QC passed: {n_passed}/{len(qc_df)}")

        # Save QC results
        qc_path = PROJECT_DIR / "data" / "qc_results.csv"
        qc_df.to_csv(qc_path, index=False)
        print(f"QC results saved to {qc_path}")


if __name__ == "__main__":
    main()
