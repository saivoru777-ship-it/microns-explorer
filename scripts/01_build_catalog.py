#!/usr/bin/env python3
"""Build neuron catalog from MICrONS via CAVEclient.

Queries cell type and proofreading tables, maps cell types,
saves catalog CSV and version metadata.

Usage:
    python scripts/01_build_catalog.py
"""

import logging
import sys
from pathlib import Path

# Project root
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.discovery import build_catalog, save_version_metadata, print_catalog_summary


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from caveclient import CAVEclient

    print("Connecting to MICrONS (minnie65_public)...")
    client = CAVEclient("minnie65_public")
    print(f"  Materialization version: {client.materialize.version}")

    # Build catalog
    catalog = build_catalog(client)

    # Save catalog
    catalog_path = PROJECT_DIR / "catalog.csv"
    catalog.to_csv(catalog_path, index=False)
    print(f"\nCatalog saved to {catalog_path}")

    # Save version metadata
    metadata = save_version_metadata(client, PROJECT_DIR / "metadata")

    # Print summary
    print_catalog_summary(catalog)

    # Quick stats for the plan
    n_proofread = catalog["is_proofread"].sum()
    print(f"\n=== Key Numbers ===")
    print(f"Total neurons: {len(catalog)}")
    print(f"Proofread neurons: {n_proofread}")
    print(f"Proofread excitatory: {len(catalog[(catalog['is_proofread']) & (catalog['cell_type_broad'] == 'excitatory')])}")
    print(f"Proofread inhibitory: {len(catalog[(catalog['is_proofread']) & (catalog['cell_type_broad'] == 'inhibitory')])}")


if __name__ == "__main__":
    main()
