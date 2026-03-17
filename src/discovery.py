"""Discover and catalog neurons from MICrONS via CAVEclient.

Queries cell type tables and proofreading status to build a complete
catalog of available neurons with their classification.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


# Cell type mapping: raw MICrONS strings → fine labels
# Based on aibs_metamodel_celltypes_v661 type strings
CELL_TYPE_FINE_MAP = {
    # Excitatory
    "23P": "23P",
    "4P": "4P",
    "5P-IT": "5PIT",
    "5P-ET": "5PET",
    "5P-NP": "5PNP",
    "6P-IT": "6PIT",
    "6P-CT": "6PCT",
    # Inhibitory
    "BC": "BC",
    "MC": "MC",
    "BPC": "BPC",
    "NGC": "NGC",
    "Lamp5": "Lamp5",
    "Sst": "Sst",
    "Pvalb": "Pvalb",
    "Vip": "Vip",
    "Sncg": "Sncg",
}


def _map_cell_type_fine(raw_type):
    """Map raw cell type string to fine label.

    Returns (fine_label, mapping_method) tuple.
    """
    if pd.isna(raw_type) or str(raw_type).strip() == "":
        return "unknown", "missing"

    raw = str(raw_type).strip()

    # Exact match first
    if raw in CELL_TYPE_FINE_MAP:
        return CELL_TYPE_FINE_MAP[raw], "exact"

    # Case-insensitive exact match
    raw_lower = raw.lower()
    for key, val in CELL_TYPE_FINE_MAP.items():
        if key.lower() == raw_lower:
            return val, "exact"

    # Heuristic substring matching
    for key, val in CELL_TYPE_FINE_MAP.items():
        if key.lower() in raw_lower:
            return val, "heuristic"

    return raw, "heuristic"


def _classify_broad(cell_type_str):
    """Map fine cell type string to broad category.

    Returns one of: 'excitatory', 'inhibitory', 'other', 'unknown'.
    Mirrors classify_cell_type_broad() from hdf5_extraction.py.
    """
    ct = str(cell_type_str).lower().strip()

    if ct in ("", "unknown", "nan", "none"):
        return "unknown"

    inh_patterns = [
        "inhibitory", "interneuron", "gaba", "pvalb", "sst",
        "vip", "lamp5", "basket", "chandelier", "martinotti",
        "bipolar", "bc", "mc", "bpc", "ngc", "sncg",
    ]
    for p in inh_patterns:
        if p in ct:
            return "inhibitory"

    exc_patterns = [
        "excitatory", "pyramidal", "spiny", "stellate",
        "23p", "4p", "5p", "6p",
    ]
    for p in exc_patterns:
        if p in ct:
            return "excitatory"

    return "other"


def build_catalog(client):
    """Query cell types + proofreading status → catalog DataFrame.

    Parameters
    ----------
    client : CAVEclient
        Initialized CAVEclient for minnie65_public.

    Returns
    -------
    pd.DataFrame with columns:
        root_id, cell_type_original, cell_type_fine, cell_type_broad,
        mapping_method, is_proofread, region
    """
    # 1. Cell type query
    print("Querying cell type table...")
    ct_df = client.materialize.query_table("aibs_metamodel_celltypes_v661")
    print(f"  {len(ct_df)} neurons with cell type annotations")

    # 2. Proofreading status query
    print("Querying proofreading status...")
    proof_df = client.materialize.query_table("proofreading_status_and_strategy")
    proofread_ids = set(proof_df["pt_root_id"].values)
    print(f"  {len(proofread_ids)} proofread neurons")

    # 3. Build catalog
    print("Building catalog...")

    # Extract root_id column (may be pt_root_id or target_id depending on table)
    if "pt_root_id" in ct_df.columns:
        root_id_col = "pt_root_id"
    elif "target_id" in ct_df.columns:
        root_id_col = "target_id"
    else:
        raise ValueError(
            f"Cannot find root_id column. Available: {ct_df.columns.tolist()}"
        )

    # Identify the cell type column
    ct_col = None
    for candidate in ["cell_type", "cell_type_coarse", "classification_system"]:
        if candidate in ct_df.columns:
            ct_col = candidate
            break
    if ct_col is None:
        # Use the first string-like column that isn't an ID
        for col in ct_df.columns:
            if col != root_id_col and ct_df[col].dtype == object:
                ct_col = col
                break
    if ct_col is None:
        raise ValueError(
            f"Cannot find cell type column. Available: {ct_df.columns.tolist()}"
        )

    # Map cell types
    fine_labels = []
    broad_labels = []
    methods = []
    for raw_type in ct_df[ct_col]:
        fine, method = _map_cell_type_fine(raw_type)
        fine_labels.append(fine)
        methods.append(method)
        broad_labels.append(_classify_broad(str(raw_type)))

    catalog = pd.DataFrame({
        "root_id": ct_df[root_id_col].values,
        "cell_type_original": ct_df[ct_col].values,
        "cell_type_fine": fine_labels,
        "cell_type_broad": broad_labels,
        "mapping_method": methods,
        "is_proofread": ct_df[root_id_col].isin(proofread_ids).values,
    })

    # Add region if available
    if "region" in ct_df.columns:
        catalog["region"] = ct_df["region"].values
    else:
        catalog["region"] = "unknown"

    # Deduplicate by root_id (keep first)
    catalog = catalog.drop_duplicates(subset="root_id", keep="first")
    catalog = catalog.reset_index(drop=True)

    return catalog


def save_version_metadata(client, output_dir):
    """Save version pinning metadata for reproducibility.

    Parameters
    ----------
    client : CAVEclient
    output_dir : str or Path
        Directory to save metadata JSON.

    Returns
    -------
    dict : metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        mat_version = client.materialize.version
    except Exception:
        mat_version = "unknown"

    try:
        import caveclient
        cave_version = caveclient.__version__
    except Exception:
        cave_version = "unknown"

    metadata = {
        "query_date": str(date.today()),
        "materialization_version": mat_version,
        "cell_type_table": "aibs_metamodel_celltypes_v661",
        "synapse_table": "synapses_pni_2",
        "proofreading_table": "proofreading_status_and_strategy",
        "voxel_resolution_nm": [4, 4, 40],
        "caveclient_version": cave_version,
        "datastack": "minnie65_public",
    }

    metadata_path = output_dir / f"batch_{date.today().isoformat()}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Metadata saved to {metadata_path}")
    return metadata


def print_catalog_summary(catalog):
    """Print summary table of catalog contents."""
    print("\n=== Catalog Summary ===")
    print(f"Total neurons: {len(catalog)}")
    print(f"Proofread: {catalog['is_proofread'].sum()}")
    print()

    # Cross-tabulation: fine type × proofread
    ct = pd.crosstab(
        catalog["cell_type_fine"],
        catalog["is_proofread"],
        margins=True,
    )
    ct.columns = [
        c if c == "All" else ("Proofread" if c else "Not proofread")
        for c in ct.columns
    ]
    print(ct.to_string())
    print()

    # Broad type summary
    broad = catalog.groupby("cell_type_broad").agg(
        total=("root_id", "count"),
        proofread=("is_proofread", "sum"),
    )
    print("Broad type summary:")
    print(broad.to_string())
