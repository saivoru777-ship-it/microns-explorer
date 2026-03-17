"""Fetch skeleton, synapses, and partner types for a single neuron.

Downloads raw data from MICrONS via CAVEclient. Returns raw DataFrames
that need to be converted by format.py before use.
"""

import logging
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VOXEL_TO_NM = np.array([4, 4, 40])
SYNAPSE_TABLE = "synapses_pni_2"
CELL_TYPE_TABLE = "aibs_metamodel_celltypes_v661"
PARTNER_BATCH_SIZE = 50

# Module-level cache for the full cell type table (download once, reuse)
_cell_type_cache = None


def _retry(func, max_attempts=3, base_delay=2.0):
    """Retry with exponential backoff on network/server errors."""
    import requests
    for attempt in range(max_attempts):
        try:
            return func()
        except (ConnectionError, TimeoutError, OSError, requests.HTTPError) as e:
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)


def load_cell_type_table(client):
    """Download the full cell type table once and cache it.

    Returns dict mapping root_id → cell_type string.
    """
    global _cell_type_cache
    if _cell_type_cache is not None:
        return _cell_type_cache

    logger.info("Downloading full cell type table (one-time)...")
    ct_df = _retry(
        lambda: client.materialize.query_table(CELL_TYPE_TABLE)
    )
    _cell_type_cache = dict(zip(
        ct_df["pt_root_id"].astype(np.int64),
        ct_df["cell_type"].astype(str),
    ))
    logger.info(f"Cached {len(_cell_type_cache)} cell type entries")
    return _cell_type_cache


def fetch_skeleton(client, root_id):
    """Download skeleton for a neuron.

    The CAVEclient skeleton service (v8.0.1) returns a dict with:
    - 'vertices': N×3 array in nm
    - 'edges': M×2 array (node index pairs)
    - 'radius': N array in nm (optional, may be absent or uniform)
    - 'compartment': N array (optional, SWC type codes)

    Parameters
    ----------
    client : CAVEclient
    root_id : int

    Returns
    -------
    dict with 'vertices_nm', 'edges', 'radius', etc., or None on failure.
    """
    try:
        sk = _retry(
            lambda: client.skeleton.get_skeleton(root_id, output_format="dict")
        )
    except Exception as e:
        error_str = str(e).lower()
        if "not found" in error_str or "root" in error_str or "invalid" in error_str:
            logger.warning(f"Skeleton not found for {root_id} (stale root)")
            return None
        logger.error(f"Skeleton fetch failed for {root_id}: {e}")
        return None

    # sk is a dict with numpy arrays
    vertices = np.array(sk["vertices"])  # already in nm
    edges = np.array(sk["edges"]) if "edges" in sk else None

    if vertices.shape[0] == 0:
        logger.warning(f"Empty skeleton for {root_id}")
        return None

    # Check for radius data
    radius = None
    has_radius = False
    if "radius" in sk and sk["radius"] is not None:
        radius = np.array(sk["radius"])
        if len(radius) == len(vertices) and np.std(radius) > 0:
            logger.info(f"Skeleton has variable radii (mean={np.mean(radius):.1f} nm)")
            has_radius = True
        else:
            logger.info(
                f"Skeleton radii uniform or missing "
                f"(std={np.std(radius) if radius is not None else 0:.1f})"
            )

    # Compartment labels (SWC type codes) if available
    compartment = None
    if "compartment" in sk and sk["compartment"] is not None:
        compartment = np.array(sk["compartment"])

    return {
        "vertices_nm": vertices,
        "edges": edges,
        "radius": radius,
        "has_variable_radius": has_radius,
        "compartment": compartment,
        "n_vertices": len(vertices),
    }


def fetch_synapses(client, root_id):
    """Download post-synaptic inputs for a neuron.

    Uses split_positions=True for efficiency. Requests coordinates in
    the dataset's native voxel resolution [4,4,40] nm/voxel so we
    can apply the standard voxel→nm conversion.

    Parameters
    ----------
    client : CAVEclient
    root_id : int

    Returns
    -------
    pd.DataFrame with columns: ctr_pt_x/y/z (voxel coords),
    pre_pt_root_id, id. Or None on failure.
    """
    try:
        syn_df = _retry(
            lambda: client.materialize.synapse_query(
                post_ids=root_id,
                split_positions=True,
            )
        )
    except Exception as e:
        logger.error(f"Synapse query failed for {root_id}: {e}")
        return None

    if syn_df is None or len(syn_df) == 0:
        logger.warning(f"No synapses found for {root_id}")
        return None

    syn_df = syn_df.copy()

    # With split_positions=True, columns are like ctr_pt_position_x/y/z
    # Map to our standard names
    col_mappings = [
        (["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"], True),
    ]

    found = False
    for cols, _ in col_mappings:
        if all(c in syn_df.columns for c in cols):
            syn_df["ctr_pt_x"] = syn_df[cols[0]]
            syn_df["ctr_pt_y"] = syn_df[cols[1]]
            syn_df["ctr_pt_z"] = syn_df[cols[2]]
            found = True
            break

    if not found:
        # Fallback: try unsplit position column
        if "ctr_pt_position" in syn_df.columns:
            positions = np.array(syn_df["ctr_pt_position"].tolist())
            syn_df["ctr_pt_x"] = positions[:, 0]
            syn_df["ctr_pt_y"] = positions[:, 1]
            syn_df["ctr_pt_z"] = positions[:, 2]
        else:
            logger.error(
                f"Unexpected synapse column format: {syn_df.columns.tolist()}"
            )
            return None

    logger.info(f"Fetched {len(syn_df)} synapses for {root_id}")
    return syn_df


def fetch_partner_types(client, pre_root_ids):
    """Look up cell types for presynaptic partners using cached full table.

    Parameters
    ----------
    client : CAVEclient
    pre_root_ids : array-like of int
        Presynaptic root IDs (not necessarily unique).

    Returns
    -------
    dict mapping pre_root_id → cell_type string.
    Missing partners get "unknown".
    """
    # Use the cached full cell type table (downloads once)
    ct_map = load_cell_type_table(client)

    unique_ids = np.unique(pre_root_ids)
    type_map = {}
    for rid in unique_ids:
        rid_int = int(rid)
        type_map[rid_int] = ct_map.get(rid_int, "unknown")

    n_typed = sum(1 for v in type_map.values() if v != "unknown")
    coverage = n_typed / len(unique_ids) if len(unique_ids) > 0 else 0
    logger.info(
        f"Partner types: {n_typed}/{len(unique_ids)} typed "
        f"({coverage:.0%} coverage)"
    )

    return type_map


def fetch_neuron(client, root_id):
    """Download skeleton, synapses, and partner info for one neuron.

    Parameters
    ----------
    client : CAVEclient
    root_id : int

    Returns
    -------
    dict with 'skeleton', 'synapse_df', 'partner_types', 'partner_coverage',
    or None if critical data is unavailable.
    """
    logger.info(f"Fetching neuron {root_id}...")

    # 1. Skeleton
    skeleton = fetch_skeleton(client, root_id)
    if skeleton is None:
        return None

    # 2. Synapses
    synapse_df = fetch_synapses(client, root_id)
    if synapse_df is None:
        return None

    # 3. Partner types
    pre_ids = synapse_df["pre_pt_root_id"].values.astype(np.int64)
    partner_types = fetch_partner_types(client, pre_ids)

    # Compute coverage
    unique_pre = np.unique(pre_ids)
    n_typed = sum(1 for rid in unique_pre if partner_types.get(int(rid), "unknown") != "unknown")
    partner_coverage = n_typed / len(unique_pre) if len(unique_pre) > 0 else 0

    return {
        "skeleton": skeleton,
        "synapse_df": synapse_df,
        "partner_types": partner_types,
        "partner_coverage": partner_coverage,
        "root_id": root_id,
        "n_synapses": len(synapse_df),
        "n_unique_partners": len(unique_pre),
    }
