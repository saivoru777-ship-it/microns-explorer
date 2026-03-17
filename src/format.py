"""Convert raw CAVEclient output to pipeline-compatible file formats.

Output formats:
- SWC: 7-column standard, coordinates in μm
- Synapse CSV: x_um, y_um, z_um (micrometers)
- Partner CSV: x_nm, y_nm, z_nm, pre_root_id, pre_cell_type, pre_vertex_idx, pre_cell_type_broad (nanometers)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VOXEL_TO_NM = np.array([4, 4, 40])


def _classify_broad(cell_type_str):
    """Map cell type string to broad category.

    Returns one of: 'excitatory', 'inhibitory', 'other', 'unknown'.

    Handles both full strings ('excitatory_neuron') and short codes ('23P', 'BC')
    from the aibs_metamodel_celltypes_v661 table.
    """
    ct = str(cell_type_str).strip()
    ct_lower = ct.lower()

    if ct_lower in ("", "unknown", "nan", "none"):
        return "unknown"

    # Exact match on known MICrONS short codes first
    excitatory_codes = {"23p", "4p", "5p-it", "5p-et", "5p-np", "6p-it", "6p-ct",
                        "5pit", "5pet", "5pnp", "6pit", "6pct"}
    inhibitory_codes = {"bc", "mc", "bpc", "ngc"}

    if ct_lower in excitatory_codes:
        return "excitatory"
    if ct_lower in inhibitory_codes:
        return "inhibitory"

    # Non-neuronal types
    non_neuronal = {"astrocyte", "microglia", "oligo", "opc", "pericyte"}
    if ct_lower in non_neuronal:
        return "other"

    # Substring patterns for full-form strings (e.g., 'excitatory_neuron')
    inh_patterns = [
        "inhibitory", "interneuron", "gaba", "pvalb", "sst",
        "vip", "lamp5", "basket", "chandelier", "martinotti",
        "bipolar",
    ]
    for p in inh_patterns:
        if p in ct_lower:
            return "inhibitory"

    exc_patterns = ["excitatory", "pyramidal", "spiny", "stellate"]
    for p in exc_patterns:
        if p in ct_lower:
            return "excitatory"

    return "other"


def make_label(cell_type_broad, cell_type_fine, root_id):
    """Generate deterministic label for a neuron.

    Format: {broad_prefix}_{fine}_{root_id}
    e.g., exc_23P_864691135848859998
    """
    prefix_map = {
        "excitatory": "exc",
        "inhibitory": "inh",
        "other": "oth",
        "unknown": "unk",
    }
    prefix = prefix_map.get(cell_type_broad, "unk")
    fine = str(cell_type_fine).replace(" ", "").replace("-", "")
    return f"{prefix}_{fine}_{root_id}"


def format_swc(skeleton_data, root_id):
    """Convert skeleton to SWC string with coordinates in μm.

    The skeleton service returns vertices already in nm.
    We convert nm → μm by dividing by 1000.

    Parameters
    ----------
    skeleton_data : dict
        From fetch.fetch_skeleton(). Contains 'vertices_nm', 'edges',
        'radius', 'has_variable_radius', 'compartment'.
    root_id : int

    Returns
    -------
    str : SWC file content
    """
    vertices_nm = skeleton_data["vertices_nm"]
    radius_data = skeleton_data.get("radius")
    compartment = skeleton_data.get("compartment")

    # nm → μm
    coords_um = vertices_nm / 1000.0

    # Build parent mapping from edge list via BFS
    n_vertices = len(vertices_nm)
    parent_ids = np.full(n_vertices, -1, dtype=int)
    edges = skeleton_data.get("edges")

    if edges is not None and edges.ndim == 2 and edges.shape[1] == 2:
        # Edge list (M×2): undirected edges, build tree via BFS from node 0
        from collections import deque
        adj = {i: [] for i in range(n_vertices)}
        for e in edges:
            adj[int(e[0])].append(int(e[1]))
            adj[int(e[1])].append(int(e[0]))

        visited = np.zeros(n_vertices, dtype=bool)
        parent_ids[0] = -1
        visited[0] = True
        queue = deque([0])
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent_ids[neighbor] = node
                    queue.append(neighbor)

    # Build SWC lines
    lines = [f"# SWC from MICrONS root_id={root_id}"]
    lines.append("# Coordinates in micrometers (um)")

    for i in range(n_vertices):
        node_id = i  # 0-indexed (matches existing MICrONS SWC convention)

        # Node type: use compartment labels if available, else default
        if compartment is not None and i < len(compartment):
            node_type = int(compartment[i])
        else:
            node_type = 1 if i == 0 else 3  # 1=soma/root, 3=dendrite

        x, y, z = coords_um[i]

        # Radius in μm (skeleton radius is in nm)
        if radius_data is not None and skeleton_data.get("has_variable_radius"):
            r_um = radius_data[i] / 1000.0
        else:
            r_um = 0.5  # default 500 nm = 0.5 μm

        # Parent: -1 for root
        pid = int(parent_ids[i])

        lines.append(
            f"{node_id} {node_type} {x:.3f} {y:.3f} {z:.3f} {r_um:.3f} {pid}"
        )

    return "\n".join(lines) + "\n"


def format_synapse_csv(synapse_df):
    """Convert synapse positions to x_um, y_um, z_um.

    Parameters
    ----------
    synapse_df : pd.DataFrame
        From fetch.fetch_synapses(). Has ctr_pt_x/y/z in voxel coords.

    Returns
    -------
    pd.DataFrame with columns: x_um, y_um, z_um
    """
    # Voxel → nm → μm
    x_nm = synapse_df["ctr_pt_x"].values * VOXEL_TO_NM[0]
    y_nm = synapse_df["ctr_pt_y"].values * VOXEL_TO_NM[1]
    z_nm = synapse_df["ctr_pt_z"].values * VOXEL_TO_NM[2]

    return pd.DataFrame({
        "x_um": x_nm / 1000.0,
        "y_um": y_nm / 1000.0,
        "z_um": z_nm / 1000.0,
    })


def format_partner_csv(synapse_df, partner_types):
    """Build partner CSV with nm coords and cell type labels.

    Parameters
    ----------
    synapse_df : pd.DataFrame
        From fetch.fetch_synapses(). Has ctr_pt_x/y/z and pre_pt_root_id.
    partner_types : dict
        Mapping pre_root_id → cell_type string.

    Returns
    -------
    pd.DataFrame with columns:
        x_nm, y_nm, z_nm, pre_root_id, pre_cell_type, pre_vertex_idx, pre_cell_type_broad
    """
    # Synapse positions in nm
    x_nm = synapse_df["ctr_pt_x"].values * VOXEL_TO_NM[0]
    y_nm = synapse_df["ctr_pt_y"].values * VOXEL_TO_NM[1]
    z_nm = synapse_df["ctr_pt_z"].values * VOXEL_TO_NM[2]

    pre_ids = synapse_df["pre_pt_root_id"].values.astype(np.int64)

    # Map partner types
    pre_cell_types = [partner_types.get(int(rid), "unknown") for rid in pre_ids]
    pre_cell_types_broad = [_classify_broad(ct) for ct in pre_cell_types]

    # pre_vertex_idx: not available from CAVEclient (that's an HDF5 concept)
    # Use -1 as placeholder; downstream code that uses this field will need adaptation
    pre_vertex_idx = np.full(len(pre_ids), -1, dtype=int)

    return pd.DataFrame({
        "x_nm": x_nm.astype(np.int64),
        "y_nm": y_nm.astype(np.int64),
        "z_nm": z_nm.astype(np.int64),
        "pre_root_id": pre_ids,
        "pre_cell_type": pre_cell_types,
        "pre_vertex_idx": pre_vertex_idx,
        "pre_cell_type_broad": pre_cell_types_broad,
    })


def save_neuron_files(label, root_id, fetch_result, output_dir):
    """Save all files for one neuron in pipeline-compatible format.

    Parameters
    ----------
    label : str
        Neuron label (e.g., "exc_23P_864691135848859998").
    root_id : int
    fetch_result : dict
        From fetch.fetch_neuron().
    output_dir : str or Path
        Directory to save files.

    Returns
    -------
    dict with file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # SWC
    swc_content = format_swc(fetch_result["skeleton"], root_id)
    swc_path = output_dir / f"{label}.swc"
    with open(swc_path, "w") as f:
        f.write(swc_content)

    # Synapse CSV
    syn_df = format_synapse_csv(fetch_result["synapse_df"])
    syn_path = output_dir / f"{label}_synapses.csv"
    syn_df.to_csv(syn_path, index=False)

    # Partner CSV
    partner_df = format_partner_csv(
        fetch_result["synapse_df"],
        fetch_result["partner_types"],
    )
    partner_path = output_dir / f"{label}_presynaptic.csv"
    partner_df.to_csv(partner_path, index=False)

    logger.info(
        f"Saved {label}: SWC ({fetch_result['skeleton']['n_vertices']} nodes), "
        f"{len(syn_df)} synapses, {len(partner_df)} partner entries"
    )

    return {
        "swc_path": str(swc_path),
        "synapse_path": str(syn_path),
        "partner_path": str(partner_path),
    }
