#!/usr/bin/env python3
"""Inhibitory residual decomposition by afferent cell type.

Decomposes the ~86% unexplained inhibitory coupling residual into
targeting (do different presynaptic classes prefer different regimes?)
and placement (do they impose different spacing rules?) components.

Pre-registered hypotheses:
  A: BC (PV-like) enriched on low-L/λ; MC (SST-like) on high-L/λ
  B: Residual concentrates in specific postsynaptic subtypes
  C: Inh→inh connections drive disproportionate residual share

Stages:
  0a: CAVE coverage boost (query additional cell type tables)
  0b: Two-level coverage gate (overall + per-presynaptic-class)
  1:  Targeting analysis (log-enrichment, shuffle, within-neuron, mixed model)
  2a: Composition mediation (four-model comparison: M0/M1/M2/M3)
  2b: Class-specific placement rules (conditional on 2a)
  +   Morphology controls

Output (results/replication_full/):
  afferent_coverage_gate.json   — coverage stats, go/no-go by subtype
  afferent_targeting.json       — log-enrichment, shuffle p-values, within-neuron
  afferent_mediation.json       — four-model table, mediation %, by subtype × metric
  afferent_class_placement.json — class-specific regime effects (conditional on 2a)
  afferent_branch_features.csv  — branch features + per-branch afferent fractions
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

PROJECT_DIR = Path(__file__).resolve().parent.parent
MICRONS_DATA = PROJECT_DIR / "data" / "neurons"
RESULTS_DIR = PROJECT_DIR / "results" / "replication_full"

sys.path.insert(0, str(Path.home() / "research" / "neurostat"))
sys.path.insert(0, str(Path.home() / "research" / "neurostat-input-clustering" / "src"))
sys.path.insert(0, str(Path.home() / "research" / "dendritic-regime-coupling"))

from neurostat.io.swc import NeuronSkeleton
from src.branch_morphometry import RM, RI

import statsmodels.formula.api as smf

# Annotation levels
LEVEL_EXACT_SUBTYPE = 1   # pre_cell_type from partner CSV (direct CAVE)
LEVEL_BOOSTED = 2         # backfilled from CAVE table or catalog
LEVEL_BROAD_ONLY = 3      # pre_cell_type_broad only (exc/inh, no subtype)
LEVEL_UNKNOWN = 4         # no annotation

# Inhibitory subtypes for fine-grained analysis
INH_SUBTYPES = ["BC", "MC", "BPC", "NGC"]
# All recognized subtypes (for annotation_level assignment)
KNOWN_SUBTYPES = ["BC", "MC", "BPC", "NGC", "23P", "4P", "5PIT", "5PET",
                  "5PNP", "6PIT", "6PCT"]

N_SHUFFLE = 1000
RNG = np.random.default_rng(42)


# ═══════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════

def _classify_subtype(cell_type_str):
    """Map a pre_cell_type string to a standardized subtype label.

    Returns (subtype, broad_class) or (None, broad_class) if no subtype.
    """
    if not isinstance(cell_type_str, str) or cell_type_str in ("unknown", ""):
        return None, "unknown"

    ct = cell_type_str.strip()

    # Direct subtype matches
    subtype_map = {
        "BC": ("BC", "inhibitory"),
        "MC": ("MC", "inhibitory"),
        "BPC": ("BPC", "inhibitory"),
        "NGC": ("NGC", "inhibitory"),
        "23P": ("23P", "excitatory"),
        "4P": ("4P", "excitatory"),
        "5P-IT": ("5PIT", "excitatory"),
        "5P-ET": ("5PET", "excitatory"),
        "5P-NP": ("5PNP", "excitatory"),
        "6P-IT": ("6PIT", "excitatory"),
        "6P-CT": ("6PCT", "excitatory"),
        "5PIT": ("5PIT", "excitatory"),
        "5PET": ("5PET", "excitatory"),
        "5PNP": ("5PNP", "excitatory"),
        "6PIT": ("6PIT", "excitatory"),
        "6PCT": ("6PCT", "excitatory"),
    }

    if ct in subtype_map:
        return subtype_map[ct]

    # Broad-only labels
    if ct in ("excitatory_neuron", "excitatory"):
        return None, "excitatory"
    if ct in ("inhibitory_neuron", "inhibitory"):
        return None, "inhibitory"

    # Non-neuronal
    if ct in ("astrocyte", "microglia", "oligo", "OPC", "pericyte"):
        return None, "other"

    return None, "unknown"


def _load_neuron_data(nl, data_dir):
    """Load skeleton, synapses, and partner data for a neuron.

    Returns (skeleton, syn_coords_nm, snap, partner_df) or None if data missing.
    """
    swc_path = data_dir / f"{nl}.swc"
    syn_path = data_dir / f"{nl}_synapses.csv"
    partner_path = data_dir / f"{nl}_presynaptic.csv"

    if not all(p.exists() for p in [swc_path, syn_path, partner_path]):
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skeleton = NeuronSkeleton.from_swc_file(str(swc_path), scale_factor=1000.0)

    dendrite_skel = skeleton.filter_by_type([1, 3, 4])
    if len(dendrite_skel.branches) < 3:
        dendrite_skel = skeleton

    syn_df = pd.read_csv(syn_path)
    syn_coords_nm = syn_df[["x_um", "y_um", "z_um"]].values * 1000.0
    snap = dendrite_skel.snap_points(syn_coords_nm, d_max=50000.0)

    if snap.valid.sum() < 10:
        return None

    partner_df = pd.read_csv(partner_path)
    return dendrite_skel, syn_coords_nm, snap, partner_df


def _match_partners(syn_coords_nm, snap, partner_df, catalog_lookup=None):
    """Match synapses to partners via KDTree, assign annotation levels.

    Returns DataFrame with columns:
        synapse_idx, branch_idx, pre_root_id, pre_cell_type, pre_cell_type_broad,
        subtype, broad_class, annotation_level
    for valid (snapped) synapses only.

    Vectorized for performance (~2,000 neurons × ~3,000 synapses each).
    """
    partner_coords = partner_df[["x_nm", "y_nm", "z_nm"]].values.astype(float)
    tree = cKDTree(partner_coords)
    _, p_indices = tree.query(syn_coords_nm)

    valid_mask = snap.valid
    valid_indices = np.where(valid_mask)[0]
    valid_branch_ids = snap.branch_ids[valid_mask]

    # Vectorized partner lookup
    matched_p_indices = p_indices[valid_indices]
    pre_root_ids = partner_df["pre_root_id"].values[matched_p_indices]
    pre_cell_types = partner_df["pre_cell_type"].values[matched_p_indices].astype(str)
    pre_broad_types = partner_df["pre_cell_type_broad"].values[matched_p_indices].astype(str)

    # Classify all at once
    n = len(valid_indices)
    subtypes = np.empty(n, dtype=object)
    broad_classes = np.empty(n, dtype=object)
    ann_levels = np.empty(n, dtype=int)

    for i in range(n):
        subtype, broad_class = _classify_subtype(pre_cell_types[i])

        if subtype is not None:
            ann_level = LEVEL_EXACT_SUBTYPE
        elif broad_class in ("excitatory", "inhibitory"):
            ann_level = LEVEL_BROAD_ONLY
        elif broad_class == "other":
            ann_level = LEVEL_UNKNOWN
        else:
            # Try catalog backfill
            rid = pre_root_ids[i]
            if catalog_lookup is not None and rid in catalog_lookup:
                cat_subtype, cat_broad = catalog_lookup[rid]
                if cat_subtype is not None:
                    subtype = cat_subtype
                    broad_class = cat_broad
                    ann_level = LEVEL_BOOSTED
                elif cat_broad in ("excitatory", "inhibitory"):
                    broad_class = cat_broad
                    ann_level = LEVEL_BROAD_ONLY
                else:
                    ann_level = LEVEL_UNKNOWN
            else:
                ann_level = LEVEL_UNKNOWN

        subtypes[i] = subtype
        broad_classes[i] = broad_class
        ann_levels[i] = ann_level

    return pd.DataFrame({
        "synapse_idx": valid_indices.astype(int),
        "branch_idx": valid_branch_ids.astype(int),
        "pre_root_id": pre_root_ids.astype(int),
        "pre_cell_type": pre_cell_types,
        "pre_cell_type_broad": pre_broad_types,
        "subtype": subtypes,
        "broad_class": broad_classes,
        "annotation_level": ann_levels,
    })


def _build_catalog_lookup(catalog_path):
    """Build dict: pre_root_id → (subtype, broad_class) from catalog.csv.

    Vectorized for performance (~91K entries).
    """
    cat = pd.read_csv(catalog_path)
    lookup = {}

    root_ids = cat["root_id"].values
    fines = cat["cell_type_fine"].fillna("").values.astype(str)
    broads = cat["cell_type_broad"].fillna("unknown").values.astype(str)

    for i in range(len(cat)):
        subtype, broad_class = _classify_subtype(fines[i])
        if subtype is None and broads[i] in ("excitatory", "inhibitory"):
            broad_class = broads[i]
        lookup[root_ids[i]] = (subtype, broad_class)

    return lookup


# ═══════════════════════════════════════════════════════════
# STAGE 0a: CAVE COVERAGE BOOST
# ═══════════════════════════════════════════════════════════

def cave_coverage_boost(spatial_df, data_dir):
    """Query CAVE for additional cell type annotations beyond the primary table.

    Attempts to boost presynaptic partner type coverage by:
    1. Querying available annotation tables
    2. Backfilling from catalog.csv
    3. Reporting coverage improvement

    Returns catalog_lookup dict for use in subsequent stages.
    """
    print("=" * 60)
    print("STAGE 0a: CAVE COVERAGE BOOST")
    print("=" * 60)

    catalog_path = PROJECT_DIR / "catalog.csv"
    catalog_lookup = _build_catalog_lookup(catalog_path)
    print(f"  Catalog lookup built: {len(catalog_lookup)} entries")

    # Try CAVE for additional tables
    cave_boost_stats = {"catalog_entries": len(catalog_lookup)}

    try:
        from caveclient import CAVEclient
        client = CAVEclient("minnie65_public")
        tables = client.materialize.get_tables()

        # Look for cell type / classification tables
        ct_tables = [t for t in tables if any(kw in t.lower()
                     for kw in ["cell_type", "celltypes", "classification",
                                "morpho", "perisomatic"])]
        cave_boost_stats["available_tables"] = ct_tables
        print(f"  CAVE tables with cell type info: {ct_tables}")

        # The primary table is aibs_metamodel_celltypes_v661
        # Check for any additional tables that might help
        primary_table = "aibs_metamodel_celltypes_v661"
        additional_tables = [t for t in ct_tables if t != primary_table]

        if additional_tables:
            print(f"  Additional tables to check: {additional_tables}")
            for tbl in additional_tables[:3]:  # limit to avoid excessive queries
                try:
                    meta = client.materialize.get_table_metadata(tbl)
                    print(f"    {tbl}: {meta.get('description', 'no description')[:80]}")
                except Exception as e:
                    print(f"    {tbl}: could not fetch metadata ({str(e)[:40]})")
        else:
            print("  No additional cell type tables found beyond primary")

        cave_boost_stats["primary_table"] = primary_table
        cave_boost_stats["additional_tables_checked"] = additional_tables[:3]

    except Exception as e:
        print(f"  CAVE query failed: {str(e)[:60]}")
        print("  Proceeding with catalog backfill only")
        cave_boost_stats["cave_error"] = str(e)[:100]

    # Compute baseline and boosted coverage on a sample
    print("\n  Computing coverage improvement on sample...")
    sample_neurons = spatial_df["neuron_label"].unique()[:50]
    n_total = 0
    n_typed_base = 0
    n_typed_boost = 0
    n_subtyped_base = 0
    n_subtyped_boost = 0

    for nl in sample_neurons:
        result = _load_neuron_data(nl, data_dir)
        if result is None:
            continue
        _, syn_coords_nm, snap, partner_df = result

        # Baseline (no catalog)
        matched_base = _match_partners(syn_coords_nm, snap, partner_df, catalog_lookup=None)
        # Boosted (with catalog)
        matched_boost = _match_partners(syn_coords_nm, snap, partner_df, catalog_lookup=catalog_lookup)

        n_total += len(matched_base)
        n_typed_base += (matched_base["annotation_level"] <= LEVEL_BROAD_ONLY).sum()
        n_typed_boost += (matched_boost["annotation_level"] <= LEVEL_BROAD_ONLY).sum()
        n_subtyped_base += (matched_base["annotation_level"] == LEVEL_EXACT_SUBTYPE).sum()
        n_subtyped_boost += (matched_boost["annotation_level"] <= LEVEL_BOOSTED).sum()

    if n_total > 0:
        cave_boost_stats["sample_n_synapses"] = int(n_total)
        cave_boost_stats["typed_pct_base"] = float(n_typed_base / n_total * 100)
        cave_boost_stats["typed_pct_boost"] = float(n_typed_boost / n_total * 100)
        cave_boost_stats["subtyped_pct_base"] = float(n_subtyped_base / n_total * 100)
        cave_boost_stats["subtyped_pct_boost"] = float(n_subtyped_boost / n_total * 100)

        print(f"    Total synapses (sample): {n_total}")
        print(f"    Typed (base): {n_typed_base / n_total * 100:.1f}%")
        print(f"    Typed (boost): {n_typed_boost / n_total * 100:.1f}%")
        print(f"    Subtyped (base): {n_subtyped_base / n_total * 100:.1f}%")
        print(f"    Subtyped (boost): {n_subtyped_boost / n_total * 100:.1f}%")

    return catalog_lookup, cave_boost_stats


# ═══════════════════════════════════════════════════════════
# STAGE 0b: TWO-LEVEL COVERAGE GATE
# ═══════════════════════════════════════════════════════════

def coverage_gate(spatial_df, data_dir, catalog_lookup):
    """Two-level coverage gate for afferent typing.

    Gate 1 (overall): For each postsynaptic subtype, require ≥50% of branches
    to have ≥3 typed afferent synapses.

    Gate 2 (per-presynaptic-class): For each postsynaptic subtype that passes
    Gate 1, check per-presynaptic-class coverage (≥30 branches, ≥10 neurons).

    Returns gate_results dict with pass/fail for each subtype.
    """
    print("\n" + "=" * 60)
    print("STAGE 0b: TWO-LEVEL COVERAGE GATE")
    print("=" * 60)

    # Get all inhibitory neurons grouped by postsynaptic subtype
    inh_df = spatial_df[spatial_df["cell_type"].str.startswith("inhibitory")].copy()
    post_subtypes = inh_df["subtype"].unique()

    gate_results = {}

    for post_st in post_subtypes:
        st_df = inh_df[inh_df["subtype"] == post_st]
        st_neurons = st_df["neuron_label"].unique()

        print(f"\n--- Postsynaptic subtype: {post_st} ({len(st_neurons)} neurons, "
              f"{len(st_df)} branches) ---")

        # Track per-branch coverage
        branch_coverage = []  # (neuron_label, branch_idx, n_typed, n_subtyped, has_BC, has_MC, ...)
        level_counts = {1: 0, 2: 0, 3: 0, 4: 0}

        for nl in st_neurons:
            result = _load_neuron_data(nl, data_dir)
            if result is None:
                continue
            _, syn_coords_nm, snap, partner_df = result
            matched = _match_partners(syn_coords_nm, snap, partner_df, catalog_lookup)

            # Count annotation levels
            for lvl in [1, 2, 3, 4]:
                level_counts[lvl] += (matched["annotation_level"] == lvl).sum()

            # Per-branch stats
            for bid, bgrp in matched.groupby("branch_idx"):
                n_typed = (bgrp["annotation_level"] <= LEVEL_BROAD_ONLY).sum()
                n_subtyped = (bgrp["annotation_level"] <= LEVEL_BOOSTED).sum()

                # Check for specific presynaptic classes
                pre_classes = {}
                for pc in INH_SUBTYPES + ["excitatory"]:
                    if pc == "excitatory":
                        pre_classes[pc] = (bgrp["broad_class"] == "excitatory").sum()
                    else:
                        pre_classes[pc] = (bgrp["subtype"] == pc).sum()

                branch_coverage.append({
                    "neuron_label": nl,
                    "branch_idx": bid,
                    "n_typed": n_typed,
                    "n_subtyped": n_subtyped,
                    "n_total": len(bgrp),
                    **{f"n_{pc}": pre_classes[pc] for pc in pre_classes},
                })

        if not branch_coverage:
            print(f"  No data for {post_st}")
            gate_results[post_st] = {"gate1_pass": False, "reason": "no data"}
            continue

        bcov_df = pd.DataFrame(branch_coverage)
        total_branches = len(bcov_df)
        branches_with_3typed = (bcov_df["n_typed"] >= 3).sum()
        pct_with_3typed = branches_with_3typed / total_branches * 100

        # Gate 1
        gate1_pass = pct_with_3typed >= 50
        print(f"  Gate 1: {branches_with_3typed}/{total_branches} branches "
              f"({pct_with_3typed:.1f}%) have ≥3 typed afferent synapses "
              f"→ {'PASS' if gate1_pass else 'FAIL'}")

        # Annotation level breakdown
        total_syn = sum(level_counts.values())
        for lvl, name in [(1, "L1:exact"), (2, "L2:boosted"), (3, "L3:broad"),
                          (4, "L4:unknown")]:
            pct = level_counts[lvl] / total_syn * 100 if total_syn > 0 else 0
            print(f"    {name}: {level_counts[lvl]} ({pct:.1f}%)")

        # Gate 2 (per-presynaptic-class)
        gate2_results = {}
        focal_classes = INH_SUBTYPES + ["excitatory"]

        for pc in focal_classes:
            col = f"n_{pc}"
            branches_with_pc = (bcov_df[col] >= 1).sum()
            neurons_with_pc = bcov_df[bcov_df[col] >= 1]["neuron_label"].nunique()

            g2_pass = branches_with_pc >= 30 and neurons_with_pc >= 10
            gate2_results[pc] = {
                "branches_with_class": int(branches_with_pc),
                "neurons_with_class": int(neurons_with_pc),
                "pass": g2_pass,
            }
            status = "PASS" if g2_pass else "UNDERPOWERED"
            print(f"  Gate 2 [{pc}]: {branches_with_pc} branches, "
                  f"{neurons_with_pc} neurons → {status}")

        gate_results[post_st] = {
            "n_neurons": len(st_neurons),
            "n_branches": total_branches,
            "gate1_pct_with_3typed": float(pct_with_3typed),
            "gate1_pass": gate1_pass,
            "gate2": gate2_results,
            "annotation_levels": {
                f"L{lvl}": int(level_counts[lvl]) for lvl in [1, 2, 3, 4]
            },
        }

    return gate_results


# ═══════════════════════════════════════════════════════════
# COMPUTE AFFERENT FEATURES (per-branch)
# ═══════════════════════════════════════════════════════════

def compute_afferent_features(spatial_df, data_dir, catalog_lookup):
    """Compute per-branch afferent fractions with annotation-level tracking.

    For each branch, computes:
      - frac_exc_afferent, frac_inh_afferent (using L1+L2+L3 synapses)
      - frac_BC_afferent, frac_MC_afferent, frac_BPC_afferent, frac_NGC_afferent
        (using L1+L2 synapses only)
      - n_typed_synapses, n_subtyped_synapses
      - soma_distance_nm (mean across synapses on the branch, for depth binning)

    Returns DataFrame with neuron_label, branch_idx, and afferent features.
    """
    print("\n" + "=" * 60)
    print("COMPUTING AFFERENT FEATURES")
    print("=" * 60)

    neuron_labels = spatial_df["neuron_label"].unique()
    all_records = []
    # Also collect per-synapse data for shuffle test
    all_synapse_records = []
    t0 = time.time()
    n_skipped = 0

    for i, nl in enumerate(neuron_labels):
        result = _load_neuron_data(nl, data_dir)
        if result is None:
            n_skipped += 1
            continue

        _, syn_coords_nm, snap, partner_df = result
        matched = _match_partners(syn_coords_nm, snap, partner_df, catalog_lookup)

        if len(matched) == 0:
            n_skipped += 1
            continue

        # Store per-synapse data for shuffle test (only for inhibitory postsynaptic)
        post_type = spatial_df[spatial_df["neuron_label"] == nl]["cell_type"].iloc[0]
        if str(post_type).startswith("inhibitory"):
            # Get per-branch soma_distance_nm from spatial_df for depth binning
            neuron_branches = spatial_df[spatial_df["neuron_label"] == nl]
            branch_to_soma_dist = dict(zip(
                neuron_branches["branch_idx"].astype(int),
                neuron_branches["soma_distance_nm"]
            ))

            syn_records = matched[["branch_idx", "subtype", "broad_class",
                                   "annotation_level"]].copy()
            syn_records["neuron_label"] = nl
            syn_records["soma_distance_nm"] = syn_records["branch_idx"].map(branch_to_soma_dist)
            all_synapse_records.append(syn_records)

        # Get regime assignments for this neuron's branches
        neuron_regime = spatial_df[spatial_df["neuron_label"] == nl]
        branch_to_regime = dict(zip(
            neuron_regime["branch_idx"].astype(int),
            neuron_regime["regime"].astype(int)
        ))

        # Per-branch aggregation
        for bid, bgrp in matched.groupby("branch_idx"):
            # Broad-class fractions (L1+L2+L3)
            typed_mask = bgrp["annotation_level"] <= LEVEL_BROAD_ONLY
            n_typed = typed_mask.sum()

            # Subtype fractions (L1+L2 only)
            subtyped_mask = bgrp["annotation_level"] <= LEVEL_BOOSTED
            n_subtyped = subtyped_mask.sum()

            if n_typed == 0:
                frac_exc = np.nan
                frac_inh = np.nan
            else:
                typed_broad = bgrp.loc[typed_mask, "broad_class"]
                frac_exc = (typed_broad == "excitatory").sum() / n_typed
                frac_inh = (typed_broad == "inhibitory").sum() / n_typed

            # Fine inhibitory fractions (L1+L2)
            fine_fracs = {}
            if n_subtyped > 0:
                subtyped_df = bgrp[subtyped_mask]
                for st in INH_SUBTYPES:
                    fine_fracs[f"frac_{st}_afferent"] = (
                        (subtyped_df["subtype"] == st).sum() / n_subtyped
                    )
            else:
                for st in INH_SUBTYPES:
                    fine_fracs[f"frac_{st}_afferent"] = np.nan

            all_records.append({
                "neuron_label": nl,
                "branch_idx": int(bid),
                "frac_exc_afferent": frac_exc,
                "frac_inh_afferent": frac_inh,
                **fine_fracs,
                "n_typed_synapses": int(n_typed),
                "n_subtyped_synapses": int(n_subtyped),
                "n_total_synapses_matched": len(bgrp),
                "regime": branch_to_regime.get(int(bid), -1),
            })

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(neuron_labels)} neurons ({elapsed:.0f}s, "
                  f"{len(all_records)} branch records)")

    elapsed = time.time() - t0
    print(f"\nDone: {len(all_records)} branches from "
          f"{len(neuron_labels) - n_skipped}/{len(neuron_labels)} neurons ({elapsed:.0f}s)")

    afferent_df = pd.DataFrame(all_records) if all_records else pd.DataFrame()
    synapse_df = pd.concat(all_synapse_records, ignore_index=True) if all_synapse_records else pd.DataFrame()

    return afferent_df, synapse_df


# ═══════════════════════════════════════════════════════════
# STAGE 1: TARGETING ANALYSIS
# ═══════════════════════════════════════════════════════════

def _precompute_neuron_shuffle_data(neuron_syn, branch_regime_map, n_bins=3):
    """Pre-compute arrays needed for fast depth-preserving shuffles.

    Returns (subtypes, broad_classes, depth_bins, regimes, n_valid) or None.
    Arrays are aligned: index i corresponds to synapse i in neuron_syn.
    """
    subtypes = neuron_syn["subtype"].values.copy()
    broad_classes = neuron_syn["broad_class"].values.copy()
    branch_ids = neuron_syn["branch_idx"].values
    soma_dists = neuron_syn["soma_distance_nm"].values

    regimes = np.array([branch_regime_map.get(int(b), -1) for b in branch_ids])

    # Compute depth bins
    valid = ~np.isnan(soma_dists)
    if valid.sum() < 5:
        return None

    valid_dists = soma_dists[valid]
    bin_edges = np.unique(np.quantile(valid_dists, np.linspace(0, 1, n_bins + 1)))

    depth_bins = np.full(len(subtypes), -1, dtype=int)
    if len(bin_edges) < 2:
        depth_bins[valid] = 0
    else:
        depth_bins[valid] = np.digitize(soma_dists[valid], bin_edges[1:-1])

    return subtypes, broad_classes, depth_bins, regimes


def _fast_shuffle_fractions(subtypes, broad_classes, depth_bins, regimes, focal_classes):
    """Perform one depth-preserving shuffle and compute regime fractions.

    Modifies subtypes/broad_classes in-place, then restores them.
    Returns dict: {class_name: {regime: fraction}}.
    """
    n = len(subtypes)

    # Save originals for restoration
    orig_subtypes = subtypes.copy()
    orig_broad = broad_classes.copy()

    # Shuffle within each depth bin
    unique_bins = np.unique(depth_bins)
    for db in unique_bins:
        if db < 0:
            continue
        mask = depth_bins == db
        idx = np.where(mask)[0]
        perm = RNG.permutation(len(idx))
        subtypes[idx] = orig_subtypes[idx[perm]]
        broad_classes[idx] = orig_broad[idx[perm]]

    # Compute fractions per regime
    result = {}
    for cls_name, cls_check in focal_classes.items():
        result[cls_name] = {}
        for r in [0, 1, 2]:
            r_mask = regimes == r
            n_regime = r_mask.sum()
            if n_regime == 0:
                continue
            if cls_check == "broad_exc":
                n_cls = (broad_classes[r_mask] == "excitatory").sum()
            elif cls_check == "broad_inh":
                n_cls = (broad_classes[r_mask] == "inhibitory").sum()
            else:
                n_cls = (subtypes[r_mask] == cls_check).sum()
            result[cls_name][r] = n_cls / n_regime

    # Restore originals
    subtypes[:] = orig_subtypes
    broad_classes[:] = orig_broad

    return result



def targeting_analysis(spatial_df, afferent_df, synapse_df):
    """Stage 1: Do different presynaptic cell types target different regimes?

    Four complementary analyses:
    1. Log-enrichment vs depth-preserving shuffle (PRIMARY)
    2. Descriptive fractions by regime (SUPPORT)
    3. Depth-preserving shuffle test (1,000 permutations per neuron)
    4. Within-neuron targeting test (chi-squared per neuron)
    5. Mixed model: frac_inh_afferent ~ C(regime) + covariates + (1|neuron)
    """
    print("\n" + "=" * 60)
    print("STAGE 1: TARGETING ANALYSIS")
    print("=" * 60)

    results = {}

    # Merge afferent features with spatial features
    merged = spatial_df.merge(afferent_df, on=["neuron_label", "branch_idx"],
                              how="inner", suffixes=("", "_aff"))

    inh_merged = merged[merged["cell_type"].str.startswith("inhibitory")].copy()
    inh_merged = inh_merged[inh_merged["regime"] >= 0]

    print(f"  Inhibitory branches with afferent data: {len(inh_merged)}")

    # ── Analysis 1 & 2: Descriptive fractions + log-enrichment ──
    print("\n  --- Descriptive afferent fractions by regime ---")

    desc_results = {}
    for post_st in sorted(inh_merged["subtype"].unique()):
        st_data = inh_merged[inh_merged["subtype"] == post_st]
        desc_results[post_st] = {}

        for frac_col in (["frac_exc_afferent", "frac_inh_afferent"] +
                         [f"frac_{st}_afferent" for st in INH_SUBTYPES]):
            regime_means = {}
            for r in [0, 1, 2]:
                vals = st_data[st_data["regime"] == r][frac_col].dropna()
                if len(vals) > 0:
                    regime_means[str(r)] = {
                        "mean": float(vals.mean()),
                        "se": float(vals.std() / np.sqrt(len(vals))),
                        "n": int(len(vals)),
                    }
            desc_results[post_st][frac_col] = regime_means

        # Print summary for BC and MC
        for key in ["frac_BC_afferent", "frac_MC_afferent"]:
            means = {r: desc_results[post_st].get(key, {}).get(str(r), {}).get("mean", np.nan)
                     for r in [0, 1, 2]}
            print(f"    {post_st} {key}: R0={means[0]:.3f}, R1={means[1]:.3f}, R2={means[2]:.3f}")

    results["descriptive_fractions"] = desc_results

    # ── Analysis 3: Depth-preserving shuffle test ──
    print("\n  --- Depth-preserving shuffle test ---")

    if len(synapse_df) == 0:
        print("    No synapse-level data available for shuffle test")
        results["shuffle_test"] = {"error": "no synapse data"}
    else:
        inh_neurons = synapse_df["neuron_label"].unique()
        print(f"    Running {N_SHUFFLE} permutations for {len(inh_neurons)} inhibitory neurons...")

        # Focal classes: string keys for fast numpy comparison
        focal_classes_check = {
            "BC": "BC", "MC": "MC", "BPC": "BPC", "NGC": "NGC",
            "excitatory": "broad_exc", "inhibitory": "broad_inh",
        }

        # Get branch-to-regime map from spatial_df
        branch_regime_all = {}
        for nl in inh_neurons:
            nr = spatial_df[spatial_df["neuron_label"] == nl]
            bmap = dict(zip(nr["branch_idx"].astype(int), nr["regime"].astype(int)))
            branch_regime_all[nl] = bmap

        # Observed and null enrichments
        # observed_enrichments: per-neuron observed fractions
        # null_matrix: per-neuron × per-permutation fractions (for proper population-level p-values)
        observed_enrichments = {cls: {r: [] for r in [0, 1, 2]} for cls in focal_classes_check}
        null_matrix = {cls: {r: [] for r in [0, 1, 2]} for cls in focal_classes_check}
        # null_matrix[cls][r] = list of lists: outer = neurons, inner = N_SHUFFLE values

        t0_shuffle = time.time()
        n_valid_neurons = 0

        for ni, nl in enumerate(inh_neurons):
            neuron_syn = synapse_df[synapse_df["neuron_label"] == nl]
            bmap = branch_regime_all.get(nl, {})

            if len(neuron_syn) < 10:
                continue

            # Pre-compute arrays for this neuron
            precomp = _precompute_neuron_shuffle_data(neuron_syn, bmap, n_bins=3)
            if precomp is None:
                continue

            subtypes, broad_classes, depth_bins, regimes = precomp
            n_valid_neurons += 1

            # Observed fractions
            obs = {}
            for cls_name, cls_check in focal_classes_check.items():
                obs[cls_name] = {}
                for r in [0, 1, 2]:
                    r_mask = regimes == r
                    n_regime = r_mask.sum()
                    if n_regime == 0:
                        continue
                    if cls_check == "broad_exc":
                        n_cls = (broad_classes[r_mask] == "excitatory").sum()
                    elif cls_check == "broad_inh":
                        n_cls = (broad_classes[r_mask] == "inhibitory").sum()
                    else:
                        n_cls = (subtypes[r_mask] == cls_check).sum()
                    obs[cls_name][r] = n_cls / n_regime

            for cls in focal_classes_check:
                for r in [0, 1, 2]:
                    if r in obs.get(cls, {}):
                        observed_enrichments[cls][r].append(obs[cls][r])

            # Null distribution: store per-permutation values for this neuron
            null_per_neuron = {cls: {r: [] for r in [0, 1, 2]} for cls in focal_classes_check}

            for _ in range(N_SHUFFLE):
                null_fracs = _fast_shuffle_fractions(
                    subtypes, broad_classes, depth_bins, regimes, focal_classes_check
                )
                for cls in focal_classes_check:
                    for r in [0, 1, 2]:
                        if r in null_fracs.get(cls, {}):
                            null_per_neuron[cls][r].append(null_fracs[cls][r])

            # Store full per-permutation vectors for population-level test
            for cls in focal_classes_check:
                for r in [0, 1, 2]:
                    if null_per_neuron[cls][r]:
                        null_matrix[cls][r].append(null_per_neuron[cls][r])

            if (ni + 1) % 50 == 0:
                print(f"      {ni+1}/{len(inh_neurons)} neurons shuffled "
                      f"({time.time()-t0_shuffle:.0f}s)")

        print(f"    Shuffle complete ({time.time()-t0_shuffle:.0f}s), "
              f"{n_valid_neurons} valid neurons")

        # Compute log-enrichment and PROPER population-level p-values
        # For each permutation i, compute the population mean by taking
        # the i-th shuffled value from each neuron, then averaging.
        # The p-value is the fraction of population null means ≥ observed population mean.
        enrichment_results = {}
        for cls in focal_classes_check:
            enrichment_results[cls] = {}
            for r in [0, 1, 2]:
                obs_vals = observed_enrichments[cls][r]
                neuron_null_lists = null_matrix[cls][r]

                if not obs_vals or not neuron_null_lists:
                    continue

                obs_mean = np.mean(obs_vals)

                # Build population-level null distribution
                # Each neuron contributed N_SHUFFLE values; take permutation i from each neuron
                n_neurons_with_data = len(neuron_null_lists)
                # Pad shorter lists if needed (some neurons may have fewer regimes)
                min_shuffles = min(len(v) for v in neuron_null_lists)
                null_pop_means = np.zeros(min_shuffles)
                for perm_i in range(min_shuffles):
                    pop_sum = sum(neuron_null_lists[ni][perm_i]
                                 for ni in range(n_neurons_with_data))
                    null_pop_means[perm_i] = pop_sum / n_neurons_with_data

                null_mean = float(np.mean(null_pop_means))

                # Log2 enrichment
                if null_mean > 0 and obs_mean > 0:
                    log2_enrich = np.log2(obs_mean / null_mean)
                else:
                    log2_enrich = np.nan

                # Empirical p-value: fraction of population null means ≥ observed
                p_enriched = float(np.mean(null_pop_means >= obs_mean))
                p_depleted = float(np.mean(null_pop_means <= obs_mean))
                p_two_sided = 2 * min(p_enriched, p_depleted)
                p_two_sided = min(p_two_sided, 1.0)  # cap at 1

                enrichment_results[cls][str(r)] = {
                    "observed_mean_frac": float(obs_mean),
                    "null_mean_frac": null_mean,
                    "null_std": float(np.std(null_pop_means)),
                    "log2_enrichment": float(log2_enrich) if not np.isnan(log2_enrich) else None,
                    "p_enriched": p_enriched,
                    "p_depleted": p_depleted,
                    "p_two_sided": p_two_sided,
                    "n_neurons": len(obs_vals),
                    "n_permutations": int(min_shuffles),
                }

                if cls in ("BC", "MC") and r in (0, 2):
                    direction = "enriched" if log2_enrich > 0 else "depleted"
                    print(f"    {cls} on regime {r}: log2={log2_enrich:.3f} ({direction}), "
                          f"p={p_two_sided:.4f}")

        results["log_enrichment"] = enrichment_results

        # Sensitivity: repeat with 2 and 4 depth bins
        print("\n  --- Sensitivity: depth bin count ---")
        sensitivity = {}
        sens_focal = {"BC": "BC", "MC": "MC"}

        for n_bins in [2, 4]:
            sens_null = {cls: {r: [] for r in [0, 1, 2]} for cls in ["BC", "MC"]}
            for nl in inh_neurons[:100]:  # subset for speed
                neuron_syn = synapse_df[synapse_df["neuron_label"] == nl]
                bmap = branch_regime_all.get(nl, {})
                if len(neuron_syn) < 10:
                    continue

                precomp = _precompute_neuron_shuffle_data(neuron_syn, bmap, n_bins=n_bins)
                if precomp is None:
                    continue
                st, bc, db, reg = precomp

                for _ in range(100):  # fewer permutations for sensitivity
                    nf = _fast_shuffle_fractions(st, bc, db, reg, sens_focal)
                    for cls in ["BC", "MC"]:
                        for r in [0, 1, 2]:
                            if r in nf.get(cls, {}):
                                sens_null[cls][r].append(nf[cls][r])

            sensitivity[str(n_bins)] = {}
            for cls in ["BC", "MC"]:
                for r in [0, 2]:
                    null_m = np.mean(sens_null[cls][r]) if sens_null[cls][r] else np.nan
                    sensitivity[str(n_bins)][f"{cls}_regime{r}_null_mean"] = (
                        float(null_m) if not np.isnan(null_m) else None
                    )
            print(f"    {n_bins} bins: BC R0 null={sensitivity[str(n_bins)].get('BC_regime0_null_mean', 'N/A')}, "
                  f"MC R2 null={sensitivity[str(n_bins)].get('MC_regime2_null_mean', 'N/A')}")

        results["depth_bin_sensitivity"] = sensitivity

    # ── Analysis 4: Within-neuron targeting (chi-squared) ──
    print("\n  --- Within-neuron targeting test ---")

    within_neuron_results = {}
    for post_st in sorted(inh_merged["subtype"].unique()):
        st_data = inh_merged[inh_merged["subtype"] == post_st]
        neurons = st_data["neuron_label"].unique()

        n_tested = 0
        n_significant = 0
        directions = {"BC_low": 0, "MC_high": 0}  # count consistent directions

        for nl in neurons:
            nl_data = st_data[st_data["neuron_label"] == nl]
            # Need branches in multiple regimes with subtyped afferents
            regimes_present = nl_data["regime"].unique()
            if len(regimes_present) < 2:
                continue

            # Build contingency: rows = regimes, cols = afferent subtypes
            # Use subtyped synapses only
            nl_with_fracs = nl_data.dropna(subset=["frac_BC_afferent"])
            if len(nl_with_fracs) < 5:
                continue

            # Aggregate synapse counts by regime × afferent class
            try:
                # Use n_subtyped_synapses weighted fractions
                agg = []
                for r in sorted(regimes_present):
                    r_data = nl_with_fracs[nl_with_fracs["regime"] == r]
                    n_bc = (r_data["frac_BC_afferent"] * r_data["n_subtyped_synapses"]).sum()
                    n_mc = (r_data["frac_MC_afferent"] * r_data["n_subtyped_synapses"]).sum()
                    n_other = r_data["n_subtyped_synapses"].sum() - n_bc - n_mc
                    agg.append([n_bc, n_mc, max(0, n_other)])

                contingency = np.array(agg)
                # Remove zero columns
                contingency = contingency[:, contingency.sum(axis=0) > 0]

                if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                    if contingency.sum() >= 10:
                        chi2, p, _, _ = stats.chi2_contingency(contingency)
                        n_tested += 1
                        if p < 0.05:
                            n_significant += 1

                        # Check directionality: is BC fraction higher on low regime?
                        bc_col = 0  # first column is BC
                        regimes_sorted = sorted(regimes_present)
                        if len(regimes_sorted) >= 2:
                            bc_low = contingency[0, bc_col] / max(contingency[0].sum(), 1)
                            bc_high = contingency[-1, bc_col] / max(contingency[-1].sum(), 1)
                            if bc_low > bc_high:
                                directions["BC_low"] += 1
                            if contingency.shape[1] >= 2:
                                mc_col = 1
                                mc_low = contingency[0, mc_col] / max(contingency[0].sum(), 1)
                                mc_high = contingency[-1, mc_col] / max(contingency[-1].sum(), 1)
                                if mc_high > mc_low:
                                    directions["MC_high"] += 1

            except Exception:
                continue

        within_neuron_results[post_st] = {
            "n_neurons_tested": n_tested,
            "n_significant_005": n_significant,
            "frac_significant": float(n_significant / n_tested) if n_tested > 0 else None,
            "directional_consistency": {
                "BC_enriched_on_low_regime": int(directions["BC_low"]),
                "MC_enriched_on_high_regime": int(directions["MC_high"]),
                "of_n_tested": n_tested,
            },
        }
        print(f"    {post_st}: {n_significant}/{n_tested} neurons significant (p<0.05), "
              f"BC→low: {directions['BC_low']}/{n_tested}, MC→high: {directions['MC_high']}/{n_tested}")

    results["within_neuron_targeting"] = within_neuron_results

    # ── Analysis 5: Mixed model ──
    print("\n  --- Mixed model: afferent fraction ~ regime ---")

    mm_results = {}
    for frac_col in ["frac_inh_afferent", "frac_exc_afferent",
                     "frac_BC_afferent", "frac_MC_afferent"]:
        work = inh_merged.dropna(subset=[frac_col, "regime", "neuron_label",
                                         "synapse_count", "total_length_nm"]).copy()
        work = work[work["regime"] >= 0]

        if len(work) < 50 or work["neuron_label"].nunique() < 5:
            print(f"    {frac_col}: insufficient data")
            continue

        try:
            formula = f"{frac_col} ~ C(regime) + synapse_count + total_length_nm"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(formula, work, groups=work["neuron_label"])
                res = model.fit(reml=True, maxiter=500)

            coefs = {}
            for key in res.params.index:
                coefs[key] = {
                    "coef": float(res.params[key]),
                    "p": float(res.pvalues[key]),
                }

            mm_results[frac_col] = {
                "coefficients": coefs,
                "n_obs": len(work),
                "n_neurons": work["neuron_label"].nunique(),
            }

            r1_coef = res.params.get("C(regime)[T.1]", np.nan)
            r2_coef = res.params.get("C(regime)[T.2]", np.nan)
            r2_p = res.pvalues.get("C(regime)[T.2]", np.nan)
            print(f"    {frac_col}: regime1={r1_coef:.4f}, regime2={r2_coef:.4f}, p(r2)={r2_p:.2e}")

        except Exception as e:
            print(f"    {frac_col}: FAILED ({str(e)[:50]})")

    results["mixed_model"] = mm_results

    return results


# ═══════════════════════════════════════════════════════════
# STAGE 2a: COMPOSITION MEDIATION (FOUR-MODEL COMPARISON)
# ═══════════════════════════════════════════════════════════

def placement_mediation(spatial_df, afferent_df):
    """Stage 2a: Does afferent composition mediate the inhibitory residual?

    Four-model comparison for each spatial metric × postsynaptic subtype:
      M0: spatial_z ~ C(regime) + synapse_count + total_length_nm + (1|neuron)
      M1: M0 + afferent fractions
      M2: M0 + partner architecture features
      M3: M0 + afferent + partner architecture

    Also: within-neuron mediation variant (neuron-mean-centered afferents).
    """
    print("\n" + "=" * 60)
    print("STAGE 2a: COMPOSITION MEDIATION (FOUR-MODEL COMPARISON)")
    print("=" * 60)

    # Load partner architecture features
    partner_path = RESULTS_DIR / "partner_architecture_features.csv"
    if not partner_path.exists():
        print("  ERROR: partner_architecture_features.csv not found")
        return {"error": "missing partner architecture features"}

    partner_df = pd.read_csv(partner_path)

    # Merge all features
    merged = spatial_df.merge(afferent_df, on=["neuron_label", "branch_idx"],
                              how="left", suffixes=("", "_aff"))

    # Partner features are already in partner_df which includes spatial features
    partner_cols = ["neuron_label", "branch_idx", "unique_partners",
                    "mean_syn_per_partner", "gini_partner", "frac_multisynaptic"]
    partner_subset = partner_df[partner_cols].drop_duplicates()
    merged = merged.merge(partner_subset, on=["neuron_label", "branch_idx"], how="left")

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    afferent_covariates = ["frac_inh_afferent", "frac_BC_afferent", "frac_MC_afferent"]
    partner_covariates = ["unique_partners", "mean_syn_per_partner",
                          "gini_partner", "frac_multisynaptic"]

    results = {}

    # Stratify by postsynaptic subtype
    inh_merged = merged[merged["cell_type"].str.startswith("inhibitory")].copy()

    for post_st in ["all_inhibitory"] + sorted(inh_merged["subtype"].unique()):
        if post_st == "all_inhibitory":
            subset = inh_merged
        else:
            subset = inh_merged[inh_merged["subtype"] == post_st]

        print(f"\n--- {post_st} (n={len(subset)}) ---")
        results[post_st] = {}

        for metric in metrics:
            base_cols = [metric, "regime", "neuron_label", "synapse_count", "total_length_nm"]

            # M0: Baseline
            work_m0 = subset.dropna(subset=base_cols).copy()
            work_m0 = work_m0[work_m0["regime"] >= 0]

            if len(work_m0) < 50 or work_m0["neuron_label"].nunique() < 5:
                print(f"  {metric}: insufficient data")
                continue

            try:
                formula_m0 = f"{metric} ~ C(regime) + synapse_count + total_length_nm"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_m0 = smf.mixedlm(formula_m0, work_m0, groups=work_m0["neuron_label"])
                    res_m0 = model_m0.fit(reml=True, maxiter=500)

                coef_m0 = res_m0.params.get("C(regime)[T.2]", np.nan)
                p_m0 = res_m0.pvalues.get("C(regime)[T.2]", np.nan)

                metric_results = {
                    "M0_coef": float(coef_m0),
                    "M0_p": float(p_m0),
                    "M0_n": len(work_m0),
                }

                # M1: + afferent fractions
                m1_cols = base_cols + afferent_covariates
                work_m1 = subset.dropna(subset=m1_cols).copy()
                work_m1 = work_m1[work_m1["regime"] >= 0]

                if len(work_m1) >= 50:
                    try:
                        aff_terms = " + ".join(afferent_covariates)
                        formula_m1 = f"{metric} ~ C(regime) + synapse_count + total_length_nm + {aff_terms}"
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model_m1 = smf.mixedlm(formula_m1, work_m1, groups=work_m1["neuron_label"])
                            res_m1 = model_m1.fit(reml=True, maxiter=500)

                        coef_m1 = res_m1.params.get("C(regime)[T.2]", np.nan)
                        med_m1 = (1 - abs(coef_m1) / abs(coef_m0)) * 100 if abs(coef_m0) > 1e-10 else np.nan

                        metric_results["M1_coef"] = float(coef_m1)
                        metric_results["M1_mediation_pct"] = float(med_m1)
                        metric_results["M1_n"] = len(work_m1)
                        metric_results["M1_afferent_coefs"] = {
                            cov: {"coef": float(res_m1.params.get(cov, np.nan)),
                                  "p": float(res_m1.pvalues.get(cov, np.nan))}
                            for cov in afferent_covariates
                        }
                        print(f"  {metric} M1: mediation={med_m1:.1f}%")
                    except Exception as e:
                        print(f"  {metric} M1: FAILED ({str(e)[:50]})")
                        metric_results["M1_error"] = str(e)[:100]

                # M2: + partner architecture
                m2_cols = base_cols + partner_covariates
                work_m2 = subset.dropna(subset=m2_cols).copy()
                work_m2 = work_m2[work_m2["regime"] >= 0]

                if len(work_m2) >= 50:
                    try:
                        part_terms = " + ".join(partner_covariates)
                        formula_m2 = f"{metric} ~ C(regime) + synapse_count + total_length_nm + {part_terms}"
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model_m2 = smf.mixedlm(formula_m2, work_m2, groups=work_m2["neuron_label"])
                            res_m2 = model_m2.fit(reml=True, maxiter=500)

                        coef_m2 = res_m2.params.get("C(regime)[T.2]", np.nan)
                        med_m2 = (1 - abs(coef_m2) / abs(coef_m0)) * 100 if abs(coef_m0) > 1e-10 else np.nan

                        metric_results["M2_coef"] = float(coef_m2)
                        metric_results["M2_mediation_pct"] = float(med_m2)
                        metric_results["M2_n"] = len(work_m2)
                        print(f"  {metric} M2: mediation={med_m2:.1f}%")
                    except Exception as e:
                        print(f"  {metric} M2: FAILED ({str(e)[:50]})")
                        metric_results["M2_error"] = str(e)[:100]

                # M3: + both afferent + partner architecture
                m3_cols = base_cols + afferent_covariates + partner_covariates
                work_m3 = subset.dropna(subset=m3_cols).copy()
                work_m3 = work_m3[work_m3["regime"] >= 0]

                if len(work_m3) >= 50:
                    try:
                        all_terms = " + ".join(afferent_covariates + partner_covariates)
                        formula_m3 = f"{metric} ~ C(regime) + synapse_count + total_length_nm + {all_terms}"
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model_m3 = smf.mixedlm(formula_m3, work_m3, groups=work_m3["neuron_label"])
                            res_m3 = model_m3.fit(reml=True, maxiter=500)

                        coef_m3 = res_m3.params.get("C(regime)[T.2]", np.nan)
                        med_m3 = (1 - abs(coef_m3) / abs(coef_m0)) * 100 if abs(coef_m0) > 1e-10 else np.nan

                        metric_results["M3_coef"] = float(coef_m3)
                        metric_results["M3_mediation_pct"] = float(med_m3)
                        metric_results["M3_n"] = len(work_m3)
                        metric_results["M3_all_coefs"] = {
                            cov: {"coef": float(res_m3.params.get(cov, np.nan)),
                                  "p": float(res_m3.pvalues.get(cov, np.nan))}
                            for cov in afferent_covariates + partner_covariates
                        }
                        print(f"  {metric} M3: mediation={med_m3:.1f}%")

                        # Interpretation
                        if "M2_mediation_pct" in metric_results:
                            m2_med = metric_results["M2_mediation_pct"]
                            if abs(med_m3 - m2_med) < 5:
                                metric_results["interpretation"] = (
                                    "afferent identity acts through partner architecture"
                                )
                            elif med_m3 > m2_med + 5:
                                metric_results["interpretation"] = (
                                    "afferent identity has placement effects "
                                    "beyond partner architecture"
                                )
                    except Exception as e:
                        print(f"  {metric} M3: FAILED ({str(e)[:50]})")
                        metric_results["M3_error"] = str(e)[:100]

                # Within-neuron mediation variant
                # Add neuron-mean-centered afferent fractions
                wn_cols = base_cols + afferent_covariates
                work_wn = subset.dropna(subset=wn_cols).copy()
                work_wn = work_wn[work_wn["regime"] >= 0]

                if len(work_wn) >= 50 and work_wn["neuron_label"].nunique() >= 5:
                    try:
                        # Center afferent fractions within each neuron
                        for cov in afferent_covariates:
                            neuron_means = work_wn.groupby("neuron_label")[cov].transform("mean")
                            work_wn[f"{cov}_centered"] = work_wn[cov] - neuron_means

                        centered_terms = " + ".join(f"{c}_centered" for c in afferent_covariates)
                        formula_wn = f"{metric} ~ C(regime) + synapse_count + total_length_nm + {centered_terms}"
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model_wn = smf.mixedlm(formula_wn, work_wn, groups=work_wn["neuron_label"])
                            res_wn = model_wn.fit(reml=True, maxiter=500)

                        coef_wn = res_wn.params.get("C(regime)[T.2]", np.nan)
                        med_wn = (1 - abs(coef_wn) / abs(coef_m0)) * 100 if abs(coef_m0) > 1e-10 else np.nan

                        metric_results["within_neuron_coef"] = float(coef_wn)
                        metric_results["within_neuron_mediation_pct"] = float(med_wn)
                        print(f"  {metric} within-neuron: mediation={med_wn:.1f}%")

                    except Exception as e:
                        print(f"  {metric} within-neuron: FAILED ({str(e)[:50]})")

                results[post_st][metric] = metric_results

            except Exception as e:
                print(f"  {metric}: FAILED baseline ({str(e)[:60]})")

    return results


# ═══════════════════════════════════════════════════════════
# STAGE 2b: CLASS-SPECIFIC PLACEMENT RULES
# ═══════════════════════════════════════════════════════════

def class_specific_placement(spatial_df, afferent_df, mediation_results):
    """Stage 2b: Do different presynaptic classes impose different spacing rules?

    Conditional on Stage 2a showing meaningful afferent mediation.
    Within each focal afferent stratum (branches with ≥3 typed synapses from that class),
    test whether spatial metrics differ by regime.
    """
    print("\n" + "=" * 60)
    print("STAGE 2b: CLASS-SPECIFIC PLACEMENT RULES")
    print("=" * 60)

    # Check if Stage 2a showed meaningful afferent effects
    has_meaningful_effect = False
    for post_st, st_results in mediation_results.items():
        if post_st == "error":
            continue
        for metric, mr in st_results.items():
            if isinstance(mr, dict) and "M1_mediation_pct" in mr:
                if abs(mr["M1_mediation_pct"]) > 5:
                    has_meaningful_effect = True
                    break
            if isinstance(mr, dict) and "M3_mediation_pct" in mr:
                if abs(mr["M3_mediation_pct"]) > 5:
                    has_meaningful_effect = True
                    break

    if not has_meaningful_effect:
        print("  Stage 2a showed no meaningful afferent mediation (all <5%)")
        print("  Stage 2b skipped — afferent composition does not reduce regime coefficient")
        print("  Interpretation: inhibitory residual is not primarily explained by afferent identity")
        return {"skipped": True, "reason": "no meaningful afferent mediation in Stage 2a"}

    # Merge features
    merged = spatial_df.merge(afferent_df, on=["neuron_label", "branch_idx"],
                              how="inner", suffixes=("", "_aff"))
    inh_merged = merged[merged["cell_type"].str.startswith("inhibitory")].copy()
    inh_merged = inh_merged[inh_merged["regime"] >= 0]

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    results = {}

    for focal_class in INH_SUBTYPES + ["excitatory"]:
        print(f"\n  --- Focal class: {focal_class} ---")

        if focal_class == "excitatory":
            frac_col = "frac_exc_afferent"
            n_col = "n_typed_synapses"
        else:
            frac_col = f"frac_{focal_class}_afferent"
            n_col = "n_subtyped_synapses"

        # Filter to branches with ≥3 typed synapses from this class
        # Approximate: frac * n_subtyped ≥ 3
        stratum = inh_merged.dropna(subset=[frac_col, n_col]).copy()
        stratum["n_class"] = stratum[frac_col] * stratum[n_col]
        stratum = stratum[stratum["n_class"] >= 3]

        if len(stratum) < 30:
            print(f"    Too few branches (n={len(stratum)}), skipping")
            results[focal_class] = {"n_branches": len(stratum), "skipped": True}
            continue

        print(f"    Branches with ≥3 {focal_class} synapses: {len(stratum)}")
        results[focal_class] = {"n_branches": len(stratum), "n_neurons": stratum["neuron_label"].nunique()}

        for metric in metrics:
            work = stratum.dropna(subset=[metric, "synapse_count", "total_length_nm",
                                          "neuron_label"]).copy()
            if len(work) < 30 or work["neuron_label"].nunique() < 5:
                continue

            try:
                formula = f"{metric} ~ C(regime) + synapse_count + total_length_nm"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = smf.mixedlm(formula, work, groups=work["neuron_label"])
                    res = model.fit(reml=True, maxiter=500)

                coef_r2 = res.params.get("C(regime)[T.2]", np.nan)
                p_r2 = res.pvalues.get("C(regime)[T.2]", np.nan)

                results[focal_class][metric] = {
                    "regime2_coef": float(coef_r2),
                    "regime2_p": float(p_r2),
                    "n_obs": len(work),
                }
                print(f"    {metric}: regime2_coef={coef_r2:.4f}, p={p_r2:.2e}")

            except Exception as e:
                print(f"    {metric}: FAILED ({str(e)[:50]})")

    return results


# ═══════════════════════════════════════════════════════════
# MORPHOLOGY CONTROLS
# ═══════════════════════════════════════════════════════════

def morphology_controls(spatial_df):
    """Test whether coupling strength correlates with morphology within subtypes.

    For each inhibitory subtype, correlate per-neuron coupling slopes (from
    random_slope_results.json) with morphological features.
    """
    print("\n" + "=" * 60)
    print("MORPHOLOGY CONTROLS")
    print("=" * 60)

    # Load random slope results to get per-neuron slopes
    rs_path = RESULTS_DIR / "random_slope_results.json"
    if not rs_path.exists():
        print("  ERROR: random_slope_results.json not found")
        return {"error": "missing random slope results"}

    with open(rs_path) as f:
        rs_data = json.load(f)

    results = {}

    # Use interval_cv_z slopes (most interpretable for spacing)
    icv = rs_data.get("interval_cv_z", {})
    if "error" in icv:
        print("  No interval_cv_z results available")
        return {"error": "no interval_cv_z slopes"}

    # Reconstruct per-neuron slopes from the model
    # We need to re-fit the random-slope model to get per-neuron BLUPs
    # Instead, compute per-neuron morphology features and correlate with
    # aggregate spacing behavior

    inh_df = spatial_df[spatial_df["cell_type"].str.startswith("inhibitory")].copy()

    # Per-neuron morphology features
    neuron_morph = inh_df.groupby(["neuron_label", "subtype"]).agg(
        total_dendritic_length=("total_length_nm", "sum"),
        mean_branch_order=("branch_order", "mean"),
        n_branches=("branch_idx", "count"),
        mean_electrotonic_length=("electrotonic_length", "mean"),
        max_soma_distance=("soma_distance_nm", "max"),
        mean_diameter=("mean_diameter_nm", "mean"),
    ).reset_index()

    # Per-neuron coupling proxy: std of interval_cv_z across regimes
    # (neurons with higher coupling show more variation in ICV across regimes)
    neuron_coupling = inh_df.groupby("neuron_label").apply(
        lambda g: pd.Series({
            "icv_range": g["interval_cv_z"].max() - g["interval_cv_z"].min(),
            "icv_std": g["interval_cv_z"].std(),
            "ce_std": g["clark_evans_z"].std(),
            "n_regimes": g["regime"].nunique(),
        }),
        include_groups=False,
    ).reset_index()

    neuron_morph = neuron_morph.merge(neuron_coupling, on="neuron_label")

    for st in sorted(neuron_morph["subtype"].unique()):
        st_data = neuron_morph[neuron_morph["subtype"] == st]
        print(f"\n  --- {st} (n={len(st_data)}) ---")

        if len(st_data) < 10:
            print("    Too few neurons")
            continue

        results[st] = {"n_neurons": len(st_data)}

        morph_features = ["total_dendritic_length", "mean_branch_order", "n_branches",
                          "mean_electrotonic_length", "max_soma_distance"]

        for morph_feat in morph_features:
            for coupling_metric in ["icv_std", "icv_range"]:
                valid = st_data.dropna(subset=[morph_feat, coupling_metric])
                if len(valid) < 10:
                    continue

                r, p = stats.spearmanr(valid[morph_feat], valid[coupling_metric])
                if coupling_metric == "icv_std":  # only print one
                    if abs(r) > 0.2 and p < 0.05:
                        print(f"    {morph_feat} ~ {coupling_metric}: "
                              f"r={r:.3f}, p={p:.3e} *")

                results[st][f"{morph_feat}_vs_{coupling_metric}"] = {
                    "spearman_r": float(r),
                    "spearman_p": float(p),
                    "n": len(valid),
                }

    return results


# ═══════════════════════════════════════════════════════════
# SUPPLEMENTARY ANALYSES
# ═══════════════════════════════════════════════════════════

def axon_conditioned_placement(spatial_df, data_dir, catalog_lookup=None):
    """Axon-conditioned placement analysis: do presynaptic classes place
    synapses differently on compact vs compartmentalized branches?

    For multisynaptic typed partners (same pre_root_id with ≥2 synapses on
    same branch), computes within-partner pairwise 3D distance and tests
    whether regime predicts placement compactness after conditioning on
    multiplicity.

    Separates two mechanisms:
    A) Arrival: BC axons preferentially innervate compact branches
    B) Placement: BC axons, once there, place boutons more tightly on
       compact branches

    Returns dict with per-subtype results and interaction model.
    """
    print("\n" + "=" * 60)
    print("AXON-CONDITIONED PLACEMENT ANALYSIS")
    print("=" * 60)

    # Collect per-partner-branch placement records
    records = []
    inh_neurons = spatial_df[spatial_df["cell_type"].str.startswith("inhibitory")][
        "neuron_label"].unique()

    # Build branch → regime + branch_length lookup from spatial_df
    branch_info = {}
    for _, row in spatial_df.iterrows():
        key = (row["neuron_label"], int(row["branch_idx"]))
        branch_info[key] = {
            "regime": int(row["regime"]) if pd.notna(row.get("regime")) else -1,
            "total_length_nm": row.get("total_length_nm", np.nan),
            "synapse_count": row.get("synapse_count", np.nan),
            "subtype": row.get("subtype", ""),
            "soma_distance_nm": row.get("soma_distance_nm", np.nan),
        }

    t0 = time.time()
    n_processed = 0

    for nl in inh_neurons:
        result = _load_neuron_data(nl, data_dir)
        if result is None:
            continue

        _, syn_coords_nm, snap, partner_df = result
        matched = _match_partners(syn_coords_nm, snap, partner_df, catalog_lookup)

        if len(matched) == 0:
            continue

        # Add 3D coordinates for valid synapses
        matched["x_nm"] = syn_coords_nm[matched["synapse_idx"].values, 0]
        matched["y_nm"] = syn_coords_nm[matched["synapse_idx"].values, 1]
        matched["z_nm"] = syn_coords_nm[matched["synapse_idx"].values, 2]

        # Filter to typed presynaptic partners (L1+L2)
        typed = matched[matched["annotation_level"] <= LEVEL_BOOSTED].copy()

        # For each branch × pre_root_id combo, check for multisynaptic contacts
        for (bid, rid), grp in typed.groupby(["branch_idx", "pre_root_id"]):
            k = len(grp)
            if k < 2:
                continue

            pre_subtype = grp["subtype"].iloc[0]
            if pre_subtype not in INH_SUBTYPES and pre_subtype not in (
                "23P", "4P", "5PIT", "5PET", "6PIT", "6PCT"):
                continue

            # Determine if presynaptic is excitatory or specific inhibitory subtype
            pre_class = pre_subtype if pre_subtype in INH_SUBTYPES else "excitatory"

            # Compute pairwise 3D distances
            coords = grp[["x_nm", "y_nm", "z_nm"]].values
            dists = []
            for i in range(k):
                for j in range(i + 1, k):
                    d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                    dists.append(d)

            mean_pw_dist = np.mean(dists)
            max_pw_dist = np.max(dists)

            bi = (nl, int(bid))
            info = branch_info.get(bi, {})

            records.append({
                "neuron_label": nl,
                "branch_idx": int(bid),
                "pre_root_id": int(rid),
                "pre_class": pre_class,
                "k_synapses": k,
                "mean_pairwise_dist_nm": mean_pw_dist,
                "max_pairwise_dist_nm": max_pw_dist,
                "regime": info.get("regime", -1),
                "total_length_nm": info.get("total_length_nm", np.nan),
                "synapse_count": info.get("synapse_count", np.nan),
                "post_subtype": info.get("subtype", ""),
                "soma_distance_nm": info.get("soma_distance_nm", np.nan),
            })

        n_processed += 1
        if n_processed % 100 == 0:
            print(f"  {n_processed}/{len(inh_neurons)} neurons ({time.time()-t0:.0f}s)")

    print(f"  Done: {n_processed} neurons, {len(records)} partner-branch records ({time.time()-t0:.0f}s)")

    if len(records) == 0:
        print("  No multisynaptic typed partners found!")
        return {"error": "no data"}

    pdf = pd.DataFrame(records)
    pdf = pdf[pdf["regime"] >= 0].copy()

    # Report sample sizes
    print("\n  Sample sizes by presynaptic class:")
    for pc in sorted(pdf["pre_class"].unique()):
        sub = pdf[pdf["pre_class"] == pc]
        print(f"    {pc}: {len(sub)} records, {sub.neuron_label.nunique()} neurons, "
              f"median k={sub.k_synapses.median():.0f}")
        for r in [0, 1, 2]:
            rsub = sub[sub["regime"] == r]
            print(f"      regime {r}: n={len(rsub)}, "
                  f"mean_dist={rsub.mean_pairwise_dist_nm.mean()/1000:.1f} um")

    results = {"sample_sizes": {}}

    # Per-class analysis
    for focal_class in ["BC", "MC", "excitatory"]:
        sub = pdf[pdf["pre_class"] == focal_class].copy()
        if len(sub) < 30:
            print(f"\n  {focal_class}: SKIPPED (n={len(sub)})")
            continue

        sub = sub.dropna(subset=["mean_pairwise_dist_nm", "total_length_nm",
                                  "k_synapses", "regime"]).copy()
        n_neurons = sub["neuron_label"].nunique()

        results["sample_sizes"][focal_class] = {
            "n_records": len(sub),
            "n_neurons": n_neurons,
            "by_regime": {
                str(r): int((sub["regime"] == r).sum()) for r in [0, 1, 2]
            },
        }

        if n_neurons < 5:
            print(f"\n  {focal_class}: SKIPPED (neurons={n_neurons})")
            continue

        print(f"\n  --- {focal_class} (n={len(sub)}, neurons={n_neurons}) ---")

        # Log-transform distance for normality
        sub["log_mean_dist"] = np.log1p(sub["mean_pairwise_dist_nm"])

        # Model: placement compactness ~ regime + k + branch_length + (1|neuron)
        try:
            formula = ("log_mean_dist ~ C(regime) + k_synapses + total_length_nm")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(formula, sub, groups=sub["neuron_label"])
                res = model.fit(reml=False)

            r1_c = float(res.params.get("C(regime)[T.1]", np.nan))
            r1_p = float(res.pvalues.get("C(regime)[T.1]", np.nan))
            r2_c = float(res.params.get("C(regime)[T.2]", np.nan))
            r2_p = float(res.pvalues.get("C(regime)[T.2]", np.nan))
            k_c = float(res.params.get("k_synapses", np.nan))
            k_p = float(res.pvalues.get("k_synapses", np.nan))

            results[focal_class] = {
                "placement_model": {
                    "regime1_coef": r1_c, "regime1_p": r1_p,
                    "regime2_coef": r2_c, "regime2_p": r2_p,
                    "k_synapses_coef": k_c, "k_synapses_p": k_p,
                    "n_obs": len(sub), "n_neurons": n_neurons,
                },
                "descriptive": {
                    str(r): {
                        "n": int((sub["regime"] == r).sum()),
                        "mean_dist_um": float(sub.loc[sub["regime"]==r, "mean_pairwise_dist_nm"].mean() / 1000),
                        "median_dist_um": float(sub.loc[sub["regime"]==r, "mean_pairwise_dist_nm"].median() / 1000),
                        "mean_k": float(sub.loc[sub["regime"]==r, "k_synapses"].mean()),
                    } for r in [0, 1, 2]
                },
            }

            print(f"    regime1: coef={r1_c:.4f}, p={r1_p:.2e}")
            print(f"    regime2: coef={r2_c:.4f}, p={r2_p:.2e}")
            print(f"    k_synapses: coef={k_c:.4f}, p={k_p:.2e}")

            # Descriptive
            for r in [0, 1, 2]:
                rsub = sub[sub["regime"] == r]
                if len(rsub) > 0:
                    print(f"    R{r}: n={len(rsub)}, mean_dist={rsub.mean_pairwise_dist_nm.mean()/1000:.2f} um, "
                          f"median_dist={rsub.mean_pairwise_dist_nm.median()/1000:.2f} um, "
                          f"mean_k={rsub.k_synapses.mean():.1f}")

        except Exception as e:
            print(f"    Model FAILED: {str(e)[:80]}")
            results[focal_class] = {"error": str(e)[:200]}

    # Interaction model: regime × presynaptic class
    bc_mc = pdf[pdf["pre_class"].isin(["BC", "MC"])].copy()
    bc_mc = bc_mc.dropna(subset=["mean_pairwise_dist_nm", "total_length_nm",
                                  "k_synapses", "regime"])
    if len(bc_mc) > 50:
        bc_mc["log_mean_dist"] = np.log1p(bc_mc["mean_pairwise_dist_nm"])
        try:
            formula = "log_mean_dist ~ C(regime) * C(pre_class) + k_synapses + total_length_nm"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(formula, bc_mc, groups=bc_mc["neuron_label"])
                res = model.fit(reml=False)

            interaction_results = {}
            for param in res.params.index:
                if "regime" in str(param) or "pre_class" in str(param):
                    interaction_results[str(param)] = {
                        "coef": float(res.params[param]),
                        "p": float(res.pvalues[param]),
                    }

            results["interaction_BC_vs_MC"] = {
                "n_obs": len(bc_mc),
                "n_neurons": bc_mc["neuron_label"].nunique(),
                "coefficients": interaction_results,
            }

            print(f"\n  --- Interaction: regime × pre_class (BC vs MC) ---")
            for param, vals in interaction_results.items():
                sig = "*" if vals["p"] < 0.05 else ""
                print(f"    {param}: coef={vals['coef']:.4f}, p={vals['p']:.2e} {sig}")

        except Exception as e:
            print(f"  Interaction model FAILED: {str(e)[:80]}")

    return results


def robust_neuron_level_targeting(spatial_df, afferent_df):
    """Neuron-level targeting with robust statistics (Suggestion 6).

    Replaces mean-based summaries with:
    - Median log₂ ratio (robust to skew)
    - Wilcoxon signed-rank test
    - Hodges-Lehmann estimator
    - Bootstrap CI by neuron
    - % positive neurons
    """
    print("\n" + "=" * 60)
    print("ROBUST NEURON-LEVEL TARGETING")
    print("=" * 60)

    merged = spatial_df.merge(afferent_df, on=["neuron_label", "branch_idx"],
                              how="inner", suffixes=("", "_aff"))
    inh_merged = merged[merged["cell_type"].str.startswith("inhibitory")].copy()
    inh_merged = inh_merged[inh_merged["regime"] >= 0]

    results = {}

    for aff_class, target_regime, label in [
        ("frac_BC_afferent", 0, "BC_compact"),
        ("frac_MC_afferent", 2, "MC_compartmentalized"),
    ]:
        neuron_ratios = []
        neuron_labels_list = []

        for nl in inh_merged["neuron_label"].unique():
            nl_data = inh_merged[inh_merged["neuron_label"] == nl]
            target = nl_data[nl_data["regime"] == target_regime][aff_class].dropna()
            other = nl_data[nl_data["regime"] != target_regime][aff_class].dropna()

            if len(target) < 1 or len(other) < 1:
                continue

            t_mean = target.mean()
            o_mean = other.mean()

            if o_mean > 0 and t_mean > 0:
                log2_ratio = np.log2(t_mean / o_mean)
            elif t_mean > 0:
                log2_ratio = 5.0  # cap
            elif o_mean > 0:
                log2_ratio = -5.0
            else:
                continue

            neuron_ratios.append(log2_ratio)
            neuron_labels_list.append(nl)

        ratios = np.array(neuron_ratios)
        n = len(ratios)

        if n < 10:
            results[label] = {"n_neurons": n, "insufficient": True}
            continue

        # Wilcoxon signed-rank test
        w_stat, w_p = stats.wilcoxon(ratios, alternative='two-sided')

        # Hodges-Lehmann estimator (median of pairwise means)
        walsh = (ratios[:, None] + ratios[None, :]) / 2
        hl = float(np.median(walsh[np.triu_indices(n)]))

        # Bootstrap CI for median
        n_boot = 1000
        boot_medians = np.array([np.median(RNG.choice(ratios, size=n, replace=True))
                                 for _ in range(n_boot)])
        ci_lo, ci_hi = np.percentile(boot_medians, [2.5, 97.5])

        # By postsynaptic subtype
        subtype_map = {}
        for nl_i, nl in enumerate(neuron_labels_list):
            st = inh_merged[inh_merged["neuron_label"] == nl]["subtype"].iloc[0]
            if st not in subtype_map:
                subtype_map[st] = []
            subtype_map[st].append(neuron_ratios[nl_i])

        subtype_summary = {}
        for st, vals_list in subtype_map.items():
            v = np.array(vals_list)
            subtype_summary[st] = {
                "n_neurons": len(v),
                "pct_positive": float((v > 0).mean() * 100),
                "median_log2_ratio": float(np.median(v)),
            }

        results[label] = {
            "n_neurons": n,
            "median_log2_ratio": float(np.median(ratios)),
            "mean_log2_ratio": float(np.mean(ratios)),
            "pct_positive": float((ratios > 0).mean() * 100),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p),
            "hodges_lehmann": float(hl),
            "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
            "by_postsynaptic_subtype": subtype_summary,
        }

        print(f"\n  {label} (n={n}):")
        print(f"    {(ratios > 0).mean()*100:.1f}% positive, "
              f"median log2={np.median(ratios):.3f}, "
              f"HL={hl:.3f}")
        print(f"    Wilcoxon p={w_p:.2e}, "
              f"95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]")

    return results


def compartment_within_regime(spatial_df, afferent_df):
    """Test BC targeting within dendritic compartments (Suggestion 5).

    Splits branches into proximal/distal using per-neuron soma_distance_nm
    median. Within each compartment, tests whether BC fraction still varies
    with regime. If it does, targeting is electrotonic (not just
    compartmental) — stronger than Schneider-Mizell 2023.

    Also repeats with branch_order tertiles as sensitivity check.
    """
    print("\n" + "=" * 60)
    print("COMPARTMENT-WITHIN-REGIME STRATIFICATION")
    print("=" * 60)

    merged = spatial_df.merge(afferent_df, on=["neuron_label", "branch_idx"],
                              how="inner", suffixes=("", "_aff"))
    inh_merged = merged[merged["cell_type"].str.startswith("inhibitory")].copy()
    inh_merged = inh_merged[inh_merged["regime"] >= 0]

    results = {}

    # ── Method 1: proximal/distal by per-neuron soma_distance_nm median ──
    print("\n  Method 1: soma_distance_nm median split")
    compartments = pd.Series("unknown", index=inh_merged.index)
    for nl in inh_merged["neuron_label"].unique():
        mask = inh_merged["neuron_label"] == nl
        dists = inh_merged.loc[mask, "soma_distance_nm"]
        valid = dists.notna()
        if valid.sum() < 2:
            continue
        med = dists[valid].median()
        compartments.loc[mask & (dists <= med)] = "proximal"
        compartments.loc[mask & (dists > med)] = "distal"

    inh_merged["compartment"] = compartments.values
    work_comp = inh_merged[inh_merged["compartment"] != "unknown"]

    # Cross-tab summary
    print("\n  Cross-tab: compartment × regime × BC fraction")
    for comp in ["proximal", "distal"]:
        for r in [0, 1, 2]:
            mask = (work_comp["compartment"] == comp) & (work_comp["regime"] == r)
            vals = work_comp.loc[mask, "frac_BC_afferent"].dropna()
            if len(vals) > 0:
                print(f"    {comp}/R{r}: BC={vals.mean():.3f}, "
                      f"MC={work_comp.loc[mask, 'frac_MC_afferent'].dropna().mean():.3f} "
                      f"(n={len(vals)})")

    results["soma_distance_split"] = {}

    for compartment in ["proximal", "distal"]:
        comp_data = work_comp[work_comp["compartment"] == compartment]
        n_branches = len(comp_data)
        n_neurons = int(comp_data["neuron_label"].nunique())

        print(f"\n  --- {compartment} ({n_branches} branches, {n_neurons} neurons) ---")

        comp_result = {"n_branches": n_branches, "n_neurons": n_neurons}

        # Mixed model: afferent fraction ~ regime within compartment
        for aff_col in ["frac_BC_afferent", "frac_MC_afferent"]:
            work = comp_data.dropna(subset=[aff_col, "regime", "neuron_label",
                                            "synapse_count", "total_length_nm"]).copy()
            work = work[work["regime"] >= 0]

            if len(work) < 50 or work["neuron_label"].nunique() < 5:
                print(f"    {aff_col}: insufficient (n={len(work)})")
                continue

            try:
                formula = f"{aff_col} ~ C(regime) + synapse_count + total_length_nm"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = smf.mixedlm(formula, work, groups=work["neuron_label"])
                    res = model.fit(reml=True, maxiter=500)

                r1_c = float(res.params.get("C(regime)[T.1]", np.nan))
                r1_p = float(res.pvalues.get("C(regime)[T.1]", np.nan))
                r2_c = float(res.params.get("C(regime)[T.2]", np.nan))
                r2_p = float(res.pvalues.get("C(regime)[T.2]", np.nan))

                comp_result[f"{aff_col}_model"] = {
                    "regime1_coef": r1_c, "regime1_p": r1_p,
                    "regime2_coef": r2_c, "regime2_p": r2_p,
                    "n_obs": len(work),
                    "n_neurons": int(work["neuron_label"].nunique()),
                }
                print(f"    {aff_col}: R1={r1_c:.4f} (p={r1_p:.2e}), "
                      f"R2={r2_c:.4f} (p={r2_p:.2e})")
            except Exception as e:
                print(f"    {aff_col}: FAILED ({str(e)[:50]})")

        # Spatial metrics within compartment (does regime coupling persist?)
        for metric in ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]:
            work = comp_data.dropna(subset=[metric, "regime", "neuron_label",
                                            "synapse_count", "total_length_nm"]).copy()
            work = work[work["regime"] >= 0]
            if len(work) < 50 or work["neuron_label"].nunique() < 5:
                continue
            try:
                formula = f"{metric} ~ C(regime) + synapse_count + total_length_nm"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = smf.mixedlm(formula, work, groups=work["neuron_label"])
                    res = model.fit(reml=True, maxiter=500)
                r2_c = float(res.params.get("C(regime)[T.2]", np.nan))
                r2_p = float(res.pvalues.get("C(regime)[T.2]", np.nan))
                comp_result[f"{metric}_regime_effect"] = {
                    "regime2_coef": r2_c, "regime2_p": r2_p,
                    "n_obs": len(work),
                }
            except Exception:
                pass

        results["soma_distance_split"][compartment] = comp_result

    # ── Method 2: branch_order tertiles ──
    print("\n  Method 2: branch_order tertile split")
    bo = inh_merged["branch_order"]
    if bo.notna().sum() > 100:
        tertiles = bo.quantile([1/3, 2/3])
        t1, t2 = tertiles.iloc[0], tertiles.iloc[1]
        bo_comp = pd.Series("mid", index=inh_merged.index)
        bo_comp[bo <= t1] = "low_order"
        bo_comp[bo > t2] = "high_order"
        inh_merged["bo_compartment"] = bo_comp.values

        results["branch_order_split"] = {}

        for bo_label in ["low_order", "high_order"]:
            comp_data = inh_merged[inh_merged["bo_compartment"] == bo_label]
            n_branches = len(comp_data)
            print(f"\n  --- {bo_label} ({n_branches} branches) ---")

            comp_result = {"n_branches": n_branches}

            for aff_col in ["frac_BC_afferent", "frac_MC_afferent"]:
                work = comp_data.dropna(subset=[aff_col, "regime", "neuron_label",
                                                "synapse_count", "total_length_nm"]).copy()
                work = work[work["regime"] >= 0]

                if len(work) < 50 or work["neuron_label"].nunique() < 5:
                    continue

                try:
                    formula = f"{aff_col} ~ C(regime) + synapse_count + total_length_nm"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = smf.mixedlm(formula, work, groups=work["neuron_label"])
                        res = model.fit(reml=True, maxiter=500)

                    r2_c = float(res.params.get("C(regime)[T.2]", np.nan))
                    r2_p = float(res.pvalues.get("C(regime)[T.2]", np.nan))

                    comp_result[f"{aff_col}_model"] = {
                        "regime2_coef": r2_c, "regime2_p": r2_p,
                        "n_obs": len(work),
                    }
                    print(f"    {aff_col}: R2={r2_c:.4f} (p={r2_p:.2e})")
                except Exception as e:
                    print(f"    {aff_col}: FAILED ({str(e)[:50]})")

            results["branch_order_split"][bo_label] = comp_result

    return results


def bootstrap_mediation_ci(spatial_df, afferent_df, n_boot=200):
    """Bootstrap CIs for mediation percentages (Suggestion 2).

    Cluster bootstrap by neuron. Uses OLS (not mixed model) for speed,
    which is valid since the bootstrap itself captures clustering.
    Reports 95% CIs around M1/M2/M3 mediation percentages.
    """
    print("\n" + "=" * 60)
    print("BOOTSTRAP MEDIATION CIs")
    print("=" * 60)

    partner_path = RESULTS_DIR / "partner_architecture_features.csv"
    if not partner_path.exists():
        print("  ERROR: partner_architecture_features.csv not found")
        return {"error": "missing partner architecture features"}

    partner_df = pd.read_csv(partner_path)

    merged = spatial_df.merge(afferent_df, on=["neuron_label", "branch_idx"],
                              how="left", suffixes=("", "_aff"))
    partner_cols = ["neuron_label", "branch_idx", "unique_partners",
                    "mean_syn_per_partner", "gini_partner", "frac_multisynaptic"]
    merged = merged.merge(partner_df[partner_cols].drop_duplicates(),
                          on=["neuron_label", "branch_idx"], how="left")

    inh_merged = merged[merged["cell_type"].str.startswith("inhibitory")].copy()
    inh_merged = inh_merged[inh_merged["regime"] >= 0]

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    aff_covs = ["frac_inh_afferent", "frac_BC_afferent", "frac_MC_afferent"]
    part_covs = ["unique_partners", "mean_syn_per_partner",
                 "gini_partner", "frac_multisynaptic"]

    all_covs = metrics + ["regime", "neuron_label", "synapse_count",
                          "total_length_nm"] + aff_covs + part_covs
    work = inh_merged.dropna(subset=all_covs).copy()
    work = work[work["regime"] >= 0]
    neurons = work["neuron_label"].unique()

    print(f"  Working data: {len(work)} branches, {len(neurons)} neurons")

    # Pre-index by neuron for fast bootstrap sampling
    neuron_idx_map = {nl: work.index[work["neuron_label"] == nl].tolist()
                      for nl in neurons}

    results = {}

    aff_terms = " + ".join(aff_covs)
    part_terms = " + ".join(part_covs)

    for metric in metrics:
        t0 = time.time()
        boot_med = {"M1": [], "M2": [], "M3": []}

        f_m0 = f"{metric} ~ C(regime) + synapse_count + total_length_nm"
        f_m1 = f_m0 + f" + {aff_terms}"
        f_m2 = f_m0 + f" + {part_terms}"
        f_m3 = f_m0 + f" + {aff_terms} + {part_terms}"

        for b in range(n_boot):
            boot_neurons = RNG.choice(neurons, size=len(neurons), replace=True)
            boot_idx = []
            for bn in boot_neurons:
                boot_idx.extend(neuron_idx_map[bn])
            boot_sample = work.loc[boot_idx].copy().reset_index(drop=True)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    c0 = smf.ols(f_m0, boot_sample).fit().params.get(
                        "C(regime)[T.2]", np.nan)
                    if abs(c0) < 1e-10 or np.isnan(c0):
                        continue
                    c1 = smf.ols(f_m1, boot_sample).fit().params.get(
                        "C(regime)[T.2]", np.nan)
                    c2 = smf.ols(f_m2, boot_sample).fit().params.get(
                        "C(regime)[T.2]", np.nan)
                    c3 = smf.ols(f_m3, boot_sample).fit().params.get(
                        "C(regime)[T.2]", np.nan)

                boot_med["M1"].append((1 - abs(c1) / abs(c0)) * 100)
                boot_med["M2"].append((1 - abs(c2) / abs(c0)) * 100)
                boot_med["M3"].append((1 - abs(c3) / abs(c0)) * 100)
            except Exception:
                continue

        elapsed = time.time() - t0
        metric_result = {}

        for m in ["M1", "M2", "M3"]:
            vals = np.array(boot_med[m])
            if len(vals) >= 20:
                ci_lo, ci_hi = np.percentile(vals, [2.5, 97.5])
                metric_result[m] = {
                    "n_successful": int(len(vals)),
                    "median": float(np.median(vals)),
                    "ci_2.5": float(ci_lo),
                    "ci_97.5": float(ci_hi),
                }
                print(f"  {metric} {m}: {np.median(vals):.1f}% "
                      f"[{ci_lo:.1f}, {ci_hi:.1f}] "
                      f"({len(vals)}/{n_boot})")

        results[metric] = metric_result
        print(f"    ({elapsed:.0f}s)")

    return results


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    t0_total = time.time()

    # Load branch features
    spatial_df = pd.read_csv(RESULTS_DIR / "all_branch_features_with_spatial.csv")
    print(f"Loaded {len(spatial_df)} branches, {spatial_df['neuron_label'].nunique()} neurons\n")

    supplementary_only = "--supplementary" in sys.argv

    if supplementary_only:
        # Load saved afferent features instead of recomputing
        aug_path = RESULTS_DIR / "afferent_branch_features.csv"
        if not aug_path.exists():
            print("ERROR: afferent_branch_features.csv not found. Run full pipeline first.")
            return
        augmented = pd.read_csv(aug_path)
        aff_cols = ["neuron_label", "branch_idx", "frac_exc_afferent",
                     "frac_inh_afferent", "frac_BC_afferent", "frac_MC_afferent",
                     "frac_BPC_afferent", "frac_NGC_afferent", "n_typed_synapses",
                     "n_subtyped_synapses", "n_total_synapses_matched"]
        avail = [c for c in aff_cols if c in augmented.columns]
        afferent_df = augmented[avail].copy()
        # Add regime from spatial_df
        regime_map = dict(zip(
            zip(spatial_df["neuron_label"], spatial_df["branch_idx"].astype(int)),
            spatial_df["regime"].astype(int),
        ))
        afferent_df["regime"] = [
            regime_map.get((nl, int(bi)), -1)
            for nl, bi in zip(afferent_df["neuron_label"],
                              afferent_df["branch_idx"])
        ]
        print("Loaded saved afferent features (supplementary mode)\n")
    else:
        # ── Stage 0a: CAVE coverage boost ──
        catalog_lookup, cave_stats = cave_coverage_boost(spatial_df, MICRONS_DATA)

        # ── Stage 0b: Coverage gate ──
        gate_results = coverage_gate(spatial_df, MICRONS_DATA, catalog_lookup)

        # Save Stage 0 results
        gate_output = {
            "cave_boost": cave_stats,
            "coverage_gate": gate_results,
        }
        with open(RESULTS_DIR / "afferent_coverage_gate.json", "w") as f:
            json.dump(gate_output, f, indent=2, default=str)
        print(f"\nSaved afferent_coverage_gate.json")

        # ── Compute afferent features ──
        afferent_df, synapse_df = compute_afferent_features(
            spatial_df, MICRONS_DATA, catalog_lookup)

        if len(afferent_df) == 0:
            print("ERROR: No afferent features computed. Aborting.")
            return

        # Save augmented branch features
        augmented = spatial_df.merge(afferent_df, on=["neuron_label", "branch_idx"],
                                     how="left")
        augmented.to_csv(RESULTS_DIR / "afferent_branch_features.csv", index=False)
        print(f"Saved afferent_branch_features.csv ({len(augmented)} rows)")

        # ── Stage 1: Targeting analysis ──
        targeting_results = targeting_analysis(spatial_df, afferent_df, synapse_df)

        with open(RESULTS_DIR / "afferent_targeting.json", "w") as f:
            json.dump(targeting_results, f, indent=2, default=str)
        print(f"\nSaved afferent_targeting.json")

        # ── Stage 2a: Placement mediation ──
        mediation_results = placement_mediation(spatial_df, afferent_df)

        with open(RESULTS_DIR / "afferent_mediation.json", "w") as f:
            json.dump(mediation_results, f, indent=2, default=str)
        print(f"Saved afferent_mediation.json")

        # ── Stage 2b: Class-specific placement ──
        placement_results = class_specific_placement(
            spatial_df, afferent_df, mediation_results)

        with open(RESULTS_DIR / "afferent_class_placement.json", "w") as f:
            json.dump(placement_results, f, indent=2, default=str)
        print(f"Saved afferent_class_placement.json")

        # ── Morphology controls ──
        morph_results = morphology_controls(spatial_df)

        mediation_output = json.load(open(RESULTS_DIR / "afferent_mediation.json"))
        mediation_output["morphology_controls"] = morph_results
        with open(RESULTS_DIR / "afferent_mediation.json", "w") as f:
            json.dump(mediation_output, f, indent=2, default=str)
        print(f"Updated afferent_mediation.json with morphology controls")

    # ── Supplementary analyses ──
    robust_results = robust_neuron_level_targeting(spatial_df, afferent_df)
    comp_results = compartment_within_regime(spatial_df, afferent_df)
    boot_results = bootstrap_mediation_ci(spatial_df, afferent_df, n_boot=200)

    # Build catalog lookup for axon-conditioned analysis
    if supplementary_only:
        cat_path = MICRONS_DATA / ".." / "catalog.csv"
        catalog_lookup = _build_catalog_lookup(cat_path) if cat_path.exists() else None
    axon_results = axon_conditioned_placement(spatial_df, MICRONS_DATA, catalog_lookup)
    with open(RESULTS_DIR / "afferent_axon_placement.json", "w") as f:
        json.dump(axon_results, f, indent=2, default=str)
    print("Saved afferent_axon_placement.json")

    supp_output = {
        "robust_neuron_level_targeting": robust_results,
        "compartment_within_regime": comp_results,
        "bootstrap_mediation_ci": boot_results,
    }
    with open(RESULTS_DIR / "afferent_supplementary.json", "w") as f:
        json.dump(supp_output, f, indent=2, default=str)
    print(f"\nSaved afferent_supplementary.json")

    elapsed = time.time() - t0_total
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
