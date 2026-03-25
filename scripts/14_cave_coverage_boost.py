#!/usr/bin/env python3
"""CAVE coverage boost for presynaptic partner subtype labels.

Previous coverage gate only reached 18% overall / 43% multi-synapse (k≥3).
This script queries additional CAVE tables not previously used:
  - aibs_metamodel_mtypes_v661_v2: has PTC (perisomatic→BC), DTC (dendrite→MC)
  - synapse_target_structure (Celii et al.): per-synapse postsynaptic compartment
    (0=spine head, 2=shaft, 6=soma) — use as structural proxy for presynaptic identity
  - baylor_gnn_cell_type_fine_model_v2: fine excitatory subtypes (49K neurons)

Mapping:
  mtypes PTC → BC (perisomatic-targeting cell)
  mtypes DTC → MC (dendrite-targeting cell)
  mtypes STC → BPC (spine-targeting, VIP-like)
  mtypes ITC → unknown (insufficient specificity)
  synapse_target_structure: pre-axon whose synapses ≥50% soma/AIS → BC-proximal

Output:
  results/replication_full/cave_boost_v2.json
  results/replication_full/presynaptic_subtype_boosted.csv  ← root_id → subtype
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "neurons"
RESULTS_DIR = PROJECT_DIR / "results" / "replication_full"

sys.path.insert(0, str(Path.home() / "research" / "neurostat"))


# ── mtype → subtype mapping ────────────────────────────────────────────────

MTYPE_TO_SUBTYPE = {
    # inhibitory morphological types → our proxy categories
    "PTC": "BC",   # perisomatic-targeting cell ≈ BC/PV-like
    "DTC": "MC",   # dendrite-targeting cell ≈ MC/SST-like
    "STC": "BPC",  # spine-targeting cell ≈ VIP-like (uncertain)
    "ITC": None,   # insufficient specificity
    # excitatory morphological types → broad excitatory
    "L2a": "excitatory", "L2b": "excitatory", "L2c": "excitatory",
    "L3a": "excitatory", "L3b": "excitatory",
    "L4a": "excitatory", "L4b": "excitatory", "L4c": "excitatory",
    "L5ET": "5PET", "L5NP": "5PNP", "L5a": "5PIT", "L5b": "5PIT",
    "L6short-a": "6PCT", "L6short-b": "6PCT",
    "L6tall-a": "6PIT", "L6tall-b": "6PIT", "L6tall-c": "6PIT",
}

# Confidence level: mtypes PTC/DTC are morphological proxies, not molecular identity
# We flag these as L2 (boosted) with source='mtypes_PTC/DTC'


def collect_all_presynaptic_root_ids():
    """Collect all unique presynaptic root IDs from partner CSVs."""
    print("  Loading presynaptic partner CSVs...")
    partner_files = list(DATA_DIR.glob("*_presynaptic.csv"))
    print(f"  {len(partner_files)} partner files")

    all_ids = set()
    cell_type_map = {}  # root_id → (subtype, broad_class, source)

    sample_df = None
    for f in partner_files:
        try:
            df = pd.read_csv(f)
            if sample_df is None:
                sample_df = df
            for _, row in df.iterrows():
                rid = row.get('pre_root_id')
                if pd.isna(rid):
                    continue
                rid = int(rid)
                all_ids.add(rid)
                # Current annotation
                cell_type = str(row.get('pre_cell_type', '')).strip()
                subtype, broad = _classify_subtype_from_csv(cell_type)
                if rid not in cell_type_map or cell_type_map[rid][0] is None:
                    cell_type_map[rid] = (subtype, broad, 'partner_csv')
        except Exception:
            continue

    if sample_df is not None:
        print(f"  Sample CSV columns: {sample_df.columns.tolist()[:8]}")
    return all_ids, cell_type_map


def _classify_subtype_from_csv(cell_type_str):
    """Map a pre_cell_type string to (subtype, broad_class)."""
    if not isinstance(cell_type_str, str) or cell_type_str in ('unknown', '', 'nan'):
        return None, 'unknown'
    s = cell_type_str.lower().strip()
    # Fine subtypes
    if 'basket' in s or s in ('bc', 'inh_bc', 'inhibitory_bc', 'bca', 'bcb', 'bcc'):
        return 'BC', 'inhibitory'
    if 'martinotti' in s or s in ('mc', 'inh_mc', 'inhibitory_mc'):
        return 'MC', 'inhibitory'
    if 'bipolar' in s or 'bpc' in s or 'vip' in s:
        return 'BPC', 'inhibitory'
    if 'neurogliaform' in s or 'ngc' in s or 'lamp5' in s:
        return 'NGC', 'inhibitory'
    if any(x in s for x in ['inhibitory', 'inh_']):
        return None, 'inhibitory'
    if any(x in s for x in ['excitatory', 'exc_', '23p', '4p', '5p', '6p', 'pyramidal']):
        sub = None
        if '23p' in s: sub = '23P'
        elif '4p' in s: sub = '4P'
        elif '5p-et' in s or '5pet' in s: sub = '5PET'
        elif '5p-it' in s or '5pit' in s: sub = '5PIT'
        elif '5p-np' in s or '5pnp' in s: sub = '5PNP'
        elif '6p-ct' in s or '6pct' in s: sub = '6PCT'
        elif '6p-it' in s or '6pit' in s: sub = '6PIT'
        return sub, 'excitatory'
    return None, 'unknown'


def query_mtypes_table(unknown_ids, client):
    """Query aibs_metamodel_mtypes_v661_v2 for unknown root IDs."""
    print(f"\n  Querying aibs_metamodel_mtypes_v661_v2 for {len(unknown_ids):,} unknown IDs...")
    try:
        # Query full table (72K rows — manageable)
        df = client.materialize.query_table(
            'aibs_metamodel_mtypes_v661_v2',
            select_columns=['pt_root_id', 'cell_type', 'classification_system']
        )
        df = df.rename(columns={'pt_root_id': 'root_id'})
        df['root_id'] = df['root_id'].astype(int)
        print(f"  Table size: {len(df):,} rows")

        # Filter to unknowns
        matches = df[df['root_id'].isin(unknown_ids)]
        print(f"  Matches in unknown set: {len(matches):,}")

        # Map to subtypes
        results = {}
        for _, row in matches.iterrows():
            rid = int(row['root_id'])
            mtype = str(row['cell_type'])
            subtype = MTYPE_TO_SUBTYPE.get(mtype)
            broad = 'inhibitory' if row['classification_system'] == 'inhibitory_neuron' else 'excitatory'
            if subtype is not None:
                results[rid] = (subtype, broad, f'mtypes_{mtype}')

        # Summary
        from collections import Counter
        by_mtype = Counter()
        for _, row in matches.iterrows():
            by_mtype[row['cell_type']] += 1
        print(f"  mtype distribution in matched unknowns:")
        for mt, n in by_mtype.most_common():
            mapped = MTYPE_TO_SUBTYPE.get(mt, '?')
            print(f"    {mt:15s} → {str(mapped):6s}  n={n:,}")

        return results, df

    except Exception as e:
        print(f"  ERROR: {e}")
        return {}, pd.DataFrame()


def query_synapse_target_structure(inh_postsynaptic_ids, inh_partner_ids, client):
    """Use synapse_target_structure to classify currently-unknown inhibitory presynaptic partners.

    For each unknown inhibitory presynaptic partner: what fraction of its synapses
    onto our inhibitory postsynaptic neurons land on soma (value=6) vs shaft (value=2)
    vs spine (value=0,1,3)?

    Classification rule (based on Celii et al. perisomatic classification):
      >40% soma/AIS target → BC-proximal (perisomatic)
      >50% shaft + <20% soma → MC-proximal (dendrite-targeting)
      >50% spine → excitatory-proximal

    NOTE: This is a structural proxy, not a direct molecular ID. Labeled L2-structural.
    """
    print(f"\n  Querying synapse_target_structure for {len(inh_partner_ids):,} unknown inh partners...")
    print(f"  (filtering to synapses onto {len(inh_postsynaptic_ids):,} inhibitory postsynaptic neurons)")

    if len(inh_partner_ids) == 0:
        return {}

    try:
        # Filter by our postsynaptic neuron set to keep query manageable
        # Query in batches since the table is very large
        BATCH = 50  # postsynaptic neurons per batch
        inh_post_list = list(inh_postsynaptic_ids)
        partner_target_counts = {}  # pre_root_id → {val: count}

        n_batches = (len(inh_post_list) + BATCH - 1) // BATCH
        print(f"  Querying in {n_batches} batches of {BATCH} postsynaptic neurons...")

        for bi, i in enumerate(range(0, len(inh_post_list), BATCH)):
            batch_post = inh_post_list[i:i+BATCH]
            if bi % 10 == 0:
                print(f"    Batch {bi+1}/{n_batches}...", flush=True)
            try:
                df = client.materialize.query_table(
                    'synapse_target_structure',
                    filter_in_dict={'post_pt_root_id': batch_post},
                    select_columns=['pre_pt_root_id', 'post_pt_root_id', 'value']
                )
                # Filter to unknown inhibitory presynaptic partners
                df['pre_pt_root_id'] = df['pre_pt_root_id'].astype(int)
                df = df[df['pre_pt_root_id'].isin(inh_partner_ids)]

                for _, row in df.iterrows():
                    pre = int(row['pre_pt_root_id'])
                    val = int(row['value'])
                    if pre not in partner_target_counts:
                        partner_target_counts[pre] = {}
                    partner_target_counts[pre][val] = partner_target_counts[pre].get(val, 0) + 1
            except Exception as e:
                if bi == 0:
                    print(f"  Batch query failed: {e}")
                continue

        print(f"  Got target structure data for {len(partner_target_counts):,} partners")

        # Classify based on target structure distribution
        SOMA_VALS = {6}           # soma
        SHAFT_VALS = {2}          # dendritic shaft
        SPINE_VALS = {0, 1, 3}    # spine variants

        results = {}
        for pre_id, val_counts in partner_target_counts.items():
            total = sum(val_counts.values())
            if total < 3:
                continue
            soma_frac = sum(val_counts.get(v, 0) for v in SOMA_VALS) / total
            shaft_frac = sum(val_counts.get(v, 0) for v in SHAFT_VALS) / total
            spine_frac = sum(val_counts.get(v, 0) for v in SPINE_VALS) / total

            if soma_frac >= 0.40:
                results[pre_id] = ('BC', 'inhibitory', 'synapse_target_struct_soma')
            elif shaft_frac >= 0.50 and soma_frac < 0.20:
                results[pre_id] = ('MC', 'inhibitory', 'synapse_target_struct_shaft')
            elif spine_frac >= 0.50:
                results[pre_id] = (None, 'excitatory', 'synapse_target_struct_spine')

        n_bc = sum(1 for v in results.values() if v[0] == 'BC')
        n_mc = sum(1 for v in results.values() if v[0] == 'MC')
        n_exc = sum(1 for v in results.values() if v[1] == 'excitatory')
        print(f"  Classified via target structure: BC={n_bc:,}, MC={n_mc:,}, exc-proximal={n_exc:,}")
        return results

    except Exception as e:
        print(f"  ERROR in synapse_target_structure query: {e}")
        return {}


def compute_coverage_improvement(cell_type_map, boosted_map, partner_files):
    """Compute before/after coverage statistics."""
    stats = {'before': {}, 'after': {}}

    for phase, use_boost in [('before', False), ('after', True)]:
        n_total = 0
        n_typed = 0
        n_subtyped = 0
        n_inh_subtyped = 0

        for f in partner_files:
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            for _, row in df.iterrows():
                rid = row.get('pre_root_id')
                if pd.isna(rid):
                    continue
                rid = int(rid)
                n_total += 1

                # Get current type
                if use_boost and rid in boosted_map:
                    subtype, broad, _ = boosted_map[rid]
                elif rid in cell_type_map:
                    subtype, broad, _ = cell_type_map[rid]
                else:
                    subtype, broad = None, 'unknown'

                if broad != 'unknown':
                    n_typed += 1
                if subtype is not None:
                    n_subtyped += 1
                    if subtype in ('BC', 'MC', 'BPC', 'NGC'):
                        n_inh_subtyped += 1

        stats[phase] = {
            'n_total': n_total,
            'typed_pct': 100 * n_typed / n_total if n_total else 0,
            'subtyped_pct': 100 * n_subtyped / n_total if n_total else 0,
            'inh_subtyped_pct': 100 * n_inh_subtyped / n_total if n_total else 0,
        }

    return stats


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CAVE COVERAGE BOOST v2")
    print("=" * 60)

    from caveclient import CAVEclient
    client = CAVEclient('minnie65_public')

    # ── 1. Collect all presynaptic partner root IDs ────────────────────────
    print("\n[1] Collecting presynaptic partner root IDs...")
    partner_files = list(DATA_DIR.glob("*_presynaptic.csv"))
    all_ids, cell_type_map = collect_all_presynaptic_root_ids()
    print(f"  Total unique presynaptic root IDs: {len(all_ids):,}")

    # Partition into typed vs unknown
    unknown_ids = {rid for rid, (sub, broad, src) in cell_type_map.items()
                   if sub is None and broad == 'unknown'}
    broad_only_ids = {rid for rid, (sub, broad, src) in cell_type_map.items()
                      if sub is None and broad in ('inhibitory', 'excitatory')}
    subtyped_ids = {rid for rid, (sub, broad, src) in cell_type_map.items()
                    if sub is not None}

    print(f"  Subtyped (L1): {len(subtyped_ids):,}")
    print(f"  Broad class only (L3): {len(broad_only_ids):,}")
    print(f"  Unknown (L4): {len(unknown_ids):,}")

    # IDs that could benefit from boost (unknown + broad-only inhibitory)
    inh_broad_ids = {rid for rid, (sub, broad, src) in cell_type_map.items()
                     if sub is None and broad == 'inhibitory'}
    boost_target_ids = unknown_ids | broad_only_ids

    print(f"\n  Boost targets: {len(boost_target_ids):,} (unknown + broad-only)")
    print(f"  Inhibitory broad-only: {len(inh_broad_ids):,}")

    # ── 2. Query aibs_metamodel_mtypes_v661_v2 ────────────────────────────
    print("\n[2] Boosting via morphological type table (PTC/DTC/STC)...")
    mtypes_boost, mtypes_df = query_mtypes_table(boost_target_ids, client)
    print(f"  New subtype assignments from mtypes: {len(mtypes_boost):,}")

    # ── 3. Identify remaining unknown inhibitory partners ──────────────────
    print("\n[3] Identifying remaining unknown inhibitory partners for structural classification...")

    # After mtypes boost, what inhibitory partners still lack a subtype?
    remaining_unknown_inh = set()
    for rid in inh_broad_ids:
        if rid not in mtypes_boost:
            remaining_unknown_inh.add(rid)
    print(f"  Remaining unknown inhibitory partners: {len(remaining_unknown_inh):,}")

    # Get inhibitory postsynaptic neuron IDs (from neuron labels)
    branch_df = pd.read_csv(RESULTS_DIR / "all_branch_features_with_spatial.csv")
    inh_labels = branch_df[branch_df['cell_type'] == 'inhibitory']['neuron_label'].unique()
    # Get root IDs for those labels
    inh_root_ids = set()
    for lbl in inh_labels:
        parts = lbl.split('_')
        for p in parts:
            if p.isdigit() and len(p) > 10:
                inh_root_ids.add(int(p))
    print(f"  Inhibitory postsynaptic neurons: {len(inh_root_ids):,}")

    # ── 4. Query synapse_target_structure ─────────────────────────────────
    print("\n[4] Classifying via synapse target structure (Celii et al.)...")
    struct_boost = {}
    if len(remaining_unknown_inh) > 0 and len(inh_root_ids) > 0:
        struct_boost = query_synapse_target_structure(
            inh_root_ids, remaining_unknown_inh, client
        )
    else:
        print("  Skipped (no remaining unknowns or no postsynaptic neurons found)")

    # ── 5. Merge boosted labels ────────────────────────────────────────────
    print("\n[5] Merging boosted labels...")
    boosted_map = dict(cell_type_map)  # copy original

    n_new_mtypes = 0
    n_new_struct = 0

    for rid, (sub, broad, src) in mtypes_boost.items():
        old_sub, old_broad, _ = boosted_map.get(rid, (None, 'unknown', 'none'))
        if old_sub is None and sub is not None:
            boosted_map[rid] = (sub, broad, src)
            n_new_mtypes += 1

    for rid, (sub, broad, src) in struct_boost.items():
        old_sub, old_broad, _ = boosted_map.get(rid, (None, 'unknown', 'none'))
        if old_sub is None:
            boosted_map[rid] = (sub, broad, src)
            n_new_struct += 1

    print(f"  New labels from mtypes:          {n_new_mtypes:,}")
    print(f"  New labels from synapse target:  {n_new_struct:,}")
    print(f"  Total newly assigned subtypes:   {n_new_mtypes + n_new_struct:,}")

    # ── 6. Coverage statistics ─────────────────────────────────────────────
    print("\n[6] Computing coverage improvement...")
    # Fast version: use the collected cell_type_map and boosted_map
    n_total = len(all_ids)

    def count_coverage(cmap):
        n_typed = sum(1 for sub, broad, _ in cmap.values() if broad != 'unknown')
        n_subtyped = sum(1 for sub, broad, _ in cmap.values() if sub is not None)
        n_inh_sub = sum(1 for sub, broad, _ in cmap.values()
                        if sub in ('BC', 'MC', 'BPC', 'NGC'))
        return n_typed, n_subtyped, n_inh_sub

    n_typed_before, n_sub_before, n_inh_before = count_coverage(cell_type_map)
    n_typed_after, n_sub_after, n_inh_after = count_coverage(boosted_map)

    print(f"\n  Coverage (unique root IDs, n={n_total:,}):")
    print(f"  {'Metric':<30} {'Before':>10} {'After':>10} {'Gain':>10}")
    print(f"  {'-'*62}")
    for label, before, after in [
        ('Typed (broad class known)', n_typed_before, n_typed_after),
        ('Subtyped (fine label known)', n_sub_before, n_sub_after),
        ('Inhibitory subtyped', n_inh_before, n_inh_after),
    ]:
        b_pct = 100 * before / n_total
        a_pct = 100 * after / n_total
        print(f"  {label:<30} {b_pct:>9.1f}% {a_pct:>9.1f}% {a_pct-b_pct:>+9.1f}%")

    # ── 7. Save boosted lookup table ──────────────────────────────────────
    print("\n[7] Saving boosted subtype lookup...")

    records = []
    for rid, (sub, broad, src) in boosted_map.items():
        records.append({
            'pre_root_id': rid,
            'subtype': sub if sub else '',
            'broad_class': broad,
            'source': src,
        })
    out_df = pd.DataFrame(records)
    out_df.to_csv(RESULTS_DIR / "presynaptic_subtype_boosted.csv", index=False)
    print(f"  Saved {len(out_df):,} rows to presynaptic_subtype_boosted.csv")

    # Summary by source
    print("\n  Label source breakdown (subtyped entries only):")
    subtyped = out_df[out_df['subtype'] != '']
    for src, grp in subtyped.groupby('source'):
        types = grp['subtype'].value_counts().to_dict()
        print(f"    {src:<40} n={len(grp):,}  types={types}")

    # ── 8. Save JSON summary ──────────────────────────────────────────────
    output = {
        'n_unique_presynaptic_partners': n_total,
        'coverage_before': {
            'typed_pct': round(100 * n_typed_before / n_total, 2),
            'subtyped_pct': round(100 * n_sub_before / n_total, 2),
            'inh_subtyped_pct': round(100 * n_inh_before / n_total, 2),
        },
        'coverage_after': {
            'typed_pct': round(100 * n_typed_after / n_total, 2),
            'subtyped_pct': round(100 * n_sub_after / n_total, 2),
            'inh_subtyped_pct': round(100 * n_inh_after / n_total, 2),
        },
        'new_assignments': {
            'from_mtypes_PTC_DTC': n_new_mtypes,
            'from_synapse_target_structure': n_new_struct,
            'total': n_new_mtypes + n_new_struct,
        },
        'tables_used': [
            'aibs_metamodel_mtypes_v661_v2 (PTC→BC, DTC→MC, STC→BPC)',
            'synapse_target_structure (Celii et al.) — soma-targeting proxy for BC',
        ],
        'mtype_mapping': MTYPE_TO_SUBTYPE,
        'notes': [
            'PTC/DTC/STC are morphological EM proxy categories, not molecular identity.',
            'Synapse target structure classification uses >40% soma target → BC-proximal.',
            'These are L2 (boosted) labels, distinct from L1 (direct CAVE annotation).',
        ]
    }

    out_path = RESULTS_DIR / "cave_boost_v2.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
