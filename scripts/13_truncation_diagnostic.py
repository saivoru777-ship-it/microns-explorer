#!/usr/bin/env python3
"""Volume-edge truncation diagnostic.

Tests whether the regime coupling results are driven by branches near the
volume boundary, where dendritic truncation could mechanically compress
synapse positions and artificially inflate spacing metrics.

Logic:
  - Parse all SWC files to get per-node 3D positions (µm)
  - Compute volume bounds from the 1st/99th percentile of all node positions
  - For each branch: compute centroid, min-distance to volume boundary
  - Merge into branch features
  - Rerun regime coupling models at boundary exclusion thresholds 0/10/25/50 µm
  - If regime 2 coefficients shift >10% relative, truncation is a concern

Output: results/replication_full/truncation_diagnostic.json
"""

import json
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "neurons"
RESULTS_DIR = PROJECT_DIR / "results" / "replication_full"

sys.path.insert(0, str(Path.home() / "research" / "neurostat"))
sys.path.insert(0, str(Path.home() / "research" / "neurostat-input-clustering" / "src"))
sys.path.insert(0, str(Path.home() / "research" / "dendritic-regime-coupling"))

from neurostat.io.swc import NeuronSkeleton
import statsmodels.formula.api as smf


# ─── helpers ────────────────────────────────────────────────────────────────

def parse_swc_nodes(swc_path):
    """Return dict node_id → (x, y, z, parent_id) in µm."""
    nodes = {}
    with open(swc_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            nid = int(parts[0])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            parent = int(parts[6])
            nodes[nid] = (x, y, z, parent)
    return nodes


def decompose_branches_from_swc(nodes):
    """Decompose SWC nodes into branches as lists of node IDs.

    A branch starts at the root or at a bifurcation point and ends at a
    leaf or the node before the next bifurcation.
    Returns list of lists of node IDs (each list = one branch).
    """
    children = defaultdict(list)
    root = None
    for nid, (x, y, z, parent) in nodes.items():
        if parent == -1:
            root = nid
        else:
            children[parent].append(nid)

    if root is None:
        return []

    branches = []
    # DFS: accumulate current branch, split at bifurcations
    stack = [(root, [root])]
    while stack:
        nid, current_branch = stack.pop()
        kids = children[nid]
        if len(kids) == 0:
            # Leaf: close branch
            branches.append(current_branch)
        elif len(kids) == 1:
            # Continuation: extend branch
            stack.append((kids[0], current_branch + [kids[0]]))
        else:
            # Bifurcation: close branch, start new ones
            branches.append(current_branch)
            for kid in kids:
                stack.append((kid, [kid]))

    return branches


def branch_centroid_and_min_dist(branch_nodes, nodes, vol_min, vol_max):
    """For a branch (list of node IDs), return (centroid, min_boundary_dist) in µm."""
    coords = np.array([(nodes[n][0], nodes[n][1], nodes[n][2])
                       for n in branch_nodes if n in nodes])
    if len(coords) == 0:
        return None, None

    centroid = coords.mean(axis=0)

    # Distance to each face of the bounding box (6 faces)
    dists_to_min = coords - vol_min  # shape (n, 3), all should be >= 0
    dists_to_max = vol_max - coords  # shape (n, 3), all should be >= 0
    all_dists = np.concatenate([dists_to_min, dists_to_max], axis=1)  # (n, 6)
    min_dist = all_dists.min()  # worst-case: the branch node closest to any wall

    return centroid, float(min_dist)


def run_model(df, exclude_below_um, metric='clark_evans_z'):
    """Run regime coupling model excluding branches within exclude_below_um of boundary."""
    sub = df.dropna(subset=[metric, 'regime', 'synapse_count', 'total_length_nm',
                             'min_boundary_dist_um'])
    if exclude_below_um > 0:
        sub = sub[sub['min_boundary_dist_um'] >= exclude_below_um]

    if len(sub) < 1000:
        return None, None, None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = smf.mixedlm(
                f"{metric} ~ C(regime) + synapse_count + total_length_nm",
                sub, groups=sub['neuron_label']
            ).fit(reml=False, method='lbfgs')
        coef_r2 = mod.params.get('C(regime)[T.2]', np.nan)
        pval_r2 = mod.pvalues.get('C(regime)[T.2]', np.nan)
        n = len(sub)
        return float(coef_r2), float(pval_r2), n
    except Exception as e:
        print(f"    model failed: {e}")
        return None, None, None


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("VOLUME EDGE TRUNCATION DIAGNOSTIC")
    print("=" * 60)

    # ── 1. Compute volume bounds ──────────────────────────────────────────
    print("\n[1] Computing volume bounds from all SWC files...")
    swc_files = list(DATA_DIR.glob("*.swc"))
    print(f"    {len(swc_files)} SWC files found")

    # Sample 500 files to get robust bounds (full set is slow)
    rng = np.random.default_rng(42)
    sample_swcs = rng.choice(swc_files, min(500, len(swc_files)), replace=False)

    all_coords = []
    for f in sample_swcs:
        nodes = parse_swc_nodes(f)
        for nid, (x, y, z, _) in nodes.items():
            all_coords.append((x, y, z))

    coords_arr = np.array(all_coords)
    # Use 1st/99th percentile as "effective volume boundary"
    vol_min = np.percentile(coords_arr, 1, axis=0)
    vol_max = np.percentile(coords_arr, 99, axis=0)
    vol_range = vol_max - vol_min

    print(f"    Volume bounds (1st–99th pct):")
    print(f"      X: {vol_min[0]:.1f} – {vol_max[0]:.1f} µm  ({vol_range[0]:.0f} µm)")
    print(f"      Y: {vol_min[1]:.1f} – {vol_max[1]:.1f} µm  ({vol_range[1]:.0f} µm)")
    print(f"      Z: {vol_min[2]:.1f} – {vol_max[2]:.1f} µm  ({vol_range[2]:.0f} µm)")

    # ── 2. Per-branch boundary distance ──────────────────────────────────
    print("\n[2] Computing per-branch min distance to volume boundary...")
    branch_records = []

    for swc_path in swc_files:
        label = swc_path.stem
        nodes = parse_swc_nodes(swc_path)
        if len(nodes) < 5:
            continue

        branches = decompose_branches_from_swc(nodes)
        for bidx, branch_node_list in enumerate(branches):
            centroid, min_dist = branch_centroid_and_min_dist(
                branch_node_list, nodes, vol_min, vol_max
            )
            if centroid is not None:
                branch_records.append({
                    'neuron_label': label,
                    'branch_idx': bidx,
                    'centroid_x': centroid[0],
                    'centroid_y': centroid[1],
                    'centroid_z': centroid[2],
                    'min_boundary_dist_um': min_dist,
                })

    boundary_df = pd.DataFrame(branch_records)
    print(f"    {len(boundary_df):,} branch-level boundary distances computed")
    print(f"    Min distance distribution:")
    for pct in [1, 5, 10, 25, 50]:
        val = boundary_df['min_boundary_dist_um'].quantile(pct / 100)
        print(f"      {pct:3d}th pct: {val:.1f} µm")

    # ── 3. Merge with branch features ─────────────────────────────────────
    print("\n[3] Merging with branch features...")
    features_path = RESULTS_DIR / "all_branch_features_with_spatial.csv"
    feat_df = pd.read_csv(features_path)
    print(f"    Branch features: {len(feat_df):,} rows")

    merged = feat_df.merge(
        boundary_df[['neuron_label', 'branch_idx', 'min_boundary_dist_um']],
        on=['neuron_label', 'branch_idx'],
        how='left'
    )
    matched = merged['min_boundary_dist_um'].notna().sum()
    print(f"    Matched branches: {matched:,} / {len(merged):,} ({100*matched/len(merged):.1f}%)")

    # Distribution of boundary distance by regime
    for reg in [0, 1, 2]:
        sub = merged[merged['regime'] == reg]['min_boundary_dist_um'].dropna()
        print(f"    Regime {reg} median boundary dist: {sub.median():.1f} µm  (n={len(sub):,})")

    # ── 4. Regime coupling models at different exclusion thresholds ────────
    print("\n[4] Running regime coupling models at exclusion thresholds...")
    thresholds = [0, 10, 25, 50]  # µm
    metrics = ['clark_evans_z', 'interval_cv_z', 'pairwise_compactness_z']
    populations = {
        'all': merged,
        'inhibitory': merged[merged['cell_type'] == 'inhibitory'],
        'excitatory': merged[merged['cell_type'] == 'excitatory'],
    }

    results = {}
    for pop_name, pop_df in populations.items():
        print(f"\n  Population: {pop_name} ({len(pop_df):,} branches)")
        results[pop_name] = {}
        for metric in metrics:
            results[pop_name][metric] = {}
            coef_baseline = None
            for thresh in thresholds:
                coef, pval, n = run_model(pop_df, thresh, metric)
                if coef is None:
                    print(f"    {metric} thresh={thresh}: FAILED")
                    continue
                if coef_baseline is None:
                    coef_baseline = coef
                shift_pct = 100 * (coef - coef_baseline) / abs(coef_baseline) if coef_baseline else 0
                print(f"    {metric} excl>{thresh:2d}µm: coef={coef:+.4f}  p={pval:.2e}  n={n:,}  shift={shift_pct:+.1f}%")
                results[pop_name][metric][f'thresh_{thresh}um'] = {
                    'coef_regime2': coef, 'pval': pval, 'n': n,
                    'shift_pct_vs_baseline': round(shift_pct, 2)
                }

    # ── 5. Summary verdict ────────────────────────────────────────────────
    print("\n[5] Verdict:")
    max_shift = 0.0
    for pop_name, pop_res in results.items():
        for metric, mres in pop_res.items():
            base_key = 'thresh_0um'
            if base_key not in mres:
                continue
            base_coef = mres[base_key]['coef_regime2']
            for thresh_key, tres in mres.items():
                if thresh_key == base_key:
                    continue
                shift = abs(tres.get('shift_pct_vs_baseline', 0))
                if shift > max_shift:
                    max_shift = shift

    verdict = 'CONCERN' if max_shift > 10 else 'CLEAN'
    verdict_detail = (
        f"Max shift {max_shift:.1f}% across all populations/metrics/thresholds. "
        f"{'Truncation may be inflating regime effects.' if verdict=='CONCERN' else 'Regime effects are robust to boundary exclusion.'}"
    )
    print(f"    {verdict}: {verdict_detail}")

    # ── 6. Save ───────────────────────────────────────────────────────────
    output = {
        'volume_bounds_um': {
            'x': [float(vol_min[0]), float(vol_max[0])],
            'y': [float(vol_min[1]), float(vol_max[1])],
            'z': [float(vol_min[2]), float(vol_max[2])],
        },
        'branch_boundary_dist_summary': {
            str(int(p)): float(boundary_df['min_boundary_dist_um'].quantile(p/100))
            for p in [1, 5, 10, 25, 50, 75, 90]
        },
        'regime_boundary_dist_median_um': {
            f'regime_{r}': float(merged[merged['regime'] == r]['min_boundary_dist_um'].median())
            for r in [0, 1, 2]
        },
        'results_by_threshold': results,
        'verdict': verdict,
        'verdict_detail': verdict_detail,
        'max_shift_pct': round(max_shift, 2),
    }

    out_path = RESULTS_DIR / "truncation_diagnostic.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
