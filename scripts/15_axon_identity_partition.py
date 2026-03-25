#!/usr/bin/env python3
"""Axon identity variance partition.

Imports data pipeline from script 12 and adds three new analyses on top of
the existing axon_conditioned_placement raw records:

  1. Variance partition: how much of within-branch placement variance is
     explained by individual axon identity vs. subtype label vs. regime?
  2. Same-axon cross-regime consistency: do axons show intrinsic placement
     tendencies (Case A) or adapt to regime (Case B)?
  3. Arrival-preference × placement-tendency correlation: do axons preferring
     compact branches also use tighter placement rules?

Output: results/replication_full/axon_identity_partition.json
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "replication_full"

# Add script 12 to path so we can import its helpers
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
sys.path.insert(0, str(Path.home() / "research" / "neurostat"))
sys.path.insert(0, str(Path.home() / "research" / "neurostat-input-clustering" / "src"))
sys.path.insert(0, str(Path.home() / "research" / "dendritic-regime-coupling"))

import statsmodels.formula.api as smf


# ── Build raw placement DataFrame using script 12 pipeline ──────────────────

def build_raw_placement_df(spatial_df, data_dir, catalog_lookup):
    """Replicate the core of axon_conditioned_placement() to get the raw pdf."""
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "inhibitory_residual",
        PROJECT_DIR / "scripts" / "12_inhibitory_residual.py"
    )
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    inh_neurons = spatial_df[spatial_df["cell_type"].str.startswith("inhibitory")][
        "neuron_label"].unique()
    print(f"  {len(inh_neurons)} inhibitory neurons")

    INH_SUBTYPES = {"BC", "MC", "BPC", "NGC"}

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

    records = []
    t0 = time.time()

    for i, nl in enumerate(inh_neurons):
        result = mod._load_neuron_data(nl, data_dir)
        if result is None:
            continue

        _, syn_coords_nm, snap, partner_df = result
        matched = mod._match_partners(syn_coords_nm, snap, partner_df, catalog_lookup)

        if len(matched) == 0:
            continue

        matched["x_nm"] = syn_coords_nm[matched["synapse_idx"].values, 0]
        matched["y_nm"] = syn_coords_nm[matched["synapse_idx"].values, 1]
        matched["z_nm"] = syn_coords_nm[matched["synapse_idx"].values, 2]

        typed = matched[matched["annotation_level"] <= mod.LEVEL_BOOSTED].copy()

        for (bid, rid), grp in typed.groupby(["branch_idx", "pre_root_id"]):
            k = len(grp)
            if k < 2:
                continue

            pre_subtype = grp["subtype"].iloc[0]
            if pre_subtype not in INH_SUBTYPES and pre_subtype not in (
                    "23P", "4P", "5PIT", "5PET", "6PIT", "6PCT"):
                continue

            pre_class = pre_subtype if pre_subtype in INH_SUBTYPES else "excitatory"
            coords = grp[["x_nm", "y_nm", "z_nm"]].values
            dists = [
                np.linalg.norm(coords[ii] - coords[jj])
                for ii in range(k) for jj in range(ii + 1, k)
            ]

            bi = (nl, int(bid))
            info = branch_info.get(bi, {})

            records.append({
                "neuron_label": nl,
                "branch_idx": int(bid),
                "pre_root_id": int(rid),
                "pre_class": pre_class,
                "k_synapses": k,
                "mean_pairwise_dist_nm": float(np.mean(dists)),
                "regime": info.get("regime", -1),
                "total_length_nm": info.get("total_length_nm", np.nan),
                "post_subtype": info.get("subtype", ""),
            })

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(inh_neurons)} neurons ({time.time()-t0:.0f}s), "
                  f"{len(records)} records")

    df = pd.DataFrame(records)
    df = df[df["regime"] >= 0].copy()
    print(f"  Done: {len(df):,} placement records from {df['neuron_label'].nunique()} neurons")
    return df


# ── Analysis 1: Variance partition ──────────────────────────────────────────

def variance_partition(pdf, subtype):
    """ICC of neuron vs axon random effects; OLS residual variance decomposition."""
    df = pdf[pdf["pre_class"] == subtype].dropna(
        subset=["mean_pairwise_dist_nm", "regime", "k_synapses", "total_length_nm"]
    ).copy()
    df["log_dist"] = np.log1p(df["mean_pairwise_dist_nm"])

    if len(df) < 50:
        return {"note": f"n={len(df)} insufficient"}

    print(f"\n  [{subtype}] Variance partition  n={len(df):,} records, "
          f"{df['pre_root_id'].nunique()} axons, {df['neuron_label'].nunique()} neurons")

    results = {}

    # Mixed models: one grouping at a time
    for group_col, label in [("neuron_label", "neuron"), ("pre_root_id", "axon")]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = smf.mixedlm(
                    "log_dist ~ C(regime) + k_synapses + total_length_nm",
                    df, groups=df[group_col]
                ).fit(reml=True, method="lbfgs")

            re_var = float(mod.cov_re.iloc[0, 0])
            resid_var = float(mod.scale)
            icc = re_var / (re_var + resid_var)
            coef_r2 = float(mod.params.get("C(regime)[T.2]", np.nan))
            pval_r2 = float(mod.pvalues.get("C(regime)[T.2]", np.nan))

            print(f"    {label:10s}: RE_var={re_var:.4f}  resid={resid_var:.4f}  "
                  f"ICC={icc:.3f}  regime2_coef={coef_r2:+.3f}  p={pval_r2:.2e}")
            results[label] = {
                "re_var": re_var, "resid_var": resid_var, "icc": icc,
                "regime2_coef": coef_r2, "regime2_pval": pval_r2,
            }
        except Exception as e:
            print(f"    {label} model failed: {e}")

    # OLS residual decomposition
    try:
        ols = smf.ols(
            "log_dist ~ C(regime) + k_synapses + total_length_nm", df
        ).fit()
        df = df.copy()
        df["resid"] = ols.resid

        total_var = df["resid"].var()
        axon_var = df.groupby("pre_root_id")["resid"].mean().var()
        neuron_var = df.groupby("neuron_label")["resid"].mean().var()

        print(f"    OLS residual decomp: total={total_var:.4f}")
        print(f"      between-axon:    {axon_var:.4f}  ({100*axon_var/total_var:.1f}% of resid)")
        print(f"      between-neuron:  {neuron_var:.4f}  ({100*neuron_var/total_var:.1f}% of resid)")

        results["ols_decomp"] = {
            "total_resid_var": float(total_var),
            "between_axon_pct": float(100 * axon_var / total_var),
            "between_neuron_pct": float(100 * neuron_var / total_var),
        }
    except Exception as e:
        print(f"    OLS decomp failed: {e}")

    results["n"] = len(df)
    results["n_axons"] = int(df["pre_root_id"].nunique())
    results["n_neurons"] = int(df["neuron_label"].nunique())
    return results


# ── Analysis 2: Same-axon cross-regime consistency ──────────────────────────

def same_axon_consistency(pdf, subtype):
    """Correlation between per-axon R0 and R2 placement medians."""
    df = pdf[pdf["pre_class"] == subtype].dropna(
        subset=["mean_pairwise_dist_nm"]
    ).copy()
    df["log_dist"] = np.log1p(df["mean_pairwise_dist_nm"])

    r0 = df[df["regime"] == 0].groupby("pre_root_id")["log_dist"].median()
    r2 = df[df["regime"] == 2].groupby("pre_root_id")["log_dist"].median()
    common = r0.index.intersection(r2.index)

    print(f"\n  [{subtype}] Same-axon consistency: {len(common)} axons with R0+R2 contacts")
    if len(common) < 10:
        return {"n_common_axons": len(common), "note": "insufficient n"}

    r0v, r2v = r0[common].values, r2[common].values
    rho, rho_p = stats.spearmanr(r0v, r2v)

    r0_med, r2_med = np.median(r0v), np.median(r2v)
    tt = np.sum((r0v < r0_med) & (r2v < r2_med))
    tb = np.sum((r0v < r0_med) & (r2v >= r2_med))
    bt = np.sum((r0v >= r0_med) & (r2v < r2_med))
    bb = np.sum((r0v >= r0_med) & (r2v >= r2_med))
    concordance = 100 * (tt + bb) / len(common)

    diff = r2v - r0v
    expands = np.sum(diff > 0)
    wsr_p = stats.wilcoxon(diff).pvalue

    print(f"    Spearman(R0, R2 placement): rho={rho:.3f}  p={rho_p:.2e}")
    print(f"    Concordance (same tendency): {concordance:.1f}%")
    print(f"    Axons expanding R0→R2: {expands}/{len(common)} ({100*expands/len(common):.0f}%)")
    print(f"    Median R2-R0 diff: {np.median(diff):.3f}  Wilcoxon p={wsr_p:.2e}")

    if rho > 0.3 and rho_p < 0.05:
        interp = "Case_A_intrinsic: stable axon-specific placement tendency"
    elif expands > 0.65 * len(common) and rho < 0.2:
        interp = "Case_B_contextual: axons adapt placement to regime"
    else:
        interp = "Mixed: intrinsic tendency + regime modulation"
    print(f"    → {interp}")

    return {
        "n_common_axons": int(len(common)),
        "spearman_rho": float(rho), "spearman_p": float(rho_p),
        "concordance_pct": float(concordance),
        "expands_pct": float(100 * expands / len(common)),
        "median_r2_minus_r0": float(np.median(diff)),
        "wilcoxon_p": float(wsr_p),
        "interpretation": interp,
        "2x2": {"tt": int(tt), "tb": int(tb), "bt": int(bt), "bb": int(bb)},
    }


# ── Analysis 3: Arrival-preference × placement correlation ──────────────────

def arrival_placement_correlation(pdf, subtype):
    """Do axons preferring compact branches also use tighter placement?"""
    df = pdf[pdf["pre_class"] == subtype].dropna(
        subset=["mean_pairwise_dist_nm"]
    ).copy()
    df["log_dist"] = np.log1p(df["mean_pairwise_dist_nm"])

    per_axon = df.groupby("pre_root_id").agg(
        n_contacts=("branch_idx", "count"),
        n_r0=("regime", lambda x: (x == 0).sum()),
        median_log_dist=("log_dist", "median"),
    )
    per_axon = per_axon[per_axon["n_contacts"] >= 3]
    per_axon["compact_pref"] = per_axon["n_r0"] / per_axon["n_contacts"]

    print(f"\n  [{subtype}] Arrival-placement correlation: {len(per_axon)} axons (≥3 contacts)")
    if len(per_axon) < 10:
        return {"note": "insufficient n"}

    rho, p = stats.spearmanr(per_axon["compact_pref"], per_axon["median_log_dist"])
    print(f"    Spearman(compact_pref, placement_tendency): rho={rho:.3f}  p={p:.2e}")

    q25, q75 = per_axon["compact_pref"].quantile([0.25, 0.75])
    prefer_compact = per_axon[per_axon["compact_pref"] >= q75]["median_log_dist"]
    prefer_distal = per_axon[per_axon["compact_pref"] <= q25]["median_log_dist"]
    if len(prefer_compact) > 1 and len(prefer_distal) > 1:
        mwu_p = stats.mannwhitneyu(prefer_compact, prefer_distal).pvalue
        print(f"    Compact-preferring (Q4) median placement: {prefer_compact.median():.3f}")
        print(f"    Distal-preferring  (Q1) median placement: {prefer_distal.median():.3f}")
        print(f"    Mann-Whitney p: {mwu_p:.2e}")
    else:
        mwu_p = np.nan

    direction = "tight" if rho < 0 else "broad"
    interp = (
        f"Axons preferring compact branches use {direction} placement "
        f"(rho={rho:.3f}, {'aligned' if rho < 0 else 'anti-aligned'} with arrival)"
    )
    print(f"    → {interp}")

    return {
        "n_axons": int(len(per_axon)),
        "spearman_rho": float(rho), "spearman_p": float(p),
        "compact_Q4_median": float(prefer_compact.median()),
        "distal_Q1_median": float(prefer_distal.median()),
        "mannwhitney_p": float(mwu_p),
        "interpretation": interp,
    }


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("AXON IDENTITY VARIANCE PARTITION")
    print("=" * 60)

    spatial_df = pd.read_csv(RESULTS_DIR / "all_branch_features_with_spatial.csv")
    cat_path = PROJECT_DIR / "catalog.csv"

    print("\n[0] Loading catalog lookup...")
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "inhibitory_residual",
        PROJECT_DIR / "scripts" / "12_inhibitory_residual.py"
    )
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    catalog_lookup = mod._build_catalog_lookup(cat_path) if cat_path.exists() else {}
    print(f"  {len(catalog_lookup)} catalog entries")

    print("\n[1] Building raw placement dataset...")
    data_dir = PROJECT_DIR / "data" / "neurons"
    pdf = build_raw_placement_df(spatial_df, data_dir, catalog_lookup)

    pdf.to_csv(RESULTS_DIR / "axon_placement_raw.csv", index=False)
    print(f"  Saved axon_placement_raw.csv ({len(pdf):,} rows)")

    print("\n  Sample sizes by class:")
    for pc in sorted(pdf["pre_class"].unique()):
        sub = pdf[pdf["pre_class"] == pc]
        print(f"    {pc}: {len(sub):,} records, {sub['pre_root_id'].nunique()} axons, "
              f"{sub['neuron_label'].nunique()} neurons")

    results = {}

    print("\n" + "=" * 50)
    print("ANALYSIS 1: VARIANCE PARTITION")
    print("=" * 50)
    for subtype in ["BC", "MC"]:
        results[f"variance_partition_{subtype}"] = variance_partition(pdf, subtype)

    print("\n" + "=" * 50)
    print("ANALYSIS 2: SAME-AXON CROSS-REGIME CONSISTENCY")
    print("=" * 50)
    for subtype in ["BC", "MC"]:
        results[f"same_axon_consistency_{subtype}"] = same_axon_consistency(pdf, subtype)

    print("\n" + "=" * 50)
    print("ANALYSIS 3: ARRIVAL-PLACEMENT CORRELATION")
    print("=" * 50)
    for subtype in ["BC", "MC"]:
        results[f"arrival_placement_corr_{subtype}"] = arrival_placement_correlation(pdf, subtype)

    out_path = RESULTS_DIR / "axon_identity_partition.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
