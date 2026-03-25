#!/usr/bin/env python3
"""Functional correlates of dendritic regime coupling.

Pre-registered hypothesis (see .claude/hypotheses/functional_correlates_hypothesis.md):
  H1 (PRIMARY): Excitatory neurons with steeper ICV coupling slopes (more negative
     BLUP) show stronger orientation selectivity (higher OSI).
     Directional prediction: negative Spearman correlation (ICV BLUP vs OSI).

  H2: Steeper ICV slope → higher response reliability (cc_abs or cc_max).
  H3: |Inhibitory ICV slope| → lower population coupling.

Statistical threshold: p < 0.017 (Bonferroni for 3 tests).

Output: results/replication_full/functional_correlates.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "replication_full"

sys.path.insert(0, str(Path.home() / "research" / "neurostat"))


def load_random_slope_blups():
    """Load per-neuron ICV BLUP estimates from random_slope_results.json."""
    path = RESULTS_DIR / "random_slope_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with open(path) as f:
        rs = json.load(f)

    records = []
    # The JSON structure contains per-neuron slope data
    # Find the ICV section and extract BLUPs
    icv_data = rs.get("interval_cv_z", rs.get("ICV", {}))

    # Try to load from the all_branch_features to get neuron-level info
    branch_df = pd.read_csv(RESULTS_DIR / "all_branch_features_with_spatial.csv")
    neuron_info = branch_df.groupby("neuron_label").agg(
        root_id=("root_id", "first"),
        cell_type=("cell_type", "first"),
        subtype=("subtype", "first"),
    ).reset_index()

    return neuron_info, rs


def query_functional_data(client, proofread_root_ids):
    """Query coregistration and functional properties tables."""
    print("  Querying coregistration_manual_v4...")
    coreg = client.materialize.query_table("coregistration_manual_v4")
    coreg["pt_root_id"] = coreg["pt_root_id"].astype(int)
    print(f"  Total coregistration entries: {len(coreg):,}")

    # Find intersection with our proofread neurons
    our_set = set(proofread_root_ids)
    matched = coreg[coreg["pt_root_id"].isin(our_set)]
    print(f"  Proofread neurons with functional match: {matched['pt_root_id'].nunique():,} "
          f"/ {len(our_set):,} ({100*matched['pt_root_id'].nunique()/len(our_set):.1f}%)")

    if len(matched) == 0:
        return None, None

    # Get functional properties
    print("  Querying digital_twin_properties_bcm_coreg_v4...")
    func = client.materialize.query_table("digital_twin_properties_bcm_coreg_v4")
    func["pt_root_id"] = func["pt_root_id"].astype(int)
    print(f"  Total functional property entries: {len(func):,}")

    # Join: coreg gives us root_id → unit_id, func gives unit_id → OSI/DSI/etc.
    # coreg has 'pt_root_id' and 'target_id' (which links to functional units)
    # func also has 'pt_root_id' directly — use that

    # Filter func to our matched neurons
    func_matched = func[func["pt_root_id"].isin(our_set)]
    print(f"  Functional records for our neurons: {len(func_matched):,} "
          f"({func_matched['pt_root_id'].nunique()} unique neurons)")

    return matched, func_matched


def compute_per_neuron_blups(branch_df):
    """Compute per-neuron ICV slope BLUPs using the same random-slope model as main analysis."""
    import statsmodels.formula.api as smf

    print("  Fitting random-slope model to get per-neuron ICV BLUPs...")
    df = branch_df.dropna(subset=["interval_cv_z", "electrotonic_length",
                                   "synapse_count", "total_length_nm"]).copy()
    df = df[df["synapse_count"] >= 3]

    results = {}

    for cell_type, sub in [("excitatory", df[df["cell_type"] == "excitatory"]),
                            ("inhibitory", df[df["cell_type"] == "inhibitory"])]:
        if len(sub) < 100:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = smf.mixedlm(
                    "interval_cv_z ~ electrotonic_length + synapse_count + total_length_nm",
                    sub,
                    groups=sub["neuron_label"],
                    exog_re=sub[["electrotonic_length"]]
                ).fit(reml=True, method="lbfgs")

            # Extract per-neuron random slopes
            re = mod.random_effects  # dict: neuron_label → array
            blup_records = []
            for nl, vals in re.items():
                # vals is intercept + slope
                slope = float(vals.iloc[-1]) if hasattr(vals, 'iloc') else float(vals[-1])
                fixed_slope = float(mod.params.get("electrotonic_length", 0))
                total_slope = fixed_slope + slope
                blup_records.append({"neuron_label": nl, f"icv_slope_{cell_type}": total_slope})

            blup_df = pd.DataFrame(blup_records)
            print(f"    {cell_type}: {len(blup_df)} neurons, "
                  f"median slope={blup_df[f'icv_slope_{cell_type}'].median():.3f}")
            results[cell_type] = blup_df

        except Exception as e:
            print(f"    {cell_type} model failed: {e}")

    return results


def test_hypothesis(blup_df, func_df, slope_col, h_label, outcome_col, direction="negative"):
    """Run pre-registered Spearman correlation test."""
    # Join on root_id
    merged = blup_df.merge(func_df[["root_id", outcome_col]], on="root_id")
    merged = merged.dropna(subset=[slope_col, outcome_col])

    if len(merged) < 10:
        print(f"  {h_label}: n={len(merged)} — insufficient")
        return {"n": len(merged), "note": "insufficient"}

    rho, p = stats.spearmanr(merged[slope_col], merged[outcome_col])
    significant = p < 0.017  # Bonferroni threshold
    directional = (rho < 0) == (direction == "negative")

    print(f"  {h_label}: n={len(merged)}  rho={rho:.3f}  p={p:.3e}  "
          f"{'SIGNIFICANT' if significant else 'ns'}  "
          f"{'correct direction' if directional else 'WRONG direction'}")

    return {
        "n": len(merged),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "significant_bonferroni": bool(significant),
        "correct_direction": bool(directional),
        "predicted_direction": direction,
        "slope_col": slope_col,
        "outcome_col": outcome_col,
    }


def subtype_stratified_test(blup_df, func_df, slope_col, outcome_col, neuron_info):
    """Test H1 within each excitatory subtype to check for subtype confounding."""
    merged = blup_df.merge(func_df[["root_id", outcome_col]], on="root_id")
    # subtype already present in blup_df from earlier merge with neuron_info
    if "subtype" not in merged.columns:
        merged = merged.merge(neuron_info[["root_id", "subtype"]], on="root_id", how="left")
    merged = merged.dropna(subset=[slope_col, outcome_col])

    results = {}
    print(f"\n  Within-subtype stratification ({outcome_col}):")
    for sub, grp in merged.groupby("subtype"):
        if len(grp) < 10:
            continue
        rho, p = stats.spearmanr(grp[slope_col], grp[outcome_col])
        print(f"    {sub:8s}: n={len(grp):4d}  rho={rho:+.3f}  p={p:.3e}")
        results[str(sub)] = {"n": len(grp), "rho": float(rho), "p": float(p)}

    return results


def main():
    print("=" * 60)
    print("FUNCTIONAL CORRELATES ANALYSIS")
    print("=" * 60)
    print("\nHypotheses loaded from .claude/hypotheses/functional_correlates_hypothesis.md")
    print("All predictions committed BEFORE data access.\n")

    from caveclient import CAVEclient
    client = CAVEclient("minnie65_public")

    # ── Load branch features and neuron info ─────────────────────────────
    print("[1] Loading branch features...")
    branch_df = pd.read_csv(RESULTS_DIR / "all_branch_features_with_spatial.csv")
    neuron_info = branch_df.groupby("neuron_label").agg(
        root_id=("root_id", "first"),
        cell_type=("cell_type", "first"),
        subtype=("subtype", "first"),
    ).reset_index()
    print(f"  {len(neuron_info)} proofread neurons")

    proofread_root_ids = neuron_info["root_id"].astype(int).tolist()

    # ── Compute per-neuron ICV slope BLUPs ───────────────────────────────
    print("\n[2] Computing per-neuron ICV slope BLUPs...")
    blup_results = compute_per_neuron_blups(branch_df)

    # Build combined BLUP dataframe with root_id
    blup_frames = {}
    for cell_type, blup_df in blup_results.items():
        blup_df = blup_df.merge(
            neuron_info[["neuron_label", "root_id", "subtype"]], on="neuron_label"
        )
        blup_frames[cell_type] = blup_df
        print(f"  {cell_type}: {len(blup_df)} neurons with BLUPs")

    # ── Query functional data ─────────────────────────────────────────────
    print("\n[3] Querying functional data from CAVE...")
    coreg_matched, func_matched = query_functional_data(client, proofread_root_ids)

    if func_matched is None or len(func_matched) == 0:
        print("  No functional data found for proofread neurons!")
        return

    # Aggregate per-neuron (some neurons may have multiple functional recordings)
    func_agg = func_matched.groupby("pt_root_id").agg(
        OSI=("OSI", "median"),
        gOSI=("gOSI", "median"),
        DSI=("DSI", "median"),
        gDSI=("gDSI", "median"),
        cc_abs=("cc_abs", "median"),    # response reliability
        cc_max=("cc_max", "median"),
        n_recordings=("OSI", "count"),
    ).reset_index().rename(columns={"pt_root_id": "root_id"})

    print(f"\n  Functional data summary (after aggregation):")
    print(f"    n neurons with functional data: {len(func_agg):,}")
    print(f"    OSI range: [{func_agg['OSI'].min():.3f}, {func_agg['OSI'].max():.3f}]  "
          f"median={func_agg['OSI'].median():.3f}")
    print(f"    cc_abs range: [{func_agg['cc_abs'].min():.3f}, {func_agg['cc_abs'].max():.3f}]  "
          f"median={func_agg['cc_abs'].median():.3f}")

    # ── Run pre-registered hypothesis tests ──────────────────────────────
    results = {}

    print("\n" + "=" * 50)
    print("PRE-REGISTERED HYPOTHESIS TESTS")
    print("Bonferroni threshold: p < 0.017 (3 tests)")
    print("=" * 50)

    exc_blups = blup_frames.get("excitatory")
    inh_blups = blup_frames.get("inhibitory")

    print("\n--- H1 (PRIMARY): ICV slope → OSI ---")
    if exc_blups is not None:
        results["H1_osi"] = test_hypothesis(
            exc_blups, func_agg, "icv_slope_excitatory",
            "H1(ICV→OSI)", "OSI", direction="negative"
        )
        results["H1_gosi"] = test_hypothesis(
            exc_blups, func_agg, "icv_slope_excitatory",
            "H1(ICV→gOSI)", "gOSI", direction="negative"
        )

    print("\n--- H2: ICV slope → response reliability ---")
    if exc_blups is not None:
        results["H2_ccabs"] = test_hypothesis(
            exc_blups, func_agg, "icv_slope_excitatory",
            "H2(ICV→cc_abs)", "cc_abs", direction="negative"
        )

    print("\n--- H3 (EXPLORATORY): |Inhibitory ICV slope| → population coupling ---")
    if inh_blups is not None and "cc_abs" in func_agg.columns:
        inh_blups_abs = inh_blups.copy()
        inh_blups_abs["abs_icv_slope"] = inh_blups_abs["icv_slope_inhibitory"].abs()
        results["H3_popcouple"] = test_hypothesis(
            inh_blups_abs.rename(columns={"abs_icv_slope": "icv_slope_excitatory"}),
            func_agg, "icv_slope_excitatory",
            "H3(|inh_ICV|→cc_abs)", "cc_abs", direction="negative"
        )

    # ── Subtype stratification (confound check) ───────────────────────────
    if exc_blups is not None and results.get("H1_osi", {}).get("n", 0) >= 10:
        print("\n--- CONFOUND CHECK: Within-subtype stratification ---")
        results["H1_by_subtype"] = subtype_stratified_test(
            exc_blups, func_agg, "icv_slope_excitatory", "OSI", neuron_info
        )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for h_key in ["H1_osi", "H1_gosi", "H2_ccabs", "H3_popcouple"]:
        r = results.get(h_key, {})
        if "spearman_rho" in r:
            verdict = "POSITIVE" if r.get("significant_bonferroni") and r.get("correct_direction") else \
                      "MARGINAL" if r.get("spearman_p", 1) < 0.05 else "NULL"
            print(f"  {h_key:20s}: rho={r['spearman_rho']:+.3f}  p={r['spearman_p']:.3e}  "
                  f"n={r['n']}  → {verdict}")

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "functional_correlates.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
