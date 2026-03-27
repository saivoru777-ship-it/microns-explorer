#!/usr/bin/env python3
"""Two targeted analyses.

Analysis 1 — Excitatory slope by cortical subtype
  Per-neuron ICV coupling slopes stratified by excitatory cell subtype
  (23P, 4P, 5P-ET, 5P-IT, 5P-NP, 6P-CT, 6P-IT). Tests whether the -2.23
  median conceals meaningful layer-specific variation. L5-ET neurons (complex
  apical dendrites, dendritic bursting) are predicted to show steeper slopes
  than L4 or L2/3 neurons if the organizational logic tracks dendritic
  computational capacity.

Analysis 2 — ICV vs inhibitory input count (superposition test)
  Tests the superposition hypothesis: if branch-level inhibitory spacing
  is the projected superposition of many axon-level self-avoiding processes,
  branches with more unique inhibitory presynaptic contributors should show
  MORE UNIFORM spacing (lower ICV_z) — because superposition of self-avoiding
  processes smooths local structure.

  Two proxies for "number of contributing inhibitory axons":
    (a) inh_count: total inhibitory synapses on branch (correlated with
        number of contributing axons)
    (b) estimated unique inhibitory partners: unique_partners × frac_inh_afferent
        (rough estimate from partner architecture features)

  Prediction: more inhibitory contributors → lower ICV_z (more uniform)
  This is directly falsifiable in existing data.

Output: results/replication_full/subtype_superposition.json
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
FIGURES_DIR = PROJECT_DIR / "figures"

sys.path.insert(0, str(Path.home() / "research" / "neurostat"))
sys.path.insert(0, str(Path.home() / "research" / "neurostat-input-clustering" / "src"))
sys.path.insert(0, str(Path.home() / "research" / "dendritic-regime-coupling"))

import statsmodels.formula.api as smf


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: EXCITATORY SLOPE BY SUBTYPE
# ════════════════════════════════════════════════════════════════════════════

def compute_slopes_with_subtype(branch_df):
    """Fit random-slope LME on ALL neurons; return per-neuron slopes with subtype."""
    df = branch_df.dropna(subset=["interval_cv_z", "electrotonic_length",
                                   "synapse_count", "total_length_nm",
                                   "neuron_label", "cell_type", "subtype"]).copy()
    df = df[df["synapse_count"] >= 3]

    print(f"  Running on {len(df):,} branches, {df['neuron_label'].nunique()} neurons")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = smf.mixedlm(
            "interval_cv_z ~ electrotonic_length + synapse_count + total_length_nm",
            df,
            groups=df["neuron_label"],
            re_formula="~electrotonic_length"
        ).fit(reml=True, method="lbfgs")

    fixed_slope = float(mod.params.get("electrotonic_length", 0))
    print(f"  Population fixed slope: {fixed_slope:.4f}")

    neuron_meta = df.drop_duplicates("neuron_label").set_index("neuron_label")[
        ["cell_type", "subtype"]
    ]

    slopes = []
    for nl, re_dict in mod.random_effects.items():
        keys = list(re_dict.keys())
        random_slope = float(re_dict[keys[1]]) if len(keys) >= 2 else \
                       float(re_dict.get("electrotonic_length", 0))
        total = fixed_slope + random_slope
        ctype = neuron_meta.loc[nl, "cell_type"] if nl in neuron_meta.index else "unknown"
        sub = neuron_meta.loc[nl, "subtype"] if nl in neuron_meta.index else "unknown"
        slopes.append({
            "neuron_label": nl,
            "total_slope": total,
            "cell_type": ctype,
            "subtype": sub,
        })

    return pd.DataFrame(slopes), fixed_slope


def analysis1_slopes_by_subtype():
    """Stratify per-neuron ICV slopes by excitatory cell subtype."""
    print("\n" + "=" * 60)
    print("ANALYSIS 1: EXCITATORY ICV SLOPE BY SUBTYPE")
    print("=" * 60)
    print("Prediction: L5-ET (complex dendrites, dendritic bursting)")
    print("  should show steeper slopes than L4 (feedforward, compact)")

    branch_df = pd.read_csv(RESULTS_DIR / "all_branch_features_with_spatial.csv")
    slopes_df, fixed_slope = compute_slopes_with_subtype(branch_df)

    exc_slopes = slopes_df[slopes_df["cell_type"] == "excitatory"].copy()
    inh_slopes = slopes_df[slopes_df["cell_type"].str.startswith("inhibitory")].copy()

    # Verify overall medians match paper
    print(f"\n  Overall check:")
    print(f"    Excitatory: n={len(exc_slopes)}  median={exc_slopes['total_slope'].median():.3f}")
    print(f"    Inhibitory: n={len(inh_slopes)}  median={inh_slopes['total_slope'].median():.3f}")

    # By excitatory subtype
    print(f"\n  Per excitatory subtype:")
    subtype_order = ["23P", "4P", "5PET", "5PIT", "5PNP", "6PCT", "6PIT"]
    subtype_results = {}

    for sub in subtype_order:
        grp = exc_slopes[exc_slopes["subtype"] == sub]
        if len(grp) < 5:
            continue
        m = grp["total_slope"].median()
        mn = grp["total_slope"].mean()
        q25, q75 = grp["total_slope"].quantile([0.25, 0.75])
        print(f"    {sub:8s}: n={len(grp):4d}  median={m:+.3f}  "
              f"IQR=[{q25:+.2f}, {q75:+.2f}]")
        subtype_results[sub] = {
            "n": len(grp),
            "median": float(m),
            "mean": float(mn),
            "q25": float(q25),
            "q75": float(q75),
            "std": float(grp["total_slope"].std()),
        }

    # Key comparison: L5-ET vs 23P vs 4P
    print(f"\n  Key comparisons (Kruskal-Wallis across all subtypes):")
    groups = [exc_slopes[exc_slopes["subtype"] == s]["total_slope"].values
              for s in subtype_order if (exc_slopes["subtype"] == s).sum() >= 5]
    kw_stat, kw_p = stats.kruskal(*groups)
    print(f"    Kruskal-Wallis H={kw_stat:.2f}  p={kw_p:.3e}")

    # Pairwise: 5P-ET vs 23P, 5P-ET vs 4P
    for s1, s2 in [("5PET", "23P"), ("5PET", "4P"), ("23P", "4P"), ("6PCT", "4P")]:
        g1 = exc_slopes[exc_slopes["subtype"] == s1]["total_slope"].values
        g2 = exc_slopes[exc_slopes["subtype"] == s2]["total_slope"].values
        if len(g1) >= 5 and len(g2) >= 5:
            u, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            d = (np.median(g1) - np.median(g2))
            print(f"    {s1} vs {s2}: Δmedian={d:+.3f}  MW p={p:.3e}  "
                  f"({s1}: {np.median(g1):.3f}  {s2}: {np.median(g2):.3f})")

    # Also: inhibitory subtypes
    print(f"\n  Per inhibitory subtype:")
    inh_subtype_results = {}
    for sub in ["BC", "MC", "BPC", "NGC"]:
        grp = slopes_df[slopes_df["subtype"] == sub]
        if len(grp) < 5:
            continue
        m = grp["total_slope"].median()
        print(f"    {sub:8s}: n={len(grp):4d}  median={m:+.3f}")
        inh_subtype_results[sub] = {
            "n": len(grp),
            "median": float(m),
            "mean": float(grp["total_slope"].mean()),
        }

    # Verdict
    print(f"\n  VERDICT:")
    if len(subtype_results) >= 3:
        et_med = subtype_results.get("5PET", {}).get("median", np.nan)
        l4_med = subtype_results.get("4P", {}).get("median", np.nan)
        l23_med = subtype_results.get("23P", {}).get("median", np.nan)
        if not np.isnan(et_med) and not np.isnan(l4_med):
            diff = et_med - l4_med
            if diff < -0.5 and kw_p < 0.05:
                verdict = f"GRADIENT EXISTS: L5-ET steeper ({et_med:.2f}) than L4 ({l4_med:.2f}), Δ={diff:.2f}"
            elif kw_p > 0.05:
                verdict = f"NO SIGNIFICANT GRADIENT: all subtypes similar (KW p={kw_p:.3e})"
            else:
                verdict = f"PARTIAL GRADIENT: KW significant but L5-ET ({et_med:.2f}) vs L4 ({l4_med:.2f}) Δ={diff:.2f}"
            print(f"    {verdict}")

    return {
        "excitatory_by_subtype": subtype_results,
        "inhibitory_by_subtype": inh_subtype_results,
        "kruskal_wallis": {"H": float(kw_stat), "p": float(kw_p)},
        "fixed_slope": float(fixed_slope),
    }


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: SUPERPOSITION TEST
# ════════════════════════════════════════════════════════════════════════════

def analysis2_superposition_test():
    """Test whether more unique inhibitory contributors → more uniform ICV.

    The superposition hypothesis: branch ICV_z is the projection of many
    self-avoiding axonal processes. More contributing axons → more superposition
    → more uniform (lower ICV_z).

    Prediction: inh_count coefficient < 0 in LME (more inh synapses = more
    contributing axons = more uniform spacing).
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 2: SUPERPOSITION TEST")
    print("=" * 60)
    print("Prediction: more inhibitory presynaptic contributors → lower ICV_z")
    print("  (superposition of self-avoiding processes → uniformity)")

    # Merge partner architecture with afferent branch features
    pa = pd.read_csv(RESULTS_DIR / "partner_architecture_features.csv")
    af = pd.read_csv(RESULTS_DIR / "afferent_branch_features.csv")

    if "regime_x" in af.columns:
        af = af.rename(columns={"regime_x": "regime"})

    # Compute estimated unique inhibitory partners
    # = unique_partners × frac_inh_afferent (rough estimate)
    pa["est_unique_inh_partners"] = pa["unique_partners"] * (1 - pa["exc_fraction"].fillna(0.5))

    # Merge
    merged = af.merge(
        pa[["neuron_label", "branch_idx", "unique_partners", "est_unique_inh_partners",
            "mean_syn_per_partner", "frac_multisynaptic"]],
        on=["neuron_label", "branch_idx"], how="left"
    )

    print(f"\n  Total branches: {len(merged):,}, inhibitory: {(merged['cell_type'].str.startswith('inhibitory')).sum():,}")

    results = {}

    for pop, label in [("inhibitory", "Inhibitory neurons (postsynaptic inh)"),
                        ("excitatory", "Excitatory neurons (postsynaptic exc)")]:
        mask = merged["cell_type"].str.startswith(pop)
        sub = merged[mask].dropna(subset=["interval_cv_z", "inh_count",
                                           "synapse_count", "total_length_nm",
                                           "neuron_label"]).copy()
        sub = sub[sub["synapse_count"] >= 3]

        print(f"\n  {label}: {len(sub):,} branches, {sub['neuron_label'].nunique()} neurons")

        # Proxy A: log(inh_count) as number of inhibitory contributors
        sub["log_inh_count"] = np.log1p(sub["inh_count"])
        sub["log_total"] = np.log1p(sub["synapse_count"])
        sub["log_length"] = np.log1p(sub["total_length_nm"])

        pop_res = {}

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod_a = smf.mixedlm(
                    "interval_cv_z ~ log_inh_count + log_total + log_length",
                    sub, groups=sub["neuron_label"]
                ).fit(reml=True, method="lbfgs")

            coef_a = float(mod_a.params.get("log_inh_count", np.nan))
            p_a = float(mod_a.pvalues.get("log_inh_count", np.nan))
            print(f"    Proxy A (log_inh_count): coef={coef_a:+.4f}  p={p_a:.3e}")
            superp_a = "SUPERPOSITION-CONSISTENT" if (coef_a < 0 and p_a < 0.05) else \
                       "NOT CONSISTENT" if p_a < 0.05 else "NS"
            print(f"      → {superp_a}")
            pop_res["proxy_a_log_inh_count"] = {"coef": coef_a, "p": p_a,
                                                  "verdict": superp_a}
        except Exception as e:
            print(f"    Proxy A failed: {e}")
            pop_res["proxy_a_log_inh_count"] = {"error": str(e)}

        # Proxy B: estimated unique inhibitory partners
        sub2 = sub.dropna(subset=["est_unique_inh_partners"]).copy()
        sub2 = sub2[sub2["est_unique_inh_partners"] >= 1]
        sub2["log_est_inh"] = np.log1p(sub2["est_unique_inh_partners"])

        if len(sub2) > 500:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod_b = smf.mixedlm(
                        "interval_cv_z ~ log_est_inh + log_total + log_length",
                        sub2, groups=sub2["neuron_label"]
                    ).fit(reml=True, method="lbfgs")

                coef_b = float(mod_b.params.get("log_est_inh", np.nan))
                p_b = float(mod_b.pvalues.get("log_est_inh", np.nan))
                print(f"    Proxy B (est unique inh partners): coef={coef_b:+.4f}  p={p_b:.3e}")
                superp_b = "SUPERPOSITION-CONSISTENT" if (coef_b < 0 and p_b < 0.05) else \
                           "NOT CONSISTENT" if p_b < 0.05 else "NS"
                print(f"      → {superp_b}")
                pop_res["proxy_b_est_unique_inh"] = {"coef": coef_b, "p": p_b,
                                                      "verdict": superp_b}
            except Exception as e:
                print(f"    Proxy B failed: {e}")

        # Quintile check (like density check)
        sub["inh_quintile"] = pd.qcut(sub["inh_count"].clip(0), 5, labels=False,
                                       duplicates="drop")
        print(f"    ICV_z by inh_count quintile:")
        for q in range(5):
            sq = sub[sub["inh_quintile"] == q]
            if len(sq) < 10:
                continue
            print(f"      Q{q+1} (inh_count med={sq['inh_count'].median():.0f}): "
                  f"ICV_z median={sq['interval_cv_z'].median():+.3f}  n={len(sq):,}")

        results[pop] = pop_res

    # Verdict
    print("\n  SUPERPOSITION VERDICT:")
    inh_a = results.get("inhibitory", {}).get("proxy_a_log_inh_count", {})
    if inh_a.get("coef", 0) < 0 and inh_a.get("p", 1) < 0.05:
        print("    SUPPORTED: inhibitory ICV becomes more uniform with more inhibitory inputs")
        print(f"    (coef={inh_a['coef']:+.4f}, p={inh_a['p']:.3e})")
        print("    Interpretation: branch-level spacing reflects superposition of axonal processes")
    else:
        p = inh_a.get("p", np.nan)
        coef = inh_a.get("coef", np.nan)
        print(f"    NOT SUPPORTED (coef={coef:+.4f}, p={p:.3e})")
        print("    Alternative: spacing uniformity is not driven by number of contributing axons")

    return results


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("SUBTYPE SLOPES AND SUPERPOSITION TEST")
    print("=" * 60)

    output = {}

    try:
        output["analysis1_subtype_slopes"] = analysis1_slopes_by_subtype()
    except Exception as e:
        print(f"\nAnalysis 1 failed: {e}")
        import traceback; traceback.print_exc()
        output["analysis1_subtype_slopes"] = {"error": str(e)}

    try:
        output["analysis2_superposition"] = analysis2_superposition_test()
    except Exception as e:
        print(f"\nAnalysis 2 failed: {e}")
        import traceback; traceback.print_exc()
        output["analysis2_superposition"] = {"error": str(e)}

    out_path = RESULTS_DIR / "subtype_superposition.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
