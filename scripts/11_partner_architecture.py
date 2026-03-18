#!/usr/bin/env python3
"""Partner architecture mediation, random-slope models, and compactness diagnostics.

Three analyses:

1. Partner architecture mediation — compute per-branch partner features
   (unique_partners, mean_syn_per_partner, gini_partner, frac_multisynaptic)
   and test whether they mediate the regime → spatial coupling more strongly
   than exc_fraction alone (1–10% mediation).

2. Random-slope mixed model — test whether the L/λ → spatial slope varies
   across neurons, and whether inhibitory neurons have steeper slopes.

3. Compactness geometry check — diagnose the non-monotonic dose-response
   in pairwise compactness by checking branch length, synapse density,
   and compactness residuals by regime.

Output (results/replication_full/):
  partner_architecture_features.csv  — branch features + 4 partner columns
  partner_architecture_mediation.json — mediation % by feature × metric × cell type
  random_slope_results.json          — fixed/random effects, per-neuron slopes
  compactness_diagnostics.json       — branch length/density by regime, residuals
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


# ═══════════════════════════════════════════════════════════
# ANALYSIS 1: Partner Architecture Features + Mediation
# ═══════════════════════════════════════════════════════════

def gini_coefficient(counts):
    """Gini coefficient of an array of counts (0 = perfect equality, 1 = max inequality)."""
    counts = np.sort(counts).astype(float)
    n = len(counts)
    if n < 2 or counts.sum() == 0:
        return np.nan
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * counts) - (n + 1) * np.sum(counts)) / (n * np.sum(counts))


def compute_partner_architecture_features(spatial_df, data_dir):
    """Compute per-branch partner architecture features for all neurons.

    For each branch with ≥3 snapped synapses that have matched partners:
      - unique_partners: count of distinct pre_root_id values
      - mean_syn_per_partner: synapse_count / unique_partners
      - gini_partner: Gini of per-partner synapse counts
      - frac_multisynaptic: fraction of partners with ≥2 synapses on this branch

    Returns DataFrame with neuron_label, branch_idx, and the 4 features.
    """
    print("=" * 60)
    print("COMPUTING PARTNER ARCHITECTURE FEATURES")
    print("=" * 60)

    neuron_labels = spatial_df["neuron_label"].unique()
    all_records = []
    t0 = time.time()
    n_skipped = 0

    for i, nl in enumerate(neuron_labels):
        swc_path = data_dir / f"{nl}.swc"
        syn_path = data_dir / f"{nl}_synapses.csv"
        partner_path = data_dir / f"{nl}_presynaptic.csv"

        if not all(p.exists() for p in [swc_path, syn_path, partner_path]):
            n_skipped += 1
            continue

        # Load skeleton
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skeleton = NeuronSkeleton.from_swc_file(str(swc_path), scale_factor=1000.0)
        dendrite_skel = skeleton.filter_by_type([1, 3, 4])
        if len(dendrite_skel.branches) < 3:
            dendrite_skel = skeleton

        # Load and snap synapses
        syn_df = pd.read_csv(syn_path)
        syn_coords_nm = syn_df[["x_um", "y_um", "z_um"]].values * 1000.0
        snap = dendrite_skel.snap_points(syn_coords_nm, d_max=50000.0)
        valid_mask = snap.valid

        if valid_mask.sum() < 10:
            n_skipped += 1
            continue

        # Load partner data and match via KDTree
        partner_df = pd.read_csv(partner_path)
        partner_coords = partner_df[["x_nm", "y_nm", "z_nm"]].values.astype(float)
        tree = cKDTree(partner_coords)
        _, p_indices = tree.query(syn_coords_nm)
        all_partner_ids = partner_df["pre_root_id"].values[p_indices]

        # Filter to valid snapped synapses
        valid_partner_ids = all_partner_ids[valid_mask]
        valid_branch_ids = snap.branch_ids[valid_mask]

        # Build a DataFrame of (branch_idx, partner_id) per synapse
        syn_level = pd.DataFrame({
            "branch_idx": valid_branch_ids,
            "partner_id": valid_partner_ids,
        })

        # Group by branch → partner to count synapses per partner per branch
        bp_counts = syn_level.groupby(["branch_idx", "partner_id"]).size().reset_index(name="syn_count")

        # Aggregate to branch level
        for bid, grp in bp_counts.groupby("branch_idx"):
            partner_counts = grp["syn_count"].values
            n_partners = len(partner_counts)
            total_syn = partner_counts.sum()

            if total_syn < 3:
                continue

            all_records.append({
                "neuron_label": nl,
                "branch_idx": int(bid),
                "unique_partners": n_partners,
                "mean_syn_per_partner": total_syn / n_partners,
                "gini_partner": gini_coefficient(partner_counts),
                "frac_multisynaptic": (partner_counts >= 2).sum() / n_partners,
            })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(neuron_labels)} neurons ({elapsed:.0f}s, "
                  f"{len(all_records)} branch records)")

    elapsed = time.time() - t0
    print(f"\nDone: {len(all_records)} branches with partner features "
          f"from {len(neuron_labels) - n_skipped} neurons ({elapsed:.0f}s)")

    if not all_records:
        print("  ERROR: No partner features computed!")
        return pd.DataFrame()

    partner_feat_df = pd.DataFrame(all_records)

    # Sanity check: unique_partners * mean_syn_per_partner ≈ total synapses
    check = partner_feat_df["unique_partners"] * partner_feat_df["mean_syn_per_partner"]
    # Merge with spatial_df to get synapse_count
    merged_check = partner_feat_df.merge(
        spatial_df[["neuron_label", "branch_idx", "synapse_count"]],
        on=["neuron_label", "branch_idx"], how="left"
    )
    valid_check = merged_check.dropna(subset=["synapse_count"])
    if len(valid_check) > 0:
        # Partner-matched synapse count may differ from total synapse_count
        # because not all synapses match a partner, but should correlate
        corr = np.corrcoef(
            valid_check["unique_partners"] * valid_check["mean_syn_per_partner"],
            valid_check["synapse_count"]
        )[0, 1]
        print(f"  Verification: corr(partner_syn, total_syn) = {corr:.4f}")

    return partner_feat_df


def run_partner_mediation(spatial_df, partner_feat_df):
    """Test whether partner architecture features mediate regime → spatial coupling.

    Uses coefficient-comparison approach:
      Reduced: spatial_z ~ C(regime) + synapse_count + total_length_nm + (1|neuron)
      Full:    + mediator variable(s)
      Mediation % = (1 - |coef_full| / |coef_reduced|) × 100
    """
    print("\n" + "=" * 60)
    print("PARTNER ARCHITECTURE MEDIATION ANALYSIS")
    print("=" * 60)

    # Merge partner features into branch features
    merged = spatial_df.merge(partner_feat_df, on=["neuron_label", "branch_idx"], how="left")

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    mediators_individual = ["unique_partners", "mean_syn_per_partner", "gini_partner", "frac_multisynaptic"]
    mediators_combined = mediators_individual  # all four together

    results = {}

    for subset_name, subset_df in [
        ("all", merged),
        ("excitatory", merged[merged["cell_type"] == "excitatory"]),
        ("inhibitory", merged[merged["cell_type"].str.startswith("inhibitory")]),
    ]:
        print(f"\n--- {subset_name} (n={len(subset_df)}) ---")
        results[subset_name] = {}

        for metric in metrics:
            base_cols = [metric, "regime", "neuron_label", "synapse_count", "total_length_nm"]

            # Reduced model (baseline)
            work_red = subset_df.dropna(subset=base_cols).copy()
            work_red = work_red[work_red["regime"] >= 0]

            if len(work_red) < 50 or work_red["neuron_label"].nunique() < 5:
                print(f"  {metric}: insufficient data")
                continue

            try:
                formula_red = f"{metric} ~ C(regime) + synapse_count + total_length_nm"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_red = smf.mixedlm(formula_red, work_red, groups=work_red["neuron_label"])
                    res_red = model_red.fit(reml=True, maxiter=500)

                coef_red = res_red.params.get("C(regime)[T.2]", np.nan)
                metric_results = {
                    "coef_reduced": float(coef_red),
                    "n_obs_reduced": len(work_red),
                    "individual_mediators": {},
                }

                # Individual mediator models
                for med in mediators_individual:
                    full_cols = base_cols + [med]
                    work_full = subset_df.dropna(subset=full_cols).copy()
                    work_full = work_full[work_full["regime"] >= 0]

                    if len(work_full) < 50:
                        continue

                    try:
                        formula_full = f"{metric} ~ C(regime) + synapse_count + total_length_nm + {med}"
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model_full = smf.mixedlm(formula_full, work_full, groups=work_full["neuron_label"])
                            res_full = model_full.fit(reml=True, maxiter=500)

                        coef_full = res_full.params.get("C(regime)[T.2]", np.nan)
                        med_coef = res_full.params.get(med, np.nan)
                        med_p = res_full.pvalues.get(med, np.nan)

                        if abs(coef_red) > 1e-10:
                            mediation_pct = (1 - abs(coef_full) / abs(coef_red)) * 100
                        else:
                            mediation_pct = np.nan

                        metric_results["individual_mediators"][med] = {
                            "coef_full": float(coef_full),
                            "mediation_pct": float(mediation_pct),
                            "mediator_coef": float(med_coef),
                            "mediator_p": float(med_p),
                            "n_obs": len(work_full),
                        }

                        print(f"  {metric} + {med}: mediation={mediation_pct:.1f}%, "
                              f"coef={med_coef:.4f}, p={med_p:.2e}")

                    except Exception as e:
                        print(f"  {metric} + {med}: FAILED ({str(e)[:50]})")

                # Combined model (all 4 partner features)
                combined_cols = base_cols + mediators_combined
                work_comb = subset_df.dropna(subset=combined_cols).copy()
                work_comb = work_comb[work_comb["regime"] >= 0]

                if len(work_comb) >= 50:
                    try:
                        med_terms = " + ".join(mediators_combined)
                        formula_comb = f"{metric} ~ C(regime) + synapse_count + total_length_nm + {med_terms}"
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model_comb = smf.mixedlm(formula_comb, work_comb, groups=work_comb["neuron_label"])
                            res_comb = model_comb.fit(reml=True, maxiter=500)

                        coef_comb = res_comb.params.get("C(regime)[T.2]", np.nan)
                        if abs(coef_red) > 1e-10:
                            combined_mediation = (1 - abs(coef_comb) / abs(coef_red)) * 100
                        else:
                            combined_mediation = np.nan

                        metric_results["combined"] = {
                            "coef_full": float(coef_comb),
                            "mediation_pct": float(combined_mediation),
                            "n_obs": len(work_comb),
                            "individual_coefs": {
                                med: float(res_comb.params.get(med, np.nan))
                                for med in mediators_combined
                            },
                            "individual_pvalues": {
                                med: float(res_comb.pvalues.get(med, np.nan))
                                for med in mediators_combined
                            },
                        }
                        print(f"  {metric} combined: mediation={combined_mediation:.1f}% "
                              f"(n={len(work_comb)})")

                    except Exception as e:
                        print(f"  {metric} combined: FAILED ({str(e)[:50]})")

                # Also compare to exc_fraction mediation for reference
                exc_cols = base_cols + ["exc_fraction"]
                work_exc = subset_df.dropna(subset=exc_cols).copy()
                work_exc = work_exc[work_exc["regime"] >= 0]
                if len(work_exc) >= 50:
                    try:
                        formula_exc = f"{metric} ~ C(regime) + synapse_count + total_length_nm + exc_fraction"
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model_exc = smf.mixedlm(formula_exc, work_exc, groups=work_exc["neuron_label"])
                            res_exc = model_exc.fit(reml=True, maxiter=500)

                        coef_exc = res_exc.params.get("C(regime)[T.2]", np.nan)
                        if abs(coef_red) > 1e-10:
                            exc_mediation = (1 - abs(coef_exc) / abs(coef_red)) * 100
                        else:
                            exc_mediation = np.nan

                        metric_results["exc_fraction_reference"] = {
                            "coef_full": float(coef_exc),
                            "mediation_pct": float(exc_mediation),
                        }
                    except Exception:
                        pass

                results[subset_name][metric] = metric_results

            except Exception as e:
                print(f"  {metric}: FAILED baseline ({str(e)[:60]})")

    return results


# ═══════════════════════════════════════════════════════════
# ANALYSIS 2: Random-Slope Mixed Model
# ═══════════════════════════════════════════════════════════

def run_random_slope_model(spatial_df):
    """Test whether the L/λ → spatial coupling slope varies across neurons.

    Fits: spatial_z ~ electrotonic_length + synapse_count + total_length_nm
                      + (electrotonic_length | neuron)

    Compares random-slope vs random-intercept-only (LRT / AIC).
    Extracts per-neuron slopes and compares inh vs exc distributions.
    """
    print("\n" + "=" * 60)
    print("RANDOM-SLOPE MIXED MODEL")
    print("=" * 60)

    metrics = ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]
    results = {}

    for metric in metrics:
        print(f"\n--- {metric} ---")
        cols = [metric, "electrotonic_length", "synapse_count", "total_length_nm",
                "neuron_label", "cell_type"]
        work = spatial_df.dropna(subset=cols[:-1]).copy()  # cell_type can be missing
        work = work[work["regime"] >= 0]

        if len(work) < 100:
            print(f"  Insufficient data (n={len(work)})")
            continue

        formula = f"{metric} ~ electrotonic_length + synapse_count + total_length_nm"

        # Random-intercept only (baseline)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_ri = smf.mixedlm(formula, work, groups=work["neuron_label"])
                res_ri = model_ri.fit(reml=True, maxiter=500)

            aic_ri = res_ri.aic
            bic_ri = res_ri.bic
            llf_ri = res_ri.llf

            print(f"  Random-intercept: AIC={aic_ri:.1f}, BIC={bic_ri:.1f}")
        except Exception as e:
            print(f"  Random-intercept FAILED: {str(e)[:60]}")
            continue

        # Random-slope model
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_rs = smf.mixedlm(
                    formula, work, groups=work["neuron_label"],
                    re_formula="~electrotonic_length"
                )
                res_rs = model_rs.fit(reml=True, maxiter=500)

            aic_rs = res_rs.aic
            bic_rs = res_rs.bic
            llf_rs = res_rs.llf

            print(f"  Random-slope:     AIC={aic_rs:.1f}, BIC={bic_rs:.1f}")
            print(f"  ΔAIC = {aic_rs - aic_ri:.1f} (negative = slope model better)")

            # Fixed effect: population-average L/λ slope
            fixed_slope = res_rs.params.get("electrotonic_length", np.nan)
            fixed_slope_p = res_rs.pvalues.get("electrotonic_length", np.nan)
            print(f"  Fixed L/λ slope: {fixed_slope:.4f}, p={fixed_slope_p:.2e}")

            # Random slope variance
            # In statsmodels, random effects covariance is in res.cov_re
            cov_re = res_rs.cov_re
            if hasattr(cov_re, "iloc"):
                slope_var = cov_re.iloc[1, 1] if cov_re.shape[0] > 1 else np.nan
                intercept_var = cov_re.iloc[0, 0]
            else:
                slope_var = np.nan
                intercept_var = np.nan

            print(f"  Random slope variance: {slope_var:.6f}")
            print(f"  Random intercept variance: {intercept_var:.6f}")

            # Extract per-neuron slopes (BLUPs)
            neuron_slopes = {}
            for neuron_id, re_dict in res_rs.random_effects.items():
                if "electrotonic_length" in re_dict:
                    neuron_slopes[neuron_id] = float(
                        fixed_slope + re_dict["electrotonic_length"]
                    )
                elif len(re_dict) > 1:
                    # Second element is the slope
                    keys = list(re_dict.keys())
                    neuron_slopes[neuron_id] = float(
                        fixed_slope + re_dict[keys[1]]
                    )

            print(f"  Per-neuron slopes extracted: {len(neuron_slopes)}")

            # Compare excitatory vs inhibitory slope distributions
            neuron_types = spatial_df.drop_duplicates("neuron_label").set_index("neuron_label")["cell_type"]

            exc_slopes = [s for n, s in neuron_slopes.items()
                          if neuron_types.get(n, "") == "excitatory"]
            inh_slopes = [s for n, s in neuron_slopes.items()
                          if str(neuron_types.get(n, "")).startswith("inhibitory")]

            slope_comparison = {}
            if len(exc_slopes) >= 5 and len(inh_slopes) >= 5:
                exc_arr = np.array(exc_slopes)
                inh_arr = np.array(inh_slopes)

                U, p_mw = stats.mannwhitneyu(exc_arr, inh_arr, alternative="two-sided")
                print(f"\n  Excitatory slopes: median={np.median(exc_arr):.4f}, "
                      f"n={len(exc_arr)}")
                print(f"  Inhibitory slopes: median={np.median(inh_arr):.4f}, "
                      f"n={len(inh_arr)}")
                print(f"  Mann-Whitney U: U={U:.0f}, p={p_mw:.2e}")

                # Cohen's d
                pooled_std = np.sqrt(
                    ((len(exc_arr) - 1) * np.var(exc_arr, ddof=1) +
                     (len(inh_arr) - 1) * np.var(inh_arr, ddof=1)) /
                    (len(exc_arr) + len(inh_arr) - 2)
                )
                cohens_d = (np.mean(inh_arr) - np.mean(exc_arr)) / pooled_std if pooled_std > 0 else np.nan
                print(f"  Cohen's d (inh - exc): {cohens_d:.4f}")

                slope_comparison = {
                    "exc_median": float(np.median(exc_arr)),
                    "exc_mean": float(np.mean(exc_arr)),
                    "exc_n": len(exc_arr),
                    "inh_median": float(np.median(inh_arr)),
                    "inh_mean": float(np.mean(inh_arr)),
                    "inh_n": len(inh_arr),
                    "mann_whitney_U": float(U),
                    "mann_whitney_p": float(p_mw),
                    "cohens_d": float(cohens_d),
                }

            # Likelihood ratio test (random-slope vs random-intercept)
            # LRT = -2 * (llf_restricted - llf_full), df = 2 (slope var + covariance)
            lrt_stat = -2 * (llf_ri - llf_rs)
            # Chi-squared with 2 df (slope variance + covariance term)
            lrt_p = stats.chi2.sf(lrt_stat, df=2)
            print(f"\n  LRT (slope vs intercept): χ²={lrt_stat:.2f}, p={lrt_p:.2e}")

            results[metric] = {
                "fixed_slope": float(fixed_slope),
                "fixed_slope_p": float(fixed_slope_p),
                "random_slope_variance": float(slope_var) if not np.isnan(slope_var) else None,
                "random_intercept_variance": float(intercept_var) if not np.isnan(intercept_var) else None,
                "aic_intercept_only": float(aic_ri),
                "aic_random_slope": float(aic_rs),
                "bic_intercept_only": float(bic_ri),
                "bic_random_slope": float(bic_rs),
                "delta_aic": float(aic_rs - aic_ri),
                "lrt_statistic": float(lrt_stat),
                "lrt_p": float(lrt_p),
                "n_neurons_with_slopes": len(neuron_slopes),
                "slope_comparison_exc_vs_inh": slope_comparison,
                "n_obs": len(work),
            }

        except Exception as e:
            print(f"  Random-slope FAILED: {str(e)[:80]}")
            results[metric] = {
                "error": str(e),
                "aic_intercept_only": float(aic_ri),
                "n_obs": len(work),
            }

    return results


# ═══════════════════════════════════════════════════════════
# ANALYSIS 3: Compactness Geometry Diagnostics
# ═══════════════════════════════════════════════════════════

def compactness_diagnostics(spatial_df):
    """Diagnose the non-monotonic compactness dose-response.

    Checks:
    1. Branch length distribution by regime
    2. Synapse density by regime
    3. Compactness residuals vs branch length (interaction check)
    """
    print("\n" + "=" * 60)
    print("COMPACTNESS GEOMETRY DIAGNOSTICS")
    print("=" * 60)

    work = spatial_df.dropna(subset=["regime", "pairwise_compactness_z", "total_length_nm",
                                      "synapse_count"]).copy()
    work = work[work["regime"] >= 0]
    work["syn_density"] = work["synapse_count"] / work["total_length_nm"]

    results = {}

    # 1. Branch length by regime
    print("\n--- Branch length by regime ---")
    length_by_regime = {}
    for r in [0, 1, 2]:
        vals = work[work["regime"] == r]["total_length_nm"].values
        length_by_regime[r] = vals
        print(f"  Regime {r}: n={len(vals)}, median={np.median(vals):.0f} nm, "
              f"mean={np.mean(vals):.0f} nm, IQR=[{np.percentile(vals, 25):.0f}, "
              f"{np.percentile(vals, 75):.0f}]")

    H, p = stats.kruskal(*[length_by_regime[r] for r in [0, 1, 2]])
    print(f"  Kruskal-Wallis: H={H:.2f}, p={p:.2e}")
    results["branch_length_by_regime"] = {
        str(r): {
            "n": len(v), "median": float(np.median(v)),
            "mean": float(np.mean(v)), "std": float(np.std(v)),
            "q25": float(np.percentile(v, 25)),
            "q75": float(np.percentile(v, 75)),
        }
        for r, v in length_by_regime.items()
    }
    results["branch_length_kw"] = {"H": float(H), "p": float(p)}

    # 2. Synapse density by regime
    print("\n--- Synapse density (syn/nm) by regime ---")
    density_by_regime = {}
    for r in [0, 1, 2]:
        vals = work[work["regime"] == r]["syn_density"].values
        density_by_regime[r] = vals
        print(f"  Regime {r}: n={len(vals)}, median={np.median(vals):.2e}, "
              f"mean={np.mean(vals):.2e}")

    H_d, p_d = stats.kruskal(*[density_by_regime[r] for r in [0, 1, 2]])
    print(f"  Kruskal-Wallis: H={H_d:.2f}, p={p_d:.2e}")
    results["synapse_density_by_regime"] = {
        str(r): {
            "n": len(v), "median": float(np.median(v)),
            "mean": float(np.mean(v)),
        }
        for r, v in density_by_regime.items()
    }
    results["synapse_density_kw"] = {"H": float(H_d), "p": float(p_d)}

    # 3. Compactness residuals vs branch length — check for regime × length interaction
    print("\n--- Compactness residual analysis ---")
    # Fit compactness ~ synapse_count + total_length_nm to get residuals
    try:
        import statsmodels.api as sm
        X = work[["synapse_count", "total_length_nm"]].copy()
        X = sm.add_constant(X)
        y = work["pairwise_compactness_z"].values
        ols = sm.OLS(y, X).fit()
        work = work.copy()
        work["compactness_resid"] = ols.resid

        # Residual means by regime
        print("  Compactness residuals by regime (after removing length + count effects):")
        resid_by_regime = {}
        for r in [0, 1, 2]:
            resids = work[work["regime"] == r]["compactness_resid"].values
            resid_by_regime[r] = resids
            print(f"    Regime {r}: mean resid = {np.mean(resids):.4f} ± {np.std(resids)/np.sqrt(len(resids)):.4f}")

        H_r, p_r = stats.kruskal(*[resid_by_regime[r] for r in [0, 1, 2]])
        print(f"    Kruskal-Wallis on residuals: H={H_r:.2f}, p={p_r:.2e}")

        results["compactness_residuals"] = {
            str(r): {
                "mean_resid": float(np.mean(v)),
                "se_resid": float(np.std(v) / np.sqrt(len(v))),
                "n": len(v),
            }
            for r, v in resid_by_regime.items()
        }
        results["compactness_residuals_kw"] = {"H": float(H_r), "p": float(p_r)}

        # Check regime × length interaction
        work["regime_1"] = (work["regime"] == 1).astype(float)
        work["regime_2"] = (work["regime"] == 2).astype(float)
        work["len_x_r1"] = work["total_length_nm"] * work["regime_1"]
        work["len_x_r2"] = work["total_length_nm"] * work["regime_2"]

        X_int = work[["synapse_count", "total_length_nm", "regime_1", "regime_2",
                       "len_x_r1", "len_x_r2"]].copy()
        X_int = sm.add_constant(X_int)
        ols_int = sm.OLS(work["pairwise_compactness_z"].values, X_int).fit()

        int_r1_coef = ols_int.params.get("len_x_r1", np.nan)
        int_r1_p = ols_int.pvalues.get("len_x_r1", np.nan)
        int_r2_coef = ols_int.params.get("len_x_r2", np.nan)
        int_r2_p = ols_int.pvalues.get("len_x_r2", np.nan)

        print(f"\n  Regime × branch_length interaction:")
        print(f"    Regime 1 × length: coef={int_r1_coef:.2e}, p={int_r1_p:.2e}")
        print(f"    Regime 2 × length: coef={int_r2_coef:.2e}, p={int_r2_p:.2e}")

        results["regime_length_interaction"] = {
            "regime_1_x_length_coef": float(int_r1_coef),
            "regime_1_x_length_p": float(int_r1_p),
            "regime_2_x_length_coef": float(int_r2_coef),
            "regime_2_x_length_p": float(int_r2_p),
        }

    except Exception as e:
        print(f"  Residual analysis failed: {str(e)[:60]}")

    # 4. Per-regime monotonicity check: mean compactness z by regime
    print("\n--- Compactness z-scores by regime (monotonicity check) ---")
    for metric in ["clark_evans_z", "interval_cv_z", "pairwise_compactness_z"]:
        vals_by_regime = {}
        for r in [0, 1, 2]:
            sub = work[work["regime"] == r]
            vals = sub[metric].dropna().values
            vals_by_regime[r] = (np.mean(vals), np.std(vals) / np.sqrt(len(vals)), len(vals))

        print(f"  {metric}:")
        for r in [0, 1, 2]:
            m, se, n = vals_by_regime[r]
            print(f"    Regime {r}: mean={m:.4f} ± {se:.4f} (n={n})")

        # Check monotonicity
        means = [vals_by_regime[r][0] for r in [0, 1, 2]]
        is_monotonic = (means[0] <= means[1] <= means[2]) or (means[0] >= means[1] >= means[2])
        print(f"    Monotonic: {is_monotonic}")

        results[f"{metric}_regime_means"] = {
            str(r): {"mean": float(m), "se": float(se), "n": int(n)}
            for r, (m, se, n) in vals_by_regime.items()
        }

    return results


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    t0_total = time.time()

    # Load branch features
    spatial_df = pd.read_csv(RESULTS_DIR / "all_branch_features_with_spatial.csv")
    print(f"Loaded {len(spatial_df)} branches, {spatial_df['neuron_label'].nunique()} neurons\n")

    # Analysis 1: Partner architecture features
    partner_feat_df = compute_partner_architecture_features(spatial_df, MICRONS_DATA)

    if len(partner_feat_df) > 0:
        # Save augmented features
        augmented = spatial_df.merge(partner_feat_df, on=["neuron_label", "branch_idx"], how="left")
        augmented.to_csv(RESULTS_DIR / "partner_architecture_features.csv", index=False)
        print(f"\nSaved partner_architecture_features.csv ({len(augmented)} rows)")

        # Analysis 1b: Mediation tests
        mediation_results = run_partner_mediation(spatial_df, partner_feat_df)
    else:
        mediation_results = {"error": "No partner features computed"}

    # Analysis 2: Random-slope model
    random_slope_results = run_random_slope_model(spatial_df)

    # Analysis 3: Compactness diagnostics
    compact_results = compactness_diagnostics(spatial_df)

    # Save all results
    with open(RESULTS_DIR / "partner_architecture_mediation.json", "w") as f:
        json.dump(mediation_results, f, indent=2, default=str)
    print(f"\nSaved partner_architecture_mediation.json")

    with open(RESULTS_DIR / "random_slope_results.json", "w") as f:
        json.dump(random_slope_results, f, indent=2, default=str)
    print(f"Saved random_slope_results.json")

    with open(RESULTS_DIR / "compactness_diagnostics.json", "w") as f:
        json.dump(compact_results, f, indent=2, default=str)
    print(f"Saved compactness_diagnostics.json")

    elapsed = time.time() - t0_total
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
