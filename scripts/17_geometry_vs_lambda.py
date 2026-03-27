#!/usr/bin/env python3
"""Two analyses to resolve the electrotonic-regime framing.

Analysis 1 — Geometry vs L/λ discriminator
  Does electrotonic length (L/λ) predict synaptic spacing and BC afferent
  fraction beyond what raw geometry already captures?

  Method: residualize L/λ on geometry features (soma_distance, branch_order,
  diameter, branch_length) to extract the "biophysical component" that geometry
  alone does not explain. Then test whether this residual L/λ adds incremental
  predictive power in LME models for (a) spacing metrics and (b) BC afferent
  fraction. Avoids collinearity from including correlated geometry features
  alongside raw L/λ.

Analysis 2 — B1/B2 cross-axon pair-correlation
  Is inhibitory synapse regularity driven by postsynaptic scaffold exclusion
  (repulsion hole across different axons → B1) or presynaptic axon tiling
  (regularity only within-axon → B2)?

  Method: for inhibitory neurons, compute nearest-neighbour distances between
  inhibitory synapses on the same branch, stratified by same-axon vs
  different-axon. If cross-axon nearest-neighbour distances show a hard-core
  exclusion zone (mode > 0, suppression at short distances relative to a
  within-branch shuffle), that is B1. If the exclusion zone is confined to
  within-axon pairs, that is B2.

Output: results/replication_full/geometry_lambda_discriminator.json
"""

import json
import sys
import warnings
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "neurons"
RESULTS_DIR = PROJECT_DIR / "results" / "replication_full"

sys.path.insert(0, str(Path.home() / "research" / "neurostat"))
sys.path.insert(0, str(Path.home() / "research" / "neurostat-input-clustering" / "src"))
sys.path.insert(0, str(Path.home() / "research" / "dendritic-regime-coupling"))

import statsmodels.formula.api as smf
from neurostat.io.swc import NeuronSkeleton


# ── Import helpers from script 12 ───────────────────────────────────────────

def _import_script12():
    spec = spec_from_file_location(
        "inhibitory_residual",
        PROJECT_DIR / "scripts" / "12_inhibitory_residual.py"
    )
    mod = module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(mod)
    return mod


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: GEOMETRY vs L/λ DISCRIMINATOR
# ════════════════════════════════════════════════════════════════════════════

def residualize_electrotonic_length(df):
    """Regress L/λ on geometry features; return residuals = biophysical component.

    The collinearity-safe approach: L/λ = f(soma_distance, branch_order,
    diameter, length) + ε. The residual ε is what L/λ captures that raw
    geometry does not.
    """
    sub = df.dropna(subset=["electrotonic_length", "soma_distance_nm",
                             "branch_order", "mean_diameter_nm", "total_length_nm"]).copy()
    sub = sub[sub["electrotonic_length"] > 0]
    sub["log_soma_dist"] = np.log1p(sub["soma_distance_nm"])
    sub["log_diameter"] = np.log1p(sub["mean_diameter_nm"])
    sub["log_length"] = np.log1p(sub["total_length_nm"])
    sub["log_elec"] = np.log(sub["electrotonic_length"])

    # OLS: log(L/λ) ~ log_soma_dist + branch_order + log_diameter + log_length
    from sklearn.linear_model import LinearRegression
    X = sub[["log_soma_dist", "branch_order", "log_diameter", "log_length"]].values
    y = sub["log_elec"].values
    reg = LinearRegression().fit(X, y)
    sub["elec_resid"] = y - reg.predict(X)

    r2 = reg.score(X, y)
    print(f"    L/λ ~ geometry R² = {r2:.3f}  (geometry explains {100*r2:.1f}% of L/λ variance)")
    print(f"    Geometry coefs: soma_dist={reg.coef_[0]:+.3f}, "
          f"order={reg.coef_[1]:+.3f}, diam={reg.coef_[2]:+.3f}, len={reg.coef_[3]:+.3f}")
    return sub, r2, reg.coef_.tolist(), float(reg.intercept_)


def run_incremental_lme(df, outcome, geom_formula_part, elec_col, group_col, label):
    """Compare geometry-only vs geometry+L/λ LME for a given outcome.

    Returns dict with geometry-only and full model coefficients and LRT p-value.
    """
    sub = df.dropna(subset=[outcome, elec_col, group_col,
                             "log_soma_dist", "branch_order",
                             "log_diameter", "log_length", "synapse_count"]).copy()
    if len(sub) < 500:
        return {"n": len(sub), "note": "insufficient"}

    covars = "synapse_count + log_soma_dist + branch_order + log_diameter + log_length"
    formula_geom = f"{outcome} ~ {covars}"
    formula_full = f"{outcome} ~ {covars} + {elec_col}"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_geom = smf.mixedlm(formula_geom, sub, groups=sub[group_col]).fit(
                reml=False, method="lbfgs")
            m_full = smf.mixedlm(formula_full, sub, groups=sub[group_col]).fit(
                reml=False, method="lbfgs")

        # Likelihood ratio test
        lrt_stat = 2 * (m_full.llf - m_geom.llf)
        lrt_p = stats.chi2.sf(lrt_stat, df=1)

        elec_coef = float(m_full.params.get(elec_col, np.nan))
        elec_p = float(m_full.pvalues.get(elec_col, np.nan))

        print(f"    {label}: n={len(sub):,}  {elec_col} coef={elec_coef:+.4f}  "
              f"p={elec_p:.3e}  LRT χ²={lrt_stat:.2f}  LRT_p={lrt_p:.3e}")

        return {
            "n": len(sub),
            "elec_coef": elec_coef,
            "elec_p": elec_p,
            "lrt_chi2": float(lrt_stat),
            "lrt_p": float(lrt_p),
            "geom_llf": float(m_geom.llf),
            "full_llf": float(m_full.llf),
        }
    except Exception as e:
        print(f"    {label}: model failed — {e}")
        return {"n": len(sub), "note": str(e)}


def analysis1_geometry_vs_lambda():
    """Test whether L/λ predicts spacing/BC fraction beyond raw geometry."""
    print("\n" + "=" * 60)
    print("ANALYSIS 1: GEOMETRY vs L/λ DISCRIMINATOR")
    print("=" * 60)

    bf = pd.read_csv(RESULTS_DIR / "afferent_branch_features.csv")
    # Rename regime columns
    if "regime_x" in bf.columns:
        bf = bf.rename(columns={"regime_x": "regime"})
    bf = bf[bf["synapse_count"] >= 3].copy()

    print(f"\n  Total branches: {len(bf):,}  neurons: {bf['neuron_label'].nunique()}")

    # ── Step 1: Residualize L/λ on geometry ──────────────────────────────
    print("\n[1a] Residualizing L/λ on geometry features...")
    bf_valid, r2_geom, coefs, intercept = residualize_electrotonic_length(bf)

    # Merge residuals back
    bf_valid = bf_valid.copy()
    elec_col = "elec_resid"  # the biophysical residual

    print("\n[1b] Also testing raw electrotonic_length (for comparison)...")

    # R²≈1 is expected: log(L/λ) = log(L) - 0.5·log(d) + const (cable formula).
    # The residualization therefore extracts ~0 residual. The incremental LME
    # tests below are numerically equivalent to testing raw L/λ with geometry
    # covariates — which asks whether the specific L/sqrt(d) combination
    # predicts spacing beyond L and d entered as independent linear predictors.
    results = {
        "geometry_r2": r2_geom,
        "geometry_coefs": coefs,
        "interpretation_note": (
            "R²=1.000 confirms L/λ is fully determined by branch length and diameter "
            "(cable formula). The incremental LME tests whether the biophysically "
            "specific L/sqrt(d) combination predicts spacing beyond L and d as "
            "independent covariates. Significant LRT = the ratio matters, not just "
            "each dimension alone."
        )
    }

    # ── Step 2: Does residualized L/λ predict spacing? ───────────────────
    print("\n[2] Incremental LME: does residual L/λ predict spacing metrics?")
    print("    (after controlling for geometry; random intercept per neuron)")

    populations = {
        "inhibitory": bf_valid[bf_valid["cell_type"] == "inhibitory"],
        "excitatory": bf_valid[bf_valid["cell_type"] == "excitatory"],
    }

    spacing_results = {}
    for pop, sub in populations.items():
        print(f"\n  Population: {pop} ({len(sub):,} branches)")
        spacing_results[pop] = {}
        for metric in ["interval_cv_z", "clark_evans_z", "pairwise_compactness_z"]:
            r = run_incremental_lme(
                sub, metric, "", elec_col, "neuron_label",
                f"{pop}/{metric}/residL"
            )
            spacing_results[pop][metric] = {"residualized_elec": r}

            # Also test raw L/λ for comparison
            sub2 = sub.copy()
            sub2["log_elec"] = np.log(sub2["electrotonic_length"].clip(1e-6))
            r_raw = run_incremental_lme(
                sub2, metric, "", "log_elec", "neuron_label",
                f"{pop}/{metric}/rawL"
            )
            spacing_results[pop][metric]["raw_elec"] = r_raw

    results["spacing_incremental"] = spacing_results

    # ── Step 3: Does residualized L/λ predict BC afferent fraction? ──────
    print("\n[3] Incremental LME: does residual L/λ predict BC afferent fraction?")
    print("    (inhibitory neurons only; branches with ≥1 typed synapse)")

    inh_bf = bf_valid[(bf_valid["cell_type"] == "inhibitory") &
                       (bf_valid["n_typed_synapses"] >= 1)].copy()

    bc_results = {}
    for col, label in [("frac_BC_afferent", "BC fraction"),
                        ("frac_inh_afferent", "Inh fraction"),
                        ("frac_exc_afferent", "Exc fraction")]:
        r = run_incremental_lme(inh_bf, col, "", elec_col, "neuron_label",
                                f"inh/{col}/residL")
        bc_results[col] = {"residualized_elec": r}

        inh_bf2 = inh_bf.copy()
        inh_bf2["log_elec"] = np.log(inh_bf2["electrotonic_length"].clip(1e-6))
        r_raw = run_incremental_lme(inh_bf2, col, "", "log_elec", "neuron_label",
                                    f"inh/{col}/rawL")
        bc_results[col]["raw_elec"] = r_raw

    results["bc_fraction_incremental"] = bc_results

    # ── Step 4: Summary interpretation ───────────────────────────────────
    print("\n[4] Summary: L/λ survival after geometry control")
    for pop in ["inhibitory", "excitatory"]:
        for metric in ["interval_cv_z", "clark_evans_z"]:
            r = spacing_results[pop][metric]["residualized_elec"]
            if "lrt_p" in r:
                survives = "SURVIVES" if r["lrt_p"] < 0.05 else "COLLAPSES"
                print(f"  {pop:12s} {metric:25s}: {survives}  "
                      f"(LRT p={r['lrt_p']:.3e}, coef={r['elec_coef']:+.4f})")

    return results


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: B1 vs B2 — CROSS-AXON PAIR-CORRELATION
# ════════════════════════════════════════════════════════════════════════════

def get_synapse_arclengths(nl, s12):
    """Return DataFrame with per-synapse arc-length positions and presynaptic identity.

    Columns: branch_idx, arc_pos_nm, pre_root_id, pre_subtype, regime
    """
    result = s12._load_neuron_data(nl, DATA_DIR)
    if result is None:
        return None
    dendrite_skel, syn_coords_nm, snap, partner_df = result
    matched = s12._match_partners(syn_coords_nm, snap, partner_df)

    # Build arc-length positions (snap.branch_positions = nm from branch start)
    valid_mask = snap.valid
    arc_positions = snap.branch_positions  # full array, indexed by synapse_idx

    rows = []
    for _, row in matched.iterrows():
        sidx = int(row["synapse_idx"])
        rows.append({
            "branch_idx": int(row["branch_idx"]),
            "arc_pos_nm": float(arc_positions[sidx]),
            "pre_root_id": int(row["pre_root_id"]),
            "pre_subtype": row["subtype"],   # BC / MC / None
            "pre_broad": row["broad_class"], # inhibitory / excitatory / unknown
        })

    df = pd.DataFrame(rows)

    # Merge regime
    bf = pd.read_csv(RESULTS_DIR / "all_branch_features_with_spatial.csv")
    bf_nl = bf[bf["neuron_label"] == nl][["branch_idx", "regime",
                                           "total_length_nm"]].copy()
    df = df.merge(bf_nl, on="branch_idx", how="left")
    return df


def compute_nn_distances_on_branch(positions_by_axon, total_length_nm):
    """For synapses on one branch, compute nearest-neighbour distances:
    same-axon and cross-axon pairs.

    positions_by_axon: dict {pre_root_id: [arc_pos_nm, ...]}
    Returns (same_nn, cross_nn) lists of nearest-neighbour distances in µm.
    """
    all_pos = []   # (arc_pos, axon_id)
    for axon_id, positions in positions_by_axon.items():
        for p in positions:
            all_pos.append((p, axon_id))

    if len(all_pos) < 2:
        return [], []

    all_pos.sort(key=lambda x: x[0])
    positions_arr = np.array([p for p, _ in all_pos])
    axon_ids = [a for _, a in all_pos]

    same_nn = []
    cross_nn = []

    for i, (pos_i, axon_i) in enumerate(all_pos):
        # search left and right for nearest same-axon and nearest cross-axon
        nearest_same = np.inf
        nearest_cross = np.inf

        for j, (pos_j, axon_j) in enumerate(all_pos):
            if i == j:
                continue
            d = abs(pos_i - pos_j) / 1000.0  # convert nm → µm
            if axon_i == axon_j:
                if d < nearest_same:
                    nearest_same = d
            else:
                if d < nearest_cross:
                    nearest_cross = d

        if nearest_same < np.inf:
            same_nn.append(nearest_same)
        if nearest_cross < np.inf:
            cross_nn.append(nearest_cross)

    return same_nn, cross_nn


def compute_shuffle_nn(positions_by_axon, n_shuffles=200):
    """Shuffle axon labels within branch and compute cross-axon NN distribution."""
    all_pos = []
    all_axon = []
    for axon_id, positions in positions_by_axon.items():
        for p in positions:
            all_pos.append(p)
            all_axon.append(axon_id)

    if len(all_pos) < 2:
        return []

    all_pos = np.array(all_pos)
    all_axon = np.array(all_axon)

    shuffle_medians = []
    rng = np.random.default_rng(42)
    for _ in range(n_shuffles):
        shuffled_axon = rng.permutation(all_axon)
        cross_nn = []
        for i in range(len(all_pos)):
            mask = shuffled_axon != shuffled_axon[i]
            if mask.sum() == 0:
                continue
            d = np.abs(all_pos[i] - all_pos[mask]).min() / 1000.0
            cross_nn.append(d)
        if cross_nn:
            shuffle_medians.append(np.median(cross_nn))

    return shuffle_medians


def analysis2_b1_b2_pair_correlation():
    """Test B1 vs B2: cross-axon vs within-axon inhibitory nearest-neighbour spacing."""
    print("\n" + "=" * 60)
    print("ANALYSIS 2: B1 vs B2 — CROSS-AXON PAIR-CORRELATION")
    print("=" * 60)
    print("B1 (postsynaptic scaffold exclusion): repulsion hole across axons")
    print("B2 (presynaptic tiling): regularity within-axon only")

    s12 = _import_script12()

    # Load inhibitory neuron list
    bf = pd.read_csv(RESULTS_DIR / "all_branch_features_with_spatial.csv")
    inh_neurons = bf[bf["cell_type"] == "inhibitory"]["neuron_label"].unique()
    print(f"\n  Processing {len(inh_neurons)} inhibitory neurons...")

    same_nn_all = {"all_regimes": [], "regime0": [], "regime1": [], "regime2": []}
    cross_nn_all = {"all_regimes": [], "regime0": [], "regime1": [], "regime2": []}
    cross_nn_shuffle = []

    neurons_processed = 0
    branches_used = 0

    for nl in inh_neurons:
        syn_df = get_synapse_arclengths(nl, s12)
        if syn_df is None or len(syn_df) < 5:
            continue

        # Filter to inhibitory presynaptic axons only
        inh_syn = syn_df[syn_df["pre_broad"] == "inhibitory"].copy()
        if len(inh_syn) < 3:
            continue

        neurons_processed += 1

        # Per branch
        for (bid, regime), bgrp in inh_syn.groupby(["branch_idx", "regime"]):
            if len(bgrp) < 2:
                continue

            # Need ≥2 distinct presynaptic axons for cross-axon NN
            positions_by_axon = {}
            for _, row in bgrp.iterrows():
                rid = row["pre_root_id"]
                if rid not in positions_by_axon:
                    positions_by_axon[rid] = []
                positions_by_axon[rid].append(row["arc_pos_nm"])

            if len(positions_by_axon) < 2:
                continue

            total_len = bgrp["total_length_nm"].iloc[0]
            same_nn, cross_nn = compute_nn_distances_on_branch(
                positions_by_axon, total_len)

            if same_nn:
                same_nn_all["all_regimes"].extend(same_nn)
                same_nn_all[f"regime{int(regime)}"].extend(same_nn)
            if cross_nn:
                cross_nn_all["all_regimes"].extend(cross_nn)
                cross_nn_all[f"regime{int(regime)}"].extend(cross_nn)

            # Shuffle null (subsample to keep runtime bounded)
            if branches_used < 5000 and len(positions_by_axon) >= 2:
                shuf = compute_shuffle_nn(positions_by_axon, n_shuffles=50)
                cross_nn_shuffle.extend(shuf)

            branches_used += 1

    print(f"\n  Neurons processed: {neurons_processed}")
    print(f"  Branches used: {branches_used:,}")
    print(f"  Same-axon NN pairs: {len(same_nn_all['all_regimes']):,}")
    print(f"  Cross-axon NN pairs: {len(cross_nn_all['all_regimes']):,}")

    # ── Statistics ───────────────────────────────────────────────────────
    results = {}

    same = np.array(same_nn_all["all_regimes"])
    cross = np.array(cross_nn_all["all_regimes"])
    shuffle = np.array(cross_nn_shuffle)

    print("\n  Nearest-neighbour distance distributions (µm):")
    print(f"    Same-axon  — median={np.median(same):.2f}  "
          f"mean={np.mean(same):.2f}  5th pct={np.percentile(same, 5):.2f}")
    print(f"    Cross-axon — median={np.median(cross):.2f}  "
          f"mean={np.mean(cross):.2f}  5th pct={np.percentile(cross, 5):.2f}")
    if len(shuffle) > 0:
        print(f"    Shuffle null (cross-axon) — median={np.median(shuffle):.2f}")

    # Mann-Whitney: is cross-axon NN larger than same-axon NN?
    # (if B1: cross-axon NN should be larger due to exclusion zone)
    u_stat, u_p = stats.mannwhitneyu(cross, same, alternative="greater")
    print(f"\n  Cross > Same NN (Mann-Whitney): U={u_stat:.0f}  p={u_p:.3e}")

    # Key test: is cross-axon NN LARGER than shuffle null?
    # B1 predicts: cross-axon NN > shuffle (the shuffle destroys axon identity
    # but preserves positions; if postsynaptic exclusion operates regardless
    # of axon identity, cross-axon NN ≈ shuffle. B1 signature is cross_nn ≈ shuffle
    # and same_nn > random baseline. B2 signature: same_nn >> cross_nn, cross_nn ≈ shuffle)
    if len(shuffle) > 0:
        u2, p2 = stats.mannwhitneyu(cross, shuffle, alternative="two-sided")
        print(f"  Cross vs Shuffle NN (Mann-Whitney): U={u2:.0f}  p={p2:.3e}")

    # Hard-core fraction: what fraction of same-axon NN < 1µm?
    same_hc_frac = (same < 1.0).mean()
    cross_hc_frac = (cross < 1.0).mean()
    print(f"\n  Hard-core fraction (<1µm):")
    print(f"    Same-axon:  {100*same_hc_frac:.1f}%")
    print(f"    Cross-axon: {100*cross_hc_frac:.1f}%")

    # By regime
    print("\n  NN distances by regime:")
    for reg in ["regime0", "regime1", "regime2"]:
        s_r = np.array(same_nn_all[reg])
        c_r = np.array(cross_nn_all[reg])
        if len(s_r) > 10 and len(c_r) > 10:
            print(f"    {reg}: same median={np.median(s_r):.2f}µm  "
                  f"cross median={np.median(c_r):.2f}µm  "
                  f"ratio={np.median(c_r)/np.median(s_r):.2f}")

    # ── Verdict ──────────────────────────────────────────────────────────
    # B1 (postsynaptic scaffold exclusion): inhibitory scaffold sites are spaced
    #   out on the postsynaptic membrane regardless of which axon makes the contact.
    #   Prediction: same-axon NN ≈ cross-axon NN (both show similar exclusion zones);
    #   cross-axon NN ≥ shuffle null (scaffold repels all inhibitory contacts).
    # B2 (presynaptic axon tiling): each axon distributes its own contacts to cover
    #   territory (within-axon self-avoidance). Different axons do not interact.
    #   Prediction: same-axon NN >> cross-axon NN; cross-axon NN ≈ shuffle null.
    #
    # ratio = cross_median / same_median:
    #   ratio ≈ 1 → B1 (cross ≈ same, exclusion regardless of axon identity)
    #   ratio << 1 → B2 (same >> cross, within-axon spacing only)
    ratio = np.median(cross) / np.median(same)
    cross_vs_shuffle_ratio = np.median(cross) / np.median(shuffle) if len(shuffle) > 0 else np.nan

    if ratio > 0.7:
        verdict = "B1"
        verdict_detail = (f"Cross-axon NN (median={np.median(cross):.2f}µm) ≈ "
                          f"same-axon NN (median={np.median(same):.2f}µm), ratio={ratio:.2f}. "
                          f"Exclusion operates regardless of axon identity → "
                          f"consistent with postsynaptic scaffold exclusion (B1).")
    elif ratio < 0.4:
        verdict = "B2"
        verdict_detail = (f"Same-axon NN (median={np.median(same):.2f}µm) >> "
                          f"cross-axon NN (median={np.median(cross):.2f}µm), ratio={ratio:.2f}. "
                          f"Spacing regularity is within-axon only; different axons pack closely "
                          f"(cross NN {cross_vs_shuffle_ratio:.2f}× shuffle null) → "
                          f"consistent with presynaptic axon tiling (B2).")
    else:
        verdict = "MIXED"
        verdict_detail = (f"Intermediate ratio={ratio:.2f}. "
                          f"Elements of both B1 (some cross-axon spacing) and B2 (stronger within-axon regularity).")

    print(f"\n  VERDICT: {verdict}")
    print(f"  {verdict_detail}")

    results = {
        "neurons_processed": neurons_processed,
        "branches_used": branches_used,
        "same_axon_nn": {
            "n": len(same),
            "median_um": float(np.median(same)),
            "mean_um": float(np.mean(same)),
            "p5_um": float(np.percentile(same, 5)),
            "p25_um": float(np.percentile(same, 25)),
        },
        "cross_axon_nn": {
            "n": len(cross),
            "median_um": float(np.median(cross)),
            "mean_um": float(np.mean(cross)),
            "p5_um": float(np.percentile(cross, 5)),
            "p25_um": float(np.percentile(cross, 25)),
        },
        "shuffle_null_median_um": float(np.median(shuffle)) if len(shuffle) > 0 else None,
        "cross_vs_same_mw_p": float(u_p),
        "hard_core_frac_1um": {
            "same_axon": float(same_hc_frac),
            "cross_axon": float(cross_hc_frac),
        },
        "cross_same_ratio": float(ratio),
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "by_regime": {}
    }

    for reg in ["regime0", "regime1", "regime2"]:
        s_r = np.array(same_nn_all[reg])
        c_r = np.array(cross_nn_all[reg])
        if len(s_r) > 10 and len(c_r) > 10:
            results["by_regime"][reg] = {
                "same_median_um": float(np.median(s_r)),
                "cross_median_um": float(np.median(c_r)),
                "ratio": float(np.median(c_r) / np.median(s_r)),
                "n_same": len(s_r),
                "n_cross": len(c_r),
            }

    return results


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("GEOMETRY vs L/λ AND B1/B2 PAIR-CORRELATION ANALYSES")
    print("=" * 60)

    output = {}

    # Analysis 1
    try:
        output["analysis1_geometry_lambda"] = analysis1_geometry_vs_lambda()
    except Exception as e:
        print(f"\nAnalysis 1 failed: {e}")
        import traceback; traceback.print_exc()
        output["analysis1_geometry_lambda"] = {"error": str(e)}

    # Analysis 2
    try:
        output["analysis2_b1_b2"] = analysis2_b1_b2_pair_correlation()
    except Exception as e:
        print(f"\nAnalysis 2 failed: {e}")
        import traceback; traceback.print_exc()
        output["analysis2_b1_b2"] = {"error": str(e)}

    # Save
    out_path = RESULTS_DIR / "geometry_lambda_discriminator.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
