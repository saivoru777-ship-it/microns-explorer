# Pre-Registered Hypothesis: Functional Correlates of Dendritic Regime Coupling

**Date**: 2026-03-25
**Status**: Pre-registered before any functional data access
**Dataset**: MICrONS coregistration_manual_v4 + digital_twin_properties_bcm_coreg_v4

---

## Background

The random-slope analysis identified a sharp cell-type dissociation: excitatory neurons have
steep ICV coupling slopes (median = -2.23), meaning inter-synapse spacing becomes dramatically
more regular as branches become more electrotonically isolated. Inhibitory neurons show
near-zero slopes (median = -0.07). This is a structural finding. The question is whether the
structural variation predicts functional differences.

---

## Primary Hypothesis

**H1: Excitatory neurons with steeper regime-coupling slopes (more negative ICV BLUP) show
stronger orientation selectivity (higher OSI).**

Directional prediction: negative correlation between per-neuron ICV BLUP and OSI
(steeper slope = more negative BLUP → higher OSI).

**Mechanistic logic:**
The ICV slope measures how much more regularly-spaced synaptic inputs become on
electrotonically isolated branches. Regular spacing on isolated compartments distributes inputs
across the branch's integrative zone, maximizing the dendritic territory engaged by each input
pattern. On branches capable of local nonlinear integration (NMDA spikes), regularly-spaced
inputs create a winner-take-all landscape: inputs from co-tuned presynaptic neurons that arrive
in a coordinated spatial pattern will preferentially drive NMDA spikes, while inputs from
differently-tuned neurons are spread across non-overlapping dendritic sub-compartments.
This geometry supports sharper feature selectivity. Neurons where this organization is strong
(steep ICV slope) should therefore show stronger orientation tuning, because their dendritic
architecture is more efficient at discriminating oriented stimuli.

The prediction is specifically for excitatory neurons because inhibitory neurons show no
ICV coupling slope variation — there is no structural variation to correlate against.

---

## Secondary Hypothesis

**H2: Excitatory neurons with steeper ICV slopes show higher response reliability
(trial-to-trial consistency of responses to preferred stimuli).**

Directional prediction: negative correlation between ICV BLUP and some response
reliability metric (e.g., signal-to-noise ratio, correlation across trials).

**Mechanistic logic:** Same dendritic geometry argument, but for reliability rather than
selectivity. Regular input spacing on isolated branches means the same stimulus reliably
activates the same dendritic subcompartments on each trial, producing more stereotyped
responses. This is a weaker and less specific prediction than H1 — OSI is preferred as
the primary outcome.

---

## Tertiary / Exploratory

**H3: The inhibitory ICV slope deviation from zero (|inhibitory BLUP|) predicts
population coupling (how correlated a neuron's activity is with the population mean).**

Directional prediction: neurons with larger |inhibitory ICV slope| (stronger inhibitory
regime coupling) are more decoupled from population activity (lower population coupling),
consistent with a stronger local inhibitory control that suppresses noise correlations.

This is more speculative — treat as exploratory, not primary.

---

## Analysis Plan

1. Query `coregistration_manual_v4` for root IDs in our 2,138 proofread neuron set
2. Pull `digital_twin_properties_bcm_coreg_v4` for matching neurons
3. Load per-neuron ICV BLUP from `results/replication_full/random_slope_results.json`
4. Restrict to excitatory neurons with both BLUP and functional data
5. Test H1: Spearman rank correlation between ICV BLUP and OSI; report rho, p, n
6. Test H2: Spearman correlation between ICV BLUP and response reliability metric
7. Test H3: Spearman correlation between |inhibitory ICV BLUP| and population coupling
8. Visualize: scatter plot of ICV BLUP vs OSI for excitatory neurons

**Statistical threshold**: p < 0.05 (Bonferroni-corrected for 3 primary tests: α = 0.017).
A correlation of r ≥ 0.15 with p < 0.017 in the matched subset is a positive result.

---

## Pre-committed Interpretation Rules

- **Positive H1** (r ≥ 0.15, p < 0.017): Report as primary structural-functional link.
  Include as a main result with scatter figure.

- **Weak positive H1** (r > 0, p < 0.05 uncorrected but > 0.017): Report as
  "consistent trend" in supporting analysis, not primary result.

- **Null H1** (r near 0 or p > 0.05): Report honestly as a negative result. The structural
  phenomenon does not predict orientation tuning in the measured coregistered population.
  Do NOT fish for other functional metrics post-hoc. The null result constrains interpretation.

- **Negative H1** (r < 0, steeper slope → lower OSI): This would be surprising and should
  trigger careful diagnostic analysis (coverage quality, confounders) before reporting.

---

## Known Confounds to Control

- Cell type: run correlation within excitatory subtypes, not pooled, to rule out
  subtype-level confounding (e.g., L5 neurons having both steeper slopes and higher OSI)
- Imaging depth: visual area and layer may confound both slope and OSI; include as covariate
- Sample size: report n carefully; some proofread neurons may lack functional matches
