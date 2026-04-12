# Presynaptic Bouton Tiling: Paper Plan

## What We're Trying to Show

**Central claim:** Individual axons distribute their synaptic boutons along postsynaptic dendritic branches via a self-avoidance tiling rule — each axon spaces its own contacts apart from each other, creating a regular lattice-like pattern. This is a general cortical wiring principle operating across all synapse types, not specific to inhibitory circuits.

**Why this matters:** Connectomics has characterized WHO connects to WHOM (connection probabilities, motifs). But HOW individual axons physically distribute their contacts on dendrites is almost entirely uncharacterized. If each axon tiles its boutons with a quantifiable spacing rule, that's a fundamental wiring constraint that shapes the spatial organization of all synaptic input.

## Background

### The existing finding (from microns-explorer paper)
The B2 result tested whether inhibitory spacing regularity comes from a postsynaptic scaffold (B1: gephyrin slots that repel ALL inhibitory contacts regardless of which axon made them) or from presynaptic tiling (B2: each axon spaces its OWN contacts, but different axons don't interact).

The test: compute nearest-neighbor distances between inhibitory synapses on the same branch, stratified by same-axon vs different-axon.
- Same-axon NN: 2.34 µm (contacts from the SAME axon are far apart)
- Cross-axon NN: 0.42 µm (contacts from DIFFERENT axons pack close together)
- Ratio: 0.18 (cross/same)

This conclusively shows B2: spacing regularity is a within-axon property. Different axons land freely near each other.

### What was missing
1. Only tested on inhibitory presynaptic partners on inhibitory postsynaptic neurons
2. No characterization of the inter-bouton spacing distribution (is it regular or random?)
3. No connection to WHY different cell subtypes show different ICV slopes
4. Not tested whether tiling explains the ~50% excitatory residual or ~75% inhibitory residual

### What we just found (script 21 results, April 2026)

**Tiling IS universal.** The B2 ratio on excitatory neurons (all synapses) = 0.177, virtually identical to the inhibitory result (0.179). On both cell types, individual axons space their boutons ~5.6x further apart than the nearest contact from any other axon.

| Population | Same-axon NN | Cross-axon NN | Ratio |
|---|---|---|---|
| Excitatory (all synapses) | 2.08 µm | 0.37 µm | 0.177 |
| Inhibitory (all synapses) | 2.30 µm | 0.18 µm | 0.078 |
| Inhibitory (inh presynaptic only) | 2.34 µm | 0.42 µm | 0.179 |

The lattice constant (mean inter-bouton spacing) shows clear regime scaling:
- Regime 0 (compact): ~1.3 µm
- Regime 1 (intermediate): ~2.4 µm
- Regime 2 (compartmentalized): ~3.0-3.7 µm

## Analysis Plan

### Block A: Universality + Controls (CURRENT PRIORITY)

**A1. Five-population B2 test** ✅ DONE (script 21)

**A2. K-preserving shuffle null** ← IMPLEMENTING NOW
For each branch, shuffle synapse POSITIONS while preserving which axon made how many contacts. If the B2 ratio matches this shuffle → "tiling" is a multiplicity artifact (k=2 axons naturally have large same-axon NN). If the ratio is smaller than shuffle → tiling is real self-avoidance.

**A3. Validate lattice assumption (inter-bouton interval CV)** ← IMPLEMENTING NOW
For axons with k≥3 boutons on one branch, compute the coefficient of variation of consecutive inter-bouton intervals:
- CV < 0.5 → regular (lattice-like) spacing
- CV ≈ 1.0 → random (exponential) spacing
- CV > 1.0 → clustered

**A4. Random-partner negative control** ← IMPLEMENTING NOW
Assign each synapse to a random "fake partner" (preserving k per partner). Re-run B2. If tiling signal disappears → it's partner-specific (real tiling). If it persists → it's generic spacing, not partner-related.

**A5. Branch-length normalization** ← IMPLEMENTING NOW
Compute lattice_constant / branch_length per axon-branch pair. If this normalized spacing is constant across regimes → "contextual adaptation" is just geometry. If it varies → adaptation is biologically real.

**A6. k≥2 data fraction reporting** ← IMPLEMENTING NOW
Report what fraction of branches and synapses are included in the tiling analysis. State explicitly that tiling operates on multi-contact partners, not the entire synapse population.

### Block B: Lattice Constant Characterization (weeks 3-4, script 22)

- Full lattice constant distributions per subtype (11 subtypes)
- Branch length scaling (partial correlation, mixed model)
- Synapse count k scaling (does adding contacts compress the lattice?)
- Per-subtype lattice constants predict ICV slopes? (n=11 correlation)

### Block C: Tiling Explains the Residual (weeks 5-6, script 23)

- Per-branch tiling features (mean within-axon NN, z-scored against shuffle)
- Four-model mediation: regime-only, +partner, +tiling, +all
- Quantitative superposition prediction: N tiling axons → branch-level ICV
  - Simulate N axons, each placing contacts at regular intervals (spacing d, random phase)
  - Compute simulated ICV as function of N
  - Overlay with empirical ICV vs number of contributing axons
  - If data follow prediction → strongest evidence for tiling as generative mechanism

### Block D: Generative Model (weeks 7-8, script 24)

- Model: sequential placement with min spacing d_min, jitter σ, scaling α
  - d_min_effective = d_min × L^α (branch-length-dependent)
- Parameters: {d_min, σ, α} per subtype
- Fit: minimize KL divergence on within-axon NN distributions
- 5-fold neuron-level cross-validation
- α ≈ 0 → fixed spacing; α ≈ 1 → contextual adaptation

## Paper Structure

**Title:** "Presynaptic bouton tiling: a general cortical wiring principle revealed by population-scale connectomics"

1. Introduction: how individual axons distribute contacts is uncharacterized; tiling hypothesis
2. Results:
   - 2.1 Tiling is universal across synapse types (Block A)
   - 2.2 The lattice constant: characteristic inter-bouton spacing (Block B)
   - 2.3 Cell-type-specific tiling parameters (Block B)
   - 2.4 Tiling as generative mechanism for spacing regularity (Block C)
   - 2.5 A parameterized model of bouton placement (Block D)
3. Discussion: tiling as wiring principle, molecular mechanisms, limitations
4. Methods

## Key Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Shuffle null matches observed ratio | FATAL — tiling is a k-artifact | Reframe as multiplicity paper |
| Inter-bouton CV ≈ 1.0 (random, not regular) | SERIOUS — no "lattice" | Drop lattice framing, use "self-avoidance" |
| Normalized spacing constant across regimes | MODERATE — adaptation is geometric | Still publishable but weaker |
| Superposition prediction doesn't fit data | MODERATE — lose strongest panel | Rest of paper stands |

## Target

**Journal:** eLife (backup: PLoS Comp Bio)
**Timeline:** 12 weeks
**Data:** MICrONS, 2,138 neurons, all existing — no new data acquisition needed
