# PyEMMA Integration Example: Markov State Model Analysis for HIV Protease

## Overview

This example demonstrates the enhanced MD-Compare capabilities with PyEMMA integration for comprehensive Markov State Model (MSM) analysis. PyEMMA provides advanced kinetic modeling, metastable state identification, and transition pathway analysis that complements the existing network analysis.

## Scientific Background

### Markov State Models for Protein Dynamics
- **Microstates**: Discrete conformational states from trajectory clustering
- **Transition Matrix**: Probabilities of state-to-state transitions
- **Implied Timescales**: Characteristic timescales of conformational processes
- **Metastable States**: Long-lived conformational macrostates
- **Kinetic Networks**: Rate constants and transition pathways

### HIV Protease Dynamics
- **Flap Opening/Closing**: Critical for inhibitor binding and release
- **Dimer Interface Dynamics**: Allosteric coupling between chains
- **Active Site Fluctuations**: Catalytic mechanism and inhibitor resistance
- **Conformational Selection**: Drug binding thermodynamics and kinetics

## Installation Requirements

```bash
# Install PyEMMA for MSM analysis
pip install pyemma

# Optional: Enhanced clustering and analysis
pip install scikit-learn matplotlib seaborn

# Verify installation
python -c "import pyemma; print(f'PyEMMA {pyemma.__version__} installed successfully')"
```

## Basic MSM Analysis

### Standard MSM Analysis with Distance Features
```bash
md-compare single \
  -t data/hiv_wt_complex.pdb \
  -x data/hiv_wt_trajectory.xtc \
  -n HIV_WT_MSM_Basic \
  -o results/hiv_msm_basic \
  --compute-msm \
  --msm-features distances \
  --msm-clusters 100 \
  --msm-lag-time 10 \
  --msm-clustering kmeans
```

### Expected Output Files
```
results/hiv_msm_basic/
├── HIV_WT_MSM_Basic_msm_summary.txt          # MSM analysis summary
├── HIV_WT_MSM_Basic_msm_analysis.json        # Complete MSM data
├── HIV_WT_MSM_Basic_discrete_trajectory.npy  # State assignments
├── HIV_WT_MSM_Basic_cluster_centers.npy      # Cluster centroids
├── HIV_WT_MSM_Basic_transition_matrix.npy    # Transition probabilities
├── HIV_WT_MSM_Basic_implied_timescales.npy   # Kinetic timescales
├── HIV_WT_MSM_Basic_msm_analysis.png         # 8-panel MSM visualization
└── ... (standard network analysis files)
```

## Advanced MSM Configuration

### High-Resolution MSM for Detailed Kinetics
```bash
md-compare single \
  -t data/hiv_wt_complex.pdb \
  -x data/hiv_wt_trajectory.xtc \
  -n HIV_WT_MSM_HighRes \
  -o results/hiv_msm_highres \
  --compute-msm \
  --msm-features distances \
  --msm-clusters 200 \
  --msm-lag-time 5 \
  --msm-clustering kmeans \
  --kinetic-timescales 10 \
  --metastable-states 8 \
  --msm-stride 1
```

### Alternative Feature Types
```bash
# Coordinate-based MSM (for small systems)
md-compare single \
  -t data/system.pdb -x data/traj.xtc -n coord_msm \
  --msm-features coordinates \
  --msm-clusters 50

# Angle-based MSM (backbone dynamics)
md-compare single \
  -t data/system.pdb -x data/traj.xtc -n angle_msm \
  --msm-features angles \
  --msm-clusters 100
```

## Combined Analysis: Networks + MSM

### Comprehensive Dynamics and Network Analysis
```bash
md-compare single \
  -t data/hiv_wt_complex.pdb \
  -x data/hiv_wt_trajectory.xtc \
  -n HIV_WT_Complete \
  -o results/hiv_comprehensive \
  --compute-dccm \
  --compute-pca \
  --compute-landscape \
  --compute-msm \
  --community-method leiden \
  --allosteric-sources A_50 B_50 A_48 B_48 \
  --allosteric-targets A_25 B_25 A_27 B_27 \
  --msm-features distances \
  --msm-clusters 150 \
  --msm-lag-time 10 \
  --kinetic-timescales 8 \
  --metastable-states 6
```

## Interpreting MSM Results

### 1. Implied Timescales Analysis
```python
import numpy as np
import matplotlib.pyplot as plt

# Load timescales
timescales = np.load('results/hiv_comprehensive/HIV_WT_Complete_implied_timescales.npy')

print("Implied Timescales Analysis:")
print(f"Dominant timescale: {timescales[0]:.2f} frames")
print(f"Secondary timescale: {timescales[1]:.2f} frames") 
print(f"Timescale separation: {timescales[0]/timescales[1]:.2f}")

# Good MSM: clear timescale separation (ratio > 3-5)
if timescales[0]/timescales[1] > 3:
    print("✓ Good timescale separation - reliable MSM")
else:
    print("⚠ Poor timescale separation - consider longer lag time")
```

### 2. Metastable State Analysis
```python
import json

# Load MSM analysis
with open('results/hiv_comprehensive/HIV_WT_Complete_msm_analysis.json', 'r') as f:
    msm_data = json.load(f)

meta_states = msm_data['metastable_states']
print("Metastable States:")
print(f"Number of metastates: {meta_states['n_metastable_states']}")

# State populations
populations = meta_states['metastable_distributions']
for i, pop in enumerate(populations):
    print(f"  Metastate {i+1}: {pop:.1%} population")

# Biological interpretation for HIV protease
print("\nBiological Interpretation:")
print("Metastate 1: Likely closed/semi-open flap conformations")
print("Metastate 2: Open flap conformations")
print("Metastate 3: Interface-dominated states")
```

### 3. Transition Network Analysis
```python
# Load transition matrix
T = np.load('results/hiv_comprehensive/HIV_WT_Complete_transition_matrix.npy')

print("Transition Matrix Analysis:")
print(f"Matrix size: {T.shape[0]} × {T.shape[1]}")
print(f"Sparsity: {(T == 0).sum() / T.size:.1%} zero entries")

# Find highest transition probabilities
flat_indices = np.argsort(T.flatten())[-10:]  # Top 10 transitions
transitions = np.unravel_index(flat_indices, T.shape)

print("Top state-to-state transitions:")
for i, (from_state, to_state) in enumerate(zip(transitions[0], transitions[1])):
    if from_state != to_state:  # Skip self-transitions
        prob = T[from_state, to_state]
        print(f"  {from_state} -> {to_state}: {prob:.4f}")
```

### 4. Kinetic Rate Analysis
```python
# Calculate transition rates (inverse timescales)
dt = 1.0  # Time step (ps, ns, etc. - depends on your trajectory)
lag_time = 10  # frames

rates = 1.0 / (timescales * lag_time * dt)
print("Transition Rates:")
for i, rate in enumerate(rates[:5]):
    print(f"  Process {i+1}: {rate:.2e} /time_unit")

# Estimate exchange timescales
exchange_times = 1.0 / rates
print("Exchange Timescales:")
for i, time in enumerate(exchange_times[:5]):
    print(f"  Process {i+1}: {time:.1f} time_units")
```

## Integration with Network Analysis

### Correlating MSM States with Network Properties
```python
# Load network centrality
import pandas as pd
centrality_df = pd.read_csv('results/hiv_comprehensive/HIV_WT_Complete_centrality.csv')

# Load discrete trajectory
dtraj = np.load('results/hiv_comprehensive/HIV_WT_Complete_discrete_trajectory.npy')

# Analyze centrality changes across MSM states
state_centralities = {}
for state in range(T.shape[0]):
    state_frames = np.where(dtraj == state)[0]
    if len(state_frames) > 0:
        # Could correlate with network properties computed at these frames
        state_centralities[state] = len(state_frames)

print("State occupancies:")
for state, count in sorted(state_centralities.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  State {state}: {count} frames ({count/len(dtraj):.1%})")
```

## Troubleshooting MSM Analysis

### Common Issues and Solutions

**1. Too Few Connected States**
```bash
# Increase connectivity threshold
md-compare single ... --msm-clusters 50  # Fewer clusters

# Or reduce lag time
md-compare single ... --msm-lag-time 5
```

**2. Poor Timescale Separation**
```bash
# Increase lag time
md-compare single ... --msm-lag-time 20

# Try different features
md-compare single ... --msm-features coordinates
```

**3. MSM Construction Fails**
```bash
# Reduce system complexity
md-compare single ... --msm-stride 2  # Use fewer frames

# Simpler clustering
md-compare single ... --msm-clustering regular_space
```

**4. Memory Issues with Large Trajectories**
```bash
# Use stride to reduce data
md-compare single ... --msm-stride 5

# Fewer clusters
md-compare single ... --msm-clusters 50
```

## Scientific Applications

### Drug Discovery Applications
1. **Binding Pathway Analysis**: MSM reveals inhibitor binding/unbinding routes
2. **Conformational Selection**: Identifies pre-existing states for drug binding
3. **Allosteric Networks**: Combined with network analysis for allosteric drug design
4. **Resistance Mechanisms**: Kinetic changes due to mutations

### HIV Protease Specific Insights
1. **Flap Dynamics**: Open/closed timescales and transition barriers
2. **Inhibitor Efficacy**: Residence times and binding kinetics
3. **Resistance Evolution**: How mutations alter kinetic networks
4. **Dimer Cooperativity**: Cross-chain communication timescales

## Advanced PyEMMA Features

### Custom Feature Selection
```python
# Example: Distance features between specific residue pairs
# This would be implemented in future versions for custom features

# Flap-tip to active site distances
feature_pairs = [
    ('A_50', 'A_25'),  # Chain A flap to active site
    ('B_50', 'B_25'),  # Chain B flap to active site
    ('A_50', 'B_50'),  # Flap-flap distance
    ('A_25', 'B_25')   # Active site coupling
]

# This capability could be added via custom feature extraction
```

### MSM Model Comparison
```bash
# Compare different lag times
for lag in 5 10 15 20; do
    md-compare single ... --msm-lag-time $lag -n msm_lag_${lag}
done

# Compare clustering methods
for method in kmeans regular_space minibatch_kmeans; do
    md-compare single ... --msm-clustering $method -n msm_${method}
done
```

## Expected Performance

### Computational Requirements
- **Small systems** (<100 residues): MSM completes in minutes
- **Medium systems** (100-300 residues): 10-30 minutes additional time
- **Large systems** (>300 residues): Consider using stride or fewer clusters

### Memory Usage
- **Distance features**: ~N² scaling with atoms (manageable for proteins)
- **Coordinate features**: Linear scaling but higher dimensional
- **Clustering**: Memory usage depends on number of clusters and features

### Scientific Validation
- **Timescale separation**: Should see clear gaps in implied timescales
- **Convergence**: Multiple lag times should give consistent results
- **Biological relevance**: MSM states should correlate with known conformations

## Future Enhancements

### Planned Features
1. **Custom feature extraction**: User-defined distance/angle sets
2. **MSM comparison tools**: Automated model selection and validation
3. **Kinetic pathway analysis**: Transition path theory integration
4. **Enhanced visualization**: Interactive MSM network plots
5. **Experimental validation**: Comparison with kinetic experiments

### Integration Opportunities
1. **Machine learning**: MSM-guided enhanced sampling
2. **Drug design**: Kinetic pharmacophore models
3. **Mutation analysis**: Kinetic network perturbation studies
4. **Multi-scale modeling**: MSM-informed coarse-graining

This PyEMMA integration transforms MD-Compare into a comprehensive platform for both structural network analysis and kinetic modeling, providing unprecedented insights into protein dynamics and function!
