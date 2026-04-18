# Comprehensive Differential Analysis Example

This example demonstrates MD-Compare's v1.5.0 comprehensive differential analysis capabilities using HIV protease drug resistance as a case study. It shows how to perform quantitative simulation vs simulation comparison across all molecular dynamics dimensions.

## Overview

**Objective**: Compare wild-type HIV protease with the V82A drug-resistant mutant to understand resistance mechanisms at the molecular level.

**Scientific Context**: The V82A mutation is a major resistance mutation that affects flap dynamics, cross-chain communication, and drug binding affinity. Understanding these changes quantitatively is crucial for drug design.

**What You'll Learn**:
- Complete differential analysis workflow
- Interpretation of comprehensive comparison results
- Scientific insights from quantitative molecular changes

## Quick Start

### Basic Comprehensive Differential Analysis
```bash
python md_compare_cli.py differential \
  -t1 hiv_wt.pdb -x1 wt_trajectory.xtc -n1 "Wild_Type" \
  -t2 hiv_v82a.pdb -x2 v82a_trajectory.xtc -n2 "V82A_Mutant" \
  -o hiv_resistance_analysis \
  --allosteric-sources A_50 B_50 \
  --allosteric-targets A_25 B_25
```

### Research-Grade Analysis with Statistical Testing
```bash
python md_compare_cli.py differential \
  -t1 hiv_wt.pdb -x1 wt_trajectory.xtc -n1 "Wild_Type" \
  -t2 hiv_v82a.pdb -x2 v82a_trajectory.xtc -n2 "V82A_Mutant" \
  -o hiv_wt_vs_v82a_comprehensive \
  --allosteric-sources A_50 A_51 B_50 B_51 \
  --allosteric-targets A_25 A_26 B_25 B_26 \
  --statistical-tests \
  --publication-figures \
  --excel-export \
  --correlation-threshold 0.15 \
  --efficiency-threshold 0.1
```

## Results Structure

The analysis creates a comprehensive, organized output structure:

```
hiv_wt_vs_v82a_comprehensive/
├── 01_individual_analyses/
│   ├── Wild_Type_results/                    # Complete v1.4.0 analysis
│   │   ├── 01_residue_interaction_networks/
│   │   ├── 02_dynamic_cross_correlations/
│   │   ├── 03_principal_component_analysis/
│   │   ├── 04_energy_landscapes/
│   │   ├── 05_markov_state_models/
│   │   ├── 06_allosteric_pathways/
│   │   │   ├── Wild_Type_communication_efficiency_matrix.csv  # FULL 198x198
│   │   │   └── Wild_Type_high_efficiency_pairs.csv
│   │   └── 07_summary_reports/
│   └── V82A_Mutant_results/                  # Same complete structure
├── 02_network_comparisons/                   # 🆕 Topology differences
│   ├── edge_weight_changes.csv
│   ├── degree_centrality_changes.csv
│   └── network_property_changes.csv
├── 03_dynamics_comparisons/                  # 🆕 DCCM & PCA differences
│   ├── dccm_difference_heatmap.png
│   ├── significant_correlation_changes.csv
│   └── pca_variance_changes.csv
├── 04_energetics_comparisons/                # 🆕 Energy landscape differences
├── 05_kinetics_comparisons/                  # 🆕 MSM & timescale differences
├── 06_allosteric_comparisons/                # 🆕 Communication efficiency differences
│   ├── pathway_disruption_analysis.csv
│   ├── hotspot_ranking_changes.csv
│   └── cross_chain_communication_changes.csv
└── 07_comprehensive_report/                  # 🆕 Executive summary
    ├── comprehensive_differential_report.html
    ├── overall_similarity_scores.csv
    └── statistical_significance_tests.csv
```

## Interpreting Results

### 1. Executive Summary

Check the main report first:
```bash
# View the HTML report
open hiv_wt_vs_v82a_comprehensive/07_comprehensive_report/comprehensive_differential_report.html

# Or check similarity scores
head hiv_wt_vs_v82a_comprehensive/07_comprehensive_report/overall_similarity_scores.csv
```

**Expected Output**:
```
Analysis_Type,Similarity_Score,Similarity_Category
network,0.72,medium
dynamics,0.65,medium  
allosteric,0.58,medium
kinetics,0.71,medium
```

**Interpretation**: Medium similarity scores (0.5-0.8) indicate significant but not complete changes - exactly what we expect for a drug resistance mutation.

### 2. Allosteric Communication Changes

The most critical results for drug resistance:

```python
import pandas as pd

# Load pathway disruption analysis
pathways = pd.read_csv('hiv_wt_vs_v82a_comprehensive/06_allosteric_comparisons/pathway_disruption_analysis.csv')

# Show most disrupted pathways
print("Most disrupted allosteric pathways:")
print(pathways.sort_values('disruption_score', ascending=False).head())

# Expected findings:
# pathway_id  source  target  efficiency_change  disruption_score  significance
# 0          A_50    A_25    -0.23             0.31            high
# 1          B_50    B_25    -0.21             0.29            high
# 2          A_50    B_25    -0.18             0.25            medium
```

**Scientific Interpretation**:
- **Flap-to-Active Site Communication Disrupted**: 23% efficiency loss in primary allosteric pathways
- **Cross-Chain Effects**: V82A mutation affects communication between protein chains
- **Drug Binding Impact**: Reduced communication to binding site explains resistance

### 3. Dynamic Correlation Changes

```python
# Load significant correlation changes
correlations = pd.read_csv('hiv_wt_vs_v82a_comprehensive/03_dynamics_comparisons/significant_correlation_changes.csv')

# Show largest changes
print("Largest correlation changes:")
print(correlations.head())

# Expected findings:
# Residue1  Residue2  Original_Correlation  New_Correlation  Correlation_Change
# res_50   res_25    0.65                 0.42             -0.23
# res_51   res_26    0.58                 0.39             -0.19
# res_82   res_84    0.31                 0.52             +0.21
```

**Scientific Interpretation**:
- **Lost Correlations**: Flaps (50-51) lose correlation with active site (25-26)
- **New Correlations**: Mutation site (82) gains correlations with neighbors
- **Functional Impact**: Disrupted cooperative motion affects enzyme function

### 4. Network Topology Changes

```python
# Load centrality changes
centrality = pd.read_csv('hiv_wt_vs_v82a_comprehensive/02_network_comparisons/degree_centrality_changes.csv')

# Show biggest hub changes
print("Residues with largest centrality changes:")
print(centrality.sort_values('Magnitude', ascending=False).head())

# Expected findings:
# Node    Centrality_Change  Change_Type  Magnitude
# A_82    +0.15             increase     0.15
# A_50    -0.12             decrease     0.12  
# B_50    -0.11             decrease     0.11
```

**Scientific Interpretation**:
- **New Network Hub**: Residue 82 becomes more central after mutation
- **Lost Network Hubs**: Flap residues (50) become less central
- **Network Reorganization**: Overall topology adapts to mutation

### 5. Statistical Significance

```python
# Load statistical results
stats = pd.read_csv('hiv_wt_vs_v82a_comprehensive/07_comprehensive_report/statistical_significance_tests.csv')

# Count significant changes
significant = stats[stats['p_value'] < 0.05]
print(f"Statistically significant changes: {len(significant)}")

# Expected: ~40-60 significant changes across all analysis types
```

## Advanced Analysis Options

### Focus on Specific Regions
```bash
# Analyze only flap and active site regions
python md_compare_cli.py differential \
  -t1 hiv_wt.pdb -x1 wt_trajectory.xtc -n1 "Wild_Type" \
  -t2 hiv_v82a.pdb -x2 v82a_trajectory.xtc -n2 "V82A_Mutant" \
  -o hiv_focused_analysis \
  --selection "protein and (resid 20-30 or resid 45-55 or resid 80-90)" \
  --correlation-threshold 0.1
```

### High-Sensitivity Analysis
```bash
# Lower thresholds for detecting subtle changes
python md_compare_cli.py differential \
  -t1 hiv_wt.pdb -x1 wt_trajectory.xtc -n1 "Wild_Type" \
  -t2 hiv_v82a.pdb -x2 v82a_trajectory.xtc -n2 "V82A_Mutant" \
  -o hiv_sensitive_analysis \
  --correlation-threshold 0.1 \
  --efficiency-threshold 0.05 \
  --bootstrap-iterations 2000
```

### Skip Computationally Expensive Analyses
```bash
# Focus on network and dynamics only
python md_compare_cli.py differential \
  -t1 hiv_wt.pdb -x1 wt_trajectory.xtc -n1 "Wild_Type" \
  -t2 hiv_v82a.pdb -x2 v82a_trajectory.xtc -n2 "V82A_Mutant" \
  -o hiv_fast_analysis \
  --no-energetics-comparison \
  --no-kinetics-comparison
```

## Scientific Applications

### 1. Drug Resistance Mechanism Understanding
**Research Question**: How does V82A mutation reduce drug efficacy?

**Key Findings from Differential Analysis**:
- Disrupted allosteric communication between flaps and active site
- Altered drug binding pocket dynamics
- Compensatory network reorganization
- Quantitative efficiency losses in critical pathways

**Impact**: Molecular-level understanding guides development of next-generation inhibitors

### 2. Target Identification for Drug Design
**Research Question**: What new allosteric sites emerge after mutation?

**Key Findings**:
- Residues with increased centrality (potential new targets)
- Alternative communication pathways (allosteric opportunities)
- Conserved network features (robust targets)

**Impact**: Identifies novel druggable sites for combination therapy

### 3. Mutation Effect Prediction
**Research Question**: Can we predict effects of other mutations?

**Key Findings**:
- Communication network vulnerability map
- Critical pathway dependencies
- Structural-functional relationships

**Impact**: Enables rational design of resistance-resistant drugs

## Validation and Next Steps

### Computational Validation
1. **Extended Simulations**: Longer trajectories for statistical robustness
2. **Free Energy Calculations**: Quantify binding affinity changes
3. **Multiple Replica Analysis**: Ensure reproducibility

### Experimental Validation
1. **Enzymatic Assays**: Confirm activity changes
2. **Binding Studies**: Validate inhibitor affinity predictions
3. **Mutagenesis**: Test key residues identified by network analysis

### Drug Design Applications
1. **Inhibitor Optimization**: Target identified vulnerable sites
2. **Allosteric Modulators**: Develop compounds targeting new sites
3. **Combination Therapy**: Multi-target approaches

## Troubleshooting

### Issue: No Significant Differences Detected
**Solution**: Lower sensitivity thresholds
```bash
--correlation-threshold 0.1 --efficiency-threshold 0.05
```

### Issue: Too Many Changes to Interpret
**Solution**: Raise significance thresholds
```bash
--correlation-threshold 0.2 --significance-threshold 0.01
```

### Issue: Analysis Takes Too Long
**Solution**: Skip expensive components
```bash
--no-energetics-comparison --no-kinetics-comparison --no-plots
```

### Issue: Memory Issues with Large Systems
**Solution**: Use efficient selections
```bash
--selection "name CA" --msm-stride 5 --landscape-bins 30
```

## Command Reference

### Essential Options
```bash
-t1, -x1, -n1              # First simulation (topology, trajectory, name)
-t2, -x2, -n2              # Second simulation (topology, trajectory, name)  
-o                         # Output directory
--allosteric-sources       # Source residues for pathway analysis
--allosteric-targets       # Target residues for pathway analysis
```

### Statistical Options
```bash
--statistical-tests        # Enable significance testing
--significance-threshold   # P-value threshold (default: 0.05)
--correlation-threshold    # Minimum correlation change (default: 0.2)
--efficiency-threshold     # Minimum efficiency change (default: 0.1)
--bootstrap-iterations     # Bootstrap samples (default: 1000)
```

### Output Options
```bash
--publication-figures      # High-resolution figures
--excel-export            # Excel workbook output
--no-html-report          # Disable HTML report
--figure-dpi              # Figure resolution (default: 300)
```

### Performance Options
```bash
--no-energetics-comparison  # Skip energy landscape analysis
--no-kinetics-comparison    # Skip MSM analysis
--no-plots                 # Skip visualization generation
--selection                # Atom selection for analysis
```

## Expected Runtime

**System**: HIV protease (~200 residues), 100 ns trajectories
**Hardware**: Modern desktop (8 cores, 16 GB RAM)

- **Basic Analysis**: 15-30 minutes
- **Full Analysis**: 45-90 minutes  
- **With Statistical Testing**: 60-120 minutes

**Scaling**: Runtime increases ~quadratically with system size. For 1000+ residue systems, consider using `--selection "name CA"` for initial analysis.

---

**This comprehensive differential analysis transforms qualitative observations of mutation effects into quantitative, statistically rigorous insights suitable for high-impact publications and drug discovery applications.**

For more examples, see the other files in this directory, and refer to the complete documentation in `DIFFERENTIAL_USAGE_GUIDE.md`.
