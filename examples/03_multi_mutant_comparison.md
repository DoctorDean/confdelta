# Example 3: Multi-Mutant Comparison Study

This example demonstrates comprehensive analysis of multiple HIV protease resistance mutants to identify common resistance mechanisms and mutation-specific network changes.

## Background

We're analyzing a panel of clinically relevant HIV protease mutants to understand how different resistance mutations affect the protein's allosteric network. This systematic study includes:

**Primary Resistance Mutations:**
- **V82A**: Active site volume expansion
- **I84V**: Active site flexibility changes  
- **L90M**: β-sheet region modification
- **M46I**: Flap region adjustment

**Secondary Resistance Mutations:**
- **L10I**: N-terminal structural changes
- **L63P**: Conformational constraint introduction
- **A71V**: Hydrophobic core modification
- **V77I**: Interface region changes

This represents a clinically realistic mutation pattern often seen in multi-drug resistant HIV.

## Configuration File Setup

### Create Multi-Simulation Configuration

```json
{
  "simulations": [
    {
      "name": "HIV_WT",
      "topology": "data/hiv_wt_complex.pdb",
      "trajectory": "data/hiv_wt_trajectory.xtc",
      "selection": "protein and not name H*",
      "description": "Wild-type HIV protease with darunavir",
      "metadata": {
        "mutation_profile": "wild-type",
        "inhibitor": "darunavir",
        "simulation_time_ns": 100,
        "temperature_K": 310
      }
    },
    {
      "name": "HIV_V82A",
      "topology": "data/hiv_v82a_complex.pdb", 
      "trajectory": "data/hiv_v82a_trajectory.xtc",
      "selection": "protein and not name H*",
      "description": "V82A primary resistance mutation",
      "metadata": {
        "mutation_profile": "V82A",
        "inhibitor": "darunavir", 
        "clinical_relevance": "major",
        "resistance_fold_change": 8.5
      }
    },
    {
      "name": "HIV_I84V", 
      "topology": "data/hiv_i84v_complex.pdb",
      "trajectory": "data/hiv_i84v_trajectory.xtc",
      "selection": "protein and not name H*",
      "description": "I84V primary resistance mutation",
      "metadata": {
        "mutation_profile": "I84V",
        "inhibitor": "darunavir",
        "clinical_relevance": "major", 
        "resistance_fold_change": 6.2
      }
    },
    {
      "name": "HIV_L90M",
      "topology": "data/hiv_l90m_complex.pdb",
      "trajectory": "data/hiv_l90m_trajectory.xtc", 
      "selection": "protein and not name H*",
      "description": "L90M primary resistance mutation",
      "metadata": {
        "mutation_profile": "L90M",
        "inhibitor": "darunavir",
        "clinical_relevance": "major",
        "resistance_fold_change": 4.1
      }
    },
    {
      "name": "HIV_M46I",
      "topology": "data/hiv_m46i_complex.pdb",
      "trajectory": "data/hiv_m46i_trajectory.xtc",
      "selection": "protein and not name H*", 
      "description": "M46I primary resistance mutation",
      "metadata": {
        "mutation_profile": "M46I",
        "inhibitor": "darunavir",
        "clinical_relevance": "major",
        "resistance_fold_change": 3.8
      }
    },
    {
      "name": "HIV_L10I_L63P", 
      "topology": "data/hiv_l10i_l63p_complex.pdb",
      "trajectory": "data/hiv_l10i_l63p_trajectory.xtc",
      "selection": "protein and not name H*",
      "description": "L10I+L63P secondary mutations",
      "metadata": {
        "mutation_profile": "L10I,L63P", 
        "inhibitor": "darunavir",
        "clinical_relevance": "secondary",
        "resistance_fold_change": 2.1
      }
    },
    {
      "name": "HIV_Multidrug",
      "topology": "data/hiv_multidrug_complex.pdb", 
      "trajectory": "data/hiv_multidrug_trajectory.xtc",
      "selection": "protein and not name H*",
      "description": "Multi-drug resistant: V82A+I84V+L90M+M46I+L10I+L63P",
      "metadata": {
        "mutation_profile": "V82A,I84V,L90M,M46I,L10I,L63P",
        "inhibitor": "darunavir", 
        "clinical_relevance": "extreme",
        "resistance_fold_change": 127.3
      }
    }
  ],
  "analysis": {
    "cutoffs": {
      "all_atom": 4.5,
      "ca_only": 8.0
    },
    "interaction_types": ["distance", "hbond"],
    "threshold": 0.2,
    "timeout_seconds": 600,
    "segments": 5,
    "preprocess": true,
    "align_selection": "name CA and resid 10-90",
    "center_selection": "protein"
  }
}
```

Save this as `hiv_resistance_panel.json`

## Command Line Usage

### Complete Multi-Mutant Analysis
```bash
md-compare compare \
  -c examples/hiv_resistance_panel.json \
  -o results/hiv_resistance_study \
  --timeout 900
```

### Focused Analysis on Active Site Region
```bash
# Modify config file to include selection
md-compare compare \
  -c examples/hiv_resistance_panel_focused.json \
  -o results/hiv_active_site_study
```

### High-Sensitivity Comparison
```bash
md-compare compare \
  -c examples/hiv_resistance_panel.json \
  -o results/hiv_sensitive_study \
  --threshold 0.15 \
  --cutoff 4.2 \
  --segments 10
```

## Expected Network Analysis Results

### 1. Network Property Comparison

#### **Network Size and Connectivity**
```
Simulation     | Nodes | Edges | Density | Communities
---------------|-------|-------|---------|-------------
HIV_WT         | 198   | 1142  | 0.0584  | 5
HIV_V82A       | 198   | 1089  | 0.0557  | 5  
HIV_I84V       | 198   | 1156  | 0.0591  | 6
HIV_L90M       | 198   | 1098  | 0.0562  | 5
HIV_M46I       | 198   | 1168  | 0.0598  | 5
HIV_L10I_L63P  | 198   | 1134  | 0.0580  | 6
HIV_Multidrug  | 198   | 1034  | 0.0529  | 7
```

**Key Observations:**
- Multi-drug resistant variant shows reduced connectivity
- Some single mutants maintain or increase connectivity
- Community structure changes with certain mutations

### 2. Centrality Analysis Patterns

#### **Active Site Residues (Asp25A, Asp25B)**
```python
# Expected centrality trends
mutations = ['WT', 'V82A', 'I84V', 'L90M', 'M46I', 'L10I_L63P', 'Multidrug']
asp25_centrality = [0.087, 0.078, 0.082, 0.079, 0.085, 0.081, 0.065]
```

- Gradual decrease in active site centrality with resistance mutations
- Multi-drug resistant variant shows severe reduction

#### **Flap Tip Residues (Ile50A, Ile50B)**
```python
ile50_centrality = [0.098, 0.089, 0.105, 0.092, 0.076, 0.091, 0.058]
```

- Variable response depending on mutation location
- M46I (flap mutation) significantly reduces flap centrality
- Multi-drug resistant shows major disruption

### 3. Resistance Mechanism Classification

#### **Type 1: Active Site Volume Mutations (V82A, I84V)**
- Increased active site cavity volume
- Reduced inhibitor complementarity
- Maintained overall network integrity

#### **Type 2: Flap Dynamics Mutations (M46I)**
- Altered flap flexibility and opening/closing
- Changed inhibitor access kinetics
- Modified allosteric networks

#### **Type 3: Allosteric Network Mutations (L90M)**
- Disrupted long-range communication
- Indirect effects on active site
- Compensatory network formation

#### **Type 4: Cumulative Effects (Multi-drug)**
- Severely compromised network integrity
- Multiple pathway disruptions
- Fitness cost vs resistance trade-off

## Output Files Analysis

### Main Results Directory Structure
```
results/hiv_resistance_study/
├── comparison_report.json               # Complete comparative analysis
├── workflow_summary.json                # Analysis summary
├── HIV_WT_centrality.csv               # Individual centrality files
├── HIV_V82A_centrality.csv
├── HIV_I84V_centrality.csv  
├── HIV_L90M_centrality.csv
├── HIV_M46I_centrality.csv
├── HIV_L10I_L63P_centrality.csv
├── HIV_Multidrug_centrality.csv
└── [contact maps and network files for each]
```

### Data Analysis Scripts

#### **1. Network Property Comparison**
```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load comparison results
with open('results/hiv_resistance_study/comparison_report.json', 'r') as f:
    results = json.load(f)

# Extract network properties
network_data = []
for sim_name, properties in results['comparative_analysis']['network_properties'].items():
    network_data.append({
        'mutation': sim_name.replace('HIV_', ''),
        'nodes': properties['n_nodes'],
        'edges': properties['n_edges'], 
        'density': properties['density'],
        'clustering': properties['clustering_coefficient'],
        'communities': properties['n_communities']
    })

df = pd.DataFrame(network_data)

# Plot network density changes
plt.figure(figsize=(10, 6))
plt.bar(df['mutation'], df['density'])
plt.title('Network Density Changes with Resistance Mutations')
plt.xlabel('Mutation') 
plt.ylabel('Network Density')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('network_density_comparison.png', dpi=300)
```

#### **2. Centrality Heatmap Analysis**
```python
import numpy as np
import seaborn as sns

# Load all centrality files
simulations = ['HIV_WT', 'HIV_V82A', 'HIV_I84V', 'HIV_L90M', 
               'HIV_M46I', 'HIV_L10I_L63P', 'HIV_Multidrug']

centrality_matrix = []
residue_keys = None

for sim in simulations:
    df = pd.read_csv(f'results/hiv_resistance_study/{sim}_centrality.csv')
    if residue_keys is None:
        residue_keys = df['residue_key'].values
    centrality_matrix.append(df['betweenness'].values)

centrality_matrix = np.array(centrality_matrix)

# Create heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(centrality_matrix, 
            xticklabels=residue_keys[::10],  # Show every 10th residue
            yticklabels=[s.replace('HIV_', '') for s in simulations],
            cmap='viridis')
plt.title('Betweenness Centrality Across Resistance Mutations')
plt.xlabel('Residue')
plt.ylabel('Mutation')
plt.tight_layout()
plt.savefig('centrality_heatmap.png', dpi=300)
```

#### **3. Resistance Correlation Analysis**
```python
# Correlate network properties with resistance levels
resistance_fold_change = {
    'WT': 1.0,
    'V82A': 8.5,
    'I84V': 6.2, 
    'L90M': 4.1,
    'M46I': 3.8,
    'L10I_L63P': 2.1,
    'Multidrug': 127.3
}

# Add resistance data to network dataframe
df['resistance_fold'] = df['mutation'].map(resistance_fold_change)

# Calculate correlations
correlation_density = df['density'].corr(df['resistance_fold'])
correlation_clustering = df['clustering'].corr(df['resistance_fold'])

print(f"Network Density vs Resistance: r = {correlation_density:.3f}")
print(f"Clustering vs Resistance: r = {correlation_clustering:.3f}")

# Scatter plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(df['resistance_fold'], df['density'])
plt.xlabel('Resistance Fold Change')
plt.ylabel('Network Density')
plt.xscale('log')
plt.title(f'Density vs Resistance (r={correlation_density:.3f})')

plt.subplot(1, 2, 2)  
plt.scatter(df['resistance_fold'], df['clustering'])
plt.xlabel('Resistance Fold Change')
plt.ylabel('Clustering Coefficient')
plt.xscale('log')
plt.title(f'Clustering vs Resistance (r={correlation_clustering:.3f})')

plt.tight_layout()
plt.savefig('resistance_correlation.png', dpi=300)
```

## Key Biological Insights

### 1. Mutation Classification

#### **High-Impact Mutations**: V82A, I84V, Multidrug
- Severe network disruption
- High resistance levels
- Significant fitness costs

#### **Moderate-Impact Mutations**: L90M, M46I  
- Selective network changes
- Moderate resistance
- Balanced resistance/fitness

#### **Low-Impact Mutations**: L10I+L63P
- Minimal network disruption
- Low resistance enhancement
- Compensatory effects

### 2. Network Resilience Patterns

#### **Redundant Pathways**
- Some mutations create alternative communication routes
- Network can compensate for local disruptions
- Multiple pathways maintain function

#### **Critical Nodes**
- Certain residues are irreplaceable communication hubs
- Mutations affecting these cause severe disruption
- Potential targets for allosteric inhibitors

### 3. Clinical Implications

#### **Resistance Mechanism Diversity**
- Different mutations use different resistance strategies
- Combination therapies must address multiple mechanisms
- Network analysis reveals hidden vulnerabilities

#### **Drug Design Targets**
- Identify new allosteric sites
- Target compensatory networks
- Design mutation-resistant inhibitors

## Validation and Follow-up

### Computational Validation:
1. **Free Energy Perturbation**: Calculate binding affinity changes
2. **Principal Component Analysis**: Identify major conformational modes
3. **Markov State Models**: Map conformational landscapes

### Experimental Validation:
1. **Resistance Assays**: Confirm predicted resistance levels
2. **Kinetic Studies**: Measure enzymatic parameter changes
3. **Structural Biology**: Validate predicted conformational changes
4. **Single-Molecule Studies**: Examine dynamic changes

This comprehensive multi-mutant analysis reveals the complex relationship between individual mutations and network-level resistance mechanisms, providing a systems-level understanding of HIV protease drug resistance.
