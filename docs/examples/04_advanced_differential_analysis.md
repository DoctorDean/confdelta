# Example 4: Advanced Differential Analysis with Pathway Mapping

This example demonstrates advanced differential analysis techniques to identify specific allosteric communication pathways that are disrupted by resistance mutations, going beyond simple contact differences to understand mechanistic details.

## Background

We're performing an in-depth analysis of how the I47V mutation (a less common but highly resistance-conferring mutation) affects specific allosteric pathways in HIV protease. This mutation:

- **Location**: Flap region, near the hinge
- **Effect**: Reduces inhibitor binding affinity by >20-fold
- **Mechanism**: Hypothesized to disrupt flap-active site communication
- **Clinical Context**: Emerges under selective pressure from new inhibitors

## Advanced Analysis Workflow

### Phase 1: Standard Differential Analysis
### Phase 2: Pathway-Specific Analysis  
### Phase 3: Dynamic Network Analysis
### Phase 4: Allosteric Mapping

## Files Required

```
data/
├── hiv_wt_complex.pdb           # Wild-type reference
├── hiv_wt_trajectory.xtc        # Wild-type dynamics
├── hiv_i47v_complex.pdb         # I47V mutant structure
├── hiv_i47v_trajectory.xtc      # I47V mutant dynamics
├── crystal_structures/          # Reference crystal structures
│   ├── hiv_apo.pdb             # Unbound protease
│   ├── hiv_inhibitor_bound.pdb  # Inhibitor-bound state
│   └── hiv_substrate_bound.pdb  # Substrate-bound state
└── README.md
```

## Phase 1: Standard Differential Analysis

### High-Resolution Differential Detection
```bash
confdelta diff \
  -t1 data/hiv_wt_complex.pdb \
  -x1 data/hiv_wt_trajectory.xtc \
  -n1 HIV_WT \
  -t2 data/hiv_i47v_complex.pdb \
  -x2 data/hiv_i47v_trajectory.xtc \
  -n2 HIV_I47V \
  -o results/wt_vs_i47v_advanced \
  --diff-threshold 0.05 \
  --cutoff 4.2 \
  --threshold 0.15 \
  --segments 10 \
  --interaction-types distance hbond salt_bridge \
  --timeout 900
```

### Multi-Timescale Analysis
```bash
# Short timescale (first 20 ns)
confdelta diff \
  -t1 data/hiv_wt_complex.pdb \
  -x1 data/hiv_wt_trajectory_0-20ns.xtc \
  -n1 HIV_WT_Short \
  -t2 data/hiv_i47v_complex.pdb \
  -x2 data/hiv_i47v_trajectory_0-20ns.xtc \
  -n2 HIV_I47V_Short \
  -o results/short_timescale \
  --diff-threshold 0.08

# Long timescale (final 20 ns)  
confdelta diff \
  -t1 data/hiv_wt_complex.pdb \
  -x1 data/hiv_wt_trajectory_80-100ns.xtc \
  -n1 HIV_WT_Long \
  -t2 data/hiv_i47v_complex.pdb \
  -x2 data/hiv_i47v_trajectory_80-100ns.xtc \
  -n2 HIV_I47V_Long \
  -o results/long_timescale \
  --diff-threshold 0.08
```

## Phase 2: Pathway-Specific Analysis

### Target Pathway Definition

We'll focus on key functional pathways in HIV protease:

1. **Flap-to-Active-Site**: Ile50A/B → Asp25A/B
2. **Inter-Chain Communication**: Chain A ↔ Chain B
3. **Inhibitor-Binding-Site**: Residues within 5Å of inhibitor
4. **Allosteric Network**: Long-range communication (>15Å apart)

### Custom Analysis Scripts

#### **1. Pathway Contact Analysis**
```python
#!/usr/bin/env python3

import numpy as np
import pandas as pd
import json
from pathlib import Path

def analyze_pathway_contacts(diff_results_file, pathway_definitions):
    """
    Analyze contact changes within specific functional pathways
    """
    # Load differential results
    with open(diff_results_file, 'r') as f:
        diff_data = json.load(f)
    
    pathway_changes = {}
    
    for pathway_name, residue_pairs in pathway_definitions.items():
        pathway_changes[pathway_name] = {
            'increased_contacts': [],
            'decreased_contacts': [],
            'total_changes': 0
        }
        
        # Check each significant contact change
        for contact in diff_data['increased_contacts']:
            res1, res2 = contact['residue_pair']
            if (res1, res2) in residue_pairs or (res2, res1) in residue_pairs:
                pathway_changes[pathway_name]['increased_contacts'].append(contact)
                pathway_changes[pathway_name]['total_changes'] += 1
        
        for contact in diff_data['decreased_contacts']:
            res1, res2 = contact['residue_pair']
            if (res1, res2) in residue_pairs or (res2, res1) in residue_pairs:
                pathway_changes[pathway_name]['decreased_contacts'].append(contact)
                pathway_changes[pathway_name]['total_changes'] += 1
    
    return pathway_changes

# Define functional pathways
flap_to_active = [
    ('A_50', 'A_25'), ('A_50', 'A_27'), ('A_48', 'A_25'),
    ('B_50', 'B_25'), ('B_50', 'B_27'), ('B_48', 'B_25'),
    ('A_50', 'B_25'), ('B_50', 'A_25')  # Inter-chain connections
]

inter_chain = [
    ('A_25', 'B_25'), ('A_26', 'B_26'), ('A_27', 'B_27'),
    ('A_30', 'B_30'), ('A_80', 'B_80'), ('A_82', 'B_82')
]

inhibitor_site = [
    ('A_25', 'A_82'), ('A_25', 'A_84'), ('A_27', 'A_82'),
    ('B_25', 'B_82'), ('B_25', 'B_84'), ('B_27', 'B_82')
]

pathway_definitions = {
    'flap_to_active': flap_to_active,
    'inter_chain': inter_chain, 
    'inhibitor_site': inhibitor_site
}

# Run pathway analysis
results = analyze_pathway_contacts(
    'results/wt_vs_i47v_advanced/differential_HIV_WT_vs_HIV_I47V.json',
    pathway_definitions
)

print("Pathway-Specific Contact Changes:")
for pathway, changes in results.items():
    print(f"\n{pathway.upper()}:")
    print(f"  Total changes: {changes['total_changes']}")
    print(f"  Increased: {len(changes['increased_contacts'])}")
    print(f"  Decreased: {len(changes['decreased_contacts'])}")
```

#### **2. Allosteric Communication Mapping**
```python
#!/usr/bin/env python3

import networkx as nx
import numpy as np

def map_allosteric_paths(wt_network_file, mut_network_file, source_residues, target_residues):
    """
    Compare allosteric communication paths between WT and mutant
    """
    # Load networks
    wt_net = nx.read_graphml(wt_network_file)
    mut_net = nx.read_graphml(mut_network_file)
    
    path_analysis = {
        'wt_paths': {},
        'mut_paths': {},
        'path_changes': {}
    }
    
    for source in source_residues:
        for target in target_residues:
            # WT paths
            try:
                wt_path = nx.shortest_path(wt_net, source, target, weight='weight')
                wt_length = nx.shortest_path_length(wt_net, source, target, weight='weight')
                path_analysis['wt_paths'][f"{source}_to_{target}"] = {
                    'path': wt_path,
                    'length': wt_length,
                    'exists': True
                }
            except nx.NetworkXNoPath:
                path_analysis['wt_paths'][f"{source}_to_{target}"] = {
                    'path': None,
                    'length': float('inf'),
                    'exists': False
                }
            
            # Mutant paths
            try:
                mut_path = nx.shortest_path(mut_net, source, target, weight='weight')
                mut_length = nx.shortest_path_length(mut_net, source, target, weight='weight')
                path_analysis['mut_paths'][f"{source}_to_{target}"] = {
                    'path': mut_path,
                    'length': mut_length,
                    'exists': True
                }
            except nx.NetworkXNoPath:
                path_analysis['mut_paths'][f"{source}_to_{target}"] = {
                    'path': None,
                    'length': float('inf'), 
                    'exists': False
                }
            
            # Compare paths
            wt_data = path_analysis['wt_paths'][f"{source}_to_{target}"]
            mut_data = path_analysis['mut_paths'][f"{source}_to_{target}"]
            
            path_analysis['path_changes'][f"{source}_to_{target}"] = {
                'length_change': mut_data['length'] - wt_data['length'],
                'path_lost': wt_data['exists'] and not mut_data['exists'],
                'path_gained': not wt_data['exists'] and mut_data['exists'],
                'route_changed': (wt_data['path'] != mut_data['path']) if (wt_data['exists'] and mut_data['exists']) else False
            }
    
    return path_analysis

# Define key functional residues
flap_residues = ['A_50', 'B_50', 'A_48', 'B_48']
active_site_residues = ['A_25', 'B_25', 'A_27', 'B_27']
mutation_site = ['A_47', 'B_47']

# Analyze communication paths
path_results = map_allosteric_paths(
    'results/wt_vs_i47v_advanced/HIV_WT_network.graphml',
    'results/wt_vs_i47v_advanced/HIV_I47V_network.graphml',
    flap_residues,
    active_site_residues
)

print("Allosteric Path Analysis:")
for path_name, changes in path_results['path_changes'].items():
    print(f"\n{path_name}:")
    print(f"  Length change: {changes['length_change']:.3f}")
    print(f"  Path lost: {changes['path_lost']}")
    print(f"  Route changed: {changes['route_changed']}")
```

## Phase 3: Dynamic Network Analysis

### Temporal Network Evolution
```python
#!/usr/bin/env python3

def analyze_temporal_networks(trajectory_segments):
    """
    Analyze how network properties evolve over simulation time
    """
    import matplotlib.pyplot as plt
    
    time_points = []
    densities = {'WT': [], 'I47V': []}
    centralities = {'WT': [], 'I47V': []}
    
    for i, (wt_seg, mut_seg) in enumerate(trajectory_segments):
        time_points.append(i * 10)  # Assuming 10 ns segments
        
        # Run confdelta on each segment
        # This would involve multiple confdelta calls with time-windowed trajectories
        
        # Extract properties (pseudo-code)
        wt_density = analyze_segment(wt_seg)['density']
        mut_density = analyze_segment(mut_seg)['density']
        
        densities['WT'].append(wt_density)
        densities['I47V'].append(mut_density)
    
    # Plot temporal evolution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(time_points, densities['WT'], label='WT', marker='o')
    plt.plot(time_points, densities['I47V'], label='I47V', marker='s')
    plt.xlabel('Time (ns)')
    plt.ylabel('Network Density')
    plt.legend()
    plt.title('Network Density Evolution')
    
    plt.subplot(1, 2, 2)
    plt.plot(time_points, centralities['WT'], label='WT', marker='o')
    plt.plot(time_points, centralities['I47V'], label='I47V', marker='s') 
    plt.xlabel('Time (ns)')
    plt.ylabel('Average Centrality')
    plt.legend()
    plt.title('Average Centrality Evolution')
    
    plt.tight_layout()
    plt.savefig('temporal_network_analysis.png', dpi=300)
    
    return time_points, densities, centralities
```

## Phase 4: Allosteric Mapping and Validation

### Critical Residue Identification
```python
#!/usr/bin/env python3

def identify_critical_residues(differential_results, centrality_threshold=0.1):
    """
    Identify residues critical for allosteric communication
    """
    import pandas as pd
    
    # Load centrality data
    wt_cent = pd.read_csv('results/wt_vs_i47v_advanced/HIV_WT_centrality.csv')
    mut_cent = pd.read_csv('results/wt_vs_i47v_advanced/HIV_I47V_centrality.csv')
    
    # Merge and calculate differences
    merged = pd.merge(wt_cent, mut_cent, on='residue_key', suffixes=('_WT', '_I47V'))
    merged['centrality_change'] = merged['betweenness_I47V'] - merged['betweenness_WT']
    
    # Identify critical residues
    critical_residues = merged[
        (abs(merged['centrality_change']) > centrality_threshold) &
        (merged['betweenness_WT'] > 0.05)  # Was important in WT
    ].copy()
    
    critical_residues['criticality_score'] = (
        abs(critical_residues['centrality_change']) * 
        critical_residues['betweenness_WT']
    )
    
    critical_residues = critical_residues.sort_values('criticality_score', ascending=False)
    
    return critical_residues

critical_res = identify_critical_residues('results/wt_vs_i47v_advanced/')
print("Most Critical Residues for I47V Resistance:")
print(critical_res[['residue_key', 'centrality_change', 'criticality_score']].head(10))
```

### Experimental Validation Predictions
```python
#!/usr/bin/env python3

def generate_validation_predictions(critical_residues, pathway_changes):
    """
    Generate specific testable predictions for experimental validation
    """
    predictions = {
        'mutagenesis_targets': [],
        'binding_assays': [],
        'kinetic_predictions': [],
        'structural_predictions': []
    }
    
    # Top residues for mutagenesis
    for _, residue_data in critical_residues.head(5).iterrows():
        res_key = residue_data['residue_key']
        change = residue_data['centrality_change']
        
        predictions['mutagenesis_targets'].append({
            'residue': res_key,
            'prediction': f"{'Increased' if change > 0 else 'Decreased'} importance in I47V",
            'experiment': f"Compare {res_key}A mutation effects in WT vs I47V background",
            'expected_result': f"{'Larger' if change > 0 else 'Smaller'} effect in I47V"
        })
    
    # Binding predictions based on pathway disruption
    flap_disruption = len(pathway_changes['flap_to_active']['decreased_contacts'])
    if flap_disruption > 3:
        predictions['binding_assays'].append({
            'prediction': "Reduced cooperative binding between flaps and active site",
            'experiment': "Measure inhibitor binding kinetics (kon, koff)",
            'expected_result': "Faster koff due to reduced cooperativity"
        })
    
    # Kinetic predictions
    inter_chain_disruption = len(pathway_changes['inter_chain']['decreased_contacts'])
    if inter_chain_disruption > 2:
        predictions['kinetic_predictions'].append({
            'prediction': "Altered dimer stability",
            'experiment': "Measure enzyme activity vs protein concentration",
            'expected_result': "Shifted monomer-dimer equilibrium"
        })
    
    return predictions

predictions = generate_validation_predictions(critical_res, pathway_results)
print("\nExperimental Validation Predictions:")
for category, pred_list in predictions.items():
    print(f"\n{category.upper()}:")
    for i, pred in enumerate(pred_list, 1):
        print(f"  {i}. {pred['prediction']}")
        print(f"     Test: {pred['experiment']}")
        print(f"     Expect: {pred['expected_result']}")
```

## Expected Advanced Results

### 1. Pathway-Specific Disruption
```
PATHWAY ANALYSIS RESULTS:
========================
Flap-to-Active Site:
  - 8 contacts decreased (WT→I47V)
  - 2 contacts increased  
  - Primary disruption: A_50→A_25 pathway
  
Inter-Chain Communication:
  - 3 contacts decreased
  - 1 contact increased
  - Dimer interface partially destabilized
  
Inhibitor Binding Site:
  - 5 contacts decreased
  - 0 contacts increased  
  - Active site connectivity reduced
```

### 2. Allosteric Path Changes
```
CRITICAL PATH DISRUPTIONS:
=========================
A_50 → A_25: Path length increased by 2.3 steps
B_50 → A_25: Alternative route emerged  
A_47 → A_25: Direct connection lost
A_47 → B_47: Inter-chain communication disrupted
```

### 3. Temporal Network Dynamics
- **Early phase (0-20ns)**: Minimal network changes
- **Middle phase (20-60ns)**: Progressive pathway disruption
- **Late phase (60-100ns)**: Stabilized alternative network

## Clinical and Drug Design Implications

### 1. Resistance Mechanism
- **Primary**: Direct disruption of flap-active site communication
- **Secondary**: Compensatory network formation
- **Tertiary**: Overall network flexibility reduction

### 2. Therapeutic Targets
1. **Allosteric inhibitors** targeting compensatory pathways
2. **Combination therapy** addressing multiple network nodes
3. **Next-generation inhibitors** resistant to I47V effects

### 3. Biomarker Development
- Network centrality patterns as resistance predictors
- Pathway integrity scores for treatment monitoring
- Allosteric signature identification

This advanced differential analysis provides mechanistic insights into resistance that go far beyond simple contact differences, enabling rational drug design and precision medicine approaches for HIV treatment.
