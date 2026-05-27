# Example 1: Single HIV Protease Simulation Analysis

This example demonstrates analyzing a single wild-type HIV protease MD simulation to understand its baseline network properties and dynamics.

## Background

We're analyzing a 100 ns molecular dynamics simulation of wild-type HIV protease in complex with the inhibitor darunavir. The system was prepared using CHARMM-GUI and simulated with GROMACS.

**System Details:**
- HIV protease dimer (99 residues per chain, chains A and B)
- Darunavir inhibitor bound in the active site
- Explicit water and ions
- 100 ns production simulation (1000 frames saved)
- Temperature: 310 K, Pressure: 1 bar

## Files Required

```
data/
├── hiv_wt_complex.pdb          # Topology file with HIV protease + inhibitor
├── hiv_wt_trajectory.xtc       # 100 ns trajectory (1000 frames)
└── README.md                   # System description
```

## Command Line Usage

### Basic Analysis
```bash
md-compare single \
  -t data/hiv_wt_complex.pdb \
  -x data/hiv_wt_trajectory.xtc \
  -n HIV_WT_Baseline \
  -o results/hiv_wt_analysis \
  --cutoff 4.5 \
  --threshold 0.2 \
  --segments 5
```

### Advanced Analysis with Multiple Interaction Types
```bash
md-compare single \
  -t data/hiv_wt_complex.pdb \
  -x data/hiv_wt_trajectory.xtc \
  -n HIV_WT_Complete \
  -o results/hiv_wt_complete \
  --interaction-types distance hbond salt_bridge \
  --cutoff 4.5 \
  --threshold 0.15 \
  --segments 10 \
  --timeout 600
```

### Dynamic Analysis with DCCM and PCA
```bash
md-compare single \
  -t data/hiv_wt_complex.pdb \
  -x data/hiv_wt_trajectory.xtc \
  -n HIV_WT_Dynamics \
  -o results/hiv_wt_dynamics \
  --compute-dccm \
  --compute-pca \
  --pca-components 15 \
  --dccm-selection "name CA" \
  --cutoff 4.5 \
  --threshold 0.2
```

### Complete Analysis with Energy Landscape
```bash
md-compare single \
  -t data/hiv_wt_complex.pdb \
  -x data/hiv_wt_trajectory.xtc \
  -n HIV_WT_Complete \
  -o results/hiv_wt_complete \
  --compute-dccm \
  --compute-pca \
  --compute-landscape \
  --pca-components 15 \
  --landscape-bins 60 \
  --landscape-temp 310 \
  --landscape-sigma 1.2 \
  --cutoff 4.5 \
  --threshold 0.2
```

### Advanced Network Analysis with Allosteric Pathways
```bash
md-compare single \
  -t data/hiv_wt_complex.pdb \
  -x data/hiv_wt_trajectory.xtc \
  -n HIV_WT_Network \
  -o results/hiv_wt_network \
  --community-method leiden \
  --allosteric-sources A_50 B_50 A_48 B_48 \
  --allosteric-targets A_25 B_25 A_27 B_27 \
  --cutoff 4.5 \
  --threshold 0.2
```

### Comprehensive Analysis (All Features)
```bash
md-compare single \
  -t data/hiv_wt_complex.pdb \
  -x data/hiv_wt_trajectory.xtc \
  -n HIV_WT_Full \
  -o results/hiv_wt_full \
  --compute-dccm \
  --compute-pca \
  --compute-landscape \
  --community-method leiden \
  --allosteric-sources A_50 B_50 A_45 B_45 A_48 B_48 \
  --allosteric-targets A_25 B_25 A_27 B_27 A_82 B_82 \
  --pca-components 20 \
  --landscape-bins 70 \
  --cutoff 4.5 \
  --threshold 0.18 \
  --segments 8
```

### High-Resolution Energy Landscape
```bash
md-compare single \
  -t data/hiv_wt_complex.pdb \
  -x data/hiv_wt_trajectory.xtc \
  -n HIV_WT_HighRes \
  -o results/hiv_wt_highres \
  --landscape-bins 80 \
  --landscape-temp 300 \
  --landscape-sigma 0.8 \
  --pca-components 20
```

### Analysis with Custom Selections
```bash
md-compare single \
  -t data/hiv_wt_complex.pdb \
  -x data/hiv_wt_trajectory.xtc \
  -n HIV_WT_Protein_Only \
  -o results/hiv_wt_protein \
  --selection "protein and not name H*" \
  --align-selection "name CA and resid 10-90" \
  --center-selection "protein" \
  --dccm-selection "name CA and protein"
```

## Expected Results

### Network Properties
- **Nodes**: ~198 residues (99 per chain)
- **Edges**: 800-1200 contacts (depending on threshold)
- **Density**: 0.04-0.06 (typical for protein networks)
- **Communities**: 4-6 communities (often corresponding to protein domains)

### Dynamic Analysis Results
- **DCCM**: Cross-correlation matrix showing residue motion coupling
- **PCA**: First 3 components typically explain 60-80% of variance
- **Motion Modes**: Collective motions including flap opening/closing
- **Correlated Regions**: Active site-flap coupling, inter-chain communication

### Energy Landscape Results
- **Free Energy Surface**: 2D landscape showing conformational stability
- **Energy Minima**: Stable conformational states (typically 2-5 for HIV protease)
- **Energy Barriers**: Transition barriers between states (5-20 kJ/mol typical)
- **Conformational Routes**: Pathways between stable states
- **Thermodynamic Profile**: Temperature-dependent conformational preferences

### Advanced Network Analysis Results
- **Community Detection**: 4-6 functional communities (domains, interfaces, active regions)
- **Allosteric Pathways**: Communication routes between flaps and active site
- **Network Robustness**: Resilience to node/edge removal (typically 0.6-0.8)
- **Critical Residues**: Statistically significant nodes (z-score > 1.96)
- **Communication Efficiency**: Quality of information flow through network

### Key Insights Expected
1. **Flap Dynamics**: High centrality for flap region residues (Ile50A, Ile50B)
2. **Active Site Hub**: Asp25A and Asp25B as central catalytic residues
3. **Inter-chain Communication**: Strong contacts across the dimer interface
4. **Allosteric Networks**: Pathways connecting inhibitor binding site to flaps
5. **Dynamic Coupling**: DCCM reveals correlated motions between flaps and active site
6. **Principal Modes**: PCA identifies dominant conformational changes

### Output Files Generated

```
results/hiv_wt_analysis/
├── HIV_WT_Baseline_distance_contacts.npy        # Contact frequency matrix
├── HIV_WT_Baseline_distance_contacts.csv        # Contact data (if <1M entries)
├── HIV_WT_Baseline_centrality.csv              # Centrality measures per residue
├── HIV_WT_Baseline_network.graphml             # Network for Cytoscape/Gephi
├── HIV_WT_Baseline_system_info.txt             # System composition details
├── HIV_WT_Baseline_dccm_matrix.npy             # Dynamic cross-correlation matrix
├── HIV_WT_Baseline_dccm_matrix.csv             # DCCM data (if not too large)
├── HIV_WT_Baseline_dccm_heatmap.png            # DCCM visualization
├── HIV_WT_Baseline_pca_eigenvalues.npy         # PCA eigenvalues
├── HIV_WT_Baseline_pca_eigenvectors.npy        # PCA eigenvectors
├── HIV_WT_Baseline_pca_variance_explained.npy  # Variance explained by each PC
├── HIV_WT_Baseline_principal_components.npy    # Trajectory projections
├── HIV_WT_Baseline_pca_summary.txt             # PCA analysis summary
├── HIV_WT_Baseline_pca_analysis.png            # PCA plots (scree, variance, etc.)
├── HIV_WT_Baseline_dynamic_summary.txt         # Dynamic analysis overview
├── HIV_WT_Baseline_energy_landscape.npy        # Free energy surface matrix
├── HIV_WT_Baseline_landscape_pc1_bins.npy      # PC1 bin edges
├── HIV_WT_Baseline_landscape_pc2_bins.npy      # PC2 bin edges
├── HIV_WT_Baseline_landscape_gradient_x.npy    # Energy gradients (forces)
├── HIV_WT_Baseline_landscape_gradient_y.npy    # Energy gradients (forces)
├── HIV_WT_Baseline_landscape_laplacian.npy     # Curvature matrix
├── HIV_WT_Baseline_landscape_summary.txt       # Energy landscape summary
├── HIV_WT_Baseline_energy_landscape.png        # Comprehensive landscape plots
├── HIV_WT_Baseline_energy_landscape_detailed.png # High-resolution contour plot
├── HIV_WT_Baseline_community_analysis.json     # Detailed community detection results
├── HIV_WT_Baseline_community_assignments.csv   # Node-to-community mapping
├── HIV_WT_Baseline_community_summary.txt       # Community detection summary
├── HIV_WT_Baseline_path_metrics.json          # Detailed path analysis
├── HIV_WT_Baseline_path_summary.txt           # Path analysis summary
├── HIV_WT_Baseline_allosteric_analysis.json    # Complete allosteric analysis
├── HIV_WT_Baseline_allosteric_summary.txt      # Allosteric pathway summary
├── HIV_WT_Baseline_allosteric_hotspots.csv     # Hotspot residues with scores
├── HIV_WT_Baseline_robustness_analysis.json    # Network robustness results
├── HIV_WT_Baseline_robustness_summary.txt      # Robustness summary
├── HIV_WT_Baseline_centrality_significance.json # Statistical significance data
├── HIV_WT_Baseline_significant_nodes.csv       # Statistically significant nodes
├── HIV_WT_Baseline_significance_summary.txt    # Significance analysis summary
├── HIV_WT_Baseline_advanced_network_analysis.png # Comprehensive network visualization
├── analysis_config.json                        # Analysis parameters used
└── workflow_summary.json                       # Complete analysis summary
```

## Interpreting Results

### 1. System Information
Check `HIV_WT_Baseline_system_info.txt`:
```
MD Simulation Analysis: HIV_WT_Baseline
==================================================

name: HIV_WT_Baseline
n_chains: 2
n_residues: 198
n_atoms: 1536
n_frames: 1000
description: Single simulation analysis: HIV_WT_Baseline

Chain Information:
  Chain A:
    Residues: 99
    Range: 1-99
    Atoms: 768

  Chain B:
    Residues: 99
    Range: 1-99
    Atoms: 768
```

### 2. Network Centrality Analysis
Key residues to examine in `HIV_WT_Baseline_centrality.csv`:

**High Betweenness Centrality** (communication hubs):
- Asp25A, Asp25B (active site catalytic residues)
- Ile50A, Ile50B (flap tips - conformational switches)
- Gly48A, Gly48B (flap hinge regions)

**High Degree Centrality** (highly connected):
- Core hydrophobic residues
- Interface residues between chains

### 3. Contact Analysis
Use the contact matrix to identify:
- **Persistent contacts** (frequency > 0.8): Structural core
- **Dynamic contacts** (frequency 0.2-0.8): Functional regions
- **Inter-chain contacts**: Dimer stability

### 4. Visualization Recommendations

#### Cytoscape Network Analysis:
1. Load `HIV_WT_Baseline_network.graphml` into Cytoscape
2. Color nodes by betweenness centrality
3. Size nodes by degree centrality
4. Identify communities with clustering algorithms

#### DCCM Analysis:
1. Examine `HIV_WT_Baseline_dccm_heatmap.png` for correlated motions
2. **Positive correlations (red)**: Residues moving in same direction
3. **Negative correlations (blue)**: Residues moving in opposite directions
4. **Strong anti-correlations**: Often indicate allosteric relationships

#### PCA Analysis:
1. Review `HIV_WT_Baseline_pca_analysis.png` for motion modes
2. **First PC**: Usually represents the most dominant motion (flap opening/closing)
3. **Eigenvalue spectrum**: Shows which modes are most important
4. **Cumulative variance**: How many PCs needed to capture most motion

#### Energy Landscape Analysis:
1. Examine `HIV_WT_Baseline_energy_landscape.png` for conformational states
2. **Energy minima (red stars)**: Stable conformational states
3. **Contour lines**: Isoenergy surfaces showing transition pathways
4. **Color gradients**: Blue (low energy/stable) to yellow (high energy/unstable)
5. **3D surface**: Overall shape of conformational landscape
6. **Gradient vectors**: Forces driving conformational changes
7. **Laplacian plot**: Curvature showing stability/instability regions

#### Advanced Network Analysis:
1. Review `HIV_WT_Baseline_advanced_network_analysis.png` comprehensive overview
2. **Community structure**: Functional domain organization
3. **Allosteric hotspots**: Critical residues for communication
4. **Network robustness**: System resilience to perturbations
5. **Communication efficiency**: Quality of information flow
6. **Statistical significance**: Identify outlier nodes (high z-scores)

#### Community Detection Analysis:
```python
# Load community assignments
import pandas as pd
comm_df = pd.read_csv('results/hiv_wt_analysis/HIV_WT_Baseline_community_assignments.csv')

# Analyze community composition
community_groups = comm_df.groupby('community_id')['node'].apply(list).to_dict()

print("Community Structure:")
for comm_id, nodes in community_groups.items():
    print(f"Community {comm_id}: {len(nodes)} nodes")
    print(f"  Example nodes: {nodes[:5]}")  # Show first 5 nodes
```

#### Allosteric Pathway Analysis:
```python
# Load allosteric hotspots
hotspots_df = pd.read_csv('results/hiv_wt_analysis/HIV_WT_Baseline_allosteric_hotspots.csv')

print("Top 10 Allosteric Hotspots:")
print(hotspots_df.head(10)[['node', 'hotspot_score', 'frequency']])

# Identify functional regions
flap_hotspots = hotspots_df[hotspots_df['node'].str.contains('_4[8-9]|_5[0-2]')]
active_hotspots = hotspots_df[hotspots_df['node'].str.contains('_2[5-7]')]

print(f"\nFlap region hotspots: {len(flap_hotspots)}")
print(f"Active site hotspots: {len(active_hotspots)}")
```

#### Network Robustness Analysis:
```python
# Load robustness data
import json
with open('results/hiv_wt_analysis/HIV_WT_Baseline_robustness_analysis.json', 'r') as f:
    robustness = json.load(f)

print("Network Robustness Metrics:")
print(f"Node attack robustness: {robustness['node_attack_robustness']:.3f}")
print(f"Random failure robustness: {robustness['random_failure_robustness']:.3f}")
print(f"Edge attack robustness: {robustness['edge_attack_robustness']:.3f}")

critical_nodes = robustness['critical_nodes']
print(f"\nTop 5 Critical Nodes:")
for node in critical_nodes[:5]:
    print(f"  {node}")
```

#### PyMOL Structure Visualization:
```python
# PyMOL script for centrality visualization
load hiv_wt_complex.pdb
# Color by betweenness centrality (from CSV file)
alter all, b=0  # Reset B-factors
# Map centrality values to B-factors
spectrum b, blue_white_red, minimum=0, maximum=1
```

### 5. Advanced Analysis Scripts

#### DCCM Analysis Script:
```python
import numpy as np
import matplotlib.pyplot as plt

# Load DCCM matrix
dccm = np.load('results/hiv_wt_analysis/HIV_WT_Baseline_dccm_matrix.npy')

# Find strongest correlations
strong_pos = np.where(dccm > 0.7)
strong_neg = np.where(dccm < -0.7)

print(f"Strong positive correlations: {len(strong_pos[0])}")
print(f"Strong negative correlations: {len(strong_neg[0])}")

# Plot correlation distribution
plt.figure(figsize=(8, 6))
plt.hist(dccm.flatten(), bins=50, alpha=0.7)
plt.xlabel('Cross-Correlation')
plt.ylabel('Frequency')
plt.title('DCCM Value Distribution')
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
plt.show()
```

#### PCA Analysis Script:
```python
import numpy as np
import matplotlib.pyplot as plt

# Load PCA results
eigenvals = np.load('results/hiv_wt_analysis/HIV_WT_Baseline_pca_eigenvalues.npy')
variance_exp = np.load('results/hiv_wt_analysis/HIV_WT_Baseline_pca_variance_explained.npy')
pc_proj = np.load('results/hiv_wt_analysis/HIV_WT_Baseline_principal_components.npy')

# Analyze the first few modes
print("Principal Component Analysis Results:")
print(f"PC1 explains {variance_exp[0]*100:.1f}% of variance")
print(f"PC2 explains {variance_exp[1]*100:.1f}% of variance")
print(f"PC3 explains {variance_exp[2]*100:.1f}% of variance")

# Plot trajectory evolution in PC space
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(pc_proj[:, 0], label='PC1')
plt.plot(pc_proj[:, 1], label='PC2')
plt.xlabel('Frame')
plt.ylabel('PC Projection')
plt.title('Trajectory Evolution')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(pc_proj[:, 0], pc_proj[:, 1], alpha=0.6, c=range(len(pc_proj)))
plt.xlabel(f'PC1 ({variance_exp[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({variance_exp[1]*100:.1f}%)')
plt.title('PC1 vs PC2 Projection')
plt.colorbar(label='Frame')
plt.tight_layout()
plt.show()
```

#### Energy Landscape Analysis Script:
```python
import numpy as np
import matplotlib.pyplot as plt

# Load energy landscape data
landscape = np.load('results/hiv_wt_analysis/HIV_WT_Baseline_energy_landscape.npy')
pc1_bins = np.load('results/hiv_wt_analysis/HIV_WT_Baseline_landscape_pc1_bins.npy')
pc2_bins = np.load('results/hiv_wt_analysis/HIV_WT_Baseline_landscape_pc2_bins.npy')
gradient_x = np.load('results/hiv_wt_analysis/HIV_WT_Baseline_landscape_gradient_x.npy')
gradient_y = np.load('results/hiv_wt_analysis/HIV_WT_Baseline_landscape_gradient_y.npy')

# Analyze energy landscape
print("Energy Landscape Analysis:")
print(f"Energy range: {np.min(landscape):.1f} to {np.max(landscape):.1f} kJ/mol")
print(f"Energy span: {np.max(landscape) - np.min(landscape):.1f} kJ/mol")

# Find energy minima
min_indices = np.unravel_index(landscape.argmin(), landscape.shape)
min_energy = landscape[min_indices]
print(f"Global minimum: {min_energy:.1f} kJ/mol")

# Analyze gradient magnitude (forces)
gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
print(f"Maximum force: {np.max(gradient_mag):.2f} kJ/mol/unit")

# Custom landscape analysis
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
pc1_centers = (pc1_bins[:-1] + pc1_bins[1:]) / 2
pc2_centers = (pc2_bins[:-1] + pc2_bins[1:]) / 2
PC1, PC2 = np.meshgrid(pc1_centers, pc2_centers)
contour = plt.contourf(PC1, PC2, landscape, levels=20, cmap='viridis')
plt.colorbar(contour, label='Energy (kJ/mol)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Energy Landscape')

plt.subplot(1, 3, 2)
plt.imshow(gradient_mag, extent=[pc1_bins[0], pc1_bins[-1], pc2_bins[0], pc2_bins[-1]], 
           origin='lower', cmap='plasma')
plt.colorbar(label='|∇G| (kJ/mol/unit)')
plt.xlabel('PC1')
plt.ylabel('PC2')  
plt.title('Force Field Magnitude')

plt.subplot(1, 3, 3)
# Energy profile along PC1 (averaging over PC2)
pc1_profile = np.mean(landscape, axis=0)
plt.plot(pc1_centers, pc1_profile, 'b-', linewidth=2)
plt.xlabel('PC1 Projection')
plt.ylabel('Average Energy (kJ/mol)')
plt.title('PC1 Energy Profile')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Identify conformational states
low_energy_threshold = np.min(landscape) + 5.0  # Within 5 kJ/mol of minimum
stable_regions = landscape < low_energy_threshold
n_stable_points = np.sum(stable_regions)
print(f"Stable conformations (within 5 kJ/mol): {n_stable_points} points")
print(f"Percentage of conformational space: {n_stable_points/landscape.size*100:.1f}%")
```

## Next Steps

1. **Compare with mutants** using Example 2
2. **Analyze different inhibitors** with the same system
3. **Investigate specific pathways** using high centrality residues
4. **Validate predictions** with experimental mutagenesis data

## Troubleshooting

### Common Issues:

**Large network timeout:**
```bash
# Reduce network size with higher threshold
md-compare single ... --threshold 0.3 --timeout 300
```

**Memory issues with dynamic analysis:**
```bash
# Disable DCCM/PCA for large systems
md-compare single ... --no-dccm --no-pca

# Or use fewer PCA components
md-compare single ... --pca-components 5
```

**DCCM computation slow:**
```bash
# Use backbone atoms only
md-compare single ... --dccm-selection "backbone"

# Or CA atoms from specific region
md-compare single ... --dccm-selection "name CA and resid 20-80"
```

**Chain detection problems:**
```bash
# Verify chain information in system_info.txt
# Check if chains are named A,B vs 1,2 in topology
```

**PCA fails:**
```bash
# Ensure sufficient frames for stable PCA
# Need at least 3x more frames than atoms for analysis
# Check that trajectory has conformational variation
```

**Energy landscape computation slow:**
```bash
# Reduce resolution for faster computation
md-compare single ... --landscape-bins 30

# Skip smoothing
md-compare single ... --landscape-sigma 0

# Disable landscape for speed
md-compare single ... --no-landscape
```

**Energy landscape shows artifacts:**
```bash
# Increase smoothing
md-compare single ... --landscape-sigma 2.0

# More bins for better resolution
md-compare single ... --landscape-bins 70

# Check temperature setting
md-compare single ... --landscape-temp 310
```

**Community detection fails:**
```bash
# Try different method
md-compare single ... --community-method louvain

# Skip if problematic
md-compare single ... --no-communities
```

**Allosteric analysis slow:**
```bash
# Specify fewer sources/targets
md-compare single ... --allosteric-sources A_50 B_50 --allosteric-targets A_25 B_25

# Skip if not needed
md-compare single ... --no-allosteric
```

**Memory issues with advanced analysis:**
```bash
# Disable resource-intensive components
md-compare single ... --no-paths --no-allosteric --community-method louvain
```

This baseline analysis establishes the foundation for understanding how mutations and different conditions affect the HIV protease network structure and dynamics.
