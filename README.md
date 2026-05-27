# MD-Compare v1.5.0: Comprehensive Protein Dynamics Analysis Platform

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-1.5.0-orange.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)

**MD-Compare** is a comprehensive toolkit for analyzing molecular dynamics simulations with advanced network analysis, conformational dynamics, and kinetic modeling capabilities. Originally designed for HIV protease research, it provides publication-ready insights into protein dynamics, allosteric mechanisms, and drug resistance pathways.

## **Key Features**

### **Multi-Scale Dynamics Analysis**
-  **Network Topology Analysis**: Communities, centrality, allosteric pathways
-  **Dynamic Cross-Correlations**: DCCM matrices and motion coupling
-  **Principal Component Analysis**: Dominant motion modes and projections  
-  **Free Energy Landscapes**: Thermodynamic conformational analysis
-  **Markov State Models**: Kinetic pathways and transition timescales
-  **Allosteric Hotspot Mapping**: Critical communication residues

### **Advanced Algorithms**
- **Community Detection**: Leiden, Louvain, Spectral, Hierarchical clustering
- **Centrality Analysis**: Betweenness, closeness, eigenvector, degree centrality
- **Network Robustness**: Attack tolerance and vulnerability assessment
- **Kinetic Modeling**: PyEMMA integration for MSM analysis
- **Statistical Validation**: Z-score significance testing and cross-validation

### **Publication-Ready Output**
- **9-panel Network Dashboard**: Comprehensive topology visualization
- **8-panel MSM Dashboard**: Complete kinetic analysis visualization
- **Excel-Compatible Exports**: CSV files for detailed data analysis
- **High-Resolution Figures**: 300 DPI publication-quality plots
- **Comprehensive Reports**: Detailed scientific summaries

## **Quick Start**

### **Installation**

MD-Compare is an installable Python package. Once published it will be
available from PyPI; until then, install from a local clone.

#### **Option 1: pip (Recommended)**
```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Core install
pip install .

# Or with optional features:
pip install ".[msm]"               # + PyEMMA (Markov State Models)
pip install ".[leiden]"            # + Leiden community detection
pip install ".[all]"               # + all optional features
pip install ".[dev]"               # + test/lint/build tooling
```

#### **Option 2: Conda (for the heavier optional dependencies)**
```bash
# Create an environment (Python 3.8-3.12 supported)
conda create -n mdcompare python=3.10
conda activate mdcompare

# PyEMMA / igraph / leidenalg install most reliably from conda-forge
conda install -c conda-forge pyemma python-igraph leidenalg mdanalysis

# Then install MD-Compare itself
pip install .
```

After installation the `md-compare` command is on your PATH; you can also
run the tool as `python -m mdcompare`.

### **Basic Usage**

#### **Complete Analysis (Network + Dynamics + Kinetics)**
```bash
md-compare single \
  -t protein.pdb \
  -x trajectory.xtc \
  -n comprehensive_analysis \
  --compute-dccm \
  --compute-pca \
  --compute-landscape \
  --compute-msm \
  --community-method leiden \
  --allosteric-sources A_50 B_50 \
  --allosteric-targets A_25 B_25
```

#### **Network Analysis Only**
```bash
md-compare single \
  -t protein.pdb \
  -x trajectory.xtc \
  -n network_analysis \
  --no-dccm --no-pca --no-msm \
  --community-method leiden
```

#### **Kinetic Modeling Focus**
```bash
md-compare single \
  -t protein.pdb \
  -x trajectory.xtc \
  -n kinetic_analysis \
  --compute-msm \
  --msm-features distances \
  --msm-clusters 150 \
  --msm-lag-time 10 \
  --kinetic-timescales 8 \
  --metastable-states 6
```

### **Analysis Workflow Overview**

#### **1. Data Preparation**
```bash
# Ensure trajectory is centered and aligned
# MD-Compare handles most trajectory formats via MDAnalysis
# Supported: .xtc, .dcd, .trr, .nc, .dtr (Desmond), .pdb
```

#### **2. Network Construction**
```bash
# Distance-based networks with customizable cutoffs
--cutoff-distance 4.5              # Contact distance in Angstroms
--contact-selection "protein"      # Atom selection for network
--interaction-types distance       # Network interaction types
```

#### **3. Topology Analysis**
```bash
# Community detection and centrality analysis
--community-method leiden          # Algorithm: leiden, louvain, spectral
--centrality-measures all          # Compute all centrality types
--network-robustness               # Assess vulnerability to attacks
```

#### **4. Dynamic Analysis**
```bash
# Cross-correlation and PCA
--compute-dccm                     # Dynamic cross-correlation matrix
--dccm-selection "name CA"         # Atoms for correlation analysis
--compute-pca                      # Principal component analysis
--pca-components 10                # Number of PC components
```

#### **5. Kinetic Modeling**
```bash
# Markov State Model construction
--compute-msm                      # Enable MSM analysis
--msm-features distances           # Feature type for clustering
--msm-clusters 100                 # Number of microstates
--msm-lag-time 10                  # Lag time (frames)
--kinetic-timescales 5             # Implied timescales to compute
--metastable-states 5              # Number of macrostates
```

#### **6. Output Generation**
```bash
# Comprehensive visualization and export
# → 9-panel network dashboard
# → 8-panel MSM analysis dashboard
# → Excel-compatible CSV files
# → High-resolution publication figures
# → Detailed scientific reports
```

## **Analysis Capabilities**

### **Network Topology Analysis**
- **Residue Interaction Networks**: Distance-based, hydrogen bonds, salt bridges
- **Community Detection**: Functional domain identification using multiple algorithms
- **Centrality Analysis**: Critical residue identification with statistical significance
- **Allosteric Pathway Mapping**: Communication route analysis between functional sites
- **Network Robustness**: Vulnerability assessment under targeted attacks

### **Dynamic Analysis**
- **DCCM Analysis**: Residue-residue cross-correlations and motion coupling
- **PCA Projections**: Principal component analysis with variance decomposition
- **Energy Landscapes**: Free energy surfaces from PC1/PC2 projections
- **Motion Mode Analysis**: Collective motions and conformational transitions

### **Kinetic Modeling (PyEMMA Integration)**
- **Markov State Models**: Microstate networks and transition probabilities
- **Implied Timescales**: Process separation and kinetic hierarchy
- **Metastable States**: Long-lived conformational macrostates (PCCA+)
- **Transition Pathways**: Dominant routes between conformational states
- **Rate Constant Analysis**: Quantitative kinetic modeling

## **Output Files**

### **Core Analysis Results**
```
analysis_results/
├── analysis_network_analysis.png          # 9-panel network dashboard
├── analysis_dccm_heatmap.png             # Dynamic cross-correlations
├── analysis_pca_analysis.png             # Principal component analysis
├── analysis_energy_landscape.png         # Free energy surface
├── analysis_msm_analysis.png             # 8-panel MSM dashboard
├── analysis_centrality.csv               # Node importance metrics
├── analysis_allosteric_hotspots.csv      # Critical communication residues
└── analysis_report.json                  # Complete analysis summary
```

### **MSM Analysis Files**
```
msm_results/
├── analysis_transition_matrix.csv         # Full transition matrix (Excel-ready)
├── analysis_transition_matrix_summary.csv # High-probability transitions only
├── analysis_implied_timescales.csv        # Kinetic hierarchy with rates
├── analysis_state_populations.csv         # Microstate importance ranking
├── analysis_metastable_assignments.csv    # Microstate → macrostate mapping
├── analysis_discrete_trajectory.npy       # State assignments over time
└── analysis_msm_summary.txt              # Comprehensive kinetic analysis
```

## **Scientific Applications**

### **HIV Protease Research**
- **Flap Dynamics**: Open/closed transition analysis and inhibitor binding
- **Drug Resistance**: Mutation effects on network topology and kinetics
- **Allosteric Mechanisms**: Cross-chain communication and cooperativity
- **Inhibitor Design**: Binding pathway analysis and residence time prediction

### **General Protein Dynamics**
- **Conformational Selection**: Pre-existing state analysis for ligand binding
- **Allosteric Networks**: Signal transduction pathway identification
- **Protein Folding**: Pathway analysis and intermediate state characterization
- **Stability Engineering**: Critical residue identification for rational design

## ⚙️ **Configuration Options**

### **Input/Output Parameters**
```bash
# Input files
-t, --topology protein.pdb         # Topology file (PDB, PSF, etc.)
-x, --trajectory traj.xtc           # Trajectory file(s) 
-n, --name analysis_name            # Analysis identifier

# Output control
--output-dir ./results/             # Output directory
--save-format png pdf               # Figure formats
--no-cleanup                        # Keep intermediate files
```

### **Network Construction Parameters**
```bash
# Contact definition
--cutoff-distance 4.5               # Distance cutoff (Å)
--contact-selection "protein"       # Atom selection for network
--interaction-types distance        # Network interaction types

# Network topology
--min-contacts 1                    # Minimum contacts per residue
--contact-method ca_distance        # Contact calculation method
```

### **Analysis Method Selection**
```bash
# Core analysis components
--compute-dccm / --no-dccm          # Dynamic cross-correlation matrix
--compute-pca / --no-pca            # Principal component analysis
--compute-landscape / --no-landscape # Free energy landscapes
--compute-msm / --no-msm            # Markov State Models

# Advanced network analysis
--compute-centrality                # Centrality measures
--compute-communities               # Community detection
--compute-robustness                # Network robustness analysis
--compute-paths                     # Shortest path analysis
```

### **Community Detection Parameters**
```bash
# Algorithm selection
--community-method leiden           # {leiden,louvain,spectral,hierarchical}
--resolution 1.0                    # Resolution parameter (leiden/louvain)
--n-communities 5                   # Target communities (spectral)

# Validation options
--community-validation              # Cross-validation of communities
--stability-analysis                # Community stability assessment
```

### **Dynamic Analysis Parameters**
```bash
# Cross-correlation analysis
--dccm-selection "name CA"          # Atoms for correlation analysis
--dccm-cutoff 0.3                   # Correlation significance threshold

# Principal component analysis  
--pca-components 10                 # Number of components to compute
--pca-selection "protein"           # Atom selection for PCA

# Energy landscape analysis
--landscape-temperature 300         # Temperature for Boltzmann weighting
--landscape-bins 50                 # Number of bins per dimension
```

### **MSM Analysis Parameters**
```bash
# Basic MSM options
--msm-lag-time 10                   # Lag time for transition matrix (frames)
--msm-clusters 100                  # Number of microstates
--msm-stride 1                      # Frame sampling stride
--msm-features distances            # {distances,coordinates,angles,dihedrals}

# Clustering options
--msm-clustering kmeans             # {kmeans,regular_space,minibatch_kmeans}
--msm-cluster-distance-cutoff 2.0   # Distance cutoff for clustering

# Kinetic analysis options
--kinetic-timescales 5              # Number of timescales to compute
--metastable-states 5               # Number of macrostates (PCCA+)
--transition-analysis               # Detailed transition analysis
--pathway-analysis                  # Dominant pathway identification

# Model validation
--msm-validation                    # Enable model validation
--cross-validation-folds 5          # Number of CV folds
--bootstrap-samples 100             # Bootstrap samples for uncertainty

# Advanced MSM options
--no-kinetics                       # Skip kinetic analysis
--no-metastable                     # Skip metastable state analysis
--save-discrete-trajectory          # Save state assignments
```

### **Allosteric Analysis Parameters**
```bash
# Allosteric pathway analysis
--allosteric-sources A_50 B_50      # Source residues (format: chain_resid)
--allosteric-targets A_25 B_25      # Target residues
--allosteric-method shortest_path   # {shortest_path,random_walk,betweenness}
--max-path-length 10                # Maximum path length to consider

# Communication analysis
--communication-efficiency          # Global efficiency measures
--local-efficiency                  # Local clustering analysis
--small-world-analysis              # Small-world network properties
```

### **Statistical Analysis Parameters**
```bash
# Significance testing
--significance-testing              # Enable statistical validation
--n-bootstrap 1000                  # Bootstrap samples
--confidence-level 0.95             # Confidence interval level
--multiple-testing-correction fdr   # {bonferroni,fdr,none}

# Comparative analysis
--compare-method ks_test            # {ks_test,t_test,mann_whitney}
--effect-size-threshold 0.2         # Minimum effect size
```

### **Visualization Parameters**
```bash
# Plot customization
--plot-style publication            # {publication,presentation,notebook}
--color-scheme viridis              # Color palette
--figure-size 12 9                  # Figure dimensions (width height)
--dpi 300                           # Resolution for raster formats

# Network visualization
--network-layout spring             # {spring,circular,kamada_kawai}
--node-size-method centrality       # {centrality,degree,uniform}
--edge-width-method weight          # {weight,uniform,significance}

# Specialized plots
--plot-network-evolution            # Time-resolved network properties
--plot-energy-landscapes            # 2D/3D energy surfaces
--plot-msm-network                  # MSM transition network
--plot-pathway-heatmaps             # Allosteric pathway visualization
```

### **Performance and Debugging Parameters**
```bash
# Performance tuning
--n-cores 4                         # Number of CPU cores
--chunk-size 1000                   # Trajectory chunk size (frames)
--memory-limit 8GB                  # Memory usage limit

# Debugging options
--verbose                           # Detailed output
--debug                             # Debug mode with extra info
--profile                           # Performance profiling
--log-level INFO                    # {DEBUG,INFO,WARNING,ERROR}
--save-intermediate                 # Save intermediate results
```

### **Platform-Specific Options**
```bash
# Windows compatibility
--use-utf8-encoding                 # Force UTF-8 encoding
--windows-path-fix                  # Handle Windows path issues

# HPC/cluster options
--batch-mode                        # Disable interactive features
--no-display                        # Disable GUI components
--scratch-dir /tmp/mdcompare        # Temporary directory
```

##  **Troubleshooting**

### **PyEMMA Installation Problems**
```bash
# Solution 1: Use conda-forge
conda install -c conda-forge pyemma

# Solution 2: Use pre-built wheels
pip install --only-binary=all pyemma

# Solution 3: Use Python 3.8-3.10
pyenv install 3.10.12
pyenv local 3.10.12
pip install pyemma
```

### **Memory Issues with Large Systems**
```bash
# Reduce memory usage
md-compare single \
  -t large_protein.pdb \
  -x large_trajectory.xtc \
  -n memory_efficient \
  --msm-stride 5 \
  --msm-clusters 50 \
  --contact-selection "name CA"
```

##  **License**

This project is licensed under the MIT License - see the LICENSE file for details.

##  **Version History**

### **v1.5.0 (Current)**
- **Differential Analysis Framework**: Full simulation-vs-simulation comparison
- **Installable Package**: `src/` layout, `pyproject.toml`, console scripts
- **Statistical Comparators**: Network, dynamics, energetics, kinetics, allosteric

### **v1.4.0**
- **PyEMMA Integration**: Complete MSM analysis framework
- **Enhanced Visualizations**: 8-panel MSM dashboard
- **Excel Export**: CSV files for transition matrices and kinetic data
- **Cross-Platform Compatibility**: Windows/macOS/Linux support

### **v1.3.1**
- **Bug Fixes**: Resolved visualization and compatibility issues
- **NetworkX Compatibility**: Support for versions 1.x-3.x
- **Enhanced Error Handling**: Comprehensive fallback strategies

### **v1.3.0** 
- **Advanced Network Analysis**: Community detection and robustness analysis
- **Statistical Validation**: Z-score significance testing
- **Enhanced Visualizations**: 9-panel analysis dashboard

---

**MD-Compare v1.5.0** - *Advancing Protein Dynamics Analysis Through Computational Innovation* 
