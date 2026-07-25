# confdelta: statistical comparison of protein conformational ensembles

![Python](https://img.shields.io/badge/python-3.10--3.13-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)

**confdelta** is a comprehensive toolkit for analyzing molecular dynamics simulations with advanced network analysis, conformational dynamics, and kinetic modeling capabilities. Originally designed for HIV protease research, it provides publication-ready insights into protein dynamics, allosteric mechanisms, and drug resistance pathways.

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
- **Kinetic Modeling**: deeptime integration for MSM analysis
- **Statistical Validation**: Z-score significance testing and cross-validation

### **Publication-Ready Output**
- **9-panel Network Dashboard**: Comprehensive topology visualization
- **8-panel MSM Dashboard**: Complete kinetic analysis visualization
- **Excel-Compatible Exports**: CSV files for detailed data analysis
- **High-Resolution Figures**: 300 DPI publication-quality plots
- **Comprehensive Reports**: Detailed scientific summaries

## **Quick Start**

### **Installation**

```bash
pip install confdelta
```

Optional extras:

```bash
pip install "confdelta[msm]"       # + deeptime (Markov State Models)
pip install "confdelta[leiden]"    # + Leiden community detection
pip install "confdelta[all]"       # + all optional features
```

Requires Python 3.10-3.13. No conda environment is needed; the core install
has no compiled optional dependencies.

After installation the `confdelta` command is on your PATH; you can also
run the tool as `python -m confdelta`.

### **Basic Usage**

Compare two ensembles:

```bash
confdelta compare -a wt.pdb wt.xtc -b mutant.pdb mutant.xtc -o results/
```

Label the conditions so the output tables read clearly:

```bash
confdelta compare \
  -a wt.pdb wt.xtc       --a-name wild_type \
  -b v82a.pdb v82a.xtc   --b-name V82A \
  -o hiv_wt_vs_v82a/
```

Analyse a single ensemble:

```bash
confdelta single -t system.pdb -x trajectory.xtc -n my_run -o results/
```

Change any analysis option through a config file:

```bash
confdelta example-config -o study.json
confdelta compare -a wt.pdb wt.xtc -b mut.pdb mut.xtc --config study.json
```

> **What confdelta reports.** A per-residue **statistical** comparison — an
> effect size (Hedges' g, or Cohen's d for single runs) with a confidence
> interval, and a p-value corrected across all residues (Benjamini–Hochberg by
> default). Effect sizes lead; p and q are secondary. With replicate ensembles
> per condition the replicate is the unit of inference; with one run per
> condition a block bootstrap over frames supplies the uncertainty, carrying a
> caveat that a single run cannot separate the condition effect from run-to-run
> variation. If the design cannot reach significance (e.g. three replicates per
> condition, whose p-value floor is 0.10), the tool says so and points to the
> effect sizes rather than reporting a misleading "nothing significant".
>
> Alongside, **descriptive** views — difference matrices, centrality and
> modularity deltas, energy-surface differences, timescale ratios, pathway
> disruption — are reported without inference. See [AUDIT.md](AUDIT.md) for the
> project's history.

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

### **Kinetic Modeling (deeptime Integration)**
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

## **Configuration**

The command line carries only what a typical run needs; every other analysis
option lives in a JSON config file.

```bash
confdelta example-config -o study.json     # every option, at its default
confdelta compare -a wt.pdb wt.xtc -b mut.pdb mut.xtc --config study.json
```

The full option reference is in **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**.
It is generated from the code and checked against it by the test suite, so it
cannot document an option that does not exist.

##  **License**

This project is licensed under the MIT License - see the LICENSE file for details.

##  **Version History**

confdelta is version **0.1.0**. It continues MD-Compare, which was previously
versioned to 1.5.0; the version was reset at the rename because the public API
is being rebuilt. The full history, including the MD-Compare releases, is in
[CHANGELOG.md](CHANGELOG.md).

---

**confdelta 0.1.0** - continues MD-Compare; see CHANGELOG.md for the lineage.
