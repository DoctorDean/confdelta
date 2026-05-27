# MD-Compare Examples: HIV Protease Drug Resistance Analysis

This directory contains comprehensive examples demonstrating how to use MD-Compare for studying HIV protease drug resistance mechanisms through network analysis. These examples progress from basic single-simulation analysis to advanced multi-mutant comparative studies.

## Example Overview

### **Example 1: Single Wild-Type Analysis**
`01_single_wildtype_analysis.md`

**Purpose**: Establish baseline network properties for wild-type HIV protease
**Use Case**: Understanding normal protein dynamics and communication networks
**Key Learning**: Basic MD-Compare usage, output interpretation, network visualization

**Command**:
```bash
md-compare single -t hiv_wt_complex.pdb -x hiv_wt_trajectory.xtc -n HIV_WT_Baseline
```

---

### **Example 2: Wild-Type vs Mutant Differential**
`02_wildtype_vs_mutant_differential.md`

**Purpose**: Direct comparison between wild-type and L90M resistant mutant
**Use Case**: Understanding how single mutations affect protein networks
**Key Learning**: Basic differential analysis, resistance mechanism identification

**Command**:
```bash
md-compare diff -t1 hiv_wt.pdb -x1 wt.xtc -n1 WT -t2 hiv_l90m.pdb -x2 mut.xtc -n2 L90M
```

---

### ** Comprehensive Differential Analysis**
`comprehensive_differential_analysis_example.md`

**Purpose**: Complete v1.5.0 differential analysis across all molecular dynamics dimensions
**Use Case**: Quantitative resistance mechanism understanding with statistical rigor
**Key Learning**: Advanced differential analysis, statistical significance, publication-quality results

**Command**:
```bash
md-compare differential -t1 hiv_wt.pdb -x1 wt.xtc -n1 "Wild_Type" \
                        -t2 hiv_v82a.pdb -x2 v82a.xtc -n2 "V82A_Mutant" \
                        -o hiv_resistance_analysis --statistical-tests --publication-figures
```

**Features**:
- Complete network topology, dynamics, energetics, kinetics, and allosteric analysis
- Statistical significance testing with p-values and effect sizes
- Professional output organization suitable for publication
- Comprehensive HTML reports and publication-quality figures

---

### **Example 3: Multi-Mutant Comparison**
`03_multi_mutant_comparison.md`

**Purpose**: Systematic analysis of multiple resistance mutations
**Use Case**: Understanding resistance mechanism diversity and commonalities
**Key Learning**: Batch analysis, pattern recognition, clinical correlation

**Configuration**: `hiv_resistance_panel.json`
**Command**:
```bash
md-compare compare -c hiv_resistance_panel.json -o hiv_resistance_study
```

---

### **Example 4: Advanced Differential Analysis** 
`04_advanced_differential_analysis.md`

**Purpose**: In-depth pathway mapping and allosteric network analysis
**Use Case**: Mechanistic understanding for drug design applications
**Key Learning**: Custom analysis scripts, pathway identification, experimental predictions

## Scientific Context

### HIV Protease Background

HIV protease is a critical enzyme for viral replication that:
- Processes viral polyproteins into mature functional proteins
- Is targeted by protease inhibitor drugs
- Develops resistance through specific mutations
- Shows complex allosteric communication networks

### Drug Resistance Challenge

**Clinical Problem**:
- Resistance mutations reduce drug efficacy
- Multiple resistance pathways exist
- Combination mutations have cumulative effects
- New drugs needed to overcome resistance

**Network Analysis Solution**:
- Maps how mutations affect protein communication
- Identifies alternative drug targets
- Predicts mutation effects
- Guides rational drug design

## Analysis Progression

### **Level 1: Exploration** (Example 1)
- Basic network properties
- System validation
- Method familiarization
- Output interpretation

### **Level 2: Basic Comparison** (Example 2) 
- Mutation impact assessment
- Basic differential contact analysis
- Resistance mechanism hypothesis
- Simple statistical analysis

### **Level 2+: Comprehensive Comparison** (Comprehensive Example)
- Complete differential analysis across all MD dimensions
- Advanced statistical significance testing
- Publication-ready results and figures
- Professional output organization

### **Level 3: Systematic** (Example 3)
- Multiple mutation patterns
- Resistance classification
- Clinical correlation
- Pattern recognition

### **Level 4: Mechanistic** (Example 4)
- Pathway mapping
- Allosteric network analysis
- Experimental predictions
- Drug design implications

## Key Scientific Concepts

### **Residue Interaction Networks (RINs)**
- Residues as nodes, contacts as edges
- Dynamic networks reflecting protein motion
- Communication pathway identification
- Allosteric mechanism mapping

### **Network Centrality Measures**
- **Betweenness**: Communication bottlenecks
- **Closeness**: Information spreading efficiency  
- **Degree**: Local connectivity hubs
- **Eigenvector**: Influence within network

### **Comprehensive Differential Analysis**
- **Network Topology Changes**: Node/edge additions and removals
- **Dynamic Correlation Differences**: DCCM matrix comparisons with statistical testing
- **Energy Landscape Alterations**: Conformational state stability changes
- **Kinetic Pathway Modifications**: MSM timescale and flux changes
- **Allosteric Communication Disruption**: Complete pathway efficiency analysis

### **Resistance Mechanisms**
- **Active Site Volume**: Direct binding site expansion
- **Flap Dynamics**: Altered inhibitor access kinetics
- **Allosteric Networks**: Long-range communication disruption
- **Cooperative Binding**: Multi-site binding effects

## 🛠Technical Requirements

### **Input Files Needed**
```
data/
├── hiv_wt_complex.pdb           # Wild-type structure
├── hiv_wt_trajectory.xtc        # WT dynamics (100 ns)
├── hiv_l90m_complex.pdb         # L90M mutant structure  
├── hiv_l90m_trajectory.xtc      # L90M dynamics
├── hiv_v82a_complex.pdb         # V82A mutant (for comprehensive example)
├── hiv_v82a_trajectory.xtc      # V82A dynamics
├── hiv_i84v_complex.pdb         # I84V mutant
├── hiv_i84v_trajectory.xtc      # I84V dynamics
├── hiv_m46i_complex.pdb         # M46I mutant
├── hiv_m46i_trajectory.xtc      # M46I dynamics
├── hiv_l10i_l63p_complex.pdb    # Double mutant
├── hiv_l10i_l63p_trajectory.xtc # Double mutant dynamics
└── hiv_multidrug_complex.pdb    # Multi-drug resistant
    hiv_multidrug_trajectory.xtc # MDR dynamics
```

### **System Requirements**
- **Memory**: 8-16 GB RAM for large networks (32 GB for comprehensive analysis)
- **CPU**: Multi-core recommended for parallel processing
- **Storage**: 5-10 GB for basic examples, 20-50 GB for comprehensive analysis
- **Time**: 10 minutes - 2 hours depending on analysis complexity

### **Software Dependencies**
- MD-Compare toolkit v1.5.0+ (installed)
- Python 3.8+ with scientific libraries
- Optional: PyMOL, Cytoscape for visualization
- Optional: Jupyter notebooks for interactive analysis

## Expected Outcomes

### **Scientific Insights**
1. **Mutation Classification**: Different resistance mechanisms
2. **Network Resilience**: Protein adaptability patterns  
3. **Allosteric Pathways**: Communication route identification
4. **Drug Targets**: Novel therapeutic intervention points
5. **Quantitative Changes**: Statistical significance of all molecular changes

### **Methodological Skills**
1. **MD-Compare Proficiency**: Complete toolkit usage including v1.5.0 features
2. **Network Analysis**: Interpretation of complex networks
3. **Statistical Analysis**: Significance testing and validation
4. **Scientific Visualization**: Publication-quality figures
5. **Comprehensive Reporting**: Professional manuscript preparation

### **Practical Applications**
1. **Drug Design**: Rational inhibitor development
2. **Resistance Prediction**: Clinical outcome forecasting
3. **Personalized Medicine**: Patient-specific treatment strategies
4. **Research Planning**: Experimental design optimization
5. **Publication Preparation**: Statistical rigor and professional presentation

## Workflow Integration

### **Research Pipeline Integration**
```
1. MD Simulations → 2. MD-Compare Analysis → 3. Statistical Analysis → 4. Hypothesis Generation
                           ↓                        ↓                    ↓
7. Publication ← 6. Manuscript Prep ← 5. Validation Studies ← Experimental Design
```

### **Iterative Analysis Cycle**
1. **Initial Analysis**: Basic network properties
2. **Comprehensive Comparison**: Full differential analysis with statistics
3. **Hypothesis Formation**: Mechanism proposals  
4. **Targeted Analysis**: Focused investigations
5. **Validation Design**: Experimental planning
6. **Results Integration**: Knowledge refinement

## Learning Objectives

After completing these examples, you will be able to:

1. **Analyze single MD simulations** for network properties
2. **Compare multiple simulations** systematically
3. **Perform comprehensive differential analysis** across all MD dimensions
4. **Apply statistical significance testing** to molecular changes
5. **Generate publication-quality results** with professional organization
6. **Identify differential contacts** and their significance
7. **Map allosteric pathways** and communication networks
8. **Correlate network changes** with experimental data
9. **Design validation experiments** based on predictions

## Getting Started

### **Recommended Learning Path**

1. **Start with Example 1** - Learn basic MD-Compare usage
2. **Try Example 2** - Understand basic differential analysis
3. **Use Comprehensive Example** - Experience full v1.5.0 capabilities
4. **Advance to Example 3** - Handle multiple simulations
5. **Complete Example 4** - Advanced mechanistic analysis

### **Quick Start for Comprehensive Analysis**
```bash
# Basic comprehensive differential analysis
python md_compare_cli.py differential \
  -t1 hiv_wt.pdb -x1 wt.xtc -n1 "Wild_Type" \
  -t2 hiv_v82a.pdb -x2 v82a.xtc -n2 "V82A_Mutant" \
  -o results
```

## Additional Resources

### **Scientific Background**
- HIV protease structure and function reviews
- Drug resistance mechanism literature
- Network analysis methodology papers
- Allosteric regulation principles

### **Technical Documentation**
- MD-Compare v1.5.0 user guide and API reference
- NetworkX documentation for graph analysis
- MDAnalysis tutorials for trajectory processing
- Statistical analysis best practices

### **Visualization Tools**
- Cytoscape for network visualization
- PyMOL for structural analysis
- Python matplotlib/seaborn for data visualization
- VMD for trajectory analysis

These examples provide a complete learning path from basic network analysis to advanced resistance mechanism studies, preparing researchers to apply MD-Compare to their own protein systems and research questions.
