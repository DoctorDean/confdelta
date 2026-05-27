# MD-Compare Installation Guide

This guide provides detailed instructions for installing MD-Compare v1.5.0 and all its dependencies across different platforms and environments.

> **Note:** MD-Compare is now an installable package. The recommended install
> is `pip install .` (or `pip install ".[all]"` for optional features) from a
> clone of the repository — this puts the `md-compare` command on your PATH.
> The `requirements.txt` route below still works but only installs the
> dependencies, not the package itself.

## 🎯 **Quick Installation**

### **Option 1: Conda (Recommended)**
```bash
# Create environment
conda create -n mdcompare python=3.10
conda activate mdcompare

# Install from conda-forge
conda install -c conda-forge pyemma python-igraph leidenalg mdanalysis

# Install remaining dependencies
pip install networkx scipy pandas matplotlib seaborn scikit-learn
```

### **Option 2: pip Installation**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```

## 📋 **System Requirements**

### **Minimum Requirements**
- Python 3.8+ (Python 3.10 recommended for PyEMMA compatibility)
- 4GB RAM (8GB+ recommended for large systems)
- 1GB disk space

### **Recommended Requirements**
- Python 3.10+
- 16GB RAM for large protein systems (>500 residues)
- SSD storage for fast I/O
- Multi-core CPU for parallel processing

### **Platform Support**
- ✅ **Windows 10/11** (x64)
- ✅ **macOS 10.15+** (Intel and Apple Silicon)
- ✅ **Linux** (Ubuntu 18.04+, CentOS 7+, others)

## 🔧 **Platform-Specific Instructions**

### **Windows Installation**

#### **Prerequisites**
```powershell
# Install Visual C++ Build Tools (if needed for compilation)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### **Standard Installation**
```powershell
# Option 1: Using Anaconda/Miniconda (recommended)
conda create -n mdcompare python=3.10
conda activate mdcompare
conda install -c conda-forge pyemma python-igraph leidenalg mdanalysis
pip install networkx scipy pandas matplotlib seaborn scikit-learn

# Option 2: Using pip only
python -m venv .venv
.venv\Scripts\activate
pip install "numpy<2.0"  # PyEMMA compatibility
pip install -r requirements.txt
```

#### **Troubleshooting Windows**
- **PyEMMA fails**: Use `conda install -c conda-forge pyemma` or `pip install --only-binary=all pyemma`
- **Encoding errors**: Ensure Windows Terminal uses UTF-8 encoding
- **Permission errors**: Run PowerShell as Administrator

### **macOS Installation**

#### **Prerequisites**
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### **Standard Installation**
```bash
# Option 1: Using conda
conda create -n mdcompare python=3.10
conda activate mdcompare
conda install -c conda-forge pyemma python-igraph leidenalg mdanalysis
pip install networkx scipy pandas matplotlib seaborn scikit-learn

# Option 2: Using pyenv and pip
brew install pyenv
pyenv install 3.10.12
pyenv local 3.10.12
pip install -r requirements.txt
```

#### **Troubleshooting macOS**
- **Compilation errors**: Ensure Xcode tools are updated
- **M1/M2 Macs**: Use conda-forge for best compatibility
- **Permission issues**: Use `--user` flag: `pip install --user -r requirements.txt`

### **Linux Installation**

#### **Prerequisites**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential

# CentOS/RHEL
sudo yum install python3-devel gcc gcc-c++ make

# Or using dnf (newer versions)
sudo dnf install python3-devel gcc gcc-c++ make
```

#### **Standard Installation**
```bash
# Option 1: Using conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n mdcompare python=3.10
conda activate mdcompare
conda install -c conda-forge pyemma python-igraph leidenalg mdanalysis
pip install networkx scipy pandas matplotlib seaborn scikit-learn

# Option 2: Using system Python
python3 -m venv mdcompare_env
source mdcompare_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### **HPC/Cluster Installation**
```bash
# Load required modules (example for SLURM)
module load python/3.10
module load gcc/9.3.0

# Create environment
python -m venv $HOME/mdcompare_env
source $HOME/mdcompare_env/bin/activate

# Install with specific compiler flags if needed
export CC=gcc
export CXX=g++
pip install -r requirements.txt
```

## 📦 **Dependency Management**

### **Core Dependencies (Always Required)**
```
MDAnalysis>=2.4.0          # Molecular dynamics analysis
networkx>=2.8               # Graph analysis
numpy>=1.21.0,<2.0          # Numerical computing (PyEMMA compat)
scipy>=1.7.0                # Scientific computing
pandas>=1.3.0               # Data manipulation
matplotlib>=3.5.0           # Plotting
seaborn>=0.11.0             # Statistical visualization
scikit-learn>=1.0.0         # Machine learning
tqdm>=4.62.0                # Progress bars
```

### **Optional Dependencies**

#### **MSM Analysis (Recommended)**
```bash
# For kinetic modeling and Markov State Models
pip install pyemma>=2.5.11

# Alternative installation methods:
conda install -c conda-forge pyemma
pip install --only-binary=all pyemma
```

#### **Advanced Network Analysis**
```bash
# For Leiden algorithm and enhanced community detection
pip install python-igraph>=0.10.0 leidenalg>=0.9.0

# Or via conda:
conda install -c conda-forge python-igraph leidenalg
```

#### **Interactive Analysis**
```bash
# For Jupyter notebooks and interactive plots
pip install jupyter>=1.0.0 ipywidgets>=7.6.0 plotly>=5.0.0
```

#### **Development Tools**
```bash
# For development and testing
pip install pytest>=6.0.0 pytest-cov>=3.0.0 black>=22.0.0 flake8>=4.0.0
```

## 🧪 **Installation Verification**

### **Basic Functionality Test**
```python
# test_installation.py
import sys
print(f"Python version: {sys.version}")

# Test core imports
try:
    import MDAnalysis
    import networkx
    import numpy
    import scipy
    import pandas
    import matplotlib
    import seaborn
    import sklearn
    print("✅ Core dependencies installed successfully")
except ImportError as e:
    print(f"❌ Core dependency missing: {e}")

# Test optional dependencies
try:
    import pyemma
    print(f"✅ PyEMMA {pyemma.__version__} available for MSM analysis")
except ImportError:
    print("⚠️ PyEMMA not available (MSM analysis disabled)")

try:
    import igraph
    import leidenalg
    print("✅ Advanced community detection available")
except ImportError:
    print("⚠️ Advanced community detection not available")

# Test MD-Compare
try:
    import md_compare_core
    print("✅ MD-Compare core module loaded successfully")
except ImportError as e:
    print(f"❌ MD-Compare import failed: {e}")

print("\n🧬 MD-Compare installation verification complete!")
```

### **Run Verification**
```bash
python test_installation.py
```

### **Feature Check**
```python
# Check available features
import md_compare
md_compare.check_dependencies(verbose=True)
```

## 🔧 **Troubleshooting**

### **Common Issues and Solutions**

#### **PyEMMA Installation Fails**
```bash
# Problem: Compilation errors, missing dependencies
# Solutions (try in order):

# 1. Use conda-forge (most reliable)
conda install -c conda-forge pyemma

# 2. Use pre-built wheels
pip install --only-binary=all pyemma

# 3. Use Python 3.8-3.10
pyenv install 3.10.12
pyenv local 3.10.12
pip install pyemma

# 4. Use different PyEMMA version
pip install pyemma==2.5.10
```

#### **NumPy Compatibility Issues**
```bash
# Problem: PyEMMA incompatible with NumPy 2.0+
# Solution: Install compatible NumPy version
pip install "numpy>=1.21.0,<2.0"
pip install pyemma
```

#### **Memory Issues**
```bash
# Problem: Out of memory during analysis
# Solutions:
# 1. Use stride to reduce data
md-compare single ... --msm-stride 5

# 2. Reduce cluster count
md-compare single ... --msm-clusters 50

# 3. Focus analysis
md-compare single ... --contact-selection "name CA"
```

#### **Performance Issues**
```bash
# Problem: Analysis too slow
# Solutions:
# 1. Skip expensive analysis
md-compare single ... --no-pca --no-landscape

# 2. Use faster algorithms
md-compare single ... --community-method louvain

# 3. Reduce network size
md-compare single ... --cutoff-distance 4.0
```

## 🚀 **Post-Installation Setup**

### **Environment Configuration**
```bash
# Add to .bashrc or .zshrc
export MDCOMPARE_DATA_DIR="$HOME/md_compare_data"
export MDCOMPARE_RESULTS_DIR="$HOME/md_compare_results"

# Create data directories
mkdir -p $MDCOMPARE_DATA_DIR $MDCOMPARE_RESULTS_DIR
```

### **Test Analysis**
```bash
# Run test analysis (requires test data)
md-compare single \
  -t example_data/protein.pdb \
  -x example_data/trajectory.xtc \
  -n test_analysis \
  --compute-msm
```

## 📚 **Next Steps**

After successful installation:

1. **Read the examples**: Check `examples/` directory for tutorials
2. **Run test analysis**: Use provided example data
3. **Configure for your system**: Adjust parameters for your protein size
4. **Explore features**: Try different analysis methods
5. **Get help**: Check documentation or open an issue

## 🆘 **Getting Help**

- **GitHub Issues**: https://github.com/DoctorDean/md-compare/issues
- **Discussions**: https://github.com/DoctorDean/md-compare/discussions
- **Email**: 

---

**Happy Analyzing!** 🧬✨
