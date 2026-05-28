# Contributing to MD-Compare

We welcome contributions to improve this molecular dynamics comparison toolkit! This document provides guidelines for contributing.

## 🤝 How to Contribute

### Reporting Issues

Before creating an issue, please:
1. Search existing issues to avoid duplicates
2. Use a clear, descriptive title
3. Provide system information (OS, Python version, dependency versions)
4. Include minimal reproducible examples with sample data
5. Describe expected vs. actual behavior

### Submitting Pull Requests

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the existing code style
   - Add docstrings for new functions
   - Include type hints where appropriate
   - Update documentation as needed

3. **Test your changes**:
   ```bash
   # Run existing tests
   pytest tests/
   
   # Test CLI functionality
   md-compare --help
   
   # Check code style
   flake8 your_new_file.py
   black your_new_file.py --check
   ```

4. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: brief description of what you did"
   ```

5. **Submit pull request** with:
   - Clear description of changes
   - Reference to related issues
   - Example usage if applicable
   - Performance impact assessment

## 🧪 Development Setup

### Environment Setup
```bash
# Clone your fork
git clone https://github.com/DoctorDean/md-compare.git
cd md-compare

# Create development environment (Python 3.8-3.12 supported)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install the package with all development tooling
pip install -e ".[dev]"            # pytest, ruff, black, mypy, build, twine
# Optional analysis features:
pip install -e ".[all]"            # MSM, Leiden, plotting, ML extras
```

Note: `requirements.txt` is generated from `pyproject.toml` and must not be
edited by hand. After changing dependencies in `pyproject.toml`, regenerate
it with `python scripts/generate_requirements.py` (CI fails if it is stale).

### Code Style

We follow these conventions:
- **PEP 8** for Python code style
- **Black** for code formatting
- **Type hints** for function parameters and returns
- **Docstrings** for all public functions (Google style)

Example function:
```python
def analyze_network_topology(
    contact_matrix: np.ndarray, 
    threshold: float = 0.2
) -> Dict[str, Union[float, int]]:
    """
    Analyze network topology properties from contact matrix.
    
    Args:
        contact_matrix: Residue contact frequency matrix (n_residues x n_residues)
        threshold: Minimum contact frequency for network edges
        
    Returns:
        Dictionary containing topology metrics including density, clustering, 
        and path length statistics
        
    Raises:
        ValueError: If contact_matrix is not square or has invalid dimensions
    """
    # Implementation here
    pass
```

### Testing Guidelines

- Write tests for new functionality
- Aim for >80% code coverage
- Use pytest fixtures for common setup
- Test edge cases and error conditions
- Include integration tests for CLI functionality

Example test:
```python
def test_contact_map_calculation():
    """Test basic contact map calculation functionality."""
    # Create mock simulation data
    sim_config = SimulationConfig(
        name="test_sim",
        topology="test_data/test_system.pdb",
        trajectory="test_data/test_traj.xtc"
    )
    
    simulation = MDSimulation(sim_config)
    assert simulation.load() == True
    
    # Test contact calculation
    analyzer = NetworkAnalyzer(AnalysisConfig())
    contact_maps = analyzer.compute_contact_maps(simulation)
    
    assert 'distance' in contact_maps
    assert contact_maps['distance'].shape == (simulation.n_residues, simulation.n_residues)
    assert np.allclose(contact_maps['distance'], contact_maps['distance'].T)
```

## 📖 Documentation

### Docstrings
- Use Google style docstrings
- Document all parameters, returns, and exceptions
- Include usage examples for complex functions
- Reference scientific methods where appropriate

### README Updates
- Update README.md for new features
- Add usage examples with sample commands
- Update parameter descriptions and troubleshooting

### Code Comments
- Explain WHY, not just WHAT
- Document complex algorithms and network analysis methods
- Reference scientific papers where appropriate

## 🔬 Scientific Contributions

### New Analysis Methods
When adding new network analysis methods:
1. Include scientific references in docstrings
2. Validate against known systems or literature benchmarks
3. Add appropriate error handling and input validation
4. Consider computational complexity and memory usage
5. Provide example usage and interpretation guidelines

### Performance Improvements
- Profile code before optimizing using built-in PerformanceMonitor
- Document performance improvements with benchmarks
- Consider memory usage for large protein systems
- Add timing information for new analysis steps

## 🐛 Bug Reports

Include this information:
```
**System Information:**
- OS: [e.g., Ubuntu 22.04, macOS 13.0, Windows 11]
- Python version: [e.g., 3.9.7]
- MDAnalysis version: [e.g., 2.4.0]
- NetworkX version: [e.g., 2.8.0]
- MD-Compare version: [e.g., 1.0.0]

**System Information:**
- Protein system: [e.g., 250 residues, 4000 atoms, multi-chain]
- File formats: [e.g., .pdb/.xtc, .gro/.dcd]
- Trajectory length: [e.g., 1000 frames, 200 ns]

**Command Used:**
```bash
md-compare single -t system.pdb -x traj.xtc -n test --threshold 0.2
```

**Error Message:**
```
Full error traceback here
```

**Expected Behavior:**
Clear description of what should happen

**Additional Context:**
Any other relevant information about the analysis or system
```

## 🚀 Feature Requests

Structure feature requests as:
1. **Scientific use case**: What research question does this address?
2. **Proposed solution**: How should the feature work?
3. **Alternatives**: Other approaches considered
4. **Implementation suggestions**: Ideas for technical implementation
5. **Example usage**: Mock CLI commands or code snippets

## 📋 Release Process

### Versioning
We use semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)  
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist
- [ ] Update version in setup.py
- [ ] Update CHANGELOG.md
- [ ] Test on multiple MD formats and systems
- [ ] Update documentation and examples
- [ ] Create release notes
- [ ] Tag release in Git

## 📞 Getting Help

- 💬 **Discussions**: Use GitHub Discussions for questions
- 🐛 **Bug Reports**: Create GitHub Issues
- 📧 **Direct Contact**: []
- 📚 **Documentation**: Check README.md and wiki

## 🙏 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Academic papers using the software
- Conference presentations

Thank you for helping improve molecular dynamics analysis research! 🧬

