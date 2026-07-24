# Installing confdelta

confdelta is a pure-Python package with no compiled components of its own.
A plain `pip install` works on every supported platform; no conda environment
and no workarounds are required.

> **What changed at 0.1.0.** The previous installation guide was built around
> conda, PyEMMA and a `numpy<2.0` pin. PyEMMA is end-of-life and was the sole
> reason for all three. It was replaced by deeptime, its actively maintained
> successor by the same authors, so the conda path, the numpy ceiling and the
> troubleshooting section are all gone.

## Requirements

- **Python 3.10, 3.11, 3.12 or 3.13.** Python 3.8 (EOL October 2024) and 3.9
  are not supported.
- Linux, macOS, or Windows.
- Memory scales with system size and trajectory length rather than with
  confdelta itself. 4 GB is enough for small systems; long trajectories
  analysed with MSMs benefit from 16 GB or more.

## Install

```bash
pip install confdelta
```

That is the whole core install. It pulls in MDAnalysis, NetworkX, NumPy,
SciPy, pandas, scikit-learn, matplotlib, seaborn and tqdm.

### Optional extras

| Extra | Installs | Enables |
|---|---|---|
| `msm` | deeptime | Markov State Model analysis |
| `leiden` | python-igraph, leidenalg | Leiden community detection |
| `viz` | plotly, ipywidgets | Interactive visualisation |
| `ml` | joblib | The experimental resistance classifier |
| `all` | all of the above | |

```bash
pip install "confdelta[msm]"      # + Markov State Models
pip install "confdelta[all]"      # everything optional
```

Extras are genuinely optional: with none of them installed the package
imports, the CLI runs, and the test suite passes with the MSM-dependent tests
skipped rather than failed. A dedicated CI job enforces this.

### Development install

```bash
git clone https://github.com/DoctorDean/confdelta.git
cd confdelta
pip install -e ".[dev,msm]"
```

The `dev` extra adds pytest, pytest-cov, ruff, black, mypy, build and twine.
It also adds **statsmodels**, which is a *test-only* dependency: it provides
the independent cross-check for confdelta's multiple-testing correction and is
deliberately not a runtime dependency (confdelta uses
`scipy.stats.false_discovery_control` instead).

## Verify the installation

```bash
python -c "import confdelta; print(confdelta.__version__)"
confdelta --help
```

To see which optional features are available in the current environment:

```python
import confdelta
confdelta.check_dependencies(verbose=True)
```

This prints the core dependencies, the optional ones, and which capabilities
each unlocks.

## Running the tests

```bash
pip install -e ".[dev,msm]"
pytest
```

The suite builds small synthetic systems in memory, so no external data
download is needed and it runs in a few seconds.

## Getting help

- Issues: https://github.com/DoctorDean/confdelta/issues
- Changelog: [CHANGELOG.md](../CHANGELOG.md)
