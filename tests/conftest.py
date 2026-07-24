"""Shared pytest fixtures for the confdelta test suite.

These fixtures build small *synthetic* MD systems on the fly so the test
suite has no dependency on external data packages (e.g. MDAnalysisTests)
and runs fast enough for CI.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Synthetic trajectory generation
# ---------------------------------------------------------------------------


def _build_universe(
    n_residues: int = 12, n_frames: int = 25, n_chains: int = 2, with_chainids: bool = True
):
    """Build a tiny in-memory MDAnalysis Universe of CA-only "residues".

    Each residue is a single CA atom. Coordinates are a smooth base
    conformation plus small per-frame Gaussian noise, which is enough to
    exercise contact-map, DCCM and PCA code paths deterministically.
    """
    import MDAnalysis as mda

    n_atoms = n_residues
    rng = np.random.default_rng(42)

    # Base conformation: residues strung along x with mild y/z wiggle so
    # sequential residues are within contact distance of each other.
    base = np.zeros((n_atoms, 3), dtype=np.float32)
    base[:, 0] = np.arange(n_atoms) * 3.6
    base[:, 1] = np.sin(np.arange(n_atoms) * 0.6) * 2.0
    base[:, 2] = np.cos(np.arange(n_atoms) * 0.6) * 2.0

    # Per-frame coordinates with small thermal noise.
    coords = np.empty((n_frames, n_atoms, 3), dtype=np.float32)
    for f in range(n_frames):
        coords[f] = base + rng.normal(0.0, 0.4, size=(n_atoms, 3)).astype(np.float32)

    universe = mda.Universe.empty(
        n_atoms,
        n_residues=n_residues,
        atom_resindex=np.arange(n_atoms),
        trajectory=True,
    )
    universe.add_TopologyAttr("name", ["CA"] * n_atoms)
    universe.add_TopologyAttr("type", ["C"] * n_atoms)
    universe.add_TopologyAttr("resname", ["ALA"] * n_residues)
    universe.add_TopologyAttr("resid", list(range(1, n_residues + 1)))
    universe.add_TopologyAttr("mass", [12.0] * n_atoms)

    # Split atoms across chains.
    per_chain = n_atoms // n_chains
    if with_chainids:
        chain_letters = []
        for i in range(n_atoms):
            chain_letters.append(chr(ord("A") + min(i // per_chain, n_chains - 1)))
        universe.add_TopologyAttr("chainID", chain_letters)
    else:
        # Only a segid is provided — exercises the chainID fallback.
        universe.add_TopologyAttr("segid", ["PROT"])

    universe.load_new(coords, order="fac")
    return universe


@pytest.fixture
def synthetic_universe():
    """A 2-chain, 12-residue, 25-frame Universe with explicit chainIDs."""
    return _build_universe(with_chainids=True)


@pytest.fixture
def chainless_universe():
    """A single-segment Universe with no chainID attribute (fallback test)."""
    return _build_universe(n_chains=1, with_chainids=False)


@pytest.fixture
def synthetic_simulation(synthetic_universe, monkeypatch):
    """A loaded MDSimulation wrapping the synthetic Universe."""
    from confdelta.core import MDSimulation, SimulationConfig

    config = SimulationConfig(
        name="synthetic",
        topology="<memory>",
        trajectory="<memory>",
        selection="all",
    )
    sim = MDSimulation(config)
    # Inject the in-memory universe, bypassing file IO.
    sim._universe = synthetic_universe
    sim._atoms = synthetic_universe.select_atoms("all")
    sim._setup_residue_mapping()
    return sim


def _write_ca_pdb(path, coords, chain="A"):
    """Write a minimal CA-only PDB. *coords* is (n_atoms, 3)."""
    lines = []
    for i, (x, y, z) in enumerate(coords, start=1):
        lines.append(
            f"ATOM  {i:>5}  CA  ALA {chain}{i:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def _base_ca_coords(n_atoms=8):
    base = np.zeros((n_atoms, 3), dtype=np.float32)
    base[:, 0] = np.arange(n_atoms) * 3.6
    base[:, 1] = np.sin(np.arange(n_atoms) * 0.6) * 2.0
    base[:, 2] = np.cos(np.arange(n_atoms) * 0.6) * 2.0
    return base


@pytest.fixture
def ca_topology_pdb(tmp_path):
    """A single-structure CA-only PDB usable as a topology template."""
    return _write_ca_pdb(tmp_path / "topology.pdb", _base_ca_coords())


@pytest.fixture
def multi_model_pdb(tmp_path):
    """A 3-model CA-only PDB (NMR / prediction style)."""
    rng = np.random.default_rng(7)
    base = _base_ca_coords()
    blocks = []
    for model in range(1, 4):
        frame = base + rng.normal(0, 0.3, size=base.shape).astype(np.float32)
        rows = [f"MODEL     {model:>4}"]
        for i, (x, y, z) in enumerate(frame, start=1):
            rows.append(
                f"ATOM  {i:>5}  CA  ALA A{i:>4}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            )
        rows.append("ENDMDL")
        blocks.append("\n".join(rows))
    path = tmp_path / "models.pdb"
    path.write_text("\n".join(blocks) + "\nEND\n")
    return path


@pytest.fixture
def structure_files(tmp_path):
    """Five separate single-structure PDBs sharing one atom count."""
    rng = np.random.default_rng(11)
    base = _base_ca_coords()
    paths = []
    for i in range(5):
        frame = base + rng.normal(0, 0.3, size=base.shape).astype(np.float32)
        paths.append(_write_ca_pdb(tmp_path / f"pred_{i}.pdb", frame))
    return paths


@pytest.fixture
def small_graph():
    """A small deterministic NetworkX graph for network-utility tests."""
    import networkx as nx

    g = nx.Graph()
    edges = [
        ("A_1", "A_2"),
        ("A_2", "A_3"),
        ("A_3", "A_4"),
        ("A_4", "A_5"),
        ("A_1", "A_3"),
        ("A_2", "A_5"),
        ("A_5", "B_1"),
        ("B_1", "B_2"),
        ("B_2", "B_3"),
        ("B_3", "B_4"),
        ("B_1", "B_3"),
    ]
    g.add_edges_from(edges)
    return g
