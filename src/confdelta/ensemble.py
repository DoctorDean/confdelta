"""Source-agnostic conformational ensembles.

An :class:`Ensemble` is a set of conformations of one molecular system. It is
the input type the analysis and comparison layers are built to accept, and it
is deliberately decoupled from any single file format.

Why this exists
---------------
The original code accepted only a topology file plus a trajectory file. That
excludes the ensembles confdelta most wants to compare in the long run:
NMR-style multi-model PDBs, AlphaFold-subsampled predictions, and structures
drawn from a generative model, none of which is a trajectory. :class:`Ensemble`
accepts all of them through four constructors, and normalises each to a single
in-memory representation (an MDAnalysis ``Universe`` carrying one or more
frames) so the rest of the package sees one thing regardless of where the
conformations came from.

    Ensemble.from_trajectory(topology, trajectory)   # MD
    Ensemble.from_pdb_models(path)                    # multi-model PDB
    Ensemble.from_structures([p1, p2, ...])           # predicted structures
    Ensemble.from_coordinates(array, topology)        # generative / in-memory

Replicates
----------
A single :class:`Ensemble` is one ensemble. An experimental *condition* may be
represented by several replicate ensembles; that is an :class:`EnsembleGroup`,
which is the type the two-condition comparison API accepts. Keeping the two
concepts distinct is what lets the statistical layer use the replicate -- not
the frame -- as the unit of inference when replicates are supplied. See
``DECISIONS.md`` D-005.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader

__all__ = ["Ensemble", "EnsembleGroup", "EnsembleError"]

# The default atom selection. Heavy atoms of the protein: excludes hydrogens
# (rarely present or reliable across ensemble sources) and solvent/ions.
DEFAULT_SELECTION = "protein and not name H*"


class EnsembleError(ValueError):
    """Raised when an ensemble cannot be constructed from the given input.

    The message is intended to be shown to the user, so it names the offending
    input and what was wrong with it.
    """


def _as_universe_from_topology(topology: str | Path | mda.Universe) -> mda.Universe:
    """Return a fresh single-frame Universe to serve as a coordinate template.

    Accepts a path or an existing Universe. When given a Universe, its atoms
    are merged into a new Universe so the caller's object is never mutated by a
    later ``load_new``.
    """
    if isinstance(topology, mda.Universe):
        return mda.Merge(topology.atoms)
    path = Path(topology)
    if not path.is_file():
        raise EnsembleError(f"Topology file not found: {path}")
    try:
        return mda.Universe(str(path))
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed, user-facing error
        raise EnsembleError(f"Could not read topology {path}: {exc}") from exc


class Ensemble:
    """A set of conformations of one molecular system.

    Construct with one of the ``from_*`` classmethods rather than calling the
    initializer directly; the initializer takes an already-loaded Universe and
    exists mainly for the classmethods and for advanced callers who have built
    a Universe themselves.

    Parameters
    ----------
    universe
        An MDAnalysis ``Universe`` carrying one or more frames.
    selection
        Atom selection applied to every analysis. Defaults to protein heavy
        atoms.
    name
        A short label used in output and messages.
    source
        Free-form description of where the conformations came from, for
        provenance (e.g. ``"trajectory: run1.xtc"``).
    metadata
        Arbitrary user metadata carried alongside the ensemble.
    """

    def __init__(
        self,
        universe: mda.Universe,
        *,
        selection: str = DEFAULT_SELECTION,
        name: str = "ensemble",
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.universe = universe
        self.selection = selection
        self.name = name
        self.source = source
        self.metadata: dict[str, Any] = dict(metadata or {})

        self._atoms = universe.select_atoms(selection)
        if len(self._atoms) == 0:
            raise EnsembleError(
                f"Selection {selection!r} matched no atoms in ensemble {name!r}. "
                f"The system has {len(universe.atoms)} atoms; check the selection "
                f"or pass a different one."
            )

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_trajectory(
        cls,
        topology: str | Path,
        trajectory: str | Path | Sequence[str | Path] | None = None,
        *,
        selection: str = DEFAULT_SELECTION,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Ensemble:
        """Build an ensemble from an MD topology and trajectory.

        Parameters
        ----------
        topology
            Topology file (PDB, PSF, GRO, ...).
        trajectory
            Trajectory file(s) (XTC, DCD, TRR, ...). If omitted, the topology's
            own coordinates are used as a single-frame ensemble.
        selection, name, metadata
            See the class docstring.

        Notes
        -----
        A list of trajectory files is read by MDAnalysis as one continuous
        trajectory. That is correct for split runs of a *single* simulation but
        is **not** how to supply replicates -- concatenating independent runs
        misrepresents them as one continuous trajectory. Use an
        :class:`EnsembleGroup` for replicates.
        """
        topology = Path(topology)
        if not topology.is_file():
            raise EnsembleError(f"Topology file not found: {topology}")

        try:
            if trajectory is None:
                universe = mda.Universe(str(topology))
                source = f"topology: {topology.name}"
            elif isinstance(trajectory, (str, Path)):
                traj_path = Path(trajectory)
                if not traj_path.is_file():
                    raise EnsembleError(f"Trajectory file not found: {traj_path}")
                universe = mda.Universe(str(topology), str(traj_path))
                source = f"trajectory: {traj_path.name}"
            else:
                traj_paths = [Path(t) for t in trajectory]
                missing = [str(t) for t in traj_paths if not t.is_file()]
                if missing:
                    raise EnsembleError(f"Trajectory file(s) not found: {', '.join(missing)}")
                universe = mda.Universe(str(topology), [str(t) for t in traj_paths])
                source = f"trajectory: {', '.join(t.name for t in traj_paths)}"
        except EnsembleError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EnsembleError(f"Could not load trajectory for {topology.name}: {exc}") from exc

        return cls(
            universe,
            selection=selection,
            name=name or topology.stem,
            source=source,
            metadata=metadata,
        )

    @classmethod
    def from_pdb_models(
        cls,
        path: str | Path,
        *,
        selection: str = DEFAULT_SELECTION,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Ensemble:
        """Build an ensemble from a multi-model PDB file.

        Each ``MODEL`` record becomes one conformation. This is the natural
        input for NMR ensembles and for AlphaFold-style predictions saved as a
        multi-model PDB.
        """
        path = Path(path)
        if not path.is_file():
            raise EnsembleError(f"PDB file not found: {path}")
        try:
            universe = mda.Universe(str(path))
        except Exception as exc:  # noqa: BLE001
            raise EnsembleError(f"Could not read PDB {path}: {exc}") from exc

        if len(universe.trajectory) < 2:
            # Not an error -- a single-model PDB is a one-conformation ensemble
            # -- but worth flagging, since the user asked for *models*.
            universe.trajectory[0]

        return cls(
            universe,
            selection=selection,
            name=name or path.stem,
            source=f"multi-model PDB: {path.name} ({len(universe.trajectory)} models)",
            metadata=metadata,
        )

    @classmethod
    def from_structures(
        cls,
        paths: Sequence[str | Path],
        *,
        selection: str = DEFAULT_SELECTION,
        name: str = "ensemble",
        metadata: dict[str, Any] | None = None,
    ) -> Ensemble:
        """Build an ensemble by stacking separate single-structure files.

        Each file contributes one conformation. The intended use is a set of
        predicted structures (for example AlphaFold subsampling output, or a
        batch from a generative model) saved as individual PDBs.

        All structures must share the same atom count; the first file provides
        the topology. A mismatch is an error rather than a silent truncation,
        because analysis on inconsistently-sized structures is meaningless.
        """
        resolved = [Path(p) for p in paths]
        if not resolved:
            raise EnsembleError("from_structures requires at least one structure file.")
        missing = [str(p) for p in resolved if not p.is_file()]
        if missing:
            raise EnsembleError(f"Structure file(s) not found: {', '.join(missing)}")

        template = _as_universe_from_topology(resolved[0])
        n_atoms = len(template.atoms)

        frames = np.empty((len(resolved), n_atoms, 3), dtype=np.float32)
        for i, structure_path in enumerate(resolved):
            try:
                structure = mda.Universe(str(structure_path))
            except Exception as exc:  # noqa: BLE001
                raise EnsembleError(f"Could not read structure {structure_path}: {exc}") from exc
            if len(structure.atoms) != n_atoms:
                raise EnsembleError(
                    f"Structure {structure_path.name} has {len(structure.atoms)} atoms but "
                    f"{resolved[0].name} has {n_atoms}. All structures in an ensemble "
                    f"must have the same atoms."
                )
            frames[i] = structure.atoms.positions

        template.load_new(frames, format=MemoryReader)
        return cls(
            template,
            selection=selection,
            name=name,
            source=f"{len(resolved)} structures: {resolved[0].name} ...",
            metadata=metadata,
        )

    @classmethod
    def from_coordinates(
        cls,
        coordinates: np.ndarray,
        topology: str | Path | mda.Universe,
        *,
        selection: str = DEFAULT_SELECTION,
        name: str = "ensemble",
        metadata: dict[str, Any] | None = None,
    ) -> Ensemble:
        """Build an ensemble from an in-memory coordinate array.

        Parameters
        ----------
        coordinates
            Array of shape ``(n_frames, n_atoms, 3)`` in angstroms.
        topology
            A topology that names the atoms the coordinates belong to: a PDB (or
            other) file path, or an existing MDAnalysis ``Universe``. A topology
            is genuinely required -- residue-level network analysis cannot be
            done on an anonymous point cloud -- but note this is a *structure*,
            not a trajectory, so the ensemble is still trajectory-format-free.

        This is the path for ensembles produced by a generative model or any
        other code that yields coordinates directly.
        """
        coordinates = np.asarray(coordinates, dtype=np.float32)
        if coordinates.ndim == 2:
            # A single conformation given as (n_atoms, 3); promote to one frame.
            coordinates = coordinates[np.newaxis, :, :]
        if coordinates.ndim != 3 or coordinates.shape[2] != 3:
            raise EnsembleError(
                "coordinates must have shape (n_frames, n_atoms, 3) or (n_atoms, 3); "
                f"got {coordinates.shape}."
            )

        template = _as_universe_from_topology(topology)
        n_atoms = len(template.atoms)
        if coordinates.shape[1] != n_atoms:
            raise EnsembleError(
                f"coordinates describe {coordinates.shape[1]} atoms but the topology "
                f"has {n_atoms}. They must match."
            )

        template.load_new(coordinates, format=MemoryReader)
        return cls(
            template,
            selection=selection,
            name=name,
            source=f"in-memory coordinates ({coordinates.shape[0]} frames)",
            metadata=metadata,
        )

    # -- properties ---------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Number of conformations in the ensemble."""
        return len(self.universe.trajectory)

    @property
    def n_atoms(self) -> int:
        """Number of atoms matched by the selection."""
        return len(self._atoms)

    @property
    def atoms(self) -> mda.AtomGroup:
        """The selected atoms."""
        return self._atoms

    def coordinates(self) -> np.ndarray:
        """Return selected-atom coordinates as ``(n_frames, n_atoms, 3)``.

        Materialises every frame into a single array. For very long
        trajectories prefer iterating ``ensemble.universe.trajectory`` and
        reading ``ensemble.atoms.positions`` per frame.
        """
        out = np.empty((self.n_frames, self.n_atoms, 3), dtype=np.float32)
        for i, _ in enumerate(self.universe.trajectory):
            out[i] = self._atoms.positions
        return out

    def summary(self) -> dict[str, Any]:
        """A small, JSON-friendly description of the ensemble."""
        return {
            "name": self.name,
            "source": self.source,
            "n_frames": self.n_frames,
            "n_atoms": self.n_atoms,
            "selection": self.selection,
            "metadata": dict(self.metadata),
        }

    # -- bridge to the existing analysis pipeline ---------------------------

    def to_simulation(self) -> Any:
        """Return an ``MDSimulation`` wrapping this ensemble.

        This is the bridge that lets the current analysis pipeline consume an
        :class:`Ensemble` unchanged, while ``MDSimulation`` is retired
        incrementally. Imported lazily to keep ``ensemble`` importable without
        pulling in the whole of ``core``.
        """
        from .core import MDSimulation, SimulationConfig

        config = SimulationConfig(
            name=self.name,
            topology=self.source or "<in-memory>",
            trajectory=self.source or "<in-memory>",
            selection=self.selection,
            description=self.source or "",
            metadata=dict(self.metadata),
        )
        simulation = MDSimulation(config)
        # Inject the already-loaded universe, bypassing file IO. This is the
        # same path the test fixtures use and is exercised end to end.
        simulation._universe = self.universe
        simulation._atoms = self._atoms
        simulation._setup_residue_mapping()
        return simulation

    def __repr__(self) -> str:
        return (
            f"Ensemble(name={self.name!r}, n_frames={self.n_frames}, "
            f"n_atoms={self.n_atoms}, selection={self.selection!r})"
        )


class EnsembleGroup:
    """One experimental condition, as one or more replicate ensembles.

    This is the unit the two-condition comparison API accepts. When it holds
    several ensembles, the statistical layer treats the replicate as the unit
    of inference; when it holds one, comparison is descriptive or falls back to
    a within-trajectory resampling scheme. See ``DECISIONS.md`` D-005.

    Every replicate must describe the same system: the selections must match in
    atom count, or per-residue comparison across replicates is undefined.
    """

    def __init__(self, ensembles: Sequence[Ensemble], *, label: str = "condition") -> None:
        ensembles = list(ensembles)
        if not ensembles:
            raise EnsembleError("An EnsembleGroup needs at least one ensemble.")

        atom_counts = {e.n_atoms for e in ensembles}
        if len(atom_counts) > 1:
            detail = ", ".join(f"{e.name}={e.n_atoms}" for e in ensembles)
            raise EnsembleError(
                f"Replicates in condition {label!r} have different selected atom counts "
                f"({detail}). Every replicate must describe the same system with the "
                f"same selection."
            )

        self.ensembles = ensembles
        self.label = label

    @classmethod
    def single(cls, ensemble: Ensemble, *, label: str | None = None) -> EnsembleGroup:
        """Wrap a single ensemble as a one-replicate condition (the common case)."""
        return cls([ensemble], label=label or ensemble.name)

    @property
    def n_replicates(self) -> int:
        return len(self.ensembles)

    @property
    def has_replicates(self) -> bool:
        """True when the condition has more than one replicate ensemble.

        The statistical layer branches on this: with replicates it uses the
        replicate as the unit of inference; without, it must fall back to
        within-trajectory resampling and say so.
        """
        return len(self.ensembles) > 1

    @property
    def n_atoms(self) -> int:
        """Selected atom count, shared by every replicate."""
        return self.ensembles[0].n_atoms

    def __len__(self) -> int:
        return len(self.ensembles)

    def __iter__(self) -> Iterator[Ensemble]:
        return iter(self.ensembles)

    def __getitem__(self, index: int) -> Ensemble:
        return self.ensembles[index]

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_replicates": self.n_replicates,
            "n_atoms": self.n_atoms,
            "replicates": [e.summary() for e in self.ensembles],
        }

    def __repr__(self) -> str:
        return (
            f"EnsembleGroup(label={self.label!r}, n_replicates={self.n_replicates}, "
            f"n_atoms={self.n_atoms})"
        )
