#!/usr/bin/env python3

"""
confdelta Utilities

This module contains utility functions and helper classes for the confdelta toolkit.
"""

import signal
import time
import warnings
from contextlib import contextmanager

import numpy as np

try:
    from MDAnalysis.analysis import align
except ImportError:
    pass

# =====================================================
# TIMEOUT UTILITIES
# =====================================================


@contextmanager
def timeout_handler(seconds):
    """
    Context manager for timing out long-running operations
    Cross-platform compatible (Windows/Unix)

    Parameters:
    -----------
    seconds : int
        Timeout duration in seconds

    Raises:
    -------
    TimeoutError : If operation exceeds timeout (Unix only)
    """
    import platform

    if platform.system() == "Windows":
        # Windows doesn't support SIGALRM, so we skip timeout functionality
        # Operations will run to completion
        print("Note: Timeout functionality not available on Windows, running without timeout")
        yield
        return

    def timeout_signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set up signal handler (Unix/Linux only)
    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)

    try:
        yield
    except TimeoutError:
        print(f"Operation timed out after {seconds} seconds")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# =====================================================
# MD PREPROCESSING
# =====================================================


class MDPreprocessor:
    """
    Handle MD trajectory preprocessing for network analysis
    """

    def __init__(self, universe, align_selection="name CA", center_selection="protein"):
        """
        Initialize MD preprocessor

        Parameters:
        -----------
        universe : MDAnalysis.Universe
            The MDAnalysis universe
        align_selection : str
            Selection string for alignment
        center_selection : str
            Selection string for centering
        """
        self.universe = universe
        self.align_selection = align_selection
        self.center_selection = center_selection

        # Get alignment and centering groups
        self.align_group = self.universe.select_atoms(align_selection)
        self.center_group = self.universe.select_atoms(center_selection)

        if len(self.align_group) == 0:
            raise RuntimeError(f"No atoms found for alignment: {align_selection}")
        if len(self.center_group) == 0:
            raise RuntimeError(f"No atoms found for centering: {center_selection}")

        # Reference structure storage
        self.reference_coordinates = None
        self.reference_center = None

        print("MD Preprocessor initialized:")
        print(f"  Alignment atoms: {len(self.align_group)}")
        print(f"  Centering atoms: {len(self.center_group)}")

    def setup_reference(self, frame=0):
        """
        Set up reference structure for alignment

        Parameters:
        -----------
        frame : int
            Frame number to use as reference
        """
        self.universe.trajectory[frame]
        self.reference_coordinates = self.align_group.positions.copy()
        self.reference_center = self.center_group.center_of_mass()

        print(f"Reference structure set from frame {frame}")

    def center_system(self):
        """Center the system at origin"""
        current_center = self.center_group.center_of_mass()
        translation = -current_center
        self.universe.atoms.translate(translation)

    def align_to_reference(self):
        """Align current frame to reference structure"""
        if self.reference_coordinates is None:
            raise RuntimeError("Reference coordinates not set. Call setup_reference first.")

        current_positions = self.align_group.positions

        try:
            rotation_matrix, rmsd = align.rotation_matrix(
                current_positions, self.reference_coordinates
            )
            self.universe.atoms.rotate(rotation_matrix)
        except Exception as e:
            warnings.warn(f"Alignment failed: {e}", stacklevel=2)


# =====================================================
# NETWORK ANALYSIS UTILITIES
# =====================================================


def calculate_network_robustness(network):
    """
    Calculate network robustness metrics

    Parameters:
    -----------
    network : networkx.Graph
        The network to analyze

    Returns:
    --------
    Dict : Robustness metrics
    """
    import networkx as nx

    if network.number_of_nodes() == 0:
        return {"error": "Empty network"}

    robustness = {}

    # Basic connectivity
    robustness["is_connected"] = nx.is_connected(network)
    robustness["n_components"] = nx.number_connected_components(network)

    if nx.is_connected(network):
        # Node connectivity
        robustness["node_connectivity"] = nx.node_connectivity(network)
        robustness["edge_connectivity"] = nx.edge_connectivity(network)
    else:
        robustness["node_connectivity"] = 0
        robustness["edge_connectivity"] = 0

    # Largest component size
    if network.number_of_nodes() > 0:
        components = list(nx.connected_components(network))
        largest_component_size = max(len(c) for c in components) if components else 0
        robustness["largest_component_fraction"] = (
            largest_component_size / network.number_of_nodes()
        )
    else:
        robustness["largest_component_fraction"] = 0

    return robustness


def analyze_network_assortativity(network):
    """
    Analyze network assortativity (degree correlation)

    Parameters:
    -----------
    network : networkx.Graph
        The network to analyze

    Returns:
    --------
    Dict : Assortativity metrics
    """
    import networkx as nx

    assortativity = {}

    try:
        # Degree assortativity
        assortativity["degree_assortativity"] = nx.degree_assortativity_coefficient(network)
    except Exception:
        assortativity["degree_assortativity"] = None

    # If nodes have chain information
    if network.number_of_nodes() > 0:
        node = list(network.nodes())[0]
        if "chain" in network.nodes[node]:
            try:
                # Chain assortativity (tendency for same-chain residues to connect)
                assortativity["chain_assortativity"] = nx.attribute_assortativity_coefficient(
                    network, "chain"
                )
            except Exception:
                assortativity["chain_assortativity"] = None

    return assortativity


# =====================================================
# CONTACT ANALYSIS UTILITIES
# =====================================================


def analyze_contact_persistence(contact_matrix, percentiles=(25, 50, 75, 90, 95)):
    """
    Analyze the persistence distribution of contacts

    Parameters:
    -----------
    contact_matrix : np.ndarray
        Contact frequency matrix
    percentiles : List[int]
        Percentiles to calculate

    Returns:
    --------
    Dict : Contact persistence statistics
    """
    # Get upper triangle (avoid double counting)
    triu_indices = np.triu_indices_from(contact_matrix, k=1)
    contact_frequencies = contact_matrix[triu_indices]

    # Remove zero contacts
    nonzero_contacts = contact_frequencies[contact_frequencies > 0]

    if len(nonzero_contacts) == 0:
        return {"error": "No contacts found"}

    stats = {
        "total_possible_contacts": len(contact_frequencies),
        "actual_contacts": len(nonzero_contacts),
        "contact_fraction": len(nonzero_contacts) / len(contact_frequencies),
        "mean_frequency": np.mean(nonzero_contacts),
        "std_frequency": np.std(nonzero_contacts),
        "min_frequency": np.min(nonzero_contacts),
        "max_frequency": np.max(nonzero_contacts),
    }

    # Calculate percentiles
    for p in percentiles:
        stats[f"percentile_{p}"] = np.percentile(nonzero_contacts, p)

    return stats


def find_highly_persistent_contacts(contact_matrix, residue_keys, threshold=0.8):
    """
    Find highly persistent contacts in the network

    Parameters:
    -----------
    contact_matrix : np.ndarray
        Contact frequency matrix
    residue_keys : List[str]
        Residue identifiers
    threshold : float
        Minimum frequency for "highly persistent" contacts

    Returns:
    --------
    List[Dict] : List of highly persistent contact pairs
    """
    persistent_contacts = []

    n_residues = len(residue_keys)

    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            frequency = contact_matrix[i, j]

            if frequency >= threshold:
                persistent_contacts.append(
                    {
                        "residue_1": residue_keys[i],
                        "residue_2": residue_keys[j],
                        "frequency": frequency,
                        "chain_1": residue_keys[i].split("_")[0],
                        "chain_2": residue_keys[j].split("_")[0],
                        "is_interchain": residue_keys[i].split("_")[0]
                        != residue_keys[j].split("_")[0],
                    }
                )

    # Sort by frequency
    persistent_contacts.sort(key=lambda x: x["frequency"], reverse=True)

    return persistent_contacts


# =====================================================
# RESIDUE ANALYSIS UTILITIES
# =====================================================


def analyze_residue_properties(universe, residue_selection="protein"):
    """
    Analyze properties of residues in the system

    Parameters:
    -----------
    universe : MDAnalysis.Universe
        The universe to analyze
    residue_selection : str
        Selection for residues to analyze

    Returns:
    --------
    Dict : Residue property analysis
    """
    residues = universe.select_atoms(residue_selection).residues

    properties = {
        "total_residues": len(residues),
        "residue_types": {},
        "chain_distribution": {},
        "hydrophobic_residues": 0,
        "charged_residues": 0,
        "polar_residues": 0,
    }

    # Residue classification
    hydrophobic = ["ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"]
    charged = ["ARG", "LYS", "ASP", "GLU", "HIS"]
    polar = ["SER", "THR", "ASN", "GLN", "TYR", "CYS"]

    for residue in residues:
        # Count by type
        resname = residue.resname
        properties["residue_types"][resname] = properties["residue_types"].get(resname, 0) + 1

        # Count by chain (segid fallback when chainID is unavailable)
        first_atom = residue.atoms[0]
        chainid = ""
        for attr in ("chainID", "segid"):
            try:
                value = getattr(first_atom, attr, "")
                if value and str(value).strip():
                    chainid = str(value).strip()
                    break
            except Exception:
                continue
        if not chainid:
            chainid = "A"
        properties["chain_distribution"][chainid] = (
            properties["chain_distribution"].get(chainid, 0) + 1
        )

        # Count by chemical property
        if resname in hydrophobic:
            properties["hydrophobic_residues"] += 1
        elif resname in charged:
            properties["charged_residues"] += 1
        elif resname in polar:
            properties["polar_residues"] += 1

    return properties


def calculate_sequence_distance_matrix(residue_keys):
    """
    Calculate sequence distance matrix for residues

    Parameters:
    -----------
    residue_keys : List[str]
        List of residue keys in format "chain_resid"

    Returns:
    --------
    np.ndarray : Sequence distance matrix
    """
    n_residues = len(residue_keys)
    distance_matrix = np.zeros((n_residues, n_residues))

    for i in range(n_residues):
        for j in range(n_residues):
            chain_i, resid_i = residue_keys[i].split("_")
            chain_j, resid_j = residue_keys[j].split("_")

            if chain_i == chain_j:
                # Same chain - calculate sequence distance
                distance_matrix[i, j] = abs(int(resid_i) - int(resid_j))
            else:
                # Different chains - set to a large value or special marker
                distance_matrix[i, j] = 9999  # or np.inf

    return distance_matrix


# =====================================================
# COMPARISON UTILITIES
# =====================================================


def calculate_network_similarity(network1, network2):
    """
    Calculate similarity between two networks

    Parameters:
    -----------
    network1, network2 : networkx.Graph
        Networks to compare

    Returns:
    --------
    Dict : Similarity metrics
    """
    import networkx as nx

    similarity = {}

    # Find common nodes
    nodes1 = set(network1.nodes())
    nodes2 = set(network2.nodes())
    common_nodes = nodes1.intersection(nodes2)

    if len(common_nodes) == 0:
        return {"error": "No common nodes between networks"}

    similarity["common_nodes"] = len(common_nodes)
    similarity["jaccard_nodes"] = len(common_nodes) / len(nodes1.union(nodes2))

    # Extract subgraphs with common nodes
    sub1 = network1.subgraph(common_nodes)
    sub2 = network2.subgraph(common_nodes)

    # Edge overlap
    edges1 = set(sub1.edges())
    edges2 = set(sub2.edges())
    common_edges = edges1.intersection(edges2)

    similarity["common_edges"] = len(common_edges)
    similarity["jaccard_edges"] = (
        len(common_edges) / len(edges1.union(edges2)) if len(edges1.union(edges2)) > 0 else 0
    )

    # Structural similarity
    if len(common_edges) > 0:
        try:
            # Graph edit distance (computationally expensive, so limit to small graphs)
            if len(common_nodes) <= 100:
                similarity["graph_edit_distance"] = nx.graph_edit_distance(sub1, sub2, timeout=30)
            else:
                similarity["graph_edit_distance"] = None
        except Exception:
            similarity["graph_edit_distance"] = None

    return similarity


def compare_centrality_distributions(centrality1, centrality2, metric="ks_test"):
    """
    Compare centrality distributions between two networks

    Parameters:
    -----------
    centrality1, centrality2 : Dict
        Centrality dictionaries from NetworkX
    metric : str
        Statistical test to use ('ks_test', 'mannwhitney', 'correlation')

    Returns:
    --------
    Dict : Statistical comparison results
    """
    try:
        from scipy import stats
    except ImportError:
        return {"error": "SciPy required for statistical tests"}

    # Find common nodes
    common_nodes = set(centrality1.keys()).intersection(set(centrality2.keys()))

    if len(common_nodes) < 3:
        return {"error": "Not enough common nodes for statistical comparison"}

    # Extract values for common nodes
    values1 = [centrality1[node] for node in common_nodes]
    values2 = [centrality2[node] for node in common_nodes]

    results = {"common_nodes": len(common_nodes)}

    if metric == "ks_test":
        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(values1, values2)
        results["ks_statistic"] = statistic
        results["ks_p_value"] = p_value

    elif metric == "mannwhitney":
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(values1, values2, alternative="two-sided")
        results["mw_statistic"] = statistic
        results["mw_p_value"] = p_value

    elif metric == "correlation":
        # Pearson correlation
        correlation, p_value = stats.pearsonr(values1, values2)
        results["correlation"] = correlation
        results["correlation_p_value"] = p_value

        # Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(values1, values2)
        results["spearman_correlation"] = spearman_corr
        results["spearman_p_value"] = spearman_p

    return results


# =====================================================
# VISUALIZATION UTILITIES
# =====================================================


def create_contact_map_figure(
    contact_matrix, residue_keys, title="Contact Map", chain_boundaries=None, figsize=(10, 8)
):
    """
    Create a publication-ready contact map figure

    Parameters:
    -----------
    contact_matrix : np.ndarray
        Contact frequency matrix
    residue_keys : List[str]
        Residue identifiers
    title : str
        Figure title
    chain_boundaries : List[int], optional
        Positions of chain boundaries
    figsize : Tuple[int, int]
        Figure size

    Returns:
    --------
    matplotlib.Figure : The created figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Matplotlib required for visualization") from exc

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(contact_matrix, origin="lower", cmap="viridis", aspect="equal")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Contact Frequency", fontsize=12)

    # Add chain boundaries if provided
    if chain_boundaries:
        for boundary in chain_boundaries:
            ax.axhline(y=boundary - 0.5, color="white", linestyle="--", alpha=0.7, linewidth=2)
            ax.axvline(x=boundary - 0.5, color="white", linestyle="--", alpha=0.7, linewidth=2)

    # Labels and title
    ax.set_xlabel("Residue Index", fontsize=12)
    ax.set_ylabel("Residue Index", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Set ticks to show every nth residue
    n_residues = len(residue_keys)
    if n_residues > 50:
        tick_spacing = max(1, n_residues // 20)  # ~20 ticks maximum
        tick_positions = range(0, n_residues, tick_spacing)
        tick_labels = [residue_keys[i] for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=8)

    plt.tight_layout()
    return fig


def create_centrality_comparison_plot(
    centrality_data, centrality_type="betweenness", simulation_names=None, figsize=(12, 6)
):
    """
    Create comparison plot of centrality measures across simulations

    Parameters:
    -----------
    centrality_data : Dict
        Centrality data from comparison analysis
    centrality_type : str
        Type of centrality to plot
    simulation_names : List[str], optional
        Names of simulations to include
    figsize : Tuple[int, int]
        Figure size

    Returns:
    --------
    matplotlib.Figure : The created figure
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Matplotlib and pandas required for visualization") from exc

    if centrality_type not in centrality_data:
        raise ValueError(f"Centrality type '{centrality_type}' not found in data")

    data = centrality_data[centrality_type]

    # Convert to DataFrame for easier plotting
    plot_data = []
    for residue, sim_values in data.items():
        for sim_name, value in sim_values.items():
            if simulation_names is None or sim_name in simulation_names:
                plot_data.append({"residue": residue, "simulation": sim_name, "centrality": value})

    if not plot_data:
        raise ValueError("No data to plot")

    df = pd.DataFrame(plot_data)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Box plot comparing distributions
    df.boxplot(column="centrality", by="simulation", ax=ax1)
    ax1.set_title(f'{centrality_type.replace("_", " ").title()} Distribution')
    ax1.set_ylabel("Centrality Value")

    # Scatter plot showing residue-wise comparison (if 2 simulations)
    unique_sims = df["simulation"].unique()
    if len(unique_sims) == 2:
        sim1_data = df[df["simulation"] == unique_sims[0]].set_index("residue")["centrality"]
        sim2_data = df[df["simulation"] == unique_sims[1]].set_index("residue")["centrality"]

        # Align data for common residues
        common_residues = sim1_data.index.intersection(sim2_data.index)
        if len(common_residues) > 0:
            x_vals = sim1_data.loc[common_residues].values
            y_vals = sim2_data.loc[common_residues].values

            ax2.scatter(x_vals, y_vals, alpha=0.6)
            ax2.plot(
                [0, max(x_vals.max(), y_vals.max())],
                [0, max(x_vals.max(), y_vals.max())],
                "r--",
                alpha=0.8,
                label="y=x",
            )
            ax2.set_xlabel(f"{unique_sims[0]} Centrality")
            ax2.set_ylabel(f"{unique_sims[1]} Centrality")
            ax2.set_title("Residue-wise Centrality Comparison")
            ax2.legend()
    else:
        ax2.axis("off")
        ax2.text(
            0.5,
            0.5,
            "Scatter plot available\nfor 2 simulations only",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    plt.tight_layout()
    return fig


# =====================================================
# FILE I/O UTILITIES
# =====================================================


def save_analysis_config(config, filepath):
    """
    Save analysis configuration to file

    Parameters:
    -----------
    config : AnalysisConfig
        Configuration to save
    filepath : str
        Path to save configuration
    """
    import json

    config_dict = {
        "cutoffs": config.cutoffs,
        "interaction_types": config.interaction_types,
        "threshold": config.threshold,
        "timeout_seconds": config.timeout_seconds,
        "segments": config.segments,
        "preprocess": config.preprocess,
        "align_selection": config.align_selection,
        "center_selection": config.center_selection,
    }

    with open(filepath, "w") as f:
        json.dump(config_dict, f, indent=2)


def load_analysis_config(filepath):
    """
    Load analysis configuration from file

    Parameters:
    -----------
    filepath : str
        Path to configuration file

    Returns:
    --------
    AnalysisConfig : Loaded configuration
    """
    import json

    from .core import AnalysisConfig

    with open(filepath) as f:
        config_dict = json.load(f)

    return AnalysisConfig(**config_dict)


def export_network_for_cytoscape(network, output_path, include_positions=True):
    """
    Export network in format suitable for Cytoscape

    Parameters:
    -----------
    network : networkx.Graph
        Network to export
    output_path : str
        Path for output file
    include_positions : bool
        Whether to include layout positions
    """
    import networkx as nx

    # Add layout positions if requested
    if include_positions and network.number_of_nodes() > 0:
        try:
            pos = nx.spring_layout(network, k=1, iterations=50)
            for node in network.nodes():
                network.nodes[node]["x"] = pos[node][0]
                network.nodes[node]["y"] = pos[node][1]
        except Exception:
            print("Warning: Could not generate layout positions")

    # Export as GraphML (Cytoscape-compatible)
    nx.write_graphml(network, output_path)
    print(f"Network exported for Cytoscape: {output_path}")


# =====================================================
# PERFORMANCE UTILITIES
# =====================================================


class PerformanceMonitor:
    """Monitor and report performance of analysis steps"""

    def __init__(self):
        self.timings = {}
        self.current_step = None
        self.start_time = None

    def start_step(self, step_name):
        """Start timing a step"""
        if self.current_step is not None:
            self.end_step()

        self.current_step = step_name
        self.start_time = time.time()

    def end_step(self):
        """End timing the current step"""
        if self.current_step is not None and self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.timings[self.current_step] = elapsed
            print(f"{self.current_step}: {elapsed:.2f} seconds")

            self.current_step = None
            self.start_time = None

    def get_summary(self):
        """Get performance summary"""
        if self.current_step:
            self.end_step()

        total_time = sum(self.timings.values())

        summary = {
            "total_time": total_time,
            "step_timings": self.timings.copy(),
            "step_percentages": {
                step: (time_val / total_time * 100) if total_time > 0 else 0
                for step, time_val in self.timings.items()
            },
        }

        return summary

    def print_summary(self):
        """Print performance summary"""
        summary = self.get_summary()

        print("\nPerformance Summary:")
        print("=" * 40)
        print(f"Total time: {summary['total_time']:.2f} seconds")
        print("\nStep breakdown:")

        for step, percentage in summary["step_percentages"].items():
            time_val = summary["step_timings"][step]
            print(f"  {step}: {time_val:.2f}s ({percentage:.1f}%)")
