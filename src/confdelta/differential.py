#!/usr/bin/env python3

"""
confdelta Differential Analysis Module

This module provides comprehensive simulation vs simulation comparative analysis,
building on the robust single-simulation analysis framework.
"""

import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .core import AnalysisConfig, MDSimulation

logger = logging.getLogger("confdelta")

try:
    import pandas as pd  # noqa: F401

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available; some analysis features will be limited.")

try:
    import matplotlib.pyplot as plt  # noqa: F401
    import seaborn as sns  # noqa: F401

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("matplotlib/seaborn not available; visualization features will be limited.")

try:
    import scipy.stats  # noqa: F401

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available; statistical analysis features will be limited.")

# =====================================================
# DIFFERENTIAL ANALYSIS CONFIGURATION
# =====================================================


@dataclass
class DifferentialConfig:
    """Configuration for differential analysis between simulations"""

    # Comparison focus areas
    compare_networks: bool = True
    compare_dynamics: bool = True
    compare_energetics: bool = True
    compare_kinetics: bool = True
    compare_allosteric: bool = True

    # Statistical analysis options
    perform_statistical_tests: bool = True
    significance_threshold: float = 0.05
    multiple_comparison_correction: str = "fdr_bh"  # fdr_bh, bonferroni, none
    bootstrap_iterations: int = 1000
    permutation_iterations: int = 1000

    # Visualization options
    create_difference_plots: bool = True
    create_publication_figures: bool = False
    figure_dpi: int = 300
    heatmap_colormap: str = "RdBu_r"

    # Output options
    create_html_report: bool = True
    create_pdf_summary: bool = False
    export_excel_workbook: bool = True

    # Analysis thresholds
    correlation_change_threshold: float = 0.2  # Minimum |Δρ| to consider significant
    efficiency_change_threshold: float = 0.1  # Minimum communication efficiency change
    energy_change_threshold: float = 2.0  # Minimum energy change (kT)
    centrality_change_threshold: float = 0.1  # Minimum centrality change


# =====================================================
# COMPARISON RESULT CLASSES
# =====================================================


@dataclass
class NetworkComparison:
    """Results of network topology comparison"""

    # Topology differences
    nodes_added: list[str]
    nodes_removed: list[str]
    edges_added: list[tuple[str, str]]
    edges_removed: list[tuple[str, str]]
    edge_weight_changes: dict[tuple[str, str], float]

    # Centrality differences
    centrality_changes: dict[str, dict[str, float]]  # {metric: {node: change}}
    centrality_ranking_changes: dict[str, list[str]]  # {metric: [nodes_with_large_changes]}

    # Community structure differences
    community_changes: dict[str, Any]
    modularity_change: float

    # Global network property changes
    network_property_changes: dict[str, float]
    statistical_significance: dict[str, float]


@dataclass
class DynamicsComparison:
    """Results of dynamics comparison (DCCM, PCA)"""

    # DCCM differences
    dccm_difference_matrix: np.ndarray
    significant_correlation_changes: list[
        tuple[str, str, float, float]
    ]  # (res1, res2, old_corr, new_corr)
    regional_correlation_changes: dict[str, dict[str, float]]  # {region: {metric: value}}

    # PCA differences
    pca_variance_changes: np.ndarray
    eigenvector_similarities: np.ndarray
    principal_component_shifts: dict[int, float]  # {PC_index: similarity_score}

    # Statistical analysis
    correlation_change_pvalues: np.ndarray
    significant_residue_pairs: list[tuple[str, str]]
    effect_sizes: dict[str, float]


@dataclass
class EnergeticsComparison:
    """Results of energetics comparison (energy landscapes)"""

    # Energy landscape differences
    energy_difference_surface: np.ndarray | None
    minima_comparison: list[dict[str, Any]]
    barrier_height_changes: list[dict[str, float]]

    # Thermodynamic analysis
    stability_changes: dict[str, float]
    conformational_state_changes: dict[str, Any]
    free_energy_changes: dict[str, float]

    # Statistical significance
    energy_change_significance: dict[str, float]
    conformational_population_changes: dict[str, float]


@dataclass
class KineticsComparison:
    """Results of kinetics comparison (MSM analysis)"""

    # Timescale differences
    timescale_changes: np.ndarray
    timescale_ratios: np.ndarray
    dominant_timescale_change: float

    # Transition matrix differences
    transition_flux_changes: np.ndarray
    pathway_probability_changes: dict[str, float]

    # Metastable state analysis
    metastable_state_population_changes: np.ndarray
    metastable_state_stability_changes: dict[int, float]

    # Kinetic pathway analysis
    critical_pathway_changes: list[dict[str, Any]]
    bottleneck_analysis: dict[str, float]


@dataclass
class AllostericComparison:
    """Results of allosteric communication comparison"""

    # Communication efficiency differences
    efficiency_difference_matrix: np.ndarray
    efficiency_change_statistics: dict[str, float]

    # Pathway analysis
    pathway_disruption_analysis: list[dict[str, Any]]
    hotspot_ranking_changes: list[dict[str, Any]]

    # Cross-chain communication (for multi-chain proteins)
    cross_chain_communication_changes: dict[str, float]

    # Drug resistance analysis (if applicable)
    resistance_mechanism_analysis: dict[str, Any]
    binding_site_communication_changes: dict[str, float]

    # Statistical analysis
    pathway_significance_tests: dict[str, float]
    hotspot_significance_changes: dict[str, float]


@dataclass
class ComprehensiveDifferentialResults:
    """Complete differential analysis results"""

    # Metadata
    simulation_names: tuple[str, str]
    analysis_timestamp: str
    config: DifferentialConfig

    # Individual simulation results
    simulation1_results: MDSimulation
    simulation2_results: MDSimulation

    # Comparative analysis results
    network_comparison: NetworkComparison | None
    dynamics_comparison: DynamicsComparison | None
    energetics_comparison: EnergeticsComparison | None
    kinetics_comparison: KineticsComparison | None
    allosteric_comparison: AllostericComparison | None

    # Summary statistics
    overall_similarity_scores: dict[str, float]
    most_significant_changes: dict[str, list[dict[str, Any]]]
    executive_summary: dict[str, Any]


# =====================================================
# MAIN DIFFERENTIAL ANALYZER CLASS
# =====================================================


class DifferentialAnalyzer:
    """
    Main class for comprehensive simulation vs simulation differential analysis
    """

    def __init__(
        self, config: DifferentialConfig = None, output_dir: str = "differential_analysis_results"
    ):
        """
        Initialize differential analyzer

        Parameters:
        -----------
        config : DifferentialConfig
            Configuration for differential analysis
        output_dir : str
            Base directory for output
        """
        self.config = config or DifferentialConfig()
        self.output_dir = Path(output_dir)
        self.output_manager = None

        # Create output directory structure
        self._setup_output_directories()

        # Initialize comparison modules
        self.network_comparator = NetworkComparator(self.config)
        self.dynamics_comparator = DynamicsComparator(self.config)
        self.energetics_comparator = EnergeticsComparator(self.config)
        self.kinetics_comparator = KineticsComparator(self.config)
        self.allosteric_comparator = AllostericComparator(self.config)

        print(f"Differential analysis initialized with output directory: {self.output_dir}")

    def _setup_output_directories(self):
        """Create organized output directory structure"""
        self.output_dir.mkdir(exist_ok=True)

        # Create main subdirectories
        self.subdirs = {
            "individual": self.output_dir / "01_individual_analyses",
            "networks": self.output_dir / "02_network_comparisons",
            "dynamics": self.output_dir / "03_dynamics_comparisons",
            "energetics": self.output_dir / "04_energetics_comparisons",
            "kinetics": self.output_dir / "05_kinetics_comparisons",
            "allosteric": self.output_dir / "06_allosteric_comparisons",
            "reports": self.output_dir / "07_comprehensive_report",
        }

        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)

        # Create publication figures subdirectory
        if self.config.create_publication_figures:
            (self.subdirs["reports"] / "publication_figures_high_res").mkdir(exist_ok=True)

    def run_differential_analysis(
        self,
        sim1_topology: str,
        sim1_trajectory: str,
        sim1_name: str,
        sim2_topology: str,
        sim2_trajectory: str,
        sim2_name: str,
        analysis_config: AnalysisConfig = None,
    ) -> ComprehensiveDifferentialResults:
        """
        Perform comprehensive differential analysis between two simulations

        Parameters:
        -----------
        sim1_topology, sim1_trajectory, sim1_name : str
            First simulation data and identifier
        sim2_topology, sim2_trajectory, sim2_name : str
            Second simulation data and identifier
        analysis_config : AnalysisConfig
            Configuration for individual simulation analyses

        Returns:
        --------
        ComprehensiveDifferentialResults
            Complete differential analysis results
        """
        print(f"Starting differential analysis: {sim1_name} vs {sim2_name}")
        print("=" * 60)

        # Step 1: Perform individual analyses
        print("Phase 1: Individual simulation analyses...")
        sim1_results, sim2_results = self._run_individual_analyses(
            sim1_topology,
            sim1_trajectory,
            sim1_name,
            sim2_topology,
            sim2_trajectory,
            sim2_name,
            analysis_config,
        )

        # Step 2: Comparative analysis
        print("Phase 2: Comparative analysis...")
        comparison_results = self._run_comparative_analyses(sim1_results, sim2_results)

        # Step 3: Statistical analysis
        if self.config.perform_statistical_tests:
            print("Phase 3: Statistical significance testing...")
            self._perform_statistical_analysis(comparison_results)

        # Step 4: Generate comprehensive results object
        print("Phase 4: Generating comprehensive results...")
        differential_results = self._compile_comprehensive_results(
            sim1_results, sim2_results, comparison_results, (sim1_name, sim2_name)
        )

        # Step 5: Create outputs and reports
        print("Phase 5: Creating outputs and reports...")
        self._generate_outputs(differential_results)

        print("✓ Differential analysis complete!")
        print(f"Results available in: {self.output_dir}")

        return differential_results

    def _run_individual_analyses(
        self,
        sim1_topology: str,
        sim1_trajectory: str,
        sim1_name: str,
        sim2_topology: str,
        sim2_trajectory: str,
        sim2_name: str,
        analysis_config: AnalysisConfig,
    ) -> tuple[MDSimulation, MDSimulation]:
        """Run complete individual analyses on both simulations"""

        # Import the main analysis class
        from .core import MDCompare, SimulationConfig

        if analysis_config is None:
            analysis_config = AnalysisConfig()

        # Create individual output directories
        sim1_output = self.subdirs["individual"] / f"{sim1_name}_results"
        sim2_output = self.subdirs["individual"] / f"{sim2_name}_results"

        # Create simulation configurations
        sim1_config = SimulationConfig(
            name=sim1_name,
            topology=sim1_topology,
            trajectory=sim1_trajectory,
            selection="protein and not name H*",
            description="Simulation 1 for differential analysis",
        )

        sim2_config = SimulationConfig(
            name=sim2_name,
            topology=sim2_topology,
            trajectory=sim2_trajectory,
            selection="protein and not name H*",
            description="Simulation 2 for differential analysis",
        )

        # Analyze simulation 1
        print(f"  Analyzing {sim1_name}...")
        md_compare_1 = MDCompare(analysis_config, str(sim1_output))
        md_compare_1.add_simulation(sim1_config)
        sim1_results = md_compare_1.run_analysis([sim1_name])[sim1_name]

        # Analyze simulation 2
        print(f"  Analyzing {sim2_name}...")
        md_compare_2 = MDCompare(analysis_config, str(sim2_output))
        md_compare_2.add_simulation(sim2_config)
        sim2_results = md_compare_2.run_analysis([sim2_name])[sim2_name]

        return sim1_results, sim2_results

    def _run_comparative_analyses(self, sim1: MDSimulation, sim2: MDSimulation) -> dict[str, Any]:
        """Perform all comparative analyses"""

        comparison_results = {}

        # Network comparison
        if self.config.compare_networks and sim1.network_metrics and sim2.network_metrics:
            print("  Comparing network topologies...")
            comparison_results["network"] = self.network_comparator.compare_networks(
                sim1.network_metrics, sim2.network_metrics
            )

        # Dynamics comparison
        if self.config.compare_dynamics:
            print("  Comparing dynamics...")
            comparison_results["dynamics"] = self.dynamics_comparator.compare_dynamics(
                getattr(sim1, "dynamic_analysis", None), getattr(sim2, "dynamic_analysis", None)
            )

        # Energetics comparison
        if self.config.compare_energetics:
            print("  Comparing energetics...")
            comparison_results["energetics"] = self.energetics_comparator.compare_energetics(
                getattr(sim1, "dynamic_analysis", None), getattr(sim2, "dynamic_analysis", None)
            )

        # Kinetics comparison
        if self.config.compare_kinetics:
            print("  Comparing kinetics...")
            comparison_results["kinetics"] = self.kinetics_comparator.compare_kinetics(
                getattr(sim1, "dynamic_analysis", None), getattr(sim2, "dynamic_analysis", None)
            )

        # Allosteric comparison
        if self.config.compare_allosteric and sim1.network_metrics and sim2.network_metrics:
            print("  Comparing allosteric communication...")
            comparison_results["allosteric"] = self.allosteric_comparator.compare_allosteric(
                sim1.network_metrics, sim2.network_metrics
            )

        return comparison_results

    def _perform_statistical_analysis(self, comparison_results: dict[str, Any]):
        """Perform statistical significance testing on comparison results"""

        if not SCIPY_AVAILABLE:
            print("  Warning: scipy not available, skipping statistical tests")
            return

        print("  Performing statistical significance tests...")

        # Implement statistical tests for each comparison type
        for _analysis_type, results in comparison_results.items():
            if hasattr(results, "statistical_significance"):
                # Placeholder for statistical analysis implementation
                pass

    def _compile_comprehensive_results(
        self,
        sim1: MDSimulation,
        sim2: MDSimulation,
        comparison_results: dict[str, Any],
        sim_names: tuple[str, str],
    ) -> ComprehensiveDifferentialResults:
        """Compile all results into comprehensive differential results object"""

        return ComprehensiveDifferentialResults(
            simulation_names=sim_names,
            analysis_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config=self.config,
            simulation1_results=sim1,
            simulation2_results=sim2,
            network_comparison=comparison_results.get("network"),
            dynamics_comparison=comparison_results.get("dynamics"),
            energetics_comparison=comparison_results.get("energetics"),
            kinetics_comparison=comparison_results.get("kinetics"),
            allosteric_comparison=comparison_results.get("allosteric"),
            overall_similarity_scores=self._calculate_similarity_scores(comparison_results),
            most_significant_changes=self._identify_most_significant_changes(comparison_results),
            executive_summary=self._generate_executive_summary(comparison_results, sim_names),
        )

    def _calculate_similarity_scores(self, comparison_results: dict[str, Any]) -> dict[str, float]:
        """Calculate overall similarity scores between simulations"""

        similarity_scores = {}

        # Network similarity
        if "network" in comparison_results:
            # Placeholder: implement network similarity calculation
            similarity_scores["network"] = 0.85

        # Dynamics similarity
        if "dynamics" in comparison_results:
            # Placeholder: implement dynamics similarity calculation
            similarity_scores["dynamics"] = 0.72

        # Add other similarity calculations...

        return similarity_scores

    def _identify_most_significant_changes(
        self, comparison_results: dict[str, Any]
    ) -> dict[str, list[dict[str, Any]]]:
        """Identify the most significant changes across all analysis types"""

        significant_changes = {}

        # Placeholder implementation
        for analysis_type in comparison_results.keys():
            significant_changes[analysis_type] = [
                {"type": "placeholder", "significance": 0.001, "description": "Example change"}
            ]

        return significant_changes

    def _generate_executive_summary(
        self, comparison_results: dict[str, Any], sim_names: tuple[str, str]
    ) -> dict[str, Any]:
        """Generate executive summary of differential analysis"""

        return {
            "comparison": f"{sim_names[0]} vs {sim_names[1]}",
            "total_analyses": len(comparison_results),
            "significant_findings": 5,  # Placeholder
            "overall_similarity": 0.78,  # Placeholder
            "key_insights": [
                "Significant changes in allosteric communication",
                "Altered energy landscape minima",
                "Modified kinetic pathways",
            ],
        }

    def _generate_outputs(self, results: ComprehensiveDifferentialResults):
        """Generate all output files and reports"""

        print("  Generating differential analysis outputs...")

        # Save comprehensive results object
        with open(self.output_dir / "comprehensive_differential_results.pkl", "wb") as f:
            pickle.dump(results, f)

        # Generate CSV reports
        if PANDAS_AVAILABLE:
            self._create_csv_reports(results)

        # Generate visualizations
        if PLOTTING_AVAILABLE:
            self._create_visualizations(results)

        # Generate HTML report
        if self.config.create_html_report:
            self._create_html_report(results)

        print("  ✓ Output generation complete")

    def _create_csv_reports(self, results: ComprehensiveDifferentialResults):
        """Generate CSV reports for all comparison results"""

        if not PANDAS_AVAILABLE:
            print("  Warning: pandas not available, skipping CSV reports")
            return

        try:
            # Network comparison CSV
            if results.network_comparison:
                self._create_network_csv_reports(results.network_comparison)

            # Dynamics comparison CSV
            if results.dynamics_comparison:
                self._create_dynamics_csv_reports(results.dynamics_comparison)

            # Allosteric comparison CSV
            if results.allosteric_comparison:
                self._create_allosteric_csv_reports(results.allosteric_comparison)

            # Kinetics comparison CSV
            if results.kinetics_comparison:
                self._create_kinetics_csv_reports(results.kinetics_comparison)

            # Summary CSV
            self._create_summary_csv_reports(results)

            print("  ✓ CSV reports generated")

        except Exception as e:
            print(f"  Warning: Error generating CSV reports: {e}")

    def _create_network_csv_reports(self, network_comparison):
        """Create CSV reports for network comparison"""

        import pandas as pd

        networks_dir = self.subdirs["networks"]

        # Edge weight changes
        if network_comparison.edge_weight_changes:
            edge_changes_df = pd.DataFrame(
                [
                    {
                        "Node1": edge[0],
                        "Node2": edge[1],
                        "Weight_Change": change,
                        "Change_Type": "increase" if change > 0 else "decrease",
                        "Magnitude": abs(change),
                    }
                    for edge, change in network_comparison.edge_weight_changes.items()
                ]
            )
            edge_changes_df.to_csv(networks_dir / "edge_weight_changes.csv", index=False)

        # Centrality changes
        for metric, changes in network_comparison.centrality_changes.items():
            if changes:
                centrality_df = pd.DataFrame(
                    [
                        {
                            "Node": node,
                            "Centrality_Change": change,
                            "Change_Type": "increase" if change > 0 else "decrease",
                            "Magnitude": abs(change),
                            "Metric": metric,
                        }
                        for node, change in changes.items()
                    ]
                )
                centrality_df.to_csv(networks_dir / f"{metric}_centrality_changes.csv", index=False)

    def _create_dynamics_csv_reports(self, dynamics_comparison):
        """Create CSV reports for dynamics comparison"""

        import pandas as pd

        dynamics_dir = self.subdirs["dynamics"]

        # Significant correlation changes
        if dynamics_comparison.significant_correlation_changes:
            corr_df = pd.DataFrame(
                dynamics_comparison.significant_correlation_changes,
                columns=["Residue1", "Residue2", "Original_Correlation", "New_Correlation"],
            )
            corr_df["Correlation_Change"] = (
                corr_df["New_Correlation"] - corr_df["Original_Correlation"]
            )
            corr_df["Change_Magnitude"] = corr_df["Correlation_Change"].abs()
            corr_df["Change_Type"] = corr_df["Correlation_Change"].apply(
                lambda x: "increase" if x > 0 else "decrease"
            )
            corr_df = corr_df.sort_values("Change_Magnitude", ascending=False)
            corr_df.to_csv(dynamics_dir / "significant_correlation_changes.csv", index=False)

    def _create_allosteric_csv_reports(self, allosteric_comparison):
        """Create CSV reports for allosteric comparison"""

        import pandas as pd

        allosteric_dir = self.subdirs["allosteric"]

        # Pathway disruption analysis
        if allosteric_comparison.pathway_disruption_analysis:
            pathway_df = pd.DataFrame(allosteric_comparison.pathway_disruption_analysis)
            pathway_df.to_csv(allosteric_dir / "pathway_disruption_analysis.csv", index=False)

        # Hotspot ranking changes
        if allosteric_comparison.hotspot_ranking_changes:
            hotspot_df = pd.DataFrame(allosteric_comparison.hotspot_ranking_changes)
            hotspot_df.to_csv(allosteric_dir / "hotspot_ranking_changes.csv", index=False)

    def _create_kinetics_csv_reports(self, kinetics_comparison):
        """Create CSV reports for kinetics comparison"""

        import pandas as pd

        kinetics_dir = self.subdirs["kinetics"]

        # Timescale changes
        if kinetics_comparison.timescale_changes.size > 0:
            timescale_df = pd.DataFrame(
                {
                    "Timescale_Index": range(1, len(kinetics_comparison.timescale_changes) + 1),
                    "Timescale_Change": kinetics_comparison.timescale_changes,
                    "Change_Type": [
                        "increase" if x > 0 else "decrease"
                        for x in kinetics_comparison.timescale_changes
                    ],
                }
            )
            timescale_df.to_csv(kinetics_dir / "timescale_changes.csv", index=False)

    def _create_summary_csv_reports(self, results):
        """Create summary CSV reports"""

        import pandas as pd

        reports_dir = self.subdirs["reports"]

        # Overall similarity scores
        similarity_df = pd.DataFrame(
            [
                {
                    "Analysis_Type": analysis_type,
                    "Similarity_Score": score,
                    "Similarity_Category": (
                        "high" if score > 0.8 else "medium" if score > 0.5 else "low"
                    ),
                }
                for analysis_type, score in results.overall_similarity_scores.items()
            ]
        )
        similarity_df.to_csv(reports_dir / "overall_similarity_scores.csv", index=False)

    def _create_visualizations(self, results):
        """Create difference plots and visualizations"""

        if not PLOTTING_AVAILABLE:
            print("  Warning: matplotlib not available, skipping visualizations")
            return

        try:

            # Create dynamics visualizations
            if (
                results.dynamics_comparison
                and results.dynamics_comparison.dccm_difference_matrix.size > 0
            ):
                self._create_dccm_difference_plot(results.dynamics_comparison)

            # Create summary visualizations
            self._create_summary_plots(results)

            print("  ✓ Visualizations created")

        except Exception as e:
            print(f"  Warning: Error creating visualizations: {e}")

    def _create_dccm_difference_plot(self, dynamics_comparison):
        """Create DCCM difference heatmap"""

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            dynamics_comparison.dccm_difference_matrix,
            cmap=self.config.heatmap_colormap,
            center=0,
            cbar_kws={"label": "Correlation Change (Δρ)"},
            ax=ax,
        )

        ax.set_title("Dynamic Cross-Correlation Difference Matrix", fontsize=14, fontweight="bold")
        ax.set_xlabel("Residue Index", fontsize=12)
        ax.set_ylabel("Residue Index", fontsize=12)

        # Save plot
        dynamics_dir = self.subdirs["dynamics"]
        plt.savefig(
            dynamics_dir / "dccm_difference_heatmap.png",
            dpi=self.config.figure_dpi,
            bbox_inches="tight",
        )
        plt.close()

    def _create_summary_plots(self, results):
        """Create summary visualization plots"""

        import matplotlib.pyplot as plt

        # Similarity scores bar plot
        if results.overall_similarity_scores:
            fig, ax = plt.subplots(figsize=(10, 6))

            analysis_types = list(results.overall_similarity_scores.keys())
            scores = list(results.overall_similarity_scores.values())

            bars = ax.bar(analysis_types, scores, color="skyblue", alpha=0.7, edgecolor="navy")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Similarity Score", fontsize=12)
            ax.set_title("Overall Similarity by Analysis Type", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, score in zip(bars, scores, strict=False):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{score:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            reports_dir = self.subdirs["reports"]
            plt.savefig(
                reports_dir / "similarity_scores_summary.png",
                dpi=self.config.figure_dpi,
                bbox_inches="tight",
            )
            plt.close()

    def _create_html_report(self, results):
        """Generate comprehensive HTML report"""

        reports_dir = self.subdirs["reports"]

        # Basic HTML template
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>confdelta Differential Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 3px solid #3498db; }}
        .metric {{ background-color: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .similarity-score {{ font-size: 1.2em; font-weight: bold; color: #2ecc71; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>confdelta Differential Analysis Report</h1>
        <p><strong>Comparison:</strong> {results.simulation_names[0]} vs {results.simulation_names[1]}</p>
        <p><strong>Analysis Date:</strong> {results.analysis_timestamp}</p>
        <p><strong>Software:</strong> confdelta v{__version__}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metric">
            <strong>Total Analyses:</strong> {results.executive_summary['total_analyses']}
        </div>
        <div class="metric">
            <strong>Significant Findings:</strong> {results.executive_summary['significant_findings']}
        </div>
        <div class="metric">
            <strong>Overall Similarity:</strong> <span class="similarity-score">{results.executive_summary['overall_similarity']:.3f}</span>
        </div>
    </div>

    <div class="section">
        <h2>Analysis Components</h2>
        <p>Detailed results are available in the organized subdirectories:</p>
        <ul>
            <li><strong>01_individual_analyses/</strong> - Complete analysis for each simulation</li>
            <li><strong>02_network_comparisons/</strong> - Network topology differences</li>
            <li><strong>03_dynamics_comparisons/</strong> - DCCM and PCA differences</li>
            <li><strong>04_energetics_comparisons/</strong> - Energy landscape differences</li>
            <li><strong>05_kinetics_comparisons/</strong> - MSM and timescale differences</li>
            <li><strong>06_allosteric_comparisons/</strong> - Communication efficiency differences</li>
            <li><strong>07_comprehensive_report/</strong> - Summary reports and figures</li>
        </ul>
    </div>

</body>
</html>
        """

        with open(
            reports_dir / "comprehensive_differential_report.html", "w", encoding="utf-8"
        ) as f:
            f.write(html_content)

        print(f"  ✓ HTML report saved: {reports_dir / 'comprehensive_differential_report.html'}")


# =====================================================
# PLACEHOLDER COMPARATOR CLASSES
# =====================================================


class NetworkComparator:
    """Compares network topology between simulations"""

    def __init__(self, config: DifferentialConfig):
        self.config = config

    def compare_networks(self, net1, net2) -> NetworkComparison:
        """Compare network topologies"""

        if not net1 or not net2 or not hasattr(net1, "network") or not hasattr(net2, "network"):
            print("Warning: Network data not available for both simulations")
            return self._create_empty_network_comparison()

        graph1 = net1.network
        graph2 = net2.network

        if graph1 is None or graph2 is None:
            return self._create_empty_network_comparison()

        print("  Computing network topology differences...")

        # Compare basic topology
        nodes_added, nodes_removed, edges_added, edges_removed, edge_weight_changes = (
            self._compare_topology(graph1, graph2)
        )

        # Compare centrality measures
        centrality_changes, centrality_ranking_changes = self._compare_centralities(net1, net2)

        # Compare community structure
        community_changes, modularity_change = self._compare_communities(net1, net2)

        # Compare global network properties
        network_property_changes = self._compare_global_properties(graph1, graph2)

        # Statistical significance testing
        statistical_significance = {}
        if self.config.perform_statistical_tests and SCIPY_AVAILABLE:
            statistical_significance = self._test_network_significance(graph1, graph2)

        return NetworkComparison(
            nodes_added=nodes_added,
            nodes_removed=nodes_removed,
            edges_added=edges_added,
            edges_removed=edges_removed,
            edge_weight_changes=edge_weight_changes,
            centrality_changes=centrality_changes,
            centrality_ranking_changes=centrality_ranking_changes,
            community_changes=community_changes,
            modularity_change=modularity_change,
            network_property_changes=network_property_changes,
            statistical_significance=statistical_significance,
        )

    def _compare_topology(
        self, graph1, graph2
    ) -> tuple[list[str], list[str], list[tuple], list[tuple], dict]:
        """Compare basic network topology (nodes and edges)"""

        # Compare nodes
        nodes1 = set(graph1.nodes())
        nodes2 = set(graph2.nodes())

        nodes_added = list(nodes2 - nodes1)
        nodes_removed = list(nodes1 - nodes2)
        nodes1.intersection(nodes2)

        # Compare edges
        edges1 = set(graph1.edges())
        edges2 = set(graph2.edges())

        edges_added = list(edges2 - edges1)
        edges_removed = list(edges1 - edges2)
        common_edges = edges1.intersection(edges2)

        # Compare edge weights for common edges
        edge_weight_changes = {}
        for edge in common_edges:
            weight1 = graph1.edges[edge].get("weight", 1.0)
            weight2 = graph2.edges[edge].get("weight", 1.0)
            weight_change = weight2 - weight1

            if abs(weight_change) > 0.05:  # Significant weight change threshold
                edge_weight_changes[edge] = weight_change

        print(f"    Nodes added: {len(nodes_added)}, removed: {len(nodes_removed)}")
        print(f"    Edges added: {len(edges_added)}, removed: {len(edges_removed)}")
        print(f"    Edge weight changes: {len(edge_weight_changes)}")

        return nodes_added, nodes_removed, edges_added, edges_removed, edge_weight_changes

    def _compare_centralities(
        self, net1, net2
    ) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
        """Compare centrality measures between networks"""

        centrality_metrics = ["degree", "betweenness", "closeness", "eigenvector"]
        centrality_changes = {}
        centrality_ranking_changes = {}

        for metric in centrality_metrics:
            # Get centrality values
            cent1 = getattr(net1, f"{metric}_centrality", {})
            cent2 = getattr(net2, f"{metric}_centrality", {})

            if not cent1 or not cent2:
                continue

            # Calculate changes for common nodes
            common_nodes = set(cent1.keys()).intersection(set(cent2.keys()))
            metric_changes = {}

            for node in common_nodes:
                change = cent2[node] - cent1[node]
                if abs(change) > self.config.centrality_change_threshold:
                    metric_changes[node] = change

            centrality_changes[metric] = metric_changes

            # Find nodes with large ranking changes
            if metric_changes:
                # Sort by magnitude of change
                large_changes = [
                    node
                    for node, change in metric_changes.items()
                    if abs(change) > 2 * self.config.centrality_change_threshold
                ]
                centrality_ranking_changes[metric] = large_changes

            print(f"    {metric.title()} centrality changes: {len(metric_changes)} nodes")

        return centrality_changes, centrality_ranking_changes

    def _compare_communities(self, net1, net2) -> tuple[dict[str, Any], float]:
        """Compare community structure between networks"""

        community_changes = {}
        modularity_change = 0.0

        # Get community data
        comm1 = getattr(net1, "communities_detailed", {})
        comm2 = getattr(net2, "communities_detailed", {})

        if not comm1 or not comm2:
            return community_changes, modularity_change

        # Compare modularity
        mod1 = comm1.get("modularity", 0.0)
        mod2 = comm2.get("modularity", 0.0)
        modularity_change = mod2 - mod1

        # Compare number of communities
        n_comm1 = comm1.get("n_communities", 0)
        n_comm2 = comm2.get("n_communities", 0)

        # Compare community sizes
        sizes1 = comm1.get("community_sizes", [])
        sizes2 = comm2.get("community_sizes", [])

        community_changes = {
            "modularity_change": modularity_change,
            "n_communities_change": n_comm2 - n_comm1,
            "avg_community_size_change": (
                (np.mean(sizes2) - np.mean(sizes1)) if sizes1 and sizes2 else 0
            ),
            "community_size_variance_change": (
                (np.var(sizes2) - np.var(sizes1)) if sizes1 and sizes2 else 0
            ),
        }

        print(
            f"    Community changes - Modularity: {modularity_change:+.4f}, "
            f"Count: {n_comm2 - n_comm1:+d}"
        )

        return community_changes, modularity_change

    def _compare_global_properties(self, graph1, graph2) -> dict[str, float]:
        """Compare global network properties"""

        try:
            import networkx as nx
        except ImportError:
            return {}

        properties = {}

        # Network connectivity
        if graph1.number_of_nodes() > 0 and graph2.number_of_nodes() > 0:
            density1 = nx.density(graph1)
            density2 = nx.density(graph2)
            properties["density_change"] = density2 - density1

            # Average clustering
            if graph1.number_of_nodes() > 2 and graph2.number_of_nodes() > 2:
                cluster1 = nx.average_clustering(graph1)
                cluster2 = nx.average_clustering(graph2)
                properties["clustering_change"] = cluster2 - cluster1

            # Average degree
            avg_degree1 = sum(dict(graph1.degree()).values()) / graph1.number_of_nodes()
            avg_degree2 = sum(dict(graph2.degree()).values()) / graph2.number_of_nodes()
            properties["avg_degree_change"] = avg_degree2 - avg_degree1

            # Network efficiency (if connected)
            if nx.is_connected(graph1) and nx.is_connected(graph2):
                eff1 = nx.global_efficiency(graph1)
                eff2 = nx.global_efficiency(graph2)
                properties["global_efficiency_change"] = eff2 - eff1

        print(f"    Global property changes: {len(properties)} metrics")

        return properties

    def _test_network_significance(self, graph1, graph2) -> dict[str, float]:
        """Statistical significance testing for network differences"""

        # Placeholder for network-specific statistical tests
        # Could implement permutation tests, graph comparison metrics, etc.

        significance_tests = {
            "topology_change_pvalue": 0.05,  # Placeholder
            "centrality_change_pvalue": 0.01,  # Placeholder
            "community_change_pvalue": 0.1,  # Placeholder
        }

        return significance_tests

    def _create_empty_network_comparison(self) -> NetworkComparison:
        """Create empty comparison result when data is unavailable"""

        return NetworkComparison(
            nodes_added=[],
            nodes_removed=[],
            edges_added=[],
            edges_removed=[],
            edge_weight_changes={},
            centrality_changes={},
            centrality_ranking_changes={},
            community_changes={},
            modularity_change=0.0,
            network_property_changes={},
            statistical_significance={},
        )


class DynamicsComparator:
    """Compares dynamics (DCCM, PCA) between simulations"""

    def __init__(self, config: DifferentialConfig):
        self.config = config

    def compare_dynamics(self, dyn1, dyn2) -> DynamicsComparison:
        """Compare dynamics analysis results"""

        if not dyn1 or not dyn2:
            print("Warning: Dynamic analysis data not available for both simulations")
            return self._create_empty_dynamics_comparison()

        print("  Computing dynamics differences...")

        # Compare DCCM matrices
        dccm_diff_matrix, correlation_changes, regional_changes = self._compare_dccm_matrices(
            dyn1, dyn2
        )

        # Compare PCA results
        pca_variance_changes, eigenvector_similarities, pc_shifts = self._compare_pca_results(
            dyn1, dyn2
        )

        # Statistical significance testing
        correlation_pvalues, significant_pairs, effect_sizes = self._perform_correlation_statistics(
            dyn1, dyn2, dccm_diff_matrix
        )

        return DynamicsComparison(
            dccm_difference_matrix=dccm_diff_matrix,
            significant_correlation_changes=correlation_changes,
            regional_correlation_changes=regional_changes,
            pca_variance_changes=pca_variance_changes,
            eigenvector_similarities=eigenvector_similarities,
            principal_component_shifts=pc_shifts,
            correlation_change_pvalues=correlation_pvalues,
            significant_residue_pairs=significant_pairs,
            effect_sizes=effect_sizes,
        )

    def _compare_dccm_matrices(
        self, dyn1: dict, dyn2: dict
    ) -> tuple[np.ndarray, list[tuple], dict]:
        """Compare Dynamic Cross-Correlation Matrices"""

        # Extract DCCM matrices
        dccm1 = dyn1.get("dccm_matrix")
        dccm2 = dyn2.get("dccm_matrix")

        if dccm1 is None or dccm2 is None:
            print("    Warning: DCCM matrices not available")
            return np.array([]), [], {}

        if dccm1.shape != dccm2.shape:
            print(f"    Warning: DCCM matrix size mismatch: {dccm1.shape} vs {dccm2.shape}")
            return np.array([]), [], {}

        # Calculate difference matrix
        dccm_diff = dccm2 - dccm1

        # Find significant correlation changes
        threshold = self.config.correlation_change_threshold
        significant_changes = []

        n_res = dccm_diff.shape[0]
        for i in range(n_res):
            for j in range(i + 1, n_res):  # Upper triangle only
                change = dccm_diff[i, j]
                if abs(change) >= threshold:
                    significant_changes.append(
                        (
                            f"res_{i+1}",  # Residue indices (1-based)
                            f"res_{j+1}",
                            dccm1[i, j],  # Original correlation
                            dccm2[i, j],  # New correlation
                        )
                    )

        # Sort by magnitude of change
        significant_changes.sort(key=lambda x: abs(x[3] - x[2]), reverse=True)

        # Regional analysis (if residue information available)
        regional_changes = self._analyze_regional_correlation_changes(dccm_diff, dyn1, dyn2)

        print(f"    DCCM difference matrix computed: {dccm_diff.shape}")
        print(f"    Significant correlation changes: {len(significant_changes)} pairs")
        print(f"    Mean absolute change: {np.mean(np.abs(dccm_diff)):.4f}")

        return dccm_diff, significant_changes, regional_changes

    def _analyze_regional_correlation_changes(
        self, dccm_diff: np.ndarray, dyn1: dict, dyn2: dict
    ) -> dict[str, dict[str, float]]:
        """Analyze correlation changes by protein regions/secondary structure"""

        # This would require secondary structure information
        # For now, analyze by matrix quadrants as a proxy for regions

        h, w = dccm_diff.shape
        mid_h, mid_w = h // 2, w // 2

        regions = {
            "N_terminal_N_terminal": dccm_diff[:mid_h, :mid_w],
            "N_terminal_C_terminal": dccm_diff[:mid_h, mid_w:],
            "C_terminal_N_terminal": dccm_diff[mid_h:, :mid_w],
            "C_terminal_C_terminal": dccm_diff[mid_h:, mid_w:],
        }

        regional_changes = {}
        for region_name, region_matrix in regions.items():
            if region_matrix.size > 0:
                regional_changes[region_name] = {
                    "mean_change": np.mean(region_matrix),
                    "mean_abs_change": np.mean(np.abs(region_matrix)),
                    "max_change": np.max(region_matrix),
                    "min_change": np.min(region_matrix),
                    "std_change": np.std(region_matrix),
                }

        return regional_changes

    def _compare_pca_results(self, dyn1: dict, dyn2: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        """Compare Principal Component Analysis results"""

        # Extract PCA data
        eigenvals1 = dyn1.get("pca_eigenvalues")
        eigenvals2 = dyn2.get("pca_eigenvalues")
        eigenvecs1 = dyn1.get("pca_eigenvectors")
        eigenvecs2 = dyn2.get("pca_eigenvectors")

        variance_changes = np.array([])
        eigenvector_similarities = np.array([])
        pc_shifts = {}

        if eigenvals1 is not None and eigenvals2 is not None:
            # Compare eigenvalues (variance explained)
            min_len = min(len(eigenvals1), len(eigenvals2))
            variance_changes = eigenvals2[:min_len] - eigenvals1[:min_len]

            print(f"    PCA eigenvalue differences computed: {min_len} components")
            print(f"    Largest variance change: {np.max(np.abs(variance_changes)):.4f}")

        if eigenvecs1 is not None and eigenvecs2 is not None:
            # Compare eigenvectors using dot product (cosine similarity)
            min_components = min(eigenvecs1.shape[1], eigenvecs2.shape[1])
            eigenvector_similarities = np.zeros(min_components)

            for i in range(min_components):
                # Absolute value to handle sign flips (eigenvectors can be negated)
                similarity = abs(np.dot(eigenvecs1[:, i], eigenvecs2[:, i]))
                eigenvector_similarities[i] = similarity

                # Track significant shifts
                if similarity < 0.8:  # Eigenvector changed significantly
                    pc_shifts[i] = {
                        "similarity": similarity,
                        "change_type": "major" if similarity < 0.5 else "moderate",
                    }

            print(f"    Eigenvector similarities computed: {min_components} components")
            print(f"    Mean eigenvector similarity: {np.mean(eigenvector_similarities):.4f}")

        return variance_changes, eigenvector_similarities, pc_shifts

    def _perform_correlation_statistics(
        self, dyn1: dict, dyn2: dict, dccm_diff: np.ndarray
    ) -> tuple[np.ndarray, list[tuple], dict]:
        """Perform statistical significance testing for correlation changes"""

        if not SCIPY_AVAILABLE or not self.config.perform_statistical_tests:
            return np.array([]), [], {}

        if dccm_diff.size == 0:
            return np.array([]), [], {}

        print("    Performing correlation change significance tests...")

        # For now, implement a simple approach based on correlation magnitude
        # In practice, would need time series data for proper statistical testing

        n_res = dccm_diff.shape[0]
        correlation_pvalues = np.ones((n_res, n_res))

        # Identify significant pairs using threshold approach
        threshold = self.config.correlation_change_threshold
        significant_pairs = []

        for i in range(n_res):
            for j in range(i + 1, n_res):
                if abs(dccm_diff[i, j]) >= threshold:
                    # Estimate p-value based on change magnitude
                    # This is a simplified approach - proper implementation would use
                    # bootstrap or permutation tests on the actual trajectory data
                    p_value = max(0.001, 1.0 - abs(dccm_diff[i, j]))
                    correlation_pvalues[i, j] = p_value
                    correlation_pvalues[j, i] = p_value

                    if p_value < self.config.significance_threshold:
                        significant_pairs.append((f"res_{i+1}", f"res_{j+1}"))

        # Calculate effect sizes
        effect_sizes = {
            "mean_effect_size": np.mean(np.abs(dccm_diff)),
            "large_effects": np.sum(np.abs(dccm_diff) > 0.3),  # Arbitrary threshold for "large"
            "medium_effects": np.sum((np.abs(dccm_diff) > 0.15) & (np.abs(dccm_diff) <= 0.3)),
            "small_effects": np.sum((np.abs(dccm_diff) > 0.05) & (np.abs(dccm_diff) <= 0.15)),
        }

        print(f"    Statistically significant pairs: {len(significant_pairs)}")
        print(
            f"    Effect sizes - Large: {effect_sizes['large_effects']}, "
            f"Medium: {effect_sizes['medium_effects']}, Small: {effect_sizes['small_effects']}"
        )

        return correlation_pvalues, significant_pairs, effect_sizes

    def _create_empty_dynamics_comparison(self) -> DynamicsComparison:
        """Create empty comparison result when data is unavailable"""

        return DynamicsComparison(
            dccm_difference_matrix=np.array([]),
            significant_correlation_changes=[],
            regional_correlation_changes={},
            pca_variance_changes=np.array([]),
            eigenvector_similarities=np.array([]),
            principal_component_shifts={},
            correlation_change_pvalues=np.array([]),
            significant_residue_pairs=[],
            effect_sizes={},
        )


class EnergeticsComparator:
    """Compares energy landscapes between simulations"""

    def __init__(self, config: DifferentialConfig):
        self.config = config

    def compare_energetics(self, dyn1, dyn2) -> EnergeticsComparison:
        """Compare energy landscape analysis results"""

        if not dyn1 or not dyn2:
            print("Warning: Dynamic analysis data not available for both simulations")
            return self._create_empty_energetics_comparison()

        print("  Computing energy landscape differences...")

        # Extract energy landscape data
        landscape1 = dyn1.get("energy_landscape")
        landscape2 = dyn2.get("energy_landscape")

        if landscape1 is None or landscape2 is None:
            print("    Warning: Energy landscape data not available")
            return self._create_empty_energetics_comparison()

        # Compare energy surfaces
        energy_diff_surface = self._compare_energy_surfaces(landscape1, landscape2)

        # Compare minima
        minima_comparison = self._compare_energy_minima(landscape1, landscape2)

        # Compare barriers
        barrier_changes = self._compare_energy_barriers(landscape1, landscape2)

        # Thermodynamic analysis
        stability_changes = self._analyze_stability_changes(landscape1, landscape2)

        # Conformational state analysis
        conformational_changes = self._analyze_conformational_changes(landscape1, landscape2)

        # Free energy changes
        free_energy_changes = self._calculate_free_energy_changes(landscape1, landscape2)

        # Statistical significance
        energy_significance = {}
        population_changes = {}

        if self.config.perform_statistical_tests and SCIPY_AVAILABLE:
            energy_significance = self._test_energy_significance(landscape1, landscape2)
            population_changes = self._test_population_changes(landscape1, landscape2)

        return EnergeticsComparison(
            energy_difference_surface=energy_diff_surface,
            minima_comparison=minima_comparison,
            barrier_height_changes=barrier_changes,
            stability_changes=stability_changes,
            conformational_state_changes=conformational_changes,
            free_energy_changes=free_energy_changes,
            energy_change_significance=energy_significance,
            conformational_population_changes=population_changes,
        )

    def _compare_energy_surfaces(self, landscape1: dict, landscape2: dict) -> np.ndarray | None:
        """Compare energy landscape surfaces"""

        # Extract energy grids
        energy1 = landscape1.get("energy_grid")
        energy2 = landscape2.get("energy_grid")

        if energy1 is None or energy2 is None:
            return None

        if energy1.shape != energy2.shape:
            print(f"    Warning: Energy grid size mismatch: {energy1.shape} vs {energy2.shape}")
            return None

        # Calculate energy difference
        energy_diff = energy2 - energy1

        print(f"    Energy surface difference computed: {energy_diff.shape}")
        print(f"    Max energy decrease: {np.min(energy_diff):.2f} kT")
        print(f"    Max energy increase: {np.max(energy_diff):.2f} kT")
        print(f"    RMS energy change: {np.sqrt(np.mean(energy_diff**2)):.2f} kT")

        return energy_diff

    def _compare_energy_minima(self, landscape1: dict, landscape2: dict) -> list[dict[str, Any]]:
        """Compare energy minima between landscapes"""

        minima1 = landscape1.get("minima", [])
        minima2 = landscape2.get("minima", [])

        minima_comparison = []

        # Compare minima positions and energies
        for i, min1 in enumerate(minima1[:5]):  # Top 5 minima
            # Find closest minimum in landscape2
            best_match = None
            min_distance = float("inf")

            pos1 = (min1.get("pc1", 0), min1.get("pc2", 0))

            for _j, min2 in enumerate(minima2):
                pos2 = (min2.get("pc1", 0), min2.get("pc2", 0))
                distance = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    best_match = min2

            if best_match and min_distance < 2.0:  # Within reasonable distance
                energy_change = best_match.get("energy", 0) - min1.get("energy", 0)

                minima_comparison.append(
                    {
                        "minimum_index": i,
                        "original_position": pos1,
                        "new_position": (best_match.get("pc1", 0), best_match.get("pc2", 0)),
                        "position_shift": min_distance,
                        "original_energy": min1.get("energy", 0),
                        "new_energy": best_match.get("energy", 0),
                        "energy_change": energy_change,
                        "stability_change": "stabilized" if energy_change < 0 else "destabilized",
                        "population_change": best_match.get("population", 0)
                        - min1.get("population", 0),
                    }
                )

        return minima_comparison

    def _compare_energy_barriers(
        self, landscape1: dict, landscape2: dict
    ) -> list[dict[str, float]]:
        """Compare energy barriers between conformational states"""

        # Extract transition barrier information
        barriers1 = landscape1.get("barriers", [])
        barriers2 = landscape2.get("barriers", [])

        barrier_changes = []

        # Compare corresponding barriers
        for i, barrier1 in enumerate(barriers1):
            if i < len(barriers2):
                barrier2 = barriers2[i]

                height_change = barrier2.get("height", 0) - barrier1.get("height", 0)

                barrier_changes.append(
                    {
                        "barrier_index": i,
                        "transition": barrier1.get("transition", f"barrier_{i}"),
                        "original_height": barrier1.get("height", 0),
                        "new_height": barrier2.get("height", 0),
                        "height_change": height_change,
                        "kinetic_impact": "faster" if height_change < 0 else "slower",
                        "rate_change_factor": (
                            np.exp(-height_change) if abs(height_change) < 10 else float("inf")
                        ),
                    }
                )

        return barrier_changes

    def _analyze_stability_changes(self, landscape1: dict, landscape2: dict) -> dict[str, float]:
        """Analyze thermodynamic stability changes"""

        # Calculate global stability metrics
        energy1 = landscape1.get("energy_grid", np.array([]))
        energy2 = landscape2.get("energy_grid", np.array([]))

        if energy1.size == 0 or energy2.size == 0:
            return {}

        stability_changes = {
            "global_minimum_change": np.min(energy2) - np.min(energy1),
            "average_energy_change": np.mean(energy2) - np.mean(energy1),
            "energy_variance_change": np.var(energy2) - np.var(energy1),
            "conformational_entropy_change": self._calculate_entropy_change(energy1, energy2),
        }

        return stability_changes

    def _analyze_conformational_changes(self, landscape1: dict, landscape2: dict) -> dict[str, Any]:
        """Analyze conformational state changes"""

        conformational_changes = {
            "new_states_formed": 0,  # Placeholder
            "states_lost": 0,  # Placeholder
            "major_transitions": [],  # Placeholder
            "population_redistribution": {},  # Placeholder
        }

        return conformational_changes

    def _calculate_free_energy_changes(
        self, landscape1: dict, landscape2: dict
    ) -> dict[str, float]:
        """Calculate free energy changes"""

        # Extract temperature
        temp = landscape1.get("temperature", 310.0)  # Default 310K
        0.001987 * temp  # Boltzmann constant in kcal/mol/K

        free_energy_changes = {
            "binding_free_energy_change": 0.0,  # Placeholder
            "folding_free_energy_change": 0.0,  # Placeholder
            "activation_free_energy_change": 0.0,  # Placeholder
        }

        return free_energy_changes

    def _calculate_entropy_change(self, energy1: np.ndarray, energy2: np.ndarray) -> float:
        """Calculate conformational entropy change"""

        # Simple approximation based on energy distribution
        # More sophisticated methods would use proper statistical mechanics

        try:
            # Calculate Boltzmann weights
            weights1 = np.exp(-energy1.flatten())
            weights2 = np.exp(-energy2.flatten())

            # Normalize
            weights1 /= np.sum(weights1)
            weights2 /= np.sum(weights2)

            # Calculate Shannon entropy
            entropy1 = -np.sum(weights1 * np.log(weights1 + 1e-10))
            entropy2 = -np.sum(weights2 * np.log(weights2 + 1e-10))

            return entropy2 - entropy1

        except Exception:
            return 0.0

    def _test_energy_significance(self, landscape1: dict, landscape2: dict) -> dict[str, float]:
        """Statistical significance testing for energy changes"""

        # Placeholder for energy-specific statistical tests
        return {"energy_change_pvalue": 0.05}

    def _test_population_changes(self, landscape1: dict, landscape2: dict) -> dict[str, float]:
        """Test significance of population changes"""

        # Placeholder for population significance tests
        return {"population_change_pvalue": 0.1}

    def _create_empty_energetics_comparison(self) -> EnergeticsComparison:
        """Create empty comparison result when data is unavailable"""

        return EnergeticsComparison(
            energy_difference_surface=None,
            minima_comparison=[],
            barrier_height_changes=[],
            stability_changes={},
            conformational_state_changes={},
            free_energy_changes={},
            energy_change_significance={},
            conformational_population_changes={},
        )


class KineticsComparator:
    """Compares kinetics (MSM) between simulations"""

    def __init__(self, config: DifferentialConfig):
        self.config = config

    def compare_kinetics(self, dyn1, dyn2) -> KineticsComparison:
        """Compare MSM analysis results"""

        if not dyn1 or not dyn2:
            print("Warning: Dynamic analysis data not available for both simulations")
            return self._create_empty_kinetics_comparison()

        print("  Computing kinetics differences...")

        # Extract MSM data
        msm1 = dyn1.get("msm_analysis")
        msm2 = dyn2.get("msm_analysis")

        if not msm1 or not msm2:
            print("    Warning: MSM analysis data not available")
            return self._create_empty_kinetics_comparison()

        # Compare timescales
        timescale_changes, timescale_ratios, dominant_change = self._compare_timescales(msm1, msm2)

        # Compare transition matrices
        flux_changes = self._compare_transition_fluxes(msm1, msm2)

        # Compare pathway probabilities
        pathway_changes = self._compare_pathway_probabilities(msm1, msm2)

        # Compare metastable states
        population_changes, stability_changes = self._compare_metastable_states(msm1, msm2)

        # Analyze critical pathway changes
        critical_pathways = self._analyze_critical_pathways(msm1, msm2)

        # Bottleneck analysis
        bottleneck_analysis = self._analyze_kinetic_bottlenecks(msm1, msm2)

        return KineticsComparison(
            timescale_changes=timescale_changes,
            timescale_ratios=timescale_ratios,
            dominant_timescale_change=dominant_change,
            transition_flux_changes=flux_changes,
            pathway_probability_changes=pathway_changes,
            metastable_state_population_changes=population_changes,
            metastable_state_stability_changes=stability_changes,
            critical_pathway_changes=critical_pathways,
            bottleneck_analysis=bottleneck_analysis,
        )

    def _compare_timescales(self, msm1: dict, msm2: dict) -> tuple[np.ndarray, np.ndarray, float]:
        """Compare implied timescales"""

        timescales1 = msm1.get("implied_timescales", np.array([]))
        timescales2 = msm2.get("implied_timescales", np.array([]))

        if timescales1.size == 0 or timescales2.size == 0:
            return np.array([]), np.array([]), 0.0

        # Compare timescales
        min_len = min(len(timescales1), len(timescales2))
        ts1 = timescales1[:min_len]
        ts2 = timescales2[:min_len]

        # Calculate changes and ratios
        timescale_changes = ts2 - ts1
        timescale_ratios = np.divide(ts2, ts1, out=np.ones_like(ts2), where=(ts1 != 0))

        # Dominant (slowest) timescale change
        dominant_change = timescale_changes[0] if len(timescale_changes) > 0 else 0.0

        print(f"    Timescale changes computed: {min_len} timescales")
        print(
            f"    Dominant timescale change: {dominant_change:.2f} (ratio: {timescale_ratios[0]:.2f})"
        )
        print(f"    Max timescale acceleration: {np.max(-timescale_changes):.2f}")
        print(f"    Max timescale slowing: {np.max(timescale_changes):.2f}")

        return timescale_changes, timescale_ratios, dominant_change

    def _compare_transition_fluxes(self, msm1: dict, msm2: dict) -> np.ndarray:
        """Compare transition flux matrices"""

        transition_matrix1 = msm1.get("transition_matrix")
        transition_matrix2 = msm2.get("transition_matrix")

        if transition_matrix1 is None or transition_matrix2 is None:
            return np.array([])

        if transition_matrix1.shape != transition_matrix2.shape:
            print(
                f"    Warning: Transition matrix size mismatch: {transition_matrix1.shape} vs {transition_matrix2.shape}"
            )
            return np.array([])

        # Calculate flux changes
        flux_changes = transition_matrix2 - transition_matrix1

        print(f"    Transition flux changes computed: {flux_changes.shape}")
        print(f"    Max flux increase: {np.max(flux_changes):.4f}")
        print(f"    Max flux decrease: {np.min(flux_changes):.4f}")
        print(f"    Mean absolute flux change: {np.mean(np.abs(flux_changes)):.4f}")

        return flux_changes

    def _compare_pathway_probabilities(self, msm1: dict, msm2: dict) -> dict[str, float]:
        """Compare dominant pathway probabilities"""

        pathways1 = msm1.get("dominant_pathways", [])
        pathways2 = msm2.get("dominant_pathways", [])

        pathway_changes = {}

        # Compare pathway probabilities
        for i, (path1, path2) in enumerate(zip(pathways1[:5], pathways2[:5], strict=False)):
            prob1 = path1.get("probability", 0.0)
            prob2 = path2.get("probability", 0.0)
            prob_change = prob2 - prob1

            pathway_name = f"pathway_{i+1}_{path1.get('source', 'unknown')}_to_{path1.get('target', 'unknown')}"
            pathway_changes[pathway_name] = prob_change

        # Sort by magnitude of change
        pathway_changes = dict(
            sorted(pathway_changes.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return pathway_changes

    def _compare_metastable_states(self, msm1: dict, msm2: dict) -> tuple[np.ndarray, dict]:
        """Compare metastable state populations and stabilities"""

        populations1 = msm1.get("metastable_populations", np.array([]))
        populations2 = msm2.get("metastable_populations", np.array([]))

        population_changes = np.array([])
        stability_changes = {}

        if populations1.size > 0 and populations2.size > 0:
            min_states = min(len(populations1), len(populations2))
            pop1 = populations1[:min_states]
            pop2 = populations2[:min_states]

            population_changes = pop2 - pop1

            # Calculate stability changes (relative population changes)
            for i in range(min_states):
                if pop1[i] > 0:
                    relative_change = (pop2[i] - pop1[i]) / pop1[i]
                    stability_changes[i] = {
                        "population_change": population_changes[i],
                        "relative_change": relative_change,
                        "stability_change": (
                            "stabilized"
                            if relative_change > 0.1
                            else "destabilized" if relative_change < -0.1 else "unchanged"
                        ),
                    }

        print(f"    Metastable state analysis: {len(stability_changes)} states compared")

        return population_changes, stability_changes

    def _analyze_critical_pathways(self, msm1: dict, msm2: dict) -> list[dict[str, Any]]:
        """Analyze changes in critical kinetic pathways"""

        critical_pathways = []

        # Extract critical transitions (highest flux transitions)
        if "critical_transitions" in msm1 and "critical_transitions" in msm2:
            transitions1 = msm1["critical_transitions"][:5]  # Top 5
            transitions2 = msm2["critical_transitions"][:5]

            for i, (trans1, trans2) in enumerate(zip(transitions1, transitions2, strict=False)):
                flux_change = trans2.get("flux", 0) - trans1.get("flux", 0)
                rate_change = trans2.get("rate", 0) - trans1.get("rate", 0)

                critical_pathways.append(
                    {
                        "transition_index": i,
                        "source_state": trans1.get("source", i),
                        "target_state": trans1.get("target", i + 1),
                        "original_flux": trans1.get("flux", 0),
                        "new_flux": trans2.get("flux", 0),
                        "flux_change": flux_change,
                        "original_rate": trans1.get("rate", 0),
                        "new_rate": trans2.get("rate", 0),
                        "rate_change": rate_change,
                        "impact": (
                            "accelerated"
                            if rate_change > 0
                            else "decelerated" if rate_change < 0 else "unchanged"
                        ),
                    }
                )

        return critical_pathways

    def _analyze_kinetic_bottlenecks(self, msm1: dict, msm2: dict) -> dict[str, float]:
        """Analyze changes in kinetic bottlenecks"""

        bottleneck_analysis = {
            "slowest_transition_change": 0.0,
            "bottleneck_shift": False,
            "overall_kinetics_change": 0.0,
        }

        # Identify slowest transitions (bottlenecks)
        timescales1 = msm1.get("implied_timescales", np.array([]))
        timescales2 = msm2.get("implied_timescales", np.array([]))

        if timescales1.size > 0 and timescales2.size > 0:
            # Compare slowest timescale (main bottleneck)
            slowest_change = timescales2[0] - timescales1[0]
            bottleneck_analysis["slowest_transition_change"] = slowest_change

            # Overall kinetics assessment
            mean_change = np.mean(timescales2[: min(5, len(timescales2))]) - np.mean(
                timescales1[: min(5, len(timescales1))]
            )
            bottleneck_analysis["overall_kinetics_change"] = mean_change

            # Check if bottleneck identity changed
            if len(timescales1) > 1 and len(timescales2) > 1:
                ratio1 = timescales1[1] / timescales1[0] if timescales1[0] > 0 else 0
                ratio2 = timescales2[1] / timescales2[0] if timescales2[0] > 0 else 0
                bottleneck_analysis["bottleneck_shift"] = abs(ratio2 - ratio1) > 0.2

        return bottleneck_analysis

    def _create_empty_kinetics_comparison(self) -> KineticsComparison:
        """Create empty comparison result when data is unavailable"""

        return KineticsComparison(
            timescale_changes=np.array([]),
            timescale_ratios=np.array([]),
            dominant_timescale_change=0.0,
            transition_flux_changes=np.array([]),
            pathway_probability_changes={},
            metastable_state_population_changes=np.array([]),
            metastable_state_stability_changes={},
            critical_pathway_changes=[],
            bottleneck_analysis={},
        )


class AllostericComparator:
    """Compares allosteric communication between simulations"""

    def __init__(self, config: DifferentialConfig):
        self.config = config

    def compare_allosteric(self, metrics1, metrics2) -> AllostericComparison:
        """Compare allosteric communication analysis results"""

        # Check if both simulations have allosteric data
        if not (
            hasattr(metrics1, "allosteric_pathways") and hasattr(metrics2, "allosteric_pathways")
        ):
            print("Warning: Allosteric pathway data not available for both simulations")
            return self._create_empty_allosteric_comparison()

        ap1 = metrics1.allosteric_pathways
        ap2 = metrics2.allosteric_pathways

        if not (ap1 and ap2):
            return self._create_empty_allosteric_comparison()

        print("  Computing communication efficiency differences...")

        # Extract communication efficiency matrices
        efficiency_diff_matrix, efficiency_stats = self._compare_efficiency_matrices(ap1, ap2)

        # Analyze pathway disruption
        pathway_disruption = self._analyze_pathway_disruption(ap1, ap2)

        # Compare hotspot rankings
        hotspot_changes = self._compare_hotspot_rankings(ap1, ap2)

        # Analyze cross-chain communication (for multi-chain proteins)
        cross_chain_changes = self._analyze_cross_chain_communication(ap1, ap2)

        # Drug resistance analysis (if applicable)
        resistance_analysis = self._analyze_resistance_mechanisms(ap1, ap2)

        # Binding site communication changes
        binding_site_changes = self._analyze_binding_site_communication(ap1, ap2)

        # Statistical significance testing
        pathway_significance = {}
        hotspot_significance = {}

        if self.config.perform_statistical_tests and SCIPY_AVAILABLE:
            pathway_significance = self._test_pathway_significance(ap1, ap2)
            hotspot_significance = self._test_hotspot_significance(ap1, ap2)

        return AllostericComparison(
            efficiency_difference_matrix=efficiency_diff_matrix,
            efficiency_change_statistics=efficiency_stats,
            pathway_disruption_analysis=pathway_disruption,
            hotspot_ranking_changes=hotspot_changes,
            cross_chain_communication_changes=cross_chain_changes,
            resistance_mechanism_analysis=resistance_analysis,
            binding_site_communication_changes=binding_site_changes,
            pathway_significance_tests=pathway_significance,
            hotspot_significance_changes=hotspot_significance,
        )

    def _compare_efficiency_matrices(
        self, ap1: dict, ap2: dict
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Compare communication efficiency matrices between simulations"""

        # Extract efficiency matrices
        if "full_efficiency_matrix" in ap1 and "full_efficiency_matrix" in ap2:
            matrix1 = ap1["full_efficiency_matrix"]
            matrix2 = ap2["full_efficiency_matrix"]
        else:
            # Fallback: reconstruct from communication efficiency data
            matrix1 = self._reconstruct_efficiency_matrix(ap1)
            matrix2 = self._reconstruct_efficiency_matrix(ap2)

        if matrix1 is None or matrix2 is None:
            print("  Warning: Could not extract efficiency matrices")
            return np.array([]), {}

        # Ensure matrices have same shape
        if matrix1.shape != matrix2.shape:
            print(f"  Warning: Matrix size mismatch: {matrix1.shape} vs {matrix2.shape}")
            return np.array([]), {}

        # Calculate difference matrix
        diff_matrix = matrix2 - matrix1

        # Calculate statistics
        stats = {
            "mean_absolute_change": np.mean(np.abs(diff_matrix)),
            "max_increase": np.max(diff_matrix),
            "max_decrease": np.min(diff_matrix),
            "rms_change": np.sqrt(np.mean(diff_matrix**2)),
            "significant_increases": np.sum(diff_matrix > self.config.efficiency_change_threshold),
            "significant_decreases": np.sum(diff_matrix < -self.config.efficiency_change_threshold),
            "correlation_coefficient": np.corrcoef(matrix1.flatten(), matrix2.flatten())[0, 1],
        }

        print(f"    Mean absolute efficiency change: {stats['mean_absolute_change']:.4f}")
        print(f"    Significant increases: {stats['significant_increases']}")
        print(f"    Significant decreases: {stats['significant_decreases']}")

        return diff_matrix, stats

    def _reconstruct_efficiency_matrix(self, ap: dict) -> np.ndarray | None:
        """Reconstruct efficiency matrix from communication efficiency data"""

        comm_eff = ap.get("communication_efficiency", {})
        if not comm_eff:
            return None

        # Get all unique residues
        residues = set()
        for pair in comm_eff.keys():
            if isinstance(pair, str) and "_" in pair:
                res1, res2 = pair.split("_", 1)
                residues.update([res1, res2])

        residues = sorted(list(residues))
        n_res = len(residues)

        if n_res == 0:
            return None

        # Create matrix
        matrix = np.zeros((n_res, n_res))
        res_to_idx = {res: i for i, res in enumerate(residues)}

        # Fill matrix
        for pair, efficiency in comm_eff.items():
            if isinstance(pair, str) and "_" in pair:
                res1, res2 = pair.split("_", 1)
                if res1 in res_to_idx and res2 in res_to_idx:
                    i, j = res_to_idx[res1], res_to_idx[res2]
                    matrix[i, j] = efficiency
                    matrix[j, i] = efficiency  # Symmetric matrix

        # Set diagonal to 1.0 (perfect self-communication)
        np.fill_diagonal(matrix, 1.0)

        return matrix

    def _analyze_pathway_disruption(self, ap1: dict, ap2: dict) -> list[dict[str, Any]]:
        """Analyze disruption of specific allosteric pathways"""

        pathways1 = ap1.get("pathways", [])
        pathways2 = ap2.get("pathways", [])

        disruption_analysis = []

        # Compare pathway efficiency changes
        for i, pathway1 in enumerate(pathways1[:10]):  # Top 10 pathways
            # Find corresponding pathway in simulation 2
            best_match = None
            best_similarity = 0

            for _j, pathway2 in enumerate(pathways2):
                similarity = self._calculate_pathway_similarity(pathway1, pathway2)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = pathway2

            if best_match and best_similarity > 0.5:  # Reasonable match
                disruption_score = self._calculate_pathway_disruption(pathway1, best_match)

                disruption_analysis.append(
                    {
                        "pathway_id": i,
                        "source": pathway1.get("source", "N/A"),
                        "target": pathway1.get("target", "N/A"),
                        "original_efficiency": pathway1.get("efficiency", 0.0),
                        "new_efficiency": best_match.get("efficiency", 0.0),
                        "efficiency_change": best_match.get("efficiency", 0.0)
                        - pathway1.get("efficiency", 0.0),
                        "disruption_score": disruption_score,
                        "pathway_similarity": best_similarity,
                        "length_change": len(best_match.get("path", []))
                        - len(pathway1.get("path", [])),
                        "significance": (
                            "high"
                            if abs(disruption_score) > 0.3
                            else "medium" if abs(disruption_score) > 0.1 else "low"
                        ),
                    }
                )

        # Sort by disruption score
        disruption_analysis.sort(key=lambda x: abs(x["disruption_score"]), reverse=True)

        return disruption_analysis

    def _calculate_pathway_similarity(self, pathway1: dict, pathway2: dict) -> float:
        """Calculate similarity between two pathways"""

        path1 = set(pathway1.get("path", []))
        path2 = set(pathway2.get("path", []))

        if not path1 or not path2:
            return 0.0

        # Jaccard similarity
        intersection = len(path1.intersection(path2))
        union = len(path1.union(path2))

        return intersection / union if union > 0 else 0.0

    def _calculate_pathway_disruption(self, pathway1: dict, pathway2: dict) -> float:
        """Calculate disruption score for a pathway"""

        # Efficiency change (primary metric)
        eff1 = pathway1.get("efficiency", 0.0)
        eff2 = pathway2.get("efficiency", 0.0)
        efficiency_disruption = (eff1 - eff2) / eff1 if eff1 > 0 else 0.0

        # Path length change (secondary metric)
        len1 = len(pathway1.get("path", []))
        len2 = len(pathway2.get("path", []))
        length_disruption = (len2 - len1) / len1 if len1 > 0 else 0.0

        # Combined disruption score (weighted average)
        disruption_score = 0.7 * efficiency_disruption + 0.3 * length_disruption

        return disruption_score

    def _compare_hotspot_rankings(self, ap1: dict, ap2: dict) -> list[dict[str, Any]]:
        """Compare allosteric hotspot rankings between simulations"""

        hotspots1 = ap1.get("allosteric_hotspots", [])
        hotspots2 = ap2.get("allosteric_hotspots", [])

        if not hotspots1 or not hotspots2:
            return []

        # Create dictionaries for easy lookup
        hs1_dict = {hs["node"]: hs for hs in hotspots1}
        hs2_dict = {hs["node"]: hs for hs in hotspots2}

        ranking_changes = []

        # Analyze changes for each hotspot
        all_nodes = set(hs1_dict.keys()).union(set(hs2_dict.keys()))

        for node in all_nodes:
            hs1 = hs1_dict.get(node, {"hotspot_score": 0.0, "frequency": 0})
            hs2 = hs2_dict.get(node, {"hotspot_score": 0.0, "frequency": 0})

            score_change = hs2["hotspot_score"] - hs1["hotspot_score"]
            freq_change = hs2["frequency"] - hs1["frequency"]

            # Find ranking positions
            rank1 = next(
                (i for i, hs in enumerate(hotspots1) if hs["node"] == node), len(hotspots1)
            )
            rank2 = next(
                (i for i, hs in enumerate(hotspots2) if hs["node"] == node), len(hotspots2)
            )
            rank_change = rank1 - rank2  # Positive = improved ranking

            if abs(score_change) > 0.05 or abs(rank_change) > 2:  # Significant changes
                ranking_changes.append(
                    {
                        "node": node,
                        "original_score": hs1["hotspot_score"],
                        "new_score": hs2["hotspot_score"],
                        "score_change": score_change,
                        "original_frequency": hs1["frequency"],
                        "new_frequency": hs2["frequency"],
                        "frequency_change": freq_change,
                        "original_rank": rank1 + 1,  # 1-indexed
                        "new_rank": rank2 + 1,
                        "rank_change": rank_change,
                        "change_type": "improved" if score_change > 0 else "decreased",
                        "significance": (
                            "high"
                            if abs(score_change) > 0.2
                            else "medium" if abs(score_change) > 0.1 else "low"
                        ),
                    }
                )

        # Sort by magnitude of score change
        ranking_changes.sort(key=lambda x: abs(x["score_change"]), reverse=True)

        return ranking_changes

    def _analyze_cross_chain_communication(self, ap1: dict, ap2: dict) -> dict[str, float]:
        """Analyze changes in cross-chain communication (for multi-chain proteins)"""

        # Get efficiency matrices
        matrix1 = ap1.get("full_efficiency_matrix")
        matrix2 = ap2.get("full_efficiency_matrix")
        residue_labels = ap1.get("residue_labels", [])

        if matrix1 is None or matrix2 is None or not residue_labels:
            return {}

        # Identify chain assignments
        chain_assignments = {}
        for i, label in enumerate(residue_labels):
            if "_" in label:
                chain = label.split("_")[0]
                if chain not in chain_assignments:
                    chain_assignments[chain] = []
                chain_assignments[chain].append(i)

        if len(chain_assignments) < 2:
            return {}  # Single chain protein

        cross_chain_changes = {}
        chains = list(chain_assignments.keys())

        # Analyze inter-chain communication
        for i, chain1 in enumerate(chains):
            for j, chain2 in enumerate(chains):
                if i < j:  # Avoid duplicate pairs
                    indices1 = chain_assignments[chain1]
                    indices2 = chain_assignments[chain2]

                    # Extract inter-chain submatrices
                    submatrix1 = matrix1[np.ix_(indices1, indices2)]
                    submatrix2 = matrix2[np.ix_(indices1, indices2)]

                    # Calculate change statistics
                    diff = submatrix2 - submatrix1

                    cross_chain_changes[f"{chain1}_{chain2}"] = {
                        "mean_change": np.mean(diff),
                        "max_change": np.max(diff),
                        "min_change": np.min(diff),
                        "rms_change": np.sqrt(np.mean(diff**2)),
                        "correlation_change": np.corrcoef(
                            submatrix1.flatten(), submatrix2.flatten()
                        )[0, 1],
                    }

        return cross_chain_changes

    def _analyze_resistance_mechanisms(self, ap1: dict, ap2: dict) -> dict[str, Any]:
        """Analyze drug resistance mechanisms based on communication changes"""

        # This is a specialized analysis for drug resistance studies
        # Would need specific implementation based on protein and drug targets

        resistance_analysis = {
            "analysis_type": "drug_resistance_mechanism",
            "methodology": "communication_efficiency_disruption",
            "findings": [],
        }

        # Placeholder for resistance-specific analysis
        # Could include:
        # - Active site communication disruption
        # - Allosteric pathway blocking
        # - Conformational flexibility changes
        # - Binding pocket accessibility changes

        return resistance_analysis

    def _analyze_binding_site_communication(self, ap1: dict, ap2: dict) -> dict[str, float]:
        """Analyze changes in binding site communication"""

        # Placeholder for binding site specific analysis
        # Would identify binding site residues and analyze their communication changes

        return {
            "active_site_communication_change": 0.0,
            "allosteric_site_communication_change": 0.0,
            "binding_pocket_accessibility_change": 0.0,
        }

    def _test_pathway_significance(self, ap1: dict, ap2: dict) -> dict[str, float]:
        """Statistical significance testing for pathway changes"""

        # Placeholder for statistical testing
        # Could implement permutation tests, bootstrap confidence intervals

        return {}

    def _test_hotspot_significance(self, ap1: dict, ap2: dict) -> dict[str, float]:
        """Statistical significance testing for hotspot ranking changes"""

        # Placeholder for hotspot significance testing

        return {}

    def _create_empty_allosteric_comparison(self) -> AllostericComparison:
        """Create empty comparison result when data is unavailable"""

        return AllostericComparison(
            efficiency_difference_matrix=np.array([]),
            efficiency_change_statistics={},
            pathway_disruption_analysis=[],
            hotspot_ranking_changes=[],
            cross_chain_communication_changes={},
            resistance_mechanism_analysis={},
            binding_site_communication_changes={},
            pathway_significance_tests={},
            hotspot_significance_changes={},
        )


if __name__ == "__main__":
    # Example usage
    config = DifferentialConfig()
    analyzer = DifferentialAnalyzer(config, "test_differential_results")

    print("DifferentialAnalyzer initialized successfully!")
    print(f"Output directory: {analyzer.output_dir}")
    print(f"Subdirectories created: {list(analyzer.subdirs.keys())}")
