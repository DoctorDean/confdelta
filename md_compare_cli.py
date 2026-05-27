#!/usr/bin/env python3

"""
MD-Compare: Command-line interface for molecular dynamics comparison analysis

This script provides the main command-line interface for comparing molecular dynamics
simulations using network analysis methods.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

try:
    from md_compare_core import (
        MDCompare, SimulationConfig, AnalysisConfig, 
        MDSimulation, NetworkAnalyzer, MDComparator
    )
    from md_compare_differential import DifferentialAnalyzer, DifferentialConfig
    from utils import PerformanceMonitor, save_analysis_config
except ImportError:
    print("Error: MD-Compare core modules not found. Ensure proper installation.")
    sys.exit(1)


def create_simulation_config_from_args(name: str, topology: str, trajectory: str, 
                                     description: str = "", selection: str = "protein and not name H*") -> SimulationConfig:
    """Create simulation configuration from command line arguments"""
    return SimulationConfig(
        name=name,
        topology=topology, 
        trajectory=trajectory,
        selection=selection,
        description=description
    )


def create_analysis_config_from_args(args) -> AnalysisConfig:
    """Create analysis configuration from command line arguments"""
    return AnalysisConfig(
        cutoffs={'all_atom': args.cutoff, 'ca_only': args.cutoff * 1.8},
        interaction_types=args.interaction_types,
        threshold=args.threshold,
        timeout_seconds=args.timeout,
        segments=args.segments,
        preprocess=not args.no_preprocess,
        align_selection=args.align_selection,
        center_selection=args.center_selection,
        compute_dccm=args.compute_dccm,
        compute_pca=args.compute_pca,
        pca_components=args.pca_components,
        dccm_selection=args.dccm_selection,
        compute_energy_landscape=args.compute_landscape,
        landscape_temperature=args.landscape_temp,
        landscape_bins=args.landscape_bins,
        landscape_sigma=args.landscape_sigma,
        compute_communities=getattr(args, 'compute_communities', True),
        community_method=getattr(args, 'community_method', 'leiden'),
        compute_paths=getattr(args, 'compute_paths', True),
        allosteric_analysis=getattr(args, 'allosteric_analysis', True),
        allosteric_source_nodes=getattr(args, 'allosteric_sources', None),
        allosteric_target_nodes=getattr(args, 'allosteric_targets', None),
        # PyEMMA MSM options
        compute_msm=getattr(args, 'compute_msm', True),
        msm_lag_time=getattr(args, 'msm_lag_time', 10),
        msm_n_clusters=getattr(args, 'msm_clusters', 100),
        msm_stride=getattr(args, 'msm_stride', 1),
        msm_feature_type=getattr(args, 'msm_features', 'distances'),
        msm_clustering_method=getattr(args, 'msm_clustering', 'kmeans'),
        compute_kinetics=getattr(args, 'compute_kinetics', True),
        kinetic_timescales_count=getattr(args, 'kinetic_timescales', 5),
        compute_metastable_states=getattr(args, 'compute_metastable_states', True),
        metastable_state_count=getattr(args, 'metastable_states', 5)
    )


def run_single_analysis(args):
    """Run analysis on a single simulation"""
    print("Running single simulation analysis...")
    
    # Create configurations
    sim_config = create_simulation_config_from_args(
        name=args.name,
        topology=args.topology,
        trajectory=args.trajectory,
        description=f"Single simulation analysis: {args.name}",
        selection=args.selection
    )
    
    analysis_config = create_analysis_config_from_args(args)
    
    # Initialize MD-Compare workflow
    md_compare = MDCompare(analysis_config, args.output)
    
    # Performance monitoring
    monitor = PerformanceMonitor()
    
    try:
        # Add simulation
        monitor.start_step("Loading simulation")
        success = md_compare.add_simulation(sim_config)
        monitor.end_step()
        
        if not success:
            print(f"Failed to load simulation: {args.name}")
            return 1
        
        # Run analysis
        monitor.start_step("Computing contact maps")
        analysis_results = md_compare.run_analysis([args.name])
        monitor.end_step()
        
        # Print results summary
        if args.name in analysis_results:
            results = analysis_results[args.name]
            print(f"\nAnalysis Results for {args.name}:")
            print(f"  Contact maps: {results['contact_maps_computed']}")
            print(f"  Network nodes: {results['network_nodes']}")
            print(f"  Network edges: {results['network_edges']}")
            print(f"  Network density: {results['network_density']:.4f}")
        
        # Save configuration
        config_path = Path(args.output) / "analysis_config.json"
        save_analysis_config(analysis_config, config_path)
        
        monitor.print_summary()
        print(f"\nResults saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_comparison_analysis(args):
    """Run comparison analysis on multiple simulations"""
    print("Running multi-simulation comparison analysis...")
    
    # Parse simulation configurations from JSON file
    try:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return 1
    
    # Create simulation configurations
    sim_configs = []
    for sim_data in config_data.get('simulations', []):
        sim_config = SimulationConfig(
            name=sim_data['name'],
            topology=sim_data['topology'],
            trajectory=sim_data['trajectory'],
            selection=sim_data.get('selection', 'protein and not name H*'),
            description=sim_data.get('description', ''),
            metadata=sim_data.get('metadata', {})
        )
        sim_configs.append(sim_config)
    
    if len(sim_configs) < 2:
        print("Error: Need at least 2 simulations for comparison")
        return 1
    
    # Create analysis configuration
    analysis_config = create_analysis_config_from_args(args)
    
    # Override with config file settings if provided
    if 'analysis' in config_data:
        analysis_data = config_data['analysis']
        for key, value in analysis_data.items():
            if hasattr(analysis_config, key):
                setattr(analysis_config, key, value)
    
    # Initialize MD-Compare workflow
    md_compare = MDCompare(analysis_config, args.output)
    
    # Performance monitoring
    monitor = PerformanceMonitor()
    
    try:
        # Run full workflow
        monitor.start_step("Full workflow")
        results = md_compare.run_full_workflow(sim_configs)
        monitor.end_step()
        
        # Print summary
        print(f"\nComparison Analysis Complete:")
        print(f"  Simulations analyzed: {len(results['simulations_analyzed'])}")
        
        if 'comparative_analysis' in results:
            comp_analysis = results['comparative_analysis']
            if 'network_properties' in comp_analysis:
                print(f"  Network property comparison: Available")
            if 'centrality_measures' in comp_analysis:
                print(f"  Centrality comparison: Available")
            if 'contact_patterns' in comp_analysis:
                print(f"  Contact pattern comparison: Available")
        
        monitor.print_summary()
        print(f"\nResults saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_comprehensive_differential_analysis(args):
    """Run comprehensive differential analysis between two simulations"""
    print(f"Running comprehensive differential analysis: {args.sim1} vs {args.sim2}")
    print("=" * 70)
    
    # Create differential configuration
    diff_config = DifferentialConfig(
        compare_networks=not args.no_network_comparison,
        compare_dynamics=not args.no_dynamics_comparison,
        compare_energetics=not args.no_energetics_comparison,
        compare_kinetics=not args.no_kinetics_comparison,
        compare_allosteric=not args.no_allosteric_comparison,
        perform_statistical_tests=args.statistical_tests,
        significance_threshold=args.significance_threshold,
        multiple_comparison_correction=args.multiple_comparison_correction,
        bootstrap_iterations=args.bootstrap_iterations,
        create_difference_plots=not args.no_plots,
        create_publication_figures=args.publication_figures,
        figure_dpi=args.figure_dpi,
        create_html_report=not args.no_html_report,
        export_excel_workbook=args.excel_export,
        correlation_change_threshold=args.correlation_threshold,
        efficiency_change_threshold=args.efficiency_threshold,
        energy_change_threshold=args.energy_threshold
    )
    
    # Create analysis configuration
    analysis_config = create_analysis_config_from_args(args)
    
    # Initialize differential analyzer
    differential_analyzer = DifferentialAnalyzer(diff_config, args.output)
    
    monitor = PerformanceMonitor()
    
    try:
        monitor.start_step("Comprehensive differential analysis")
        
        # Run comprehensive differential analysis
        results = differential_analyzer.run_differential_analysis(
            args.topology1, args.trajectory1, args.sim1,
            args.topology2, args.trajectory2, args.sim2,
            analysis_config
        )
        
        monitor.end_step()
        
        # Print executive summary
        print("\n" + "="*70)
        print("EXECUTIVE SUMMARY")
        print("="*70)
        
        exec_summary = results.executive_summary
        print(f"Comparison: {exec_summary['comparison']}")
        print(f"Total analyses performed: {exec_summary['total_analyses']}")
        print(f"Significant findings: {exec_summary['significant_findings']}")
        print(f"Overall similarity score: {exec_summary['overall_similarity']:.3f}")
        
        print(f"\nKey insights:")
        for insight in exec_summary['key_insights']:
            print(f"  • {insight}")
        
        print(f"\nSimilarity scores by analysis type:")
        for analysis_type, score in results.overall_similarity_scores.items():
            print(f"  • {analysis_type.title()}: {score:.3f}")
        
        print(f"\nMost significant changes:")
        for analysis_type, changes in results.most_significant_changes.items():
            if changes:
                print(f"  • {analysis_type.title()}: {len(changes)} significant changes")
                top_change = changes[0] if changes else None
                if top_change:
                    print(f"    - {top_change.get('description', 'N/A')} (p={top_change.get('significance', 'N/A')})")
        
        monitor.print_summary()
        print(f"\nComplete results available in: {args.output}")
        print(f"Individual analyses: {args.output}/01_individual_analyses/")
        print(f"Differential results: {args.output}/02_network_comparisons/ through 06_allosteric_comparisons/")
        print(f"Comprehensive report: {args.output}/07_comprehensive_report/")
        
        return 0
        
    except Exception as e:
        print(f"Error during comprehensive differential analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_differential_analysis(args):
    """Run basic differential analysis between two specific simulations (legacy)"""
    print(f"Running basic differential analysis: {args.sim1} vs {args.sim2}")
    print("Note: For comprehensive differential analysis, use 'differential' command instead of 'diff'")
    
    # Load both simulations
    sim1_config = create_simulation_config_from_args(
        name=args.sim1,
        topology=args.topology1,
        trajectory=args.trajectory1,
        selection=args.selection
    )
    
    sim2_config = create_simulation_config_from_args(
        name=args.sim2,
        topology=args.topology2,
        trajectory=args.trajectory2,
        selection=args.selection
    )
    
    analysis_config = create_analysis_config_from_args(args)
    
    # Initialize workflow
    md_compare = MDCompare(analysis_config, args.output)
    
    monitor = PerformanceMonitor()
    
    try:
        # Add simulations
        monitor.start_step("Loading simulations")
        success1 = md_compare.add_simulation(sim1_config)
        success2 = md_compare.add_simulation(sim2_config)
        monitor.end_step()
        
        if not (success1 and success2):
            print("Failed to load one or both simulations")
            return 1
        
        # Run analyses
        monitor.start_step("Running analyses")
        md_compare.run_analysis([args.sim1, args.sim2])
        monitor.end_step()
        
        # Run comparison
        monitor.start_step("Running comparison")
        comparison_results = md_compare.run_comparison([args.sim1, args.sim2])
        monitor.end_step()
        
        # Run differential analysis
        monitor.start_step("Differential analysis")
        differential_results = md_compare.comparator.find_differential_contacts(
            args.sim1, args.sim2, args.diff_threshold
        )
        
        # Save differential results
        diff_output = Path(args.output) / f"differential_{args.sim1}_vs_{args.sim2}.json"
        with open(diff_output, 'w') as f:
            json.dump(differential_results, f, indent=2, default=str)
        
        monitor.end_step()
        
        # Print summary
        print(f"\nBasic Differential Analysis Results:")
        print(f"  Increased contacts: {differential_results['n_increased']}")
        print(f"  Decreased contacts: {differential_results['n_decreased']}")
        print(f"  Threshold: {differential_results['threshold_difference']}")
        
        if differential_results['n_increased'] > 0:
            top_increase = differential_results['increased_contacts'][0]
            print(f"  Top increase: {top_increase['residue_pair']} (+{top_increase['difference']:.3f})")
        
        if differential_results['n_decreased'] > 0:
            top_decrease = differential_results['decreased_contacts'][0]
            print(f"  Top decrease: {top_decrease['residue_pair']} ({top_decrease['difference']:.3f})")
        
        monitor.print_summary()
        print(f"\nResults saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error during differential analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


def create_example_config(output_path: str):
    """Create an example configuration file"""
    example_config = {
        "simulations": [
            {
                "name": "wildtype",
                "topology": "wt_system.pdb",
                "trajectory": "wt_trajectory.xtc",
                "selection": "protein and not name H*",
                "description": "Wild-type simulation",
                "metadata": {
                    "temperature": 300,
                    "pressure": 1.0,
                    "force_field": "CHARMM36"
                }
            },
            {
                "name": "mutant",
                "topology": "mut_system.pdb", 
                "trajectory": "mut_trajectory.xtc",
                "selection": "protein and not name H*",
                "description": "Mutant simulation",
                "metadata": {
                    "mutations": ["L10F", "G48V"],
                    "temperature": 300,
                    "pressure": 1.0,
                    "force_field": "CHARMM36"
                }
            }
        ],
        "analysis": {
            "cutoffs": {
                "all_atom": 4.5,
                "ca_only": 8.0
            },
            "interaction_types": ["distance", "hbond"],
            "threshold": 0.2,
            "timeout_seconds": 300,
            "segments": 5,
            "preprocess": true,
            "align_selection": "name CA",
            "center_selection": "protein"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print(f"Example configuration saved to: {output_path}")


def main():
    """Main entry point for MD-Compare CLI"""
    
    # Main parser
    parser = argparse.ArgumentParser(
        description="MD-Compare: Comprehensive toolkit for comparing molecular dynamics simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Single simulation analysis  
  md-compare single -t system.pdb -x trajectory.xtc -n my_sim -o results/

  # Multiple simulation comparison
  md-compare compare -c simulations.json -o comparison_results/

  # Basic differential analysis (legacy)
  md-compare diff -t1 wt.pdb -x1 wt.xtc -n1 wildtype \\
                  -t2 mut.pdb -x2 mut.xtc -n2 mutant -o diff_results/

  # Comprehensive differential analysis (recommended)
  md-compare differential -t1 wt.pdb -x1 wt.xtc -n1 "Wild_Type" \\
                         -t2 mut.pdb -x2 mut.xtc -n2 "V82A_Mutant" \\
                         -o hiv_wt_vs_v82a --statistical-tests --publication-figures \\
                         --allosteric-sources A_50 B_50 --allosteric-targets A_25 B_25

  # Drug resistance analysis with focused comparisons
  md-compare differential -t1 hiv_apo.pdb -x1 apo.xtc -n1 "Apo_State" \\
                         -t2 hiv_inhibitor.pdb -x2 inhibitor.xtc -n2 "Inhibitor_Bound" \\
                         -o drug_binding_analysis --excel-export \\
                         --correlation-threshold 0.15 --efficiency-threshold 0.05

  # Generate example configuration
  md-compare example-config -o example_simulations.json

For more information, visit: https://github.com/yourusername/md-compare
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Analysis mode')
    
    # Single simulation analysis
    single_parser = subparsers.add_parser(
        'single', 
        help='Analyze a single simulation'
    )
    single_parser.add_argument('-t', '--topology', required=True,
                              help='Topology file (PDB, GRO, etc.)')
    single_parser.add_argument('-x', '--trajectory', required=True,
                              help='Trajectory file (XTC, DCD, etc.)')
    single_parser.add_argument('-n', '--name', required=True,
                              help='Name for this simulation')
    single_parser.add_argument('-o', '--output', default='md_compare_results',
                              help='Output directory')
    single_parser.add_argument('--selection', default='protein and not name H*',
                              help='Atom selection string')
    single_parser.add_argument('--cutoff', type=float, default=4.5,
                              help='Contact distance cutoff (Å)')
    single_parser.add_argument('--threshold', type=float, default=0.2,
                              help='Contact frequency threshold for network edges')
    single_parser.add_argument('--interaction-types', nargs='+',
                              choices=['distance', 'hbond', 'salt_bridge'],
                              default=['distance'],
                              help='Types of interactions to analyze')
    single_parser.add_argument('--segments', type=int, default=5,
                              help='Number of trajectory segments for statistical analysis')
    single_parser.add_argument('--timeout', type=int, default=300,
                              help='Timeout for expensive computations (seconds)')
    single_parser.add_argument('--no-preprocess', action='store_true',
                              help='Disable MD preprocessing')
    single_parser.add_argument('--align-selection', default='name CA',
                              help='Selection for structural alignment')
    single_parser.add_argument('--center-selection', default='protein',
                              help='Selection for centering')
    single_parser.add_argument('--compute-dccm', action='store_true', default=True,
                              help='Compute Dynamic Cross-Correlation Matrix (default: True)')
    single_parser.add_argument('--no-dccm', dest='compute_dccm', action='store_false',
                              help='Skip DCCM computation')
    single_parser.add_argument('--compute-pca', action='store_true', default=True,
                              help='Compute Principal Component Analysis (default: True)')
    single_parser.add_argument('--no-pca', dest='compute_pca', action='store_false',
                              help='Skip PCA computation')
    single_parser.add_argument('--pca-components', type=int, default=10,
                              help='Number of principal components to compute (default: 10)')
    single_parser.add_argument('--dccm-selection', default='name CA',
                              help='Atom selection for dynamic analysis (default: "name CA")')
    single_parser.add_argument('--compute-landscape', action='store_true', default=True,
                              help='Compute energy landscape from PC1/PC2 (default: True)')
    single_parser.add_argument('--no-landscape', dest='compute_landscape', action='store_false',
                              help='Skip energy landscape computation')
    single_parser.add_argument('--landscape-temp', type=float, default=310.0,
                              help='Temperature for energy landscape (K, default: 310)')
    single_parser.add_argument('--landscape-bins', type=int, default=50,
                              help='Number of bins for energy landscape (default: 50)')
    single_parser.add_argument('--landscape-sigma', type=float, default=1.0,
                              help='Gaussian smoothing sigma for landscape (default: 1.0)')
    single_parser.add_argument('--community-method', default='leiden',
                              choices=['leiden', 'louvain', 'spectral', 'hierarchical'],
                              help='Community detection method (default: leiden)')
    single_parser.add_argument('--no-communities', dest='compute_communities', action='store_false',
                              help='Skip community detection')
    single_parser.add_argument('--no-paths', dest='compute_paths', action='store_false', 
                              help='Skip detailed path analysis')
    single_parser.add_argument('--no-allosteric', dest='allosteric_analysis', action='store_false',
                              help='Skip allosteric pathway analysis')
    single_parser.add_argument('--allosteric-sources', nargs='+', 
                              help='Specific residues to use as allosteric sources (e.g., A_50 B_50)')
    single_parser.add_argument('--allosteric-targets', nargs='+',
                              help='Specific residues to use as allosteric targets (e.g., A_25 B_25)')
    # PyEMMA Markov State Model options
    single_parser.add_argument('--compute-msm', dest='compute_msm', action='store_true', default=True,
                              help='Compute Markov State Model analysis (default: enabled)')
    single_parser.add_argument('--no-msm', dest='compute_msm', action='store_false',
                              help='Skip Markov State Model analysis')
    single_parser.add_argument('--msm-lag-time', type=int, default=10,
                              help='Lag time for MSM construction in frames (default: 10)')
    single_parser.add_argument('--msm-clusters', type=int, default=100,
                              help='Number of clusters for MSM discretization (default: 100)')
    single_parser.add_argument('--msm-stride', type=int, default=1,
                              help='Stride for coordinate extraction (default: 1)')
    single_parser.add_argument('--msm-features', default='distances',
                              choices=['distances', 'coordinates', 'angles', 'dihedrals'],
                              help='Feature type for MSM (default: distances)')
    single_parser.add_argument('--msm-clustering', default='kmeans',
                              choices=['kmeans', 'minibatch_kmeans', 'regular_space'],
                              help='Clustering method for MSM (default: kmeans)')
    single_parser.add_argument('--no-kinetics', dest='compute_kinetics', action='store_false',
                              help='Skip kinetic analysis')
    single_parser.add_argument('--kinetic-timescales', type=int, default=5,
                              help='Number of kinetic timescales to compute (default: 5)')
    single_parser.add_argument('--no-metastable', dest='compute_metastable_states', action='store_false',
                              help='Skip metastable state analysis')
    single_parser.add_argument('--metastable-states', type=int, default=5,
                              help='Number of metastable states for PCCA+ (default: 5)')
    
    # Multiple simulation comparison
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare multiple simulations'
    )
    compare_parser.add_argument('-c', '--config', required=True,
                               help='JSON configuration file with simulation details')
    compare_parser.add_argument('-o', '--output', default='md_compare_results',
                               help='Output directory')
    compare_parser.add_argument('--cutoff', type=float, default=4.5,
                               help='Contact distance cutoff (Å)')
    compare_parser.add_argument('--threshold', type=float, default=0.2,
                               help='Contact frequency threshold')
    compare_parser.add_argument('--interaction-types', nargs='+',
                               choices=['distance', 'hbond', 'salt_bridge'],
                               default=['distance'],
                               help='Types of interactions to analyze')
    compare_parser.add_argument('--segments', type=int, default=5,
                               help='Number of trajectory segments')
    compare_parser.add_argument('--timeout', type=int, default=300,
                               help='Timeout for expensive computations (seconds)')
    compare_parser.add_argument('--no-preprocess', action='store_true',
                               help='Disable MD preprocessing')
    compare_parser.add_argument('--align-selection', default='name CA',
                               help='Selection for structural alignment')
    compare_parser.add_argument('--center-selection', default='protein',
                               help='Selection for centering')
    compare_parser.add_argument('--compute-dccm', action='store_true', default=True,
                               help='Compute Dynamic Cross-Correlation Matrix (default: True)')
    compare_parser.add_argument('--no-dccm', dest='compute_dccm', action='store_false',
                               help='Skip DCCM computation')
    compare_parser.add_argument('--compute-pca', action='store_true', default=True,
                               help='Compute Principal Component Analysis (default: True)')
    compare_parser.add_argument('--no-pca', dest='compute_pca', action='store_false',
                               help='Skip PCA computation')
    compare_parser.add_argument('--pca-components', type=int, default=10,
                               help='Number of principal components to compute (default: 10)')
    compare_parser.add_argument('--dccm-selection', default='name CA',
                               help='Atom selection for dynamic analysis (default: "name CA")')
    compare_parser.add_argument('--compute-landscape', action='store_true', default=True,
                               help='Compute energy landscape from PC1/PC2 (default: True)')
    compare_parser.add_argument('--no-landscape', dest='compute_landscape', action='store_false',
                               help='Skip energy landscape computation')
    compare_parser.add_argument('--landscape-temp', type=float, default=310.0,
                               help='Temperature for energy landscape (K, default: 310)')
    compare_parser.add_argument('--landscape-bins', type=int, default=50,
                               help='Number of bins for energy landscape (default: 50)')
    compare_parser.add_argument('--landscape-sigma', type=float, default=1.0,
                               help='Gaussian smoothing sigma for landscape (default: 1.0)')
    compare_parser.add_argument('--community-method', default='leiden',
                               choices=['leiden', 'louvain', 'spectral', 'hierarchical'],
                               help='Community detection method (default: leiden)')
    compare_parser.add_argument('--no-communities', dest='compute_communities', action='store_false',
                               help='Skip community detection')
    compare_parser.add_argument('--no-paths', dest='compute_paths', action='store_false',
                               help='Skip detailed path analysis') 
    compare_parser.add_argument('--no-allosteric', dest='allosteric_analysis', action='store_false',
                               help='Skip allosteric pathway analysis')
    compare_parser.add_argument('--allosteric-sources', nargs='+',
                               help='Specific residues to use as allosteric sources (e.g., A_50 B_50)')
    compare_parser.add_argument('--allosteric-targets', nargs='+',
                               help='Specific residues to use as allosteric targets (e.g., A_25 B_25)')
    # PyEMMA options (same as single)
    compare_parser.add_argument('--compute-msm', dest='compute_msm', action='store_true', default=True,
                               help='Compute Markov State Model analysis (default: enabled)')
    compare_parser.add_argument('--no-msm', dest='compute_msm', action='store_false',
                               help='Skip Markov State Model analysis')
    compare_parser.add_argument('--msm-lag-time', type=int, default=10,
                               help='Lag time for MSM construction in frames (default: 10)')
    compare_parser.add_argument('--msm-clusters', type=int, default=100,
                               help='Number of clusters for MSM discretization (default: 100)')
    compare_parser.add_argument('--msm-stride', type=int, default=1,
                               help='Stride for coordinate extraction (default: 1)')
    compare_parser.add_argument('--msm-features', default='distances',
                               choices=['distances', 'coordinates', 'angles', 'dihedrals'],
                               help='Feature type for MSM (default: distances)')
    compare_parser.add_argument('--msm-clustering', default='kmeans',
                               choices=['kmeans', 'minibatch_kmeans', 'regular_space'],
                               help='Clustering method for MSM (default: kmeans)')
    compare_parser.add_argument('--no-kinetics', dest='compute_kinetics', action='store_false',
                               help='Skip kinetic analysis')
    compare_parser.add_argument('--kinetic-timescales', type=int, default=5,
                               help='Number of kinetic timescales to compute (default: 5)')
    compare_parser.add_argument('--no-metastable', dest='compute_metastable_states', action='store_false',
                               help='Skip metastable state analysis')
    compare_parser.add_argument('--metastable-states', type=int, default=5,
                               help='Number of metastable states for PCCA+ (default: 5)')
    
    # Differential analysis
    diff_parser = subparsers.add_parser(
        'diff',
        help='Differential analysis between two simulations'
    )
    diff_parser.add_argument('-t1', '--topology1', required=True,
                            help='Topology file for first simulation')
    diff_parser.add_argument('-x1', '--trajectory1', required=True,
                            help='Trajectory file for first simulation')
    diff_parser.add_argument('-n1', '--sim1', required=True,
                            help='Name for first simulation')
    diff_parser.add_argument('-t2', '--topology2', required=True,
                            help='Topology file for second simulation')
    diff_parser.add_argument('-x2', '--trajectory2', required=True,
                            help='Trajectory file for second simulation')
    diff_parser.add_argument('-n2', '--sim2', required=True,
                            help='Name for second simulation')
    diff_parser.add_argument('-o', '--output', default='md_compare_results',
                            help='Output directory')
    diff_parser.add_argument('--selection', default='protein and not name H*',
                            help='Atom selection string')
    diff_parser.add_argument('--cutoff', type=float, default=4.5,
                            help='Contact distance cutoff (Å)')
    diff_parser.add_argument('--threshold', type=float, default=0.2,
                            help='Contact frequency threshold')
    diff_parser.add_argument('--diff-threshold', type=float, default=0.1,
                            help='Minimum difference for significant contacts')
    diff_parser.add_argument('--interaction-types', nargs='+',
                            choices=['distance', 'hbond', 'salt_bridge'],
                            default=['distance'],
                            help='Types of interactions to analyze')
    diff_parser.add_argument('--segments', type=int, default=5,
                            help='Number of trajectory segments')
    diff_parser.add_argument('--timeout', type=int, default=300,
                            help='Timeout for expensive computations (seconds)')
    diff_parser.add_argument('--no-preprocess', action='store_true',
                            help='Disable MD preprocessing')
    diff_parser.add_argument('--align-selection', default='name CA',
                            help='Selection for structural alignment')
    diff_parser.add_argument('--center-selection', default='protein',
                            help='Selection for centering')
    diff_parser.add_argument('--compute-dccm', action='store_true', default=True,
                            help='Compute Dynamic Cross-Correlation Matrix (default: True)')
    diff_parser.add_argument('--no-dccm', dest='compute_dccm', action='store_false',
                            help='Skip DCCM computation')
    diff_parser.add_argument('--compute-pca', action='store_true', default=True,
                            help='Compute Principal Component Analysis (default: True)')
    diff_parser.add_argument('--no-pca', dest='compute_pca', action='store_false',
                            help='Skip PCA computation')
    diff_parser.add_argument('--pca-components', type=int, default=10,
                            help='Number of principal components to compute (default: 10)')
    diff_parser.add_argument('--dccm-selection', default='name CA',
                            help='Atom selection for dynamic analysis (default: "name CA")')
    diff_parser.add_argument('--compute-landscape', action='store_true', default=True,
                            help='Compute energy landscape from PC1/PC2 (default: True)')
    diff_parser.add_argument('--no-landscape', dest='compute_landscape', action='store_false',
                            help='Skip energy landscape computation')
    diff_parser.add_argument('--landscape-temp', type=float, default=310.0,
                            help='Temperature for energy landscape (K, default: 310)')
    diff_parser.add_argument('--landscape-bins', type=int, default=50,
                            help='Number of bins for energy landscape (default: 50)')
    diff_parser.add_argument('--landscape-sigma', type=float, default=1.0,
                            help='Gaussian smoothing sigma for landscape (default: 1.0)')
    diff_parser.add_argument('--community-method', default='leiden',
                            choices=['leiden', 'louvain', 'spectral', 'hierarchical'],
                            help='Community detection method (default: leiden)')
    diff_parser.add_argument('--no-communities', dest='compute_communities', action='store_false',
                            help='Skip community detection')
    diff_parser.add_argument('--no-paths', dest='compute_paths', action='store_false',
                            help='Skip detailed path analysis')
    diff_parser.add_argument('--no-allosteric', dest='allosteric_analysis', action='store_false',
                            help='Skip allosteric pathway analysis')
    diff_parser.add_argument('--allosteric-sources', nargs='+',
                            help='Specific residues to use as allosteric sources (e.g., A_50 B_50)')
    diff_parser.add_argument('--allosteric-targets', nargs='+',
                            help='Specific residues to use as allosteric targets (e.g., A_25 B_25)')
    # PyEMMA options (same as others)
    diff_parser.add_argument('--compute-msm', dest='compute_msm', action='store_true', default=True,
                            help='Compute Markov State Model analysis (default: enabled)')
    diff_parser.add_argument('--no-msm', dest='compute_msm', action='store_false',
                            help='Skip Markov State Model analysis')
    diff_parser.add_argument('--msm-lag-time', type=int, default=10,
                            help='Lag time for MSM construction in frames (default: 10)')
    diff_parser.add_argument('--msm-clusters', type=int, default=100,
                            help='Number of clusters for MSM discretization (default: 100)')
    diff_parser.add_argument('--msm-stride', type=int, default=1,
                            help='Stride for coordinate extraction (default: 1)')
    diff_parser.add_argument('--msm-features', default='distances',
                            choices=['distances', 'coordinates', 'angles', 'dihedrals'],
                            help='Feature type for MSM (default: distances)')
    diff_parser.add_argument('--msm-clustering', default='kmeans',
                            choices=['kmeans', 'minibatch_kmeans', 'regular_space'],
                            help='Clustering method for MSM (default: kmeans)')
    diff_parser.add_argument('--no-kinetics', dest='compute_kinetics', action='store_false',
                            help='Skip kinetic analysis')
    diff_parser.add_argument('--kinetic-timescales', type=int, default=5,
                            help='Number of kinetic timescales to compute (default: 5)')
    diff_parser.add_argument('--no-metastable', dest='compute_metastable_states', action='store_false',
                            help='Skip metastable state analysis')
    diff_parser.add_argument('--metastable-states', type=int, default=5,
                            help='Number of metastable states for PCCA+ (default: 5)')
    
    # Comprehensive differential analysis
    differential_parser = subparsers.add_parser(
        'differential',
        help='Comprehensive differential analysis between two simulations',
        description='Perform comprehensive comparison including network topology, dynamics, energetics, kinetics, and allosteric communication'
    )
    
    # Required arguments
    differential_parser.add_argument('-t1', '--topology1', required=True,
                                   help='Topology file for first simulation (PDB, GRO, etc.)')
    differential_parser.add_argument('-x1', '--trajectory1', required=True,
                                   help='Trajectory file for first simulation (XTC, DCD, etc.)')
    differential_parser.add_argument('-n1', '--sim1', required=True,
                                   help='Name/identifier for first simulation')
    differential_parser.add_argument('-t2', '--topology2', required=True,
                                   help='Topology file for second simulation')
    differential_parser.add_argument('-x2', '--trajectory2', required=True,
                                   help='Trajectory file for second simulation')
    differential_parser.add_argument('-n2', '--sim2', required=True,
                                   help='Name/identifier for second simulation')
    differential_parser.add_argument('-o', '--output', default='differential_analysis_results',
                                   help='Output directory (default: differential_analysis_results)')
    
    # Analysis focus options
    differential_parser.add_argument('--no-network-comparison', action='store_true',
                                   help='Skip network topology comparison')
    differential_parser.add_argument('--no-dynamics-comparison', action='store_true',
                                   help='Skip dynamics (DCCM/PCA) comparison')
    differential_parser.add_argument('--no-energetics-comparison', action='store_true',
                                   help='Skip energetics (landscape) comparison')
    differential_parser.add_argument('--no-kinetics-comparison', action='store_true',
                                   help='Skip kinetics (MSM) comparison')
    differential_parser.add_argument('--no-allosteric-comparison', action='store_true',
                                   help='Skip allosteric communication comparison')
    
    # Statistical analysis options
    differential_parser.add_argument('--statistical-tests', action='store_true', default=True,
                                   help='Perform statistical significance testing (default: True)')
    differential_parser.add_argument('--no-statistical-tests', dest='statistical_tests', action='store_false',
                                   help='Skip statistical significance testing')
    differential_parser.add_argument('--significance-threshold', type=float, default=0.05,
                                   help='P-value threshold for significance (default: 0.05)')
    differential_parser.add_argument('--multiple-comparison-correction', default='fdr_bh',
                                   choices=['fdr_bh', 'bonferroni', 'none'],
                                   help='Multiple comparison correction method (default: fdr_bh)')
    differential_parser.add_argument('--bootstrap-iterations', type=int, default=1000,
                                   help='Number of bootstrap iterations (default: 1000)')
    
    # Visualization options
    differential_parser.add_argument('--no-plots', action='store_true',
                                   help='Skip difference plot generation')
    differential_parser.add_argument('--publication-figures', action='store_true',
                                   help='Generate high-resolution publication-quality figures')
    differential_parser.add_argument('--figure-dpi', type=int, default=300,
                                   help='Figure resolution in DPI (default: 300)')
    
    # Output options
    differential_parser.add_argument('--no-html-report', action='store_true',
                                   help='Skip HTML report generation')
    differential_parser.add_argument('--excel-export', action='store_true',
                                   help='Export results to Excel workbook')
    
    # Analysis thresholds
    differential_parser.add_argument('--correlation-threshold', type=float, default=0.2,
                                   help='Minimum correlation change to consider significant (default: 0.2)')
    differential_parser.add_argument('--efficiency-threshold', type=float, default=0.1,
                                   help='Minimum communication efficiency change (default: 0.1)')
    differential_parser.add_argument('--energy-threshold', type=float, default=2.0,
                                   help='Minimum energy change in kT (default: 2.0)')
    
    # Standard analysis configuration (inherit from single mode)
    differential_parser.add_argument('--selection', default='protein and not name H*',
                                   help='Atom selection string (default: "protein and not name H*")')
    differential_parser.add_argument('--cutoff', type=float, default=4.5,
                                   help='Contact distance cutoff in Å (default: 4.5)')
    differential_parser.add_argument('--threshold', type=float, default=0.2,
                                   help='Contact frequency threshold (default: 0.2)')
    differential_parser.add_argument('--community-method', default='leiden',
                                   choices=['leiden', 'louvain', 'spectral', 'hierarchical'],
                                   help='Community detection method (default: leiden)')
    differential_parser.add_argument('--allosteric-sources', nargs='+',
                                   help='Allosteric source residues (e.g., A_50 B_50)')
    differential_parser.add_argument('--allosteric-targets', nargs='+',
                                   help='Allosteric target residues (e.g., A_25 B_25)')
    differential_parser.add_argument('--msm-lag-time', type=int, default=10,
                                   help='MSM lag time in frames (default: 10)')
    differential_parser.add_argument('--msm-clusters', type=int, default=100,
                                   help='Number of MSM clusters (default: 100)')
    differential_parser.add_argument('--landscape-temp', type=float, default=310.0,
                                   help='Temperature for energy landscape in K (default: 310)')
    differential_parser.add_argument('--landscape-bins', type=int, default=50,
                                   help='Energy landscape bins (default: 50)')
    
    # Missing attributes needed by create_analysis_config_from_args
    differential_parser.add_argument('--interaction-types', nargs='+',
                                   choices=['distance', 'hbond', 'salt_bridge'],
                                   default=['distance'],
                                   help='Types of interactions to analyze (default: distance)')
    differential_parser.add_argument('--segments', type=int, default=5,
                                   help='Number of trajectory segments (default: 5)')
    differential_parser.add_argument('--timeout', type=int, default=300,
                                   help='Timeout for expensive computations (default: 300s)')
    differential_parser.add_argument('--no-preprocess', action='store_true',
                                   help='Disable MD preprocessing')
    differential_parser.add_argument('--align-selection', default='name CA',
                                   help='Selection for structural alignment (default: "name CA")')
    differential_parser.add_argument('--center-selection', default='protein',
                                   help='Selection for centering (default: "protein")')
    differential_parser.add_argument('--compute-dccm', action='store_true', default=True,
                                   help='Compute Dynamic Cross-Correlation Matrix (default: True)')
    differential_parser.add_argument('--no-dccm', dest='compute_dccm', action='store_false',
                                   help='Skip DCCM computation')
    differential_parser.add_argument('--compute-pca', action='store_true', default=True,
                                   help='Compute Principal Component Analysis (default: True)')
    differential_parser.add_argument('--no-pca', dest='compute_pca', action='store_false',
                                   help='Skip PCA computation')
    differential_parser.add_argument('--pca-components', type=int, default=10,
                                   help='Number of principal components (default: 10)')
    differential_parser.add_argument('--dccm-selection', default='name CA',
                                   help='Atom selection for dynamic analysis (default: "name CA")')
    differential_parser.add_argument('--compute-landscape', action='store_true', default=True,
                                   help='Compute energy landscape (default: True)')
    differential_parser.add_argument('--no-landscape', dest='compute_landscape', action='store_false',
                                   help='Skip energy landscape computation')
    differential_parser.add_argument('--landscape-sigma', type=float, default=1.0,
                                   help='Gaussian smoothing sigma for landscape (default: 1.0)')
    differential_parser.add_argument('--compute-communities', action='store_true', default=True,
                                   help='Compute community detection (default: True)')
    differential_parser.add_argument('--no-communities', dest='compute_communities', action='store_false',
                                   help='Skip community detection')
    differential_parser.add_argument('--compute-paths', action='store_true', default=True,
                                   help='Compute path analysis (default: True)')
    differential_parser.add_argument('--no-paths', dest='compute_paths', action='store_false',
                                   help='Skip path analysis')
    differential_parser.add_argument('--allosteric-analysis', action='store_true', default=True,
                                   help='Perform allosteric analysis (default: True)')
    differential_parser.add_argument('--no-allosteric', dest='allosteric_analysis', action='store_false',
                                   help='Skip allosteric analysis')
    differential_parser.add_argument('--compute-msm', action='store_true', default=True,
                                   help='Compute Markov State Model (default: True)')
    differential_parser.add_argument('--no-msm', dest='compute_msm', action='store_false',
                                   help='Skip MSM computation')
    differential_parser.add_argument('--msm-stride', type=int, default=1,
                                   help='MSM stride for coordinate extraction (default: 1)')
    differential_parser.add_argument('--msm-features', default='distances',
                                   choices=['distances', 'coordinates', 'angles', 'dihedrals'],
                                   help='MSM feature type (default: distances)')
    differential_parser.add_argument('--msm-clustering', default='kmeans',
                                   choices=['kmeans', 'minibatch_kmeans', 'regular_space'],
                                   help='MSM clustering method (default: kmeans)')
    differential_parser.add_argument('--compute-kinetics', action='store_true', default=True,
                                   help='Compute kinetic analysis (default: True)')
    differential_parser.add_argument('--no-kinetics', dest='compute_kinetics', action='store_false',
                                   help='Skip kinetic analysis')
    differential_parser.add_argument('--kinetic-timescales', type=int, default=5,
                                   help='Number of kinetic timescales (default: 5)')
    differential_parser.add_argument('--compute-metastable-states', action='store_true', default=True,
                                   help='Compute metastable states (default: True)')
    differential_parser.add_argument('--no-metastable', dest='compute_metastable_states', action='store_false',
                                   help='Skip metastable state analysis')
    differential_parser.add_argument('--metastable-states', type=int, default=5,
                                   help='Number of metastable states (default: 5)')
    
    # Example configuration generator
    example_parser = subparsers.add_parser(
        'example-config',
        help='Generate an example configuration file'
    )
    example_parser.add_argument('-o', '--output', default='example_config.json',
                               help='Output path for example configuration')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return 1
    
    # Route to appropriate function
    if args.mode == 'single':
        return run_single_analysis(args)
    elif args.mode == 'compare':
        return run_comparison_analysis(args)
    elif args.mode == 'diff':
        return run_differential_analysis(args)
    elif args.mode == 'differential':
        return run_comprehensive_differential_analysis(args)
    elif args.mode == 'example-config':
        create_example_config(args.output)
        return 0
    else:
        print(f"Unknown mode: {args.mode}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
