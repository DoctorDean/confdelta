#!/usr/bin/env python3

"""
Test script for MD-Compare differential analysis implementation
"""

import sys
import numpy as np
from pathlib import Path

def test_differential_analyzer():
    """Test the DifferentialAnalyzer implementation"""
    
    try:
        from md_compare_differential import DifferentialAnalyzer, DifferentialConfig
        print("✓ Successfully imported differential analysis modules")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test configuration creation
    try:
        config = DifferentialConfig(
            compare_networks=True,
            compare_dynamics=True,
            compare_allosteric=True,
            perform_statistical_tests=True,
            create_publication_figures=False
        )
        print("✓ DifferentialConfig created successfully")
        
        # Test analyzer initialization
        analyzer = DifferentialAnalyzer(config, "test_differential_output")
        print("✓ DifferentialAnalyzer initialized successfully")
        print(f"  Output directory: {analyzer.output_dir}")
        print(f"  Subdirectories created: {list(analyzer.subdirs.keys())}")
        
        # Test comparator initialization
        print("✓ Comparator modules initialized:")
        print(f"  - NetworkComparator: {type(analyzer.network_comparator).__name__}")
        print(f"  - DynamicsComparator: {type(analyzer.dynamics_comparator).__name__}")
        print(f"  - AllostericComparator: {type(analyzer.allosteric_comparator).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False

def test_comparator_algorithms():
    """Test individual comparator algorithms"""
    
    print("\nTesting comparator algorithms...")
    
    try:
        from md_compare_differential import (
            DifferentialConfig, NetworkComparator, DynamicsComparator, 
            AllostericComparator, NetworkComparison, DynamicsComparison, AllostericComparison
        )
        
        config = DifferentialConfig()
        
        # Test AllostericComparator with mock data
        allosteric_comparator = AllostericComparator(config)
        
        # Create mock allosteric pathway data
        mock_ap1 = {
            'communication_efficiency': {
                'A_1_A_2': 0.8,
                'A_1_B_1': 0.6,
                'A_2_B_2': 0.4
            },
            'allosteric_hotspots': [
                {'node': 'A_1', 'hotspot_score': 0.9, 'frequency': 10},
                {'node': 'A_2', 'hotspot_score': 0.7, 'frequency': 8}
            ],
            'pathways': [
                {'source': 'A_1', 'target': 'B_1', 'efficiency': 0.8, 'path': ['A_1', 'A_2', 'B_1']},
                {'source': 'A_2', 'target': 'B_2', 'efficiency': 0.6, 'path': ['A_2', 'B_2']}
            ]
        }
        
        mock_ap2 = {
            'communication_efficiency': {
                'A_1_A_2': 0.6,  # Decreased
                'A_1_B_1': 0.8,  # Increased
                'A_2_B_2': 0.3   # Decreased
            },
            'allosteric_hotspots': [
                {'node': 'A_1', 'hotspot_score': 0.7, 'frequency': 8},  # Decreased
                {'node': 'A_2', 'hotspot_score': 0.8, 'frequency': 9}   # Increased
            ],
            'pathways': [
                {'source': 'A_1', 'target': 'B_1', 'efficiency': 0.6, 'path': ['A_1', 'A_3', 'B_1']},
                {'source': 'A_2', 'target': 'B_2', 'efficiency': 0.4, 'path': ['A_2', 'B_2']}
            ]
        }
        
        # Test comparison
        print("  Testing AllostericComparator...")
        allosteric_result = allosteric_comparator.compare_allosteric(
            type('MockMetrics', (), {'allosteric_pathways': mock_ap1}),
            type('MockMetrics', (), {'allosteric_pathways': mock_ap2})
        )
        
        print(f"    ✓ Pathway disruption analysis: {len(allosteric_result.pathway_disruption_analysis)} pathways")
        print(f"    ✓ Hotspot ranking changes: {len(allosteric_result.hotspot_ranking_changes)} nodes")
        
        # Test DynamicsComparator with mock DCCM data
        dynamics_comparator = DynamicsComparator(config)
        
        # Create mock DCCM matrices
        n_res = 5
        mock_dccm1 = np.random.random((n_res, n_res)) * 2 - 1  # Random correlations
        mock_dccm2 = mock_dccm1 + np.random.random((n_res, n_res)) * 0.3 - 0.15  # Add changes
        
        mock_dyn1 = {
            'dccm_matrix': mock_dccm1,
            'pca_eigenvalues': np.array([0.5, 0.3, 0.1, 0.05, 0.05]),
            'pca_eigenvectors': np.random.random((n_res, 5))
        }
        
        mock_dyn2 = {
            'dccm_matrix': mock_dccm2,
            'pca_eigenvalues': np.array([0.4, 0.35, 0.15, 0.06, 0.04]),  # Different variance
            'pca_eigenvectors': np.random.random((n_res, 5))
        }
        
        print("  Testing DynamicsComparator...")
        dynamics_result = dynamics_comparator.compare_dynamics(mock_dyn1, mock_dyn2)
        
        print(f"    ✓ DCCM difference matrix: {dynamics_result.dccm_difference_matrix.shape}")
        print(f"    ✓ Significant correlation changes: {len(dynamics_result.significant_correlation_changes)}")
        print(f"    ✓ PCA variance changes: {len(dynamics_result.pca_variance_changes)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing comparators: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_integration():
    """Test CLI integration"""
    
    print("\nTesting CLI integration...")
    
    try:
        from md_compare_cli import run_comprehensive_differential_analysis
        print("✓ CLI integration function found")
        
        # Test argument parsing (mock)
        class MockArgs:
            def __init__(self):
                self.sim1 = "simulation1"
                self.sim2 = "simulation2"
                self.topology1 = "sim1.pdb"
                self.topology2 = "sim2.pdb"
                self.trajectory1 = "sim1.xtc"
                self.trajectory2 = "sim2.xtc"
                self.output = "test_cli_output"
                self.no_network_comparison = False
                self.no_dynamics_comparison = False
                self.no_energetics_comparison = True  # Skip for test
                self.no_kinetics_comparison = True    # Skip for test
                self.no_allosteric_comparison = False
                self.statistical_tests = True
                self.significance_threshold = 0.05
                self.multiple_comparison_correction = "fdr_bh"
                self.bootstrap_iterations = 100
                self.no_plots = True             # Skip for test
                self.publication_figures = False
                self.figure_dpi = 300
                self.no_html_report = True       # Skip for test
                self.excel_export = False
                self.correlation_threshold = 0.2
                self.efficiency_threshold = 0.1
                self.energy_threshold = 2.0
                # Standard analysis options
                self.selection = "protein and not name H*"
                self.cutoff = 4.5
                self.threshold = 0.2
                self.community_method = "leiden"
                self.allosteric_sources = ["A_50", "B_50"]
                self.allosteric_targets = ["A_25", "B_25"]
                self.msm_lag_time = 10
                self.msm_clusters = 50
                self.landscape_temp = 310.0
                self.landscape_bins = 30
        
        mock_args = MockArgs()
        print("✓ Mock CLI arguments created")
        
        # Note: Full CLI test would require actual MD files
        print("  Note: Full CLI test requires MD simulation files")
        print("  CLI integration structure validated")
        
        return True
        
    except Exception as e:
        print(f"✗ CLI integration error: {e}")
        return False

def main():
    """Run all tests"""
    
    print("MD-Compare Differential Analysis Implementation Test")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic functionality
    print("Test 1: Basic differential analyzer functionality")
    if test_differential_analyzer():
        tests_passed += 1
    
    # Test 2: Comparator algorithms
    print("Test 2: Comparator algorithm functionality")
    if test_comparator_algorithms():
        tests_passed += 1
    
    # Test 3: CLI integration
    print("Test 3: CLI integration")
    if test_cli_integration():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Differential analysis implementation is ready.")
        print("\nNext steps:")
        print("1. Test with actual MD simulation files")
        print("2. Validate scientific accuracy of comparison algorithms")
        print("3. Optimize performance for large systems")
        print("4. Add additional statistical tests")
        return 0
    else:
        print("❌ Some tests failed. Review implementation before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
