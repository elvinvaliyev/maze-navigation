#!/usr/bin/env python3
"""
Comprehensive Analysis Runner for Maze Navigation Experiments

This script runs all analysis modules and generates a complete analysis report
including visualizations, statistical analysis, performance metrics, and comparative analysis.
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path to import analysis modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.visualization_engine import VisualizationEngine
from analysis.statistical_analysis import StatisticalAnalyzer
from analysis.performance_metrics import PerformanceAnalyzer
from analysis.comparative_analysis import ComparativeAnalyzer

def create_analysis_directory():
    """Create analysis directory structure."""
    analysis_dir = "analysis"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    # Create subdirectories for different types of outputs
    subdirs = ['visualizations', 'reports', 'tables']
    for subdir in subdirs:
        subdir_path = os.path.join(analysis_dir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
    
    return analysis_dir

def run_visualization_analysis():
    """Run comprehensive visualization analysis."""
    print("="*80)
    print("RUNNING VISUALIZATION ANALYSIS")
    print("="*80)
    
    try:
        viz_engine = VisualizationEngine()
        viz_engine.generate_all_visualizations()
        print("✓ Visualization analysis completed successfully!")
        return True
    except Exception as e:
        print(f"✗ Visualization analysis failed: {e}")
        return False

def run_statistical_analysis():
    """Run comprehensive statistical analysis."""
    print("\n" + "="*80)
    print("RUNNING STATISTICAL ANALYSIS")
    print("="*80)
    
    try:
        stat_analyzer = StatisticalAnalyzer()
        results = stat_analyzer.generate_comprehensive_report()
        print("✓ Statistical analysis completed successfully!")
        return results
    except Exception as e:
        print(f"✗ Statistical analysis failed: {e}")
        return None

def run_performance_analysis():
    """Run comprehensive performance analysis."""
    print("\n" + "="*80)
    print("RUNNING PERFORMANCE ANALYSIS")
    print("="*80)
    
    try:
        perf_analyzer = PerformanceAnalyzer()
        results = perf_analyzer.generate_comprehensive_performance_report()
        print("✓ Performance analysis completed successfully!")
        return results
    except Exception as e:
        print(f"✗ Performance analysis failed: {e}")
        return None

def run_comparative_analysis():
    """Run comprehensive comparative analysis."""
    print("\n" + "="*80)
    print("RUNNING COMPARATIVE ANALYSIS")
    print("="*80)
    
    try:
        comp_analyzer = ComparativeAnalyzer()
        results = comp_analyzer.generate_comprehensive_comparison_report()
        print("✓ Comparative analysis completed successfully!")
        return results
    except Exception as e:
        print(f"✗ Comparative analysis failed: {e}")
        return None

def generate_summary_report(stat_results, perf_results, comp_results):
    """Generate a summary report of all analyses."""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    summary_file = "analysis/reports/comprehensive_summary_report.txt"
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE MAZE NAVIGATION ANALYSIS SUMMARY REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset overview
        f.write("DATASET OVERVIEW:\n")
        f.write("-" * 40 + "\n")
        f.write("This analysis covers comprehensive maze navigation experiments\n")
        f.write("comparing four different agent types across multiple conditions.\n\n")
        
        # Key findings
        f.write("KEY FINDINGS:\n")
        f.write("-" * 40 + "\n")
        
        if stat_results:
            f.write("1. STATISTICAL SIGNIFICANCE:\n")
            for metric, results in stat_results['anova'].items():
                f.write(f"   {metric}: F={results['f_stat']:.3f}, p={results['p_value']:.6f} {results['significance']}\n")
            f.write("\n")
        
        if perf_results:
            f.write("2. PERFORMANCE RANKINGS:\n")
            for i, (agent, score) in enumerate(perf_results['composite_scores'].head(3).iterrows(), 1):
                f.write(f"   {i}. {agent}: {score['composite_score']:.4f}\n")
            f.write("\n")
        
        if comp_results:
            f.write("3. AGENT SPECIALIZATIONS:\n")
            for agent, profile in comp_results['agent_profiles'].items():
                if profile['strengths']:
                    best_strength = max(profile['strengths'], key=lambda x: x[1])
                    f.write(f"   {agent}: Best at {best_strength[0]} (z={best_strength[1]:.2f})\n")
            f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 40 + "\n")
        f.write("1. For maximum reward collection: Use the agent with highest avg_reward\n")
        f.write("2. For highest exit rate: Use the agent with highest exit_rate\n")
        f.write("3. For optimal risk-adjusted returns: Use the agent with highest avg_risk_adjusted_return\n")
        f.write("4. For most efficient navigation: Use the agent with highest avg_path_efficiency\n\n")
        
        # Methodology
        f.write("METHODOLOGY:\n")
        f.write("-" * 40 + "\n")
        f.write("This analysis employed:\n")
        f.write("- Comprehensive statistical testing (ANOVA, t-tests, effect sizes)\n")
        f.write("- Performance metrics calculation (composite scores, risk-adjusted returns)\n")
        f.write("- Comparative analysis (agent-to-agent comparisons)\n")
        f.write("- Visualization generation (dashboards, heatmaps, trend analysis)\n\n")
        
        # Files generated
        f.write("FILES GENERATED:\n")
        f.write("-" * 40 + "\n")
        f.write("Visualizations:\n")
        f.write("- comprehensive_dashboard.png\n")
        f.write("- swap_probability_analysis.png\n")
        f.write("- reward_configuration_analysis.png\n")
        f.write("- agent_radar_comparison.png\n")
        f.write("- statistical_significance.png\n")
        f.write("- maze_complexity_analysis.png\n")
        f.write("- performance_trends.png\n")
        f.write("- comparative_analysis.png\n\n")
        
        f.write("Reports:\n")
        f.write("- This comprehensive summary report\n")
        f.write("- Statistical analysis output (console)\n")
        f.write("- Performance analysis output (console)\n")
        f.write("- Comparative analysis output (console)\n")
    
    print(f"✓ Summary report saved to {summary_file}")

def main():
    """Main function to run all analyses."""
    print("="*80)
    print("COMPREHENSIVE MAZE NAVIGATION ANALYSIS")
    print("="*80)
    print(f"Starting analysis at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create analysis directory
    analysis_dir = create_analysis_directory()
    print(f"✓ Analysis directory created: {analysis_dir}")
    
    # Track results
    results = {}
    
    # Run all analyses
    start_time = time.time()
    
    # 1. Visualization analysis
    viz_success = run_visualization_analysis()
    results['visualization'] = viz_success
    
    # 2. Statistical analysis
    stat_results = run_statistical_analysis()
    results['statistical'] = stat_results
    
    # 3. Performance analysis
    perf_results = run_performance_analysis()
    results['performance'] = perf_results
    
    # 4. Comparative analysis
    comp_results = run_comparative_analysis()
    results['comparative'] = comp_results
    
    # Generate summary report
    generate_summary_report(stat_results, perf_results, comp_results)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETION SUMMARY")
    print("="*80)
    
    print(f"Total analysis time: {total_time:.2f} seconds")
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nAnalysis Results:")
    for analysis_type, result in results.items():
        status = "✓ SUCCESS" if result else "✗ FAILED"
        print(f"  {analysis_type.capitalize()}: {status}")
    
    print(f"\nGenerated files:")
    print(f"  - Visualizations: analysis/*.png")
    print(f"  - Summary report: analysis/reports/comprehensive_summary_report.txt")
    print(f"  - Console output: Check terminal for detailed results")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main() 