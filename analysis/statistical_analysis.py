"""
Statistical analysis module for maze navigation experiments.

This module provides comprehensive statistical analysis of agent performance,
including significance testing, confidence intervals, and effect size calculations.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticalAnalyzer:
    """Statistical analysis for agent performance comparison."""
    
    def __init__(self):
        self.results = None
        self.alpha = 0.05  # Significance level
        
    def load_results(self, results_file: str = "comprehensive_results.csv"):
        """Load experiment results for analysis."""
        try:
            self.results = pd.read_csv(results_file)
            print(f"Loaded {len(self.results)} results from {results_file}")
            return True
        except FileNotFoundError:
            print(f"Results file {results_file} not found")
            return False
    
    def agent_performance_comparison(self) -> Dict:
        """Compare agent performance using statistical tests."""
        if self.results is None:
            return {}
        
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement']
        comparisons = {}
        
        for metric in metrics:
            print(f"\n=== {metric.upper()} Analysis ===")
            
            # Get data for each agent
            agent_data = {}
            for agent in self.results['agent'].unique():
                agent_data[agent] = self.results[self.results['agent'] == agent][metric].values
            
            # ANOVA test
            if len(agent_data) > 2:
                f_stat, p_value = stats.f_oneway(*agent_data.values())
                print(f"ANOVA: F={f_stat:.3f}, p={p_value:.3f}")
                if p_value < self.alpha:
                    print("*** SIGNIFICANT DIFFERENCE ***")
                
                comparisons[metric] = {
                    'anova_f': f_stat,
                    'anova_p': p_value,
                    'significant': p_value < self.alpha
                }
            
            # Pairwise t-tests
            agents = list(agent_data.keys())
            pairwise_results = []
            
            for i in range(len(agents)):
                for j in range(i+1, len(agents)):
                    agent1, agent2 = agents[i], agents[j]
                    t_stat, p_value = stats.ttest_ind(agent_data[agent1], agent_data[agent2])
                    mean1, mean2 = np.mean(agent_data[agent1]), np.mean(agent_data[agent2])
                    
                    print(f"{agent1} vs {agent2}: t={t_stat:.3f}, p={p_value:.3f}")
                    if p_value < self.alpha:
                        print(f"  *** SIGNIFICANT: {agent1} ({mean1:.3f}) vs {agent2} ({mean2:.3f}) ***")
                    
                    pairwise_results.append({
                        'agent1': agent1,
                        'agent2': agent2,
                        't_stat': t_stat,
                        'p_value': p_value,
                        'mean1': mean1,
                        'mean2': mean2,
                        'significant': p_value < self.alpha
                    })
            
            if metric in comparisons:
                comparisons[metric]['pairwise'] = pairwise_results
        
        return comparisons
    
    def effect_size_analysis(self) -> Dict:
        """Calculate effect sizes for agent comparisons."""
        if self.results is None:
            return {}
        
        effect_sizes = {}
        metrics = ['avg_reward', 'exit_rate', 'survival_rate']
        
        for metric in metrics:
            agent_data = {}
            for agent in self.results['agent'].unique():
                agent_data[agent] = self.results[self.results['agent'] == agent][metric].values
            
            agents = list(agent_data.keys())
            metric_effects = []
            
            for i in range(len(agents)):
                for j in range(i+1, len(agents)):
                    agent1, agent2 = agents[i], agents[j]
                    
                    # Cohen's d effect size
                    pooled_std = np.sqrt(((len(agent_data[agent1]) - 1) * np.var(agent_data[agent1]) + 
                                        (len(agent_data[agent2]) - 1) * np.var(agent_data[agent2])) / 
                                       (len(agent_data[agent1]) + len(agent_data[agent2]) - 2))
                    
                    cohens_d = (np.mean(agent_data[agent1]) - np.mean(agent_data[agent2])) / pooled_std
                    
                    # Interpret effect size
                    if abs(cohens_d) < 0.2:
                        effect_magnitude = "small"
                    elif abs(cohens_d) < 0.5:
                        effect_magnitude = "medium"
                    else:
                        effect_magnitude = "large"
                    
                    metric_effects.append({
                        'agent1': agent1,
                        'agent2': agent2,
                        'cohens_d': cohens_d,
                        'magnitude': effect_magnitude
                    })
            
            effect_sizes[metric] = metric_effects
        
        return effect_sizes
    
    def confidence_intervals(self) -> Dict:
        """Calculate confidence intervals for agent performance."""
        if self.results is None:
            return {}
        
        ci_results = {}
        metrics = ['avg_reward', 'exit_rate', 'survival_rate']
        
        for metric in metrics:
            metric_cis = {}
            for agent in self.results['agent'].unique():
                agent_data = self.results[self.results['agent'] == agent][metric].values
                mean_val = np.mean(agent_data)
                std_err = stats.sem(agent_data)
                ci_lower, ci_upper = stats.t.interval(0.95, len(agent_data)-1, loc=mean_val, scale=std_err)
                
                metric_cis[agent] = {
                    'mean': mean_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'std_error': std_err
                }
            
            ci_results[metric] = metric_cis
        
        return ci_results
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate a comprehensive statistical report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*60)
        
        if not self.load_results():
            return {}
        
        # Run all analyses
        comparisons = self.agent_performance_comparison()
        effect_sizes = self.effect_size_analysis()
        confidence_intervals = self.confidence_intervals()
        
        # Print summary
        print("\n" + "="*60)
        print("STATISTICAL SUMMARY")
        print("="*60)
        
        for metric, comp in comparisons.items():
            if 'significant' in comp and comp['significant']:
                print(f"✓ {metric}: Significant differences found (p < {self.alpha})")
            else:
                print(f"✗ {metric}: No significant differences")
        
        # Save detailed results
        report = {
            'comparisons': comparisons,
            'effect_sizes': effect_sizes,
            'confidence_intervals': confidence_intervals,
            'significance_level': self.alpha
        }
        
        return report 