#!/usr/bin/env python3
"""
Advanced Statistical Analysis for Maze Navigation Experiments

This module provides comprehensive statistical analysis including:
- Descriptive statistics
- Hypothesis testing
- Effect size calculations
- Confidence intervals
- ANOVA and post-hoc tests
- Correlation analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, tukey_hsd
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    def __init__(self, data_path='../comprehensive_results.csv'):
        """Initialize the statistical analyzer with data."""
        self.df = pd.read_csv(data_path)
        self.agents = self.df['agent'].unique()
        self.metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'avg_risk_adjusted_return', 'avg_path_efficiency']
        
    def generate_descriptive_statistics(self):
        """Generate comprehensive descriptive statistics."""
        print("="*80)
        print("DESCRIPTIVE STATISTICS")
        print("="*80)
        
        for metric in self.metrics:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print("-" * 50)
            
            stats_df = self.df.groupby('agent')[metric].agg([
                'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'
            ]).round(4)
            
            print(stats_df)
            
            # Overall statistics
            overall_mean = self.df[metric].mean()
            overall_std = self.df[metric].std()
            print(f"\nOverall: Mean = {overall_mean:.4f}, Std = {overall_std:.4f}")
    
    def perform_anova_analysis(self):
        """Perform one-way ANOVA for each metric across agents."""
        print("\n" + "="*80)
        print("ONE-WAY ANOVA ANALYSIS")
        print("="*80)
        
        anova_results = {}
        
        for metric in self.metrics:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print("-" * 50)
            
            # Prepare data for ANOVA
            groups = [self.df[self.df['agent'] == agent][metric].values for agent in self.agents]
            
            # Perform ANOVA
            f_stat, p_value = f_oneway(*groups)
            
            # Calculate effect size (eta-squared)
            ss_between = 0
            ss_total = 0
            grand_mean = self.df[metric].mean()
            
            for group in groups:
                ss_between += len(group) * (group.mean() - grand_mean) ** 2
                ss_total += np.sum((group - grand_mean) ** 2)
            
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            print(f"F-statistic: {f_stat:.4f}")
            print(f"p-value: {p_value:.6f}")
            print(f"Effect size (η²): {eta_squared:.4f}")
            
            # Interpret effect size
            if eta_squared < 0.01:
                effect_interpretation = "Small"
            elif eta_squared < 0.06:
                effect_interpretation = "Medium"
            else:
                effect_interpretation = "Large"
            
            print(f"Effect size interpretation: {effect_interpretation}")
            
            # Significance
            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"
            else:
                significance = "ns"
            
            print(f"Significance: {significance}")
            
            anova_results[metric] = {
                'f_stat': f_stat,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'significance': significance
            }
        
        return anova_results
    
    def perform_post_hoc_analysis(self):
        """Perform Tukey's HSD post-hoc test for significant ANOVAs."""
        print("\n" + "="*80)
        print("POST-HOC ANALYSIS (TUKEY'S HSD)")
        print("="*80)
        
        post_hoc_results = {}
        
        for metric in self.metrics:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print("-" * 50)
            
            # Prepare data
            groups = [self.df[self.df['agent'] == agent][metric].values for agent in self.agents]
            
            # Perform Tukey's HSD
            result = tukey_hsd(*groups)
            
            print("Pairwise comparisons:")
            for i, agent1 in enumerate(self.agents):
                for j, agent2 in enumerate(self.agents):
                    if i < j:  # Only show each pair once
                        diff = result.statistic[i, j]
                        p_val = result.pvalue[i, j]
                        
                        if p_val < 0.001:
                            sig = "***"
                        elif p_val < 0.01:
                            sig = "**"
                        elif p_val < 0.05:
                            sig = "*"
                        else:
                            sig = "ns"
                        
                        print(f"  {agent1} vs {agent2}: diff={diff:.4f}, p={p_val:.6f} {sig}")
            
            post_hoc_results[metric] = result
        
        return post_hoc_results
    
    def calculate_effect_sizes(self):
        """Calculate Cohen's d effect sizes for all agent pairs."""
        print("\n" + "="*80)
        print("EFFECT SIZE ANALYSIS (COHEN'S D)")
        print("="*80)
        
        effect_sizes = {}
        
        for metric in self.metrics:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print("-" * 50)
            
            metric_effects = {}
            
            for i, agent1 in enumerate(self.agents):
                for j, agent2 in enumerate(self.agents):
                    if i < j:  # Only calculate each pair once
                        data1 = self.df[self.df['agent'] == agent1][metric]
                        data2 = self.df[self.df['agent'] == agent2][metric]
                        
                        # Calculate Cohen's d
                        pooled_std = np.sqrt(((len(data1) - 1) * data1.var() + (len(data2) - 1) * data2.var()) / 
                                           (len(data1) + len(data2) - 2))
                        
                        cohens_d = (data1.mean() - data2.mean()) / pooled_std
                        
                        # Interpret effect size
                        if abs(cohens_d) < 0.2:
                            interpretation = "Small"
                        elif abs(cohens_d) < 0.5:
                            interpretation = "Medium"
                        elif abs(cohens_d) < 0.8:
                            interpretation = "Large"
                        else:
                            interpretation = "Very Large"
                        
                        print(f"  {agent1} vs {agent2}: d={cohens_d:.3f} ({interpretation})")
                        
                        metric_effects[f"{agent1}_vs_{agent2}"] = {
                            'cohens_d': cohens_d,
                            'interpretation': interpretation
                        }
            
            effect_sizes[metric] = metric_effects
        
        return effect_sizes
    
    def perform_correlation_analysis(self):
        """Analyze correlations between different metrics."""
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)
        
        # Calculate correlation matrix
        corr_matrix = self.df[self.metrics].corr()
        
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(3))
        
        # Test significance of correlations
        print("\nCorrelation Significance Tests:")
        print("-" * 50)
        
        for i, metric1 in enumerate(self.metrics):
            for j, metric2 in enumerate(self.metrics):
                if i < j:  # Only test each pair once
                    corr, p_val = stats.pearsonr(self.df[metric1], self.df[metric2])
                    
                    if p_val < 0.001:
                        sig = "***"
                    elif p_val < 0.01:
                        sig = "**"
                    elif p_val < 0.05:
                        sig = "*"
                    else:
                        sig = "ns"
                    
                    print(f"  {metric1} vs {metric2}: r={corr:.3f}, p={p_val:.6f} {sig}")
        
        return corr_matrix
    
    def analyze_swap_probability_effects(self):
        """Analyze the effects of swap probability on performance."""
        print("\n" + "="*80)
        print("SWAP PROBABILITY EFFECT ANALYSIS")
        print("="*80)
        
        swap_effects = {}
        
        for metric in self.metrics:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print("-" * 50)
            
            # Calculate correlation with swap probability
            corr, p_val = stats.pearsonr(self.df['swap_prob'], self.df[metric])
            
            print(f"Correlation with swap probability: r={corr:.3f}, p={p_val:.6f}")
            
            # Analyze by agent
            print("\nBy Agent:")
            for agent in self.agents:
                agent_data = self.df[self.df['agent'] == agent]
                agent_corr, agent_p = stats.pearsonr(agent_data['swap_prob'], agent_data[metric])
                
                if agent_p < 0.001:
                    sig = "***"
                elif agent_p < 0.01:
                    sig = "**"
                elif agent_p < 0.05:
                    sig = "*"
                else:
                    sig = "ns"
                
                print(f"  {agent}: r={agent_corr:.3f}, p={agent_p:.6f} {sig}")
            
            swap_effects[metric] = {
                'overall_corr': corr,
                'overall_p': p_val,
                'agent_effects': {}
            }
            
            for agent in self.agents:
                agent_data = self.df[self.df['agent'] == agent]
                agent_corr, agent_p = stats.pearsonr(agent_data['swap_prob'], agent_data[metric])
                swap_effects[metric]['agent_effects'][agent] = {
                    'correlation': agent_corr,
                    'p_value': agent_p
                }
        
        return swap_effects
    
    def calculate_confidence_intervals(self, confidence=0.95):
        """Calculate confidence intervals for all metrics by agent."""
        print(f"\n" + "="*80)
        print(f"CONFIDENCE INTERVALS ({confidence*100}%)")
        print("="*80)
        
        ci_results = {}
        
        for metric in self.metrics:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print("-" * 50)
            
            metric_cis = {}
            
            for agent in self.agents:
                agent_data = self.df[self.df['agent'] == agent][metric]
                
                # Calculate confidence interval
                mean_val = agent_data.mean()
                std_err = stats.sem(agent_data)
                ci_lower, ci_upper = stats.t.interval(confidence, len(agent_data)-1, 
                                                     loc=mean_val, scale=std_err)
                
                print(f"  {agent}: {mean_val:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
                
                metric_cis[agent] = {
                    'mean': mean_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'std_error': std_err
                }
            
            ci_results[metric] = metric_cis
        
        return ci_results
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive statistical report."""
        print("="*80)
        print("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
        print("="*80)
        
        # Run all analyses
        self.generate_descriptive_statistics()
        anova_results = self.perform_anova_analysis()
        post_hoc_results = self.perform_post_hoc_analysis()
        effect_sizes = self.calculate_effect_sizes()
        corr_matrix = self.perform_correlation_analysis()
        swap_effects = self.analyze_swap_probability_effects()
        ci_results = self.calculate_confidence_intervals()
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY OF KEY FINDINGS")
        print("="*80)
        
        print("\n1. ANOVA Results Summary:")
        for metric, results in anova_results.items():
            print(f"   {metric}: F={results['f_stat']:.3f}, p={results['p_value']:.6f} {results['significance']}")
        
        print("\n2. Largest Effect Sizes:")
        for metric, effects in effect_sizes.items():
            max_effect = max(effects.values(), key=lambda x: abs(x['cohens_d']))
            print(f"   {metric}: {max_effect['cohens_d']:.3f} ({max_effect['interpretation']})")
        
        print("\n3. Strongest Correlations with Swap Probability:")
        for metric, effects in swap_effects.items():
            print(f"   {metric}: r={effects['overall_corr']:.3f}, p={effects['overall_p']:.6f}")
        
        return {
            'anova': anova_results,
            'post_hoc': post_hoc_results,
            'effect_sizes': effect_sizes,
            'correlations': corr_matrix,
            'swap_effects': swap_effects,
            'confidence_intervals': ci_results
        }

if __name__ == "__main__":
    # Create and run statistical analyzer
    analyzer = StatisticalAnalyzer()
    results = analyzer.generate_comprehensive_report() 