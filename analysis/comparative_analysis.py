#!/usr/bin/env python3
"""
Comparative Analysis for Maze Navigation Agents

This module provides comprehensive comparative analysis including:
- Agent-to-agent comparisons
- Performance across different conditions
- Behavioral pattern analysis
- Strengths and weaknesses identification
- Recommendation systems
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ComparativeAnalyzer:
    def __init__(self, data_path='../comprehensive_results.csv'):
        """Initialize the comparative analyzer with data."""
        self.df = pd.read_csv(data_path)
        self.agents = self.df['agent'].unique()
        self.metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'avg_risk_adjusted_return', 'avg_path_efficiency']
        
    def create_agent_comparison_matrix(self):
        """Create a comprehensive comparison matrix between all agents."""
        print("="*80)
        print("AGENT COMPARISON MATRIX")
        print("="*80)
        
        comparison_matrix = {}
        
        for metric in self.metrics:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print("-" * 50)
            
            metric_matrix = {}
            
            for i, agent1 in enumerate(self.agents):
                for j, agent2 in enumerate(self.agents):
                    if i < j:  # Only compare each pair once
                        data1 = self.df[self.df['agent'] == agent1][metric]
                        data2 = self.df[self.df['agent'] == agent2][metric]
                        
                        # Calculate comparison statistics
                        mean_diff = data1.mean() - data2.mean()
                        std_diff = np.sqrt(data1.var() + data2.var())
                        effect_size = mean_diff / std_diff if std_diff != 0 else 0
                        
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(data1, data2)
                        
                        # Determine winner
                        if mean_diff > 0:
                            winner = agent1
                            advantage = f"{agent1} +{mean_diff:.4f}"
                        else:
                            winner = agent2
                            advantage = f"{agent2} +{abs(mean_diff):.4f}"
                        
                        print(f"  {agent1} vs {agent2}:")
                        print(f"    Difference: {mean_diff:.4f}")
                        print(f"    Effect Size: {effect_size:.3f}")
                        print(f"    p-value: {p_value:.6f}")
                        print(f"    Winner: {winner}")
                        print(f"    Advantage: {advantage}")
                        print()
                        
                        metric_matrix[f"{agent1}_vs_{agent2}"] = {
                            'difference': mean_diff,
                            'effect_size': effect_size,
                            'p_value': p_value,
                            'winner': winner,
                            'advantage': advantage
                        }
            
            comparison_matrix[metric] = metric_matrix
        
        return comparison_matrix
    
    def analyze_agent_strengths_weaknesses(self):
        """Identify strengths and weaknesses for each agent."""
        print("\n" + "="*80)
        print("AGENT STRENGTHS AND WEAKNESSES ANALYSIS")
        print("="*80)
        
        agent_profiles = {}
        
        for agent in self.agents:
            print(f"\n{agent}:")
            print("-" * 50)
            
            agent_data = self.df[self.df['agent'] == agent]
            
            # Calculate relative performance for each metric
            strengths = []
            weaknesses = []
            
            for metric in self.metrics:
                agent_mean = agent_data[metric].mean()
                overall_mean = self.df[metric].mean()
                overall_std = self.df[metric].std()
                
                z_score = (agent_mean - overall_mean) / overall_std if overall_std != 0 else 0
                
                if z_score > 0.5:  # Significantly above average
                    strengths.append((metric, z_score, agent_mean))
                elif z_score < -0.5:  # Significantly below average
                    weaknesses.append((metric, z_score, agent_mean))
            
            print("  Strengths:")
            for metric, z_score, value in sorted(strengths, key=lambda x: x[1], reverse=True):
                print(f"    {metric}: {value:.4f} (z={z_score:.2f})")
            
            print("  Weaknesses:")
            for metric, z_score, value in sorted(weaknesses, key=lambda x: x[1]):
                print(f"    {metric}: {value:.4f} (z={z_score:.2f})")
            
            agent_profiles[agent] = {
                'strengths': strengths,
                'weaknesses': weaknesses,
                'overall_performance': agent_data[self.metrics].mean().mean()
            }
        
        return agent_profiles
    
    def analyze_condition_specific_performance(self):
        """Analyze how agents perform under different conditions."""
        print("\n" + "="*80)
        print("CONDITION-SPECIFIC PERFORMANCE ANALYSIS")
        print("="*80)
        
        conditions = {
            'High Swap Probability (≥0.5)': self.df['swap_prob'] >= 0.5,
            'Low Swap Probability (≤0.2)': self.df['swap_prob'] <= 0.2,
            'Medium Swap Probability (0.2-0.5)': (self.df['swap_prob'] > 0.2) & (self.df['swap_prob'] < 0.5),
            'Big Reward Difference': self.df['reward_config'].str.contains('big'),
            'Small Reward Difference': self.df['reward_config'].str.contains('small'),
            'Equal Rewards': self.df['reward_config'].str.contains('equal'),
            'Complex Mazes': self.df['maze'].str.contains('complex'),
            'Simple Mazes': self.df['maze'].str.contains('simple')
        }
        
        condition_performance = {}
        
        for condition_name, condition_mask in conditions.items():
            print(f"\n{condition_name}:")
            print("-" * 50)
            
            condition_data = self.df[condition_mask]
            if len(condition_data) == 0:
                print("  No data for this condition")
                continue
            
            condition_results = {}
            
            for metric in self.metrics:
                metric_ranking = condition_data.groupby('agent')[metric].mean().sort_values(ascending=False)
                
                print(f"  {metric}:")
                for i, (agent, value) in enumerate(metric_ranking.items(), 1):
                    print(f"    {i}. {agent}: {value:.4f}")
                
                condition_results[metric] = metric_ranking
            
            condition_performance[condition_name] = condition_results
        
        return condition_performance
    
    def create_performance_profiles(self):
        """Create detailed performance profiles for each agent."""
        print("\n" + "="*80)
        print("DETAILED PERFORMANCE PROFILES")
        print("="*80)
        
        profiles = {}
        
        for agent in self.agents:
            print(f"\n{agent} - PERFORMANCE PROFILE:")
            print("=" * 60)
            
            agent_data = self.df[self.df['agent'] == agent]
            
            # Basic statistics
            print("  Basic Statistics:")
            for metric in self.metrics:
                mean_val = agent_data[metric].mean()
                std_val = agent_data[metric].std()
                min_val = agent_data[metric].min()
                max_val = agent_data[metric].max()
                
                print(f"    {metric}: {mean_val:.4f} ± {std_val:.4f} [{min_val:.4f}, {max_val:.4f}]")
            
            # Performance consistency
            print("\n  Performance Consistency:")
            for metric in self.metrics:
                cv = agent_data[metric].std() / agent_data[metric].mean() if agent_data[metric].mean() != 0 else 0
                print(f"    {metric} CV: {cv:.4f}")
            
            # Best and worst conditions
            print("\n  Best Conditions:")
            for metric in self.metrics:
                best_condition = agent_data.loc[agent_data[metric].idxmax()]
                print(f"    {metric}: {best_condition['maze']} + {best_condition['reward_config']} + swap={best_condition['swap_prob']}")
            
            print("\n  Worst Conditions:")
            for metric in self.metrics:
                worst_condition = agent_data.loc[agent_data[metric].idxmin()]
                print(f"    {metric}: {worst_condition['maze']} + {worst_condition['reward_config']} + swap={worst_condition['swap_prob']}")
            
            profiles[agent] = {
                'basic_stats': {metric: {
                    'mean': agent_data[metric].mean(),
                    'std': agent_data[metric].std(),
                    'min': agent_data[metric].min(),
                    'max': agent_data[metric].max()
                } for metric in self.metrics},
                'consistency': {metric: agent_data[metric].std() / agent_data[metric].mean() if agent_data[metric].mean() != 0 else 0 for metric in self.metrics}
            }
        
        return profiles
    
    def generate_agent_recommendations(self):
        """Generate recommendations for when to use each agent."""
        print("\n" + "="*80)
        print("AGENT RECOMMENDATION SYSTEM")
        print("="*80)
        
        recommendations = {}
        
        # Analyze best agents for different scenarios
        scenarios = {
            'Maximum Reward Collection': 'avg_reward',
            'Highest Exit Rate': 'exit_rate',
            'Best Survival Rate': 'survival_rate',
            'Optimal Risk-Adjusted Returns': 'avg_risk_adjusted_return',
            'Most Efficient Navigation': 'avg_path_efficiency'
        }
        
        for scenario, metric in scenarios.items():
            print(f"\n{scenario}:")
            print("-" * 50)
            
            # Overall best
            overall_best = self.df.groupby('agent')[metric].mean().idxmax()
            overall_value = self.df.groupby('agent')[metric].mean().max()
            print(f"  Overall Best: {overall_best} ({overall_value:.4f})")
            
            # Best by condition
            print("  Best by Condition:")
            
            # By swap probability
            for swap_prob in sorted(self.df['swap_prob'].unique()):
                swap_data = self.df[self.df['swap_prob'] == swap_prob]
                if len(swap_data) > 0:
                    best_agent = swap_data.groupby('agent')[metric].mean().idxmax()
                    best_value = swap_data.groupby('agent')[metric].mean().max()
                    print(f"    Swap Prob {swap_prob}: {best_agent} ({best_value:.4f})")
            
            # By reward configuration
            for config in self.df['reward_config'].unique():
                config_data = self.df[self.df['reward_config'] == config]
                if len(config_data) > 0:
                    best_agent = config_data.groupby('agent')[metric].mean().idxmax()
                    best_value = config_data.groupby('agent')[metric].mean().max()
                    print(f"    {config}: {best_agent} ({best_value:.4f})")
            
            recommendations[scenario] = {
                'overall_best': overall_best,
                'overall_value': overall_value
            }
        
        return recommendations
    
    def create_comparative_visualizations(self, save_path='../analysis/comparative_analysis.png'):
        """Create comparative visualization plots."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comparative Agent Analysis', fontsize=16, fontweight='bold')
        
        # 1. Performance comparison across metrics
        ax1 = axes[0,0]
        metric_means = self.df.groupby('agent')[self.metrics].mean()
        metric_means.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Performance Across All Metrics')
        ax1.set_xlabel('Agent')
        ax1.set_ylabel('Normalized Performance')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Performance stability (coefficient of variation)
        ax2 = axes[0,1]
        stability_data = []
        for agent in self.agents:
            agent_data = self.df[self.df['agent'] == agent]
            cv_values = [agent_data[metric].std() / agent_data[metric].mean() if agent_data[metric].mean() != 0 else 0 for metric in self.metrics]
            stability_data.append(cv_values)
        
        stability_df = pd.DataFrame(stability_data, index=self.agents, columns=self.metrics)
        stability_df.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Performance Stability (Lower is Better)')
        ax2.set_xlabel('Agent')
        ax2.set_ylabel('Coefficient of Variation')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Performance by swap probability
        ax3 = axes[0,2]
        for agent in self.agents:
            agent_data = self.df[self.df['agent'] == agent]
            ax3.plot(agent_data['swap_prob'], agent_data['avg_risk_adjusted_return'], 
                    marker='o', label=agent, linewidth=2, markersize=6)
        ax3.set_title('Risk-Adjusted Returns vs Swap Probability')
        ax3.set_xlabel('Swap Probability')
        ax3.set_ylabel('Risk-Adjusted Returns')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance by reward configuration
        ax4 = axes[1,0]
        pivot_data = self.df.groupby(['reward_config', 'agent'])['avg_reward'].mean().unstack()
        pivot_data.plot(kind='bar', ax=ax4, width=0.8)
        ax4.set_title('Average Reward by Reward Configuration')
        ax4.set_xlabel('Reward Configuration')
        ax4.set_ylabel('Average Reward')
        ax4.legend(title='Agent')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Performance by maze complexity
        ax5 = axes[1,1]
        pivot_data = self.df.groupby(['maze', 'agent'])['exit_rate'].mean().unstack()
        pivot_data.plot(kind='bar', ax=ax5, width=0.8)
        ax5.set_title('Exit Rate by Maze')
        ax5.set_xlabel('Maze')
        ax5.set_ylabel('Exit Rate')
        ax5.legend(title='Agent')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Agent ranking heatmap
        ax6 = axes[1,2]
        ranking_data = []
        for metric in self.metrics:
            rankings = self.df.groupby('agent')[metric].mean().rank(ascending=False)
            ranking_data.append(rankings)
        
        ranking_df = pd.DataFrame(ranking_data, index=self.metrics, columns=self.agents)
        sns.heatmap(ranking_df, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax6, cbar_kws={'label': 'Rank (1=Best)'})
        ax6.set_title('Agent Rankings Across Metrics')
        ax6.set_xlabel('Agent')
        ax6.set_ylabel('Metric')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparative analysis visualization saved to {save_path}")
    
    def generate_comprehensive_comparison_report(self):
        """Generate a comprehensive comparative analysis report."""
        print("="*80)
        print("COMPREHENSIVE COMPARATIVE ANALYSIS REPORT")
        print("="*80)
        
        # Run all analyses
        comparison_matrix = self.create_agent_comparison_matrix()
        agent_profiles = self.analyze_agent_strengths_weaknesses()
        condition_performance = self.analyze_condition_specific_performance()
        performance_profiles = self.create_performance_profiles()
        recommendations = self.generate_agent_recommendations()
        
        # Create visualizations
        self.create_comparative_visualizations()
        
        # Summary
        print("\n" + "="*80)
        print("COMPARATIVE ANALYSIS SUMMARY")
        print("="*80)
        
        print("\n1. Key Agent Comparisons:")
        for metric in self.metrics:
            best_agent = self.df.groupby('agent')[metric].mean().idxmax()
            worst_agent = self.df.groupby('agent')[metric].mean().idxmin()
            best_value = self.df.groupby('agent')[metric].mean().max()
            worst_value = self.df.groupby('agent')[metric].mean().min()
            
            print(f"   {metric}: {best_agent} ({best_value:.4f}) vs {worst_agent} ({worst_value:.4f})")
        
        print("\n2. Agent Specializations:")
        for agent in self.agents:
            strengths = agent_profiles[agent]['strengths']
            if strengths:
                best_strength = max(strengths, key=lambda x: x[1])
                print(f"   {agent}: Best at {best_strength[0]} (z={best_strength[1]:.2f})")
        
        print("\n3. Top Recommendations:")
        for scenario, metric in [('Maximum Reward', 'avg_reward'), ('Best Exit Rate', 'exit_rate'), ('Optimal Risk-Adjusted Returns', 'avg_risk_adjusted_return')]:
            best_agent = self.df.groupby('agent')[metric].mean().idxmax()
            print(f"   {scenario}: {best_agent}")
        
        return {
            'comparison_matrix': comparison_matrix,
            'agent_profiles': agent_profiles,
            'condition_performance': condition_performance,
            'performance_profiles': performance_profiles,
            'recommendations': recommendations
        }

if __name__ == "__main__":
    # Create and run comparative analyzer
    analyzer = ComparativeAnalyzer()
    results = analyzer.generate_comprehensive_comparison_report() 