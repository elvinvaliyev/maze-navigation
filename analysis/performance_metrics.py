#!/usr/bin/env python3
"""
Advanced Performance Metrics Analysis for Maze Navigation

This module provides comprehensive performance analysis including:
- Multi-criteria ranking systems
- Risk-adjusted performance metrics
- Efficiency calculations
- Performance stability analysis
- Agent specialization analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalyzer:
    def __init__(self, data_path='../comprehensive_results.csv'):
        """Initialize the performance analyzer with data."""
        self.df = pd.read_csv(data_path)
        self.agents = self.df['agent'].unique()
        self.metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'avg_risk_adjusted_return', 'avg_path_efficiency']
        
    def calculate_composite_score(self, weights=None):
        """Calculate a composite performance score using weighted metrics."""
        if weights is None:
            # Default weights: equal importance
            weights = {
                'avg_reward': 0.2,
                'exit_rate': 0.25,
                'survival_rate': 0.25,
                'avg_risk_adjusted_return': 0.2,
                'avg_path_efficiency': 0.1
            }
        
        print("="*80)
        print("COMPOSITE PERFORMANCE SCORE ANALYSIS")
        print("="*80)
        print(f"Weights: {weights}")
        
        # Normalize metrics to 0-1 scale
        normalized_scores = {}
        for metric in self.metrics:
            min_val = self.df[metric].min()
            max_val = self.df[metric].max()
            normalized_scores[metric] = (self.df[metric] - min_val) / (max_val - min_val)
        
        # Calculate weighted composite score
        composite_score = sum(weights[metric] * normalized_scores[metric] for metric in self.metrics)
        
        # Group by agent and calculate statistics
        agent_scores = self.df.groupby('agent').apply(
            lambda x: pd.Series({
                'composite_score': composite_score[x.index].mean(),
                'score_std': composite_score[x.index].std(),
                'score_min': composite_score[x.index].min(),
                'score_max': composite_score[x.index].max()
            })
        ).sort_values('composite_score', ascending=False)
        
        print("\nComposite Performance Rankings:")
        print("-" * 50)
        for i, (agent, row) in enumerate(agent_scores.iterrows(), 1):
            print(f"{i}. {agent}: {row['composite_score']:.4f} ± {row['score_std']:.4f}")
        
        return agent_scores, composite_score
    
    def analyze_performance_stability(self):
        """Analyze the stability/consistency of agent performance."""
        print("\n" + "="*80)
        print("PERFORMANCE STABILITY ANALYSIS")
        print("="*80)
        
        stability_metrics = {}
        
        for metric in self.metrics:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print("-" * 50)
            
            metric_stability = {}
            
            for agent in self.agents:
                agent_data = self.df[self.df['agent'] == agent][metric]
                
                # Calculate stability metrics
                mean_val = agent_data.mean()
                std_val = agent_data.std()
                cv = std_val / mean_val if mean_val != 0 else 0  # Coefficient of variation
                
                # Calculate interquartile range
                q75, q25 = np.percentile(agent_data, [75, 25])
                iqr = q75 - q25
                
                # Calculate range
                data_range = agent_data.max() - agent_data.min()
                
                print(f"  {agent}:")
                print(f"    Mean: {mean_val:.4f}")
                print(f"    Std: {std_val:.4f}")
                print(f"    CV: {cv:.4f}")
                print(f"    IQR: {iqr:.4f}")
                print(f"    Range: {data_range:.4f}")
                
                metric_stability[agent] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv,
                    'iqr': iqr,
                    'range': data_range
                }
            
            stability_metrics[metric] = metric_stability
        
        return stability_metrics
    
    def calculate_risk_adjusted_metrics(self):
        """Calculate various risk-adjusted performance metrics."""
        print("\n" + "="*80)
        print("RISK-ADJUSTED PERFORMANCE METRICS")
        print("="*80)
        
        risk_metrics = {}
        
        for agent in self.agents:
            print(f"\n{agent}:")
            print("-" * 30)
            
            agent_data = self.df[self.df['agent'] == agent]
            
            # Calculate risk-adjusted metrics
            reward_mean = agent_data['avg_reward'].mean()
            reward_std = agent_data['avg_reward'].std()
            
            # Sharpe-like ratio (reward per unit of risk)
            sharpe_ratio = reward_mean / reward_std if reward_std != 0 else 0
            
            # Sortino-like ratio (using downside deviation)
            downside_returns = agent_data['avg_reward'][agent_data['avg_reward'] < reward_mean]
            downside_dev = np.sqrt(np.mean((downside_returns - reward_mean) ** 2)) if len(downside_returns) > 0 else 0
            sortino_ratio = reward_mean / downside_dev if downside_dev != 0 else 0
            
            # Calmar-like ratio (reward per unit of maximum drawdown)
            cumulative_rewards = agent_data['avg_reward'].cumsum()
            running_max = cumulative_rewards.expanding().max()
            drawdown = (cumulative_rewards - running_max) / running_max
            max_drawdown = abs(drawdown.min()) if drawdown.min() < 0 else 0
            calmar_ratio = reward_mean / max_drawdown if max_drawdown != 0 else 0
            
            print(f"  Sharpe-like ratio: {sharpe_ratio:.4f}")
            print(f"  Sortino-like ratio: {sortino_ratio:.4f}")
            print(f"  Calmar-like ratio: {calmar_ratio:.4f}")
            
            risk_metrics[agent] = {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'reward_mean': reward_mean,
                'reward_std': reward_std,
                'max_drawdown': max_drawdown
            }
        
        return risk_metrics
    
    def analyze_agent_specialization(self):
        """Analyze which agents specialize in different types of tasks."""
        print("\n" + "="*80)
        print("AGENT SPECIALIZATION ANALYSIS")
        print("="*80)
        
        specialization_data = {}
        
        # Analyze performance across different conditions
        conditions = {
            'High Risk (High Swap Prob)': self.df['swap_prob'] >= 0.5,
            'Low Risk (Low Swap Prob)': self.df['swap_prob'] <= 0.2,
            'Big Reward Difference': self.df['reward_config'].str.contains('big'),
            'Small Reward Difference': self.df['reward_config'].str.contains('small'),
            'Equal Rewards': self.df['reward_config'].str.contains('equal')
        }
        
        for condition_name, condition_mask in conditions.items():
            print(f"\n{condition_name}:")
            print("-" * 50)
            
            condition_data = self.df[condition_mask]
            if len(condition_data) == 0:
                print("  No data for this condition")
                continue
            
            # Calculate best agent for each metric in this condition
            best_agents = {}
            for metric in self.metrics:
                best_agent = condition_data.groupby('agent')[metric].mean().idxmax()
                best_value = condition_data.groupby('agent')[metric].mean().max()
                best_agents[metric] = (best_agent, best_value)
                print(f"  Best {metric}: {best_agent} ({best_value:.4f})")
            
            specialization_data[condition_name] = best_agents
        
        return specialization_data
    
    def calculate_efficiency_metrics(self):
        """Calculate various efficiency metrics."""
        print("\n" + "="*80)
        print("EFFICIENCY METRICS ANALYSIS")
        print("="*80)
        
        efficiency_data = {}
        
        for agent in self.agents:
            print(f"\n{agent}:")
            print("-" * 30)
            
            agent_data = self.df[self.df['agent'] == agent]
            
            # Reward efficiency (reward per step)
            reward_per_step = agent_data['avg_reward'] / agent_data['avg_steps']
            reward_efficiency = reward_per_step.mean()
            
            # Exit efficiency (exit rate per step)
            exit_per_step = agent_data['exit_rate'] / agent_data['avg_steps']
            exit_efficiency = exit_per_step.mean()
            
            # Survival efficiency (survival rate per step)
            survival_per_step = agent_data['survival_rate'] / agent_data['avg_steps']
            survival_efficiency = survival_per_step.mean()
            
            # Risk-adjusted efficiency
            risk_efficiency = agent_data['avg_risk_adjusted_return'] / agent_data['avg_steps']
            risk_efficiency_mean = risk_efficiency.mean()
            
            print(f"  Reward per step: {reward_efficiency:.6f}")
            print(f"  Exit rate per step: {exit_efficiency:.6f}")
            print(f"  Survival rate per step: {survival_efficiency:.6f}")
            print(f"  Risk-adjusted return per step: {risk_efficiency_mean:.6f}")
            
            efficiency_data[agent] = {
                'reward_per_step': reward_efficiency,
                'exit_per_step': exit_efficiency,
                'survival_per_step': survival_efficiency,
                'risk_per_step': risk_efficiency_mean
            }
        
        return efficiency_data
    
    def create_performance_rankings(self):
        """Create comprehensive performance rankings across all metrics."""
        print("\n" + "="*80)
        print("COMPREHENSIVE PERFORMANCE RANKINGS")
        print("="*80)
        
        rankings = {}
        
        for metric in self.metrics:
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print("-" * 50)
            
            # Calculate rankings
            metric_rankings = self.df.groupby('agent')[metric].mean().sort_values(ascending=False)
            
            for i, (agent, value) in enumerate(metric_rankings.items(), 1):
                print(f"  {i}. {agent}: {value:.4f}")
            
            rankings[metric] = metric_rankings
        
        # Overall ranking based on average rank across all metrics
        print("\nOVERALL RANKING (Average Rank Across All Metrics):")
        print("-" * 50)
        
        agent_ranks = {}
        for agent in self.agents:
            ranks = []
            for metric in self.metrics:
                rank = rankings[metric].index.get_loc(agent) + 1
                ranks.append(rank)
            avg_rank = np.mean(ranks)
            agent_ranks[agent] = avg_rank
        
        overall_ranking = sorted(agent_ranks.items(), key=lambda x: x[1])
        for i, (agent, avg_rank) in enumerate(overall_ranking, 1):
            print(f"  {i}. {agent}: {avg_rank:.2f}")
        
        return rankings, overall_ranking
    
    def analyze_performance_trends(self):
        """Analyze performance trends across different experimental parameters."""
        print("\n" + "="*80)
        print("PERFORMANCE TREND ANALYSIS")
        print("="*80)
        
        trends = {}
        
        # Analyze trends across swap probabilities
        print("\nPerformance Trends by Swap Probability:")
        print("-" * 50)
        
        for metric in self.metrics:
            print(f"\n{metric}:")
            for agent in self.agents:
                agent_data = self.df[self.df['agent'] == agent]
                correlation, p_value = stats.pearsonr(agent_data['swap_prob'], agent_data[metric])
                
                trend_direction = "increasing" if correlation > 0 else "decreasing"
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                print(f"  {agent}: {trend_direction} (r={correlation:.3f}, p={p_value:.6f} {significance})")
        
        # Analyze trends across reward configurations
        print("\nPerformance by Reward Configuration:")
        print("-" * 50)
        
        for metric in self.metrics:
            print(f"\n{metric}:")
            pivot_data = self.df.groupby(['reward_config', 'agent'])[metric].mean().unstack()
            
            for agent in self.agents:
                if agent in pivot_data.columns:
                    values = pivot_data[agent].dropna()
                    if len(values) > 1:
                        # Calculate trend across reward configs
                        config_order = ['equal', 'small', 'big']
                        ordered_values = [values.get(config, np.nan) for config in config_order if config in values.index]
                        if len(ordered_values) >= 2 and not any(np.isnan(ordered_values)):
                            correlation, p_value = stats.pearsonr(range(len(ordered_values)), ordered_values)
                            trend_direction = "increasing" if correlation > 0 else "decreasing"
                            print(f"  {agent}: {trend_direction} across reward configs (r={correlation:.3f})")
        
        return trends
    
    def generate_comprehensive_performance_report(self):
        """Generate a comprehensive performance analysis report."""
        print("="*80)
        print("COMPREHENSIVE PERFORMANCE ANALYSIS REPORT")
        print("="*80)
        
        # Run all analyses
        composite_scores, _ = self.calculate_composite_score()
        stability_metrics = self.analyze_performance_stability()
        risk_metrics = self.calculate_risk_adjusted_metrics()
        specialization_data = self.analyze_agent_specialization()
        efficiency_metrics = self.calculate_efficiency_metrics()
        rankings, overall_ranking = self.create_performance_rankings()
        trends = self.analyze_performance_trends()
        
        # Summary
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        print("\n1. Overall Best Performers:")
        for i, (agent, score) in enumerate(composite_scores.head(3).iterrows(), 1):
            print(f"   {i}. {agent}: {score['composite_score']:.4f}")
        
        print("\n2. Most Stable Performers (Lowest CV):")
        stability_ranking = []
        for agent in self.agents:
            avg_cv = np.mean([stability_metrics[metric][agent]['cv'] for metric in self.metrics])
            stability_ranking.append((agent, avg_cv))
        
        stability_ranking.sort(key=lambda x: x[1])
        for i, (agent, cv) in enumerate(stability_ranking[:3], 1):
            print(f"   {i}. {agent}: CV={cv:.4f}")
        
        print("\n3. Best Risk-Adjusted Performers:")
        risk_ranking = sorted(risk_metrics.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)
        for i, (agent, metrics) in enumerate(risk_ranking[:3], 1):
            print(f"   {i}. {agent}: Sharpe={metrics['sharpe_ratio']:.4f}")
        
        print("\n4. Most Efficient Performers:")
        efficiency_ranking = sorted(efficiency_metrics.items(), key=lambda x: x[1]['risk_per_step'], reverse=True)
        for i, (agent, metrics) in enumerate(efficiency_ranking[:3], 1):
            print(f"   {i}. {agent}: Risk/Step={metrics['risk_per_step']:.6f}")
        
        return {
            'composite_scores': composite_scores,
            'stability_metrics': stability_metrics,
            'risk_metrics': risk_metrics,
            'specialization_data': specialization_data,
            'efficiency_metrics': efficiency_metrics,
            'rankings': rankings,
            'overall_ranking': overall_ranking,
            'trends': trends
        }

if __name__ == "__main__":
    # Create and run performance analyzer
    analyzer = PerformanceAnalyzer()
    results = analyzer.generate_comprehensive_performance_report() 