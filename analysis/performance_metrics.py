"""
Performance metrics module for maze navigation experiments.

This module provides comprehensive performance analysis including
efficiency metrics, learning curves, and performance rankings.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceAnalyzer:
    """Performance analysis for maze navigation agents."""
    
    def __init__(self):
        self.results = None
        
    def load_results(self, results_file: str = "comprehensive_results.csv"):
        """Load experiment results for analysis."""
        try:
            self.results = pd.read_csv(results_file)
            print(f"Loaded {len(self.results)} results from {results_file}")
            return True
        except FileNotFoundError:
            print(f"Results file {results_file} not found")
            return False
    
    def add_composite_improvement(self, w1=0.4, w2=0.2, w3=0.3, w4=0.1):
        """Compute and add a composite improvement metric to the results DataFrame."""
        if self.results is None:
            return
        # Compute deltas between first and last 20% of episodes per condition
        def compute_composite(row_group):
            N = len(row_group)
            window = max(1, int(N * 0.2))
            if N < 2 * window:
                return np.nan
            first = row_group.iloc[:window]
            last = row_group.iloc[-window:]
            d_reward = np.mean(last['avg_reward']) - np.mean(first['avg_reward'])
            d_exit = np.mean(last['exit_rate']) - np.mean(first['exit_rate'])
            d_survival = np.mean(last['survival_rate']) - np.mean(first['survival_rate'])
            d_eff = np.mean(last['avg_path_efficiency']) - np.mean(first['avg_path_efficiency'])
            return w1 * d_reward + w2 * d_exit + w3 * d_survival + w4 * d_eff
        # Group by agent/maze/reward/step/swap (or whatever columns exist)
        group_cols = [c for c in ['agent','maze','reward_config','step_budget','swap_prob'] if c in self.results.columns]
        composite_scores = self.results.groupby(group_cols, group_keys=False).apply(compute_composite)
        self.results['composite_improvement'] = composite_scores.values
    
    def calculate_efficiency_metrics(self) -> Dict:
        """Calculate efficiency metrics for each agent."""
        if self.results is None:
            return {}
        self.add_composite_improvement()
        efficiency_metrics = {}
        
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            
            # Reward efficiency (reward per step)
            avg_reward_per_step = np.mean(agent_data['avg_reward'] / agent_data['avg_steps'])
            
            # Path efficiency (collected rewards vs possible rewards)
            avg_collection_rate = np.mean(agent_data['avg_collected_rewards'] / 2.0)  # 2 rewards possible
            
            # Survival efficiency (survival rate vs steps used)
            avg_survival_efficiency = np.mean(agent_data['survival_rate'] * (1 - agent_data['avg_steps'] / agent_data['max_steps']))
            
            # Composite learning efficiency
            avg_learning_efficiency = np.nanmean(agent_data['composite_improvement'] / 100)  # per episode
            
            efficiency_metrics[agent] = {
                'reward_per_step': avg_reward_per_step,
                'collection_rate': avg_collection_rate,
                'survival_efficiency': avg_survival_efficiency,
                'learning_efficiency': avg_learning_efficiency,
                'overall_efficiency': (avg_reward_per_step + avg_collection_rate + avg_survival_efficiency + avg_learning_efficiency) / 4
            }
        
        return efficiency_metrics
    
    def agent_rankings(self) -> Dict:
        """Generate performance rankings for agents."""
        if self.results is None:
            return {}
        
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'composite_improvement', 'avg_collected_rewards']
        rankings = {}
        
        for metric in metrics:
            # Calculate mean performance for each agent
            agent_means = self.results.groupby('agent')[metric].mean().sort_values(ascending=False)
            
            rankings[metric] = {
                'ranking': agent_means.index.tolist(),
                'scores': agent_means.values.tolist()
            }
        
        # Overall ranking (average rank across all metrics)
        overall_ranks = {}
        for agent in self.results['agent'].unique():
            agent_ranks = []
            for metric in metrics:
                agent_means = self.results.groupby('agent')[metric].mean().sort_values(ascending=False)
                rank = list(agent_means.index).index(agent) + 1
                agent_ranks.append(rank)
            overall_ranks[agent] = np.mean(agent_ranks)
        
        # Sort by overall rank
        overall_ranking = sorted(overall_ranks.items(), key=lambda x: x[1])
        rankings['overall'] = {
            'ranking': [agent for agent, _ in overall_ranking],
            'scores': [score for _, score in overall_ranking]
        }
        
        return rankings
    
    def environment_analysis(self) -> Dict:
        """Analyze performance across different environments."""
        if self.results is None:
            return {}
        
        env_analysis = {}
        
        # Maze analysis
        maze_performance = self.results.groupby('maze').agg({
            'avg_reward': 'mean',
            'exit_rate': 'mean',
            'survival_rate': 'mean',
            'composite_improvement': 'mean'
        }).round(3)
        
        env_analysis['maze_performance'] = maze_performance
        
        # Reward configuration analysis
        reward_performance = self.results.groupby('reward_config').agg({
            'avg_reward': 'mean',
            'exit_rate': 'mean',
            'survival_rate': 'mean',
            'composite_improvement': 'mean'
        }).round(3)
        
        env_analysis['reward_performance'] = reward_performance
        
        # Swap probability analysis
        swap_performance = self.results.groupby('swap_prob').agg({
            'avg_reward': 'mean',
            'exit_rate': 'mean',
            'survival_rate': 'mean',
            'composite_improvement': 'mean'
        }).round(3)
        
        env_analysis['swap_performance'] = swap_performance
        
        return env_analysis
    
    def learning_curve_analysis(self) -> Dict:
        """Analyze learning curves and improvement patterns."""
        if self.results is None:
            return {}
        
        learning_analysis = {}
        
        # Learning improvement by agent
        agent_learning = self.results.groupby('agent')['composite_improvement'].agg(['mean', 'std', 'min', 'max']).round(3)
        learning_analysis['agent_learning'] = agent_learning
        
        # Learning improvement by environment
        maze_learning = self.results.groupby('maze')['composite_improvement'].mean().sort_values(ascending=False)
        learning_analysis['maze_learning'] = maze_learning
        
        # Learning improvement by reward configuration
        reward_learning = self.results.groupby('reward_config')['composite_improvement'].mean().sort_values(ascending=False)
        learning_analysis['reward_learning'] = reward_learning
        
        return learning_analysis
    
    def generate_comprehensive_performance_report(self) -> Dict:
        """Generate a comprehensive performance report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("="*60)
        
        if not self.load_results():
            return {}
        
        # Run all analyses
        efficiency_metrics = self.calculate_efficiency_metrics()
        rankings = self.agent_rankings()
        env_analysis = self.environment_analysis()
        learning_analysis = self.learning_curve_analysis()
        
        # Print efficiency metrics
        print("\n" + "="*40)
        print("EFFICIENCY METRICS")
        print("="*40)
        for agent, metrics in efficiency_metrics.items():
            print(f"\n{agent}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.3f}")
        
        # Print rankings
        print("\n" + "="*40)
        print("AGENT RANKINGS")
        print("="*40)
        for metric, ranking_data in rankings.items():
            print(f"\n{metric.upper()}:")
            for i, (agent, score) in enumerate(zip(ranking_data['ranking'], ranking_data['scores']), 1):
                print(f"  {i}. {agent}: {score:.3f}")
        
        # Print environment analysis
        print("\n" + "="*40)
        print("ENVIRONMENT ANALYSIS")
        print("="*40)
        print("\nMaze Performance:")
        print(env_analysis['maze_performance'])
        
        print("\nReward Configuration Performance:")
        print(env_analysis['reward_performance'])
        
        print("\nSwap Probability Performance:")
        print(env_analysis['swap_performance'])
        
        # Save comprehensive report
        report = {
            'efficiency_metrics': efficiency_metrics,
            'rankings': rankings,
            'environment_analysis': env_analysis,
            'learning_analysis': learning_analysis
        }
        
        return report 