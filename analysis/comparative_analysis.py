"""
Comparative analysis module for maze navigation experiments.

This module provides detailed comparison between different agents,
environments, and experimental conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class ComparativeAnalyzer:
    """Comparative analysis for maze navigation experiments."""
    
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
    
    def agent_comparison_matrix(self) -> pd.DataFrame:
        """Create a comparison matrix between all agents."""
        if self.results is None:
            return pd.DataFrame()
        
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement', 'avg_collected_rewards']
        comparison_data = []
        
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            row = {'agent': agent}
            
            for metric in metrics:
                row[f'{metric}_mean'] = agent_data[metric].mean()
                row[f'{metric}_std'] = agent_data[metric].std()
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def environment_comparison(self) -> Dict:
        """Compare performance across different environments."""
        if self.results is None:
            return {}
        
        comparisons = {}
        
        # Maze comparison
        maze_comparison = self.results.groupby('maze').agg({
            'avg_reward': ['mean', 'std'],
            'exit_rate': ['mean', 'std'],
            'survival_rate': ['mean', 'std'],
            'learning_improvement': ['mean', 'std']
        }).round(3)
        
        comparisons['maze'] = maze_comparison
        
        # Reward configuration comparison
        reward_comparison = self.results.groupby('reward_config').agg({
            'avg_reward': ['mean', 'std'],
            'exit_rate': ['mean', 'std'],
            'survival_rate': ['mean', 'std'],
            'learning_improvement': ['mean', 'std']
        }).round(3)
        
        comparisons['reward_config'] = reward_comparison
        
        # Swap probability comparison
        swap_comparison = self.results.groupby('swap_prob').agg({
            'avg_reward': ['mean', 'std'],
            'exit_rate': ['mean', 'std'],
            'survival_rate': ['mean', 'std'],
            'learning_improvement': ['mean', 'std']
        }).round(3)
        
        comparisons['swap_prob'] = swap_comparison
        
        return comparisons
    
    def agent_environment_interaction(self) -> Dict:
        """Analyze how agents perform differently across environments."""
        if self.results is None:
            return {}
        
        interactions = {}
        
        # Agent-Maze interactions
        agent_maze = self.results.groupby(['agent', 'maze']).agg({
            'avg_reward': 'mean',
            'exit_rate': 'mean',
            'survival_rate': 'mean',
            'learning_improvement': 'mean'
        }).round(3)
        
        interactions['agent_maze'] = agent_maze
        
        # Agent-Reward interactions
        agent_reward = self.results.groupby(['agent', 'reward_config']).agg({
            'avg_reward': 'mean',
            'exit_rate': 'mean',
            'survival_rate': 'mean',
            'learning_improvement': 'mean'
        }).round(3)
        
        interactions['agent_reward'] = agent_reward
        
        # Agent-Swap interactions
        agent_swap = self.results.groupby(['agent', 'swap_prob']).agg({
            'avg_reward': 'mean',
            'exit_rate': 'mean',
            'survival_rate': 'mean',
            'learning_improvement': 'mean'
        }).round(3)
        
        interactions['agent_swap'] = agent_swap
        
        return interactions
    
    def best_performing_conditions(self) -> Dict:
        """Find the best performing conditions for each agent."""
        if self.results is None:
            return {}
        
        best_conditions = {}
        
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            
            # Best maze
            best_maze = agent_data.groupby('maze')['avg_reward'].mean().idxmax()
            best_maze_score = agent_data.groupby('maze')['avg_reward'].mean().max()
            
            # Best reward configuration
            best_reward = agent_data.groupby('reward_config')['avg_reward'].mean().idxmax()
            best_reward_score = agent_data.groupby('reward_config')['avg_reward'].mean().max()
            
            # Best swap probability
            best_swap = agent_data.groupby('swap_prob')['avg_reward'].mean().idxmax()
            best_swap_score = agent_data.groupby('swap_prob')['avg_reward'].mean().max()
            
            best_conditions[agent] = {
                'best_maze': (best_maze, best_maze_score),
                'best_reward_config': (best_reward, best_reward_score),
                'best_swap_prob': (best_swap, best_swap_score)
            }
        
        return best_conditions
    
    def worst_performing_conditions(self) -> Dict:
        """Find the worst performing conditions for each agent."""
        if self.results is None:
            return {}
        
        worst_conditions = {}
        
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            
            # Worst maze
            worst_maze = agent_data.groupby('maze')['avg_reward'].mean().idxmin()
            worst_maze_score = agent_data.groupby('maze')['avg_reward'].mean().min()
            
            # Worst reward configuration
            worst_reward = agent_data.groupby('reward_config')['avg_reward'].mean().idxmin()
            worst_reward_score = agent_data.groupby('reward_config')['avg_reward'].mean().min()
            
            # Worst swap probability
            worst_swap = agent_data.groupby('swap_prob')['avg_reward'].mean().idxmin()
            worst_swap_score = agent_data.groupby('swap_prob')['avg_reward'].mean().min()
            
            worst_conditions[agent] = {
                'worst_maze': (worst_maze, worst_maze_score),
                'worst_reward_config': (worst_reward, worst_reward_score),
                'worst_swap_prob': (worst_swap, worst_swap_score)
            }
        
        return worst_conditions
    
    def generate_comprehensive_comparison_report(self) -> Dict:
        """Generate a comprehensive comparison report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE COMPARATIVE ANALYSIS")
        print("="*60)
        
        if not self.load_results():
            return {}
        
        # Run all comparisons
        comparison_matrix = self.agent_comparison_matrix()
        env_comparisons = self.environment_comparison()
        interactions = self.agent_environment_interaction()
        best_conditions = self.best_performing_conditions()
        worst_conditions = self.worst_performing_conditions()
        
        # Print comparison matrix
        print("\n" + "="*40)
        print("AGENT COMPARISON MATRIX")
        print("="*40)
        print(comparison_matrix)
        
        # Print best conditions
        print("\n" + "="*40)
        print("BEST PERFORMING CONDITIONS")
        print("="*40)
        for agent, conditions in best_conditions.items():
            print(f"\n{agent}:")
            print(f"  Best Maze: {conditions['best_maze'][0]} (score: {conditions['best_maze'][1]:.3f})")
            print(f"  Best Reward Config: {conditions['best_reward_config'][0]} (score: {conditions['best_reward_config'][1]:.3f})")
            print(f"  Best Swap Prob: {conditions['best_swap_prob'][0]} (score: {conditions['best_swap_prob'][1]:.3f})")
        
        # Print worst conditions
        print("\n" + "="*40)
        print("WORST PERFORMING CONDITIONS")
        print("="*40)
        for agent, conditions in worst_conditions.items():
            print(f"\n{agent}:")
            print(f"  Worst Maze: {conditions['worst_maze'][0]} (score: {conditions['worst_maze'][1]:.3f})")
            print(f"  Worst Reward Config: {conditions['worst_reward_config'][0]} (score: {conditions['worst_reward_config'][1]:.3f})")
            print(f"  Worst Swap Prob: {conditions['worst_swap_prob'][0]} (score: {conditions['worst_swap_prob'][1]:.3f})")
        
        # Save comprehensive report
        report = {
            'comparison_matrix': comparison_matrix,
            'environment_comparisons': env_comparisons,
            'interactions': interactions,
            'best_conditions': best_conditions,
            'worst_conditions': worst_conditions
        }
        
        return report 