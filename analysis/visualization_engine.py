"""
Visualization engine module for maze navigation experiments.

This module provides comprehensive visualization capabilities for
experiment results, including plots, charts, and interactive visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os

class VisualizationEngine:
    """Visualization engine for maze navigation experiments."""
    
    def __init__(self):
        self.results = None
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def load_results(self, results_file: str = None):
        """Load experiment results for visualization."""
        if results_file is None:
            results_file = os.path.join(self.output_dir, "comprehensive_results.csv")
        try:
            self.results = pd.read_csv(results_file)
            print(f"Loaded {len(self.results)} results from {results_file}")
            return True
        except FileNotFoundError:
            print(f"Results file {results_file} not found")
            return False
    
    def create_agent_performance_plot(self):
        """Create a comprehensive agent performance plot."""
        if self.results is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement']
        titles = ['Average Reward', 'Exit Rate', 'Survival Rate', 'Learning Improvement']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            
            # Create box plot
            sns.boxplot(data=self.results, x='agent', y=metric, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Agent')
            ax.set_ylabel(title)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'agent_performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_environment_analysis_plots(self):
        """Create plots analyzing performance across environments."""
        if self.results is None:
            return
        
        # Maze performance
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement']
        titles = ['Average Reward by Maze', 'Exit Rate by Maze', 'Survival Rate by Maze', 'Learning Improvement by Maze']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            
            maze_performance = self.results.groupby('maze')[metric].mean().sort_values(ascending=False)
            maze_performance.plot(kind='bar', ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Maze')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'maze_performance_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Reward configuration performance
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            
            reward_performance = self.results.groupby('reward_config')[metric].mean().sort_values(ascending=False)
            reward_performance.plot(kind='bar', ax=ax)
            ax.set_title(title.replace('Maze', 'Reward Config'))
            ax.set_xlabel('Reward Configuration')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'reward_config_performance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_heatmap_visualizations(self):
        """Create heatmap visualizations for agent-environment interactions."""
        if self.results is None:
            return
        
        # Agent-Maze heatmap
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement']
        titles = ['Average Reward', 'Exit Rate', 'Survival Rate', 'Learning Improvement']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            
            # Create pivot table for heatmap
            pivot_data = self.results.pivot_table(
                values=metric, 
                index='agent', 
                columns='maze', 
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax)
            ax.set_title(f'{title} - Agent vs Maze')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'agent_maze_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Agent-Reward Config heatmap
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            
            pivot_data = self.results.pivot_table(
                values=metric, 
                index='agent', 
                columns='reward_config', 
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax)
            ax.set_title(f'{title} - Agent vs Reward Config')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'agent_reward_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_learning_curves(self):
        """Create learning curve visualizations."""
        if self.results is None:
            return
        
        # Learning improvement by agent
        plt.figure(figsize=(12, 8))
        
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            plt.scatter(range(len(agent_data)), agent_data['learning_improvement'], 
                       label=agent, alpha=0.7)
        
        plt.title('Learning Improvement by Agent')
        plt.xlabel('Experiment Index')
        plt.ylabel('Learning Improvement')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'learning_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_statistical_summary_plots(self):
        """Create statistical summary plots."""
        if self.results is None:
            return
        
        # Confidence intervals for agent performance
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement']
        titles = ['Average Reward', 'Exit Rate', 'Survival Rate', 'Learning Improvement']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            
            # Calculate confidence intervals
            agent_stats = []
            for agent in self.results['agent'].unique():
                agent_data = self.results[self.results['agent'] == agent][metric]
                mean_val = agent_data.mean()
                std_err = agent_data.sem()
                agent_stats.append({
                    'agent': agent,
                    'mean': mean_val,
                    'std_err': std_err
                })
            
            # Plot with error bars
            agents = [stat['agent'] for stat in agent_stats]
            means = [stat['mean'] for stat in agent_stats]
            std_errs = [stat['std_err'] for stat in agent_stats]
            
            ax.errorbar(agents, means, yerr=std_errs, fmt='o', capsize=5)
            ax.set_title(f'{title} with 95% CI')
            ax.set_xlabel('Agent')
            ax.set_ylabel(title)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'statistical_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualization types."""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*60)
        
        if not self.load_results():
            return
        
        print("Creating agent performance plots...")
        self.create_agent_performance_plot()
        
        print("Creating environment analysis plots...")
        self.create_environment_analysis_plots()
        
        print("Creating heatmap visualizations...")
        self.create_heatmap_visualizations()
        
        print("Creating learning curves...")
        self.create_learning_curves()
        
        print("Creating statistical summary plots...")
        self.create_statistical_summary_plots()
        
        print("✓ All visualizations generated successfully!")
        print(f"Output directory: {self.output_dir}")
        
        # List generated files
        generated_files = [
            'agent_performance_comparison.png',
            'maze_performance_analysis.png',
            'reward_config_performance.png',
            'agent_maze_heatmap.png',
            'agent_reward_heatmap.png',
            'learning_curves.png',
            'statistical_summary.png'
        ]
        
        print("\nGenerated visualization files:")
        for file in generated_files:
            filepath = os.path.join(self.output_dir, file)
            if os.path.exists(filepath):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} (not found)") 