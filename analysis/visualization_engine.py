#!/usr/bin/env python3
"""
Advanced Visualization Engine for Maze Navigation Analysis

This module provides comprehensive visualization capabilities including:
- Performance heatmaps
- Statistical comparisons
- Agent behavior analysis
- Reward configuration impact
- Swap probability effects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VisualizationEngine:
    def __init__(self, data_path='../comprehensive_results.csv'):
        """Initialize the visualization engine with data."""
        self.df = pd.read_csv(data_path)
        self.setup_colors()
        
    def setup_colors(self):
        """Setup color schemes for different agents."""
        self.agent_colors = {
            'Model-Based Greedy': '#FF6B6B',
            'Model-Based Survival': '#4ECDC4', 
            'SR-Greedy': '#45B7D1',
            'SR-Reasonable': '#96CEB4'
        }
        
        self.metric_colors = {
            'avg_reward': '#FF6B6B',
            'exit_rate': '#4ECDC4',
            'survival_rate': '#45B7D1',
            'avg_risk_adjusted_return': '#96CEB4',
            'avg_path_efficiency': '#FFEAA7'
        }
    
    def create_comprehensive_dashboard(self, save_path='../analysis/comprehensive_dashboard.png'):
        """Create a comprehensive dashboard with all key metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive Agent Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Average Reward by Agent
        self._plot_metric_by_agent('avg_reward', axes[0,0], 'Average Reward')
        
        # 2. Exit Rate by Agent
        self._plot_metric_by_agent('exit_rate', axes[0,1], 'Exit Rate')
        
        # 3. Survival Rate by Agent
        self._plot_metric_by_agent('survival_rate', axes[0,2], 'Survival Rate')
        
        # 4. Risk-Adjusted Returns by Agent
        self._plot_metric_by_agent('avg_risk_adjusted_return', axes[1,0], 'Risk-Adjusted Returns')
        
        # 5. Path Efficiency by Agent
        self._plot_metric_by_agent('avg_path_efficiency', axes[1,1], 'Path Efficiency')
        
        # 6. Performance Heatmap
        self._create_performance_heatmap(axes[1,2])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Dashboard saved to {save_path}")
    
    def _plot_metric_by_agent(self, metric, ax, title):
        """Plot a specific metric by agent."""
        data = self.df.groupby('agent')[metric].agg(['mean', 'std']).reset_index()
        
        bars = ax.bar(data['agent'], data['mean'], 
                     yerr=data['std'], 
                     color=[self.agent_colors[agent] for agent in data['agent']],
                     alpha=0.8, capsize=5)
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    def _create_performance_heatmap(self, ax):
        """Create a performance heatmap across all metrics."""
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'avg_risk_adjusted_return', 'avg_path_efficiency']
        pivot_data = self.df.groupby('agent')[metrics].mean()
        
        # Normalize data for better visualization
        normalized_data = (pivot_data - pivot_data.min()) / (pivot_data.max() - pivot_data.min())
        
        sns.heatmap(normalized_data.T, annot=True, fmt='.2f', cmap='RdYlGn', 
                   ax=ax, cbar_kws={'label': 'Normalized Performance'})
        ax.set_title('Normalized Performance Heatmap', fontweight='bold')
        ax.set_xlabel('Agent')
        ax.set_ylabel('Metric')
    
    def create_swap_probability_analysis(self, save_path='../analysis/swap_probability_analysis.png'):
        """Analyze the impact of swap probabilities on agent performance."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Swap Probability Impact Analysis', fontsize=16, fontweight='bold')
        
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'avg_risk_adjusted_return']
        titles = ['Average Reward', 'Exit Rate', 'Survival Rate', 'Risk-Adjusted Returns']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            for agent in self.df['agent'].unique():
                agent_data = self.df[self.df['agent'] == agent]
                ax.plot(agent_data['swap_prob'], agent_data[metric], 
                       marker='o', linewidth=2, markersize=8,
                       label=agent, color=self.agent_colors[agent])
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Swap Probability')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Swap probability analysis saved to {save_path}")
    
    def create_reward_configuration_analysis(self, save_path='../analysis/reward_configuration_analysis.png'):
        """Analyze performance across different reward configurations."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Reward Configuration Impact Analysis', fontsize=16, fontweight='bold')
        
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'avg_risk_adjusted_return']
        titles = ['Average Reward', 'Exit Rate', 'Survival Rate', 'Risk-Adjusted Returns']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            pivot_data = self.df.groupby(['reward_config', 'agent'])[metric].mean().unstack()
            
            pivot_data.plot(kind='bar', ax=ax, color=[self.agent_colors[agent] for agent in pivot_data.columns])
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Reward Configuration')
            ax.set_ylabel(title)
            ax.legend(title='Agent')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Reward configuration analysis saved to {save_path}")
    
    def create_agent_comparison_radar(self, save_path='../analysis/agent_radar_comparison.png'):
        """Create radar charts comparing all agents across metrics."""
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'avg_risk_adjusted_return', 'avg_path_efficiency']
        metric_labels = ['Avg Reward', 'Exit Rate', 'Survival Rate', 'Risk-Adjusted', 'Path Efficiency']
        
        # Normalize data
        normalized_data = {}
        for agent in self.df['agent'].unique():
            agent_data = self.df[self.df['agent'] == agent][metrics].mean()
            normalized_data[agent] = (agent_data - agent_data.min()) / (agent_data.max() - agent_data.min())
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for agent, values in normalized_data.items():
            values = values.tolist()
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=agent, color=self.agent_colors[agent])
            ax.fill(angles, values, alpha=0.25, color=self.agent_colors[agent])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Agent Performance Comparison (Normalized)', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Radar comparison saved to {save_path}")
    
    def create_statistical_significance_plot(self, save_path='../analysis/statistical_significance.png'):
        """Create a plot showing statistical significance between agents."""
        from scipy import stats
        
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'avg_risk_adjusted_return']
        metric_names = ['Average Reward', 'Exit Rate', 'Survival Rate', 'Risk-Adjusted Returns']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Significance Analysis (p-values)', fontsize=16, fontweight='bold')
        
        agents = self.df['agent'].unique()
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            
            # Calculate p-values between all agent pairs
            p_values = np.zeros((len(agents), len(agents)))
            for j, agent1 in enumerate(agents):
                for k, agent2 in enumerate(agents):
                    if j != k:
                        data1 = self.df[self.df['agent'] == agent1][metric]
                        data2 = self.df[self.df['agent'] == agent2][metric]
                        _, p_val = stats.ttest_ind(data1, data2)
                        p_values[j, k] = p_val
            
            # Create heatmap
            im = ax.imshow(p_values, cmap='RdYlBu_r', vmin=0, vmax=0.05)
            ax.set_xticks(range(len(agents)))
            ax.set_yticks(range(len(agents)))
            ax.set_xticklabels([agent.split()[-1] for agent in agents], rotation=45)
            ax.set_yticklabels([agent.split()[-1] for agent in agents])
            ax.set_title(f'{name} - p-values')
            
            # Add text annotations
            for j in range(len(agents)):
                for k in range(len(agents)):
                    if j != k:
                        text = ax.text(k, j, f'{p_values[j, k]:.3f}',
                                     ha="center", va="center", color="black", fontsize=8)
            
            plt.colorbar(im, ax=ax, label='p-value')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Statistical significance plot saved to {save_path}")
    
    def create_maze_complexity_analysis(self, save_path='../analysis/maze_complexity_analysis.png'):
        """Analyze performance across different maze complexities."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Maze Complexity Impact Analysis', fontsize=16, fontweight='bold')
        
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'avg_risk_adjusted_return']
        titles = ['Average Reward', 'Exit Rate', 'Survival Rate', 'Risk-Adjusted Returns']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            pivot_data = self.df.groupby(['maze', 'agent'])[metric].mean().unstack()
            
            pivot_data.plot(kind='bar', ax=ax, color=[self.agent_colors[agent] for agent in pivot_data.columns])
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Maze')
            ax.set_ylabel(title)
            ax.legend(title='Agent')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Maze complexity analysis saved to {save_path}")
    
    def create_performance_trends(self, save_path='../analysis/performance_trends.png'):
        """Create trend analysis across different experimental conditions."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Trends Across Experimental Conditions', fontsize=16, fontweight='bold')
        
        # 1. Performance by swap probability and reward config
        ax1 = axes[0,0]
        pivot1 = self.df.groupby(['swap_prob', 'reward_config'])['avg_risk_adjusted_return'].mean().unstack()
        pivot1.plot(kind='line', marker='o', ax=ax1)
        ax1.set_title('Risk-Adjusted Returns by Swap Prob & Reward Config')
        ax1.set_xlabel('Swap Probability')
        ax1.set_ylabel('Risk-Adjusted Returns')
        ax1.legend(title='Reward Config')
        ax1.grid(True, alpha=0.3)
        
        # 2. Exit rate by maze and agent
        ax2 = axes[0,1]
        pivot2 = self.df.groupby(['maze', 'agent'])['exit_rate'].mean().unstack()
        pivot2.plot(kind='bar', ax=ax2, color=[self.agent_colors[agent] for agent in pivot2.columns])
        ax2.set_title('Exit Rate by Maze and Agent')
        ax2.set_xlabel('Maze')
        ax2.set_ylabel('Exit Rate')
        ax2.legend(title='Agent')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Survival rate by swap probability
        ax3 = axes[1,0]
        for agent in self.df['agent'].unique():
            agent_data = self.df[self.df['agent'] == agent]
            ax3.plot(agent_data['swap_prob'], agent_data['survival_rate'], 
                    marker='s', linewidth=2, label=agent, color=self.agent_colors[agent])
        ax3.set_title('Survival Rate by Swap Probability')
        ax3.set_xlabel('Swap Probability')
        ax3.set_ylabel('Survival Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Path efficiency by reward configuration
        ax4 = axes[1,1]
        pivot4 = self.df.groupby(['reward_config', 'agent'])['avg_path_efficiency'].mean().unstack()
        pivot4.plot(kind='bar', ax=ax4, color=[self.agent_colors[agent] for agent in pivot4.columns])
        ax4.set_title('Path Efficiency by Reward Config and Agent')
        ax4.set_xlabel('Reward Configuration')
        ax4.set_ylabel('Path Efficiency')
        ax4.legend(title='Agent')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance trends saved to {save_path}")
    
    def generate_all_visualizations(self):
        """Generate all visualization types."""
        print("Generating comprehensive visualizations...")
        
        self.create_comprehensive_dashboard()
        self.create_swap_probability_analysis()
        self.create_reward_configuration_analysis()
        self.create_agent_comparison_radar()
        self.create_statistical_significance_plot()
        self.create_maze_complexity_analysis()
        self.create_performance_trends()
        
        print("All visualizations completed!")

if __name__ == "__main__":
    # Create and run visualization engine
    viz_engine = VisualizationEngine()
    viz_engine.generate_all_visualizations() 