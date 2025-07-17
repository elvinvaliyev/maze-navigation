"""
Learning trajectory analysis module for maze navigation experiments.

This module analyzes how agents learn over time, including learning curves,
plateau detection, and transfer learning analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')
import os
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

class LearningTrajectoryAnalyzer:
    """Learning trajectory analysis for agent performance over time."""
    
    def __init__(self):
        self.results = None
        self.learning_data = {}
        
    def load_results(self, results_file: str = "results/comprehensive_results.csv"):
        """Load experiment results for analysis."""
        try:
            self.results = pd.read_csv(results_file)
            # — if the CSV has only reward_improvement, make sure learning_improvement exists —
            if 'learning_improvement' not in self.results.columns and 'reward_improvement' in self.results.columns:
                self.results['learning_improvement'] = self.results['reward_improvement']
            print(f"Loaded {len(self.results)} results for learning trajectory analysis")
            return True
        except FileNotFoundError:
            print(f"Results file {results_file} not found")
            return False
    
    def analyze_learning_curves(self) -> Dict:
        """Analyze learning curves for each agent across different conditions."""
        if self.results is None:
            return {}
        
        learning_analysis = {}
        
        # Analyze learning improvement across different conditions
        conditions = ['maze', 'reward_config', 'swap_prob', 'step_budget']
        
        for condition in conditions:
            print(f"\n=== LEARNING ANALYSIS BY {condition.upper()} ===")
            condition_learning = {}
            
            for agent in self.results['agent'].unique():
                agent_data = self.results[self.results['agent'] == agent]
                
                # Group by condition and analyze learning improvement
                learning_by_condition = agent_data.groupby(condition)['learning_improvement'].agg([
                    'mean', 'std', 'count'
                ]).round(3)
                
                # Calculate learning rate (improvement per episode)
                learning_by_condition['learning_rate'] = (
                    learning_by_condition['mean'] / learning_by_condition['count']
                )
                
                # Find best and worst learning conditions
                best_condition = learning_by_condition['mean'].idxmax()
                worst_condition = learning_by_condition['mean'].idxmin()
                
                condition_learning[agent] = {
                    'learning_by_condition': learning_by_condition.to_dict(),
                    'best_condition': (best_condition, learning_by_condition.loc[best_condition, 'mean']),
                    'worst_condition': (worst_condition, learning_by_condition.loc[worst_condition, 'mean']),
                    'avg_learning_rate': learning_by_condition['learning_rate'].mean()
                }
                
                print(f"{agent}:")
                print(f"  Best {condition}: {best_condition} ({learning_by_condition.loc[best_condition, 'mean']:.3f})")
                print(f"  Worst {condition}: {worst_condition} ({learning_by_condition.loc[worst_condition, 'mean']:.3f})")
                print(f"  Avg learning rate: {learning_by_condition['learning_rate'].mean():.3f}")
            
            learning_analysis[condition] = condition_learning
        
        return learning_analysis
    
    def detect_learning_plateaus(self) -> Dict:
        """Detect when agents reach learning plateaus."""
        if self.results is None:
            return {}
        
        plateau_analysis = {}
        
        for agent in self.results['agent'].unique():
            print(f"\n=== PLATEAU ANALYSIS FOR {agent} ===")
            agent_data = self.results[self.results['agent'] == agent]
            
            # Analyze learning improvement distribution
            learning_improvements = agent_data['learning_improvement'].values
            
            # Detect plateaus using statistical methods
            mean_improvement = np.mean(learning_improvements)
            std_improvement = np.std(learning_improvements)
            
            # Define plateau as improvement within 1 standard deviation of mean
            plateau_threshold = mean_improvement + std_improvement
            plateau_cases = learning_improvements <= plateau_threshold
            
            plateau_percentage = np.mean(plateau_cases) * 100
            
            # Analyze plateau by conditions
            plateau_by_condition = {}
            for condition in ['maze', 'reward_config', 'swap_prob', 'step_budget']:
                condition_plateaus = agent_data.groupby(condition)['learning_improvement'].apply(
                    lambda x: np.mean(x <= (x.mean() + x.std())) * 100
                )
                plateau_by_condition[condition] = condition_plateaus.to_dict()
            
            plateau_analysis[agent] = {
                'overall_plateau_percentage': plateau_percentage,
                'mean_improvement': mean_improvement,
                'std_improvement': std_improvement,
                'plateau_by_condition': plateau_by_condition,
                'plateau_threshold': plateau_threshold
            }
            
            print(f"Overall plateau percentage: {plateau_percentage:.1f}%")
            print(f"Mean improvement: {mean_improvement:.3f}")
            print(f"Plateau threshold: {plateau_threshold:.3f}")
        
        return plateau_analysis
    
    def analyze_learning_transfer(self) -> Dict:
        """Analyze transfer learning between different conditions."""
        if self.results is None:
            return {}
        
        transfer_analysis = {}
        
        # Analyze transfer between mazes
        print("\n=== TRANSFER LEARNING ANALYSIS ===")
        
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            
            # Compare learning improvement between mazes
            maze_learning = agent_data.groupby('maze')['learning_improvement'].agg(['mean', 'std'])
            
            # Calculate transfer efficiency (how well learning transfers)
            if len(maze_learning) > 1:
                learning_variance = maze_learning['mean'].var()
                learning_mean = maze_learning['mean'].mean()
                transfer_efficiency = 1 - (learning_variance / (learning_mean + 1e-6))
            else:
                transfer_efficiency = 1.0
            
            # Analyze transfer between reward configurations
            reward_learning = agent_data.groupby('reward_config')['learning_improvement'].agg(['mean', 'std'])
            if len(reward_learning) > 1:
                reward_variance = reward_learning['mean'].var()
                reward_mean = reward_learning['mean'].mean()
                reward_transfer_efficiency = 1 - (reward_variance / (reward_mean + 1e-6))
            else:
                reward_transfer_efficiency = 1.0
            
            transfer_analysis[agent] = {
                'maze_transfer_efficiency': transfer_efficiency,
                'reward_transfer_efficiency': reward_transfer_efficiency,
                'maze_learning': maze_learning.to_dict(),
                'reward_learning': reward_learning.to_dict()
            }
            
            print(f"{agent}:")
            print(f"  Maze transfer efficiency: {transfer_efficiency:.3f}")
            print(f"  Reward transfer efficiency: {reward_transfer_efficiency:.3f}")
        
        return transfer_analysis
    
    def analyze_learning_rate_evolution(self) -> Dict:
        """Analyze how learning rates evolve over different conditions."""
        if self.results is None:
            return {}
        
        evolution_analysis = {}
        
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            
            # Analyze learning rate evolution across step budgets
            budget_learning = agent_data.groupby('step_budget')['learning_improvement'].mean()
            
            # Fit learning curve model
            try:
                # Simple exponential decay model: y = a * (1 - exp(-b * x))
                def learning_model(x, a, b):
                    return a * (1 - np.exp(-b * x))
                
                x_data = budget_learning.index.values
                y_data = budget_learning.values
                
                # Normalize x data
                x_norm = (x_data - x_data.min()) / (x_data.max() - x_data.min())
                
                popt, pcov = curve_fit(learning_model, x_norm, y_data, maxfev=1000)
                a_fit, b_fit = popt
                
                # Calculate learning rate parameters
                max_learning = a_fit
                learning_rate = b_fit
                half_life = np.log(2) / b_fit if b_fit > 0 else float('inf')
                
            except (RuntimeError, ValueError):
                max_learning = np.max(y_data)
                learning_rate = 0
                half_life = float('inf')
            
            evolution_analysis[agent] = {
                'max_learning': max_learning,
                'learning_rate': learning_rate,
                'half_life': half_life,
                'budget_learning': budget_learning.to_dict()
            }
            
            print(f"{agent} learning evolution:")
            print(f"  Max learning potential: {max_learning:.3f}")
            print(f"  Learning rate: {learning_rate:.3f}")
            print(f"  Half-life: {half_life:.1f} budget units")
        
        return evolution_analysis
    
    def create_learning_visualizations(self):
        """Create visualizations for learning trajectory analysis."""
        if self.results is None:
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create learning curves plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Learning Trajectory Analysis', fontsize=16, fontweight='bold')
        
        # 1. Learning improvement by agent
        ax1 = axes[0, 0]
        learning_by_agent = self.results.groupby('agent')['learning_improvement'].agg(['mean', 'std'])
        agents = learning_by_agent.index
        means = learning_by_agent['mean'].values
        stds = learning_by_agent['std'].values
        
        bars = ax1.bar(agents, means, yerr=stds, capsize=5, alpha=0.7)
        ax1.set_xlabel('Agent')
        ax1.set_ylabel('Learning Improvement')
        ax1.set_title('Learning Improvement by Agent')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        # 2. Learning improvement by step budget
        ax2 = axes[0, 1]
        budget_learning = self.results.groupby('step_budget')['learning_improvement'].mean()
        ax2.plot(budget_learning.index, budget_learning.values, marker='o', linewidth=2)
        ax2.set_xlabel('Step Budget')
        ax2.set_ylabel('Learning Improvement')
        ax2.set_title('Learning Improvement vs Step Budget')
        ax2.grid(True, alpha=0.3)
        
        # 3. Learning improvement by swap probability
        ax3 = axes[1, 0]
        swap_learning = self.results.groupby('swap_prob')['learning_improvement'].mean()
        ax3.plot(swap_learning.index, swap_learning.values, marker='s', linewidth=2)
        ax3.set_xlabel('Swap Probability')
        ax3.set_ylabel('Learning Improvement')
        ax3.set_title('Learning Improvement vs Swap Probability')
        ax3.grid(True, alpha=0.3)
        
        # 4. Learning improvement by maze
        ax4 = axes[1, 1]
        maze_learning = self.results.groupby('maze')['learning_improvement'].mean()
        bars = ax4.bar(maze_learning.index, maze_learning.values, alpha=0.7)
        ax4.set_xlabel('Maze')
        ax4.set_ylabel('Learning Improvement')
        ax4.set_title('Learning Improvement by Maze')
        
        # Add value labels
        for bar, value in zip(bars, maze_learning.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'learning_trajectory_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create agent-specific learning plots
        self._create_agent_learning_plots()
    
    def _create_agent_learning_plots(self):
        """Create agent-specific learning plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Agent-Specific Learning Analysis', fontsize=16, fontweight='bold')
        
        # 1. Learning improvement by agent and step budget
        ax1 = axes[0, 0]
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            learning_by_budget = agent_data.groupby('step_budget')['learning_improvement'].mean()
            ax1.plot(learning_by_budget.index, learning_by_budget.values, marker='o', label=agent, linewidth=2)
        
        ax1.set_xlabel('Step Budget')
        ax1.set_ylabel('Learning Improvement')
        ax1.set_title('Learning Improvement vs Step Budget by Agent')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Learning improvement by agent and swap probability
        ax2 = axes[0, 1]
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            learning_by_swap = agent_data.groupby('swap_prob')['learning_improvement'].mean()
            ax2.plot(learning_by_swap.index, learning_by_swap.values, marker='s', label=agent, linewidth=2)
        
        ax2.set_xlabel('Swap Probability')
        ax2.set_ylabel('Learning Improvement')
        ax2.set_title('Learning Improvement vs Swap Probability by Agent')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Learning improvement by agent and maze
        ax3 = axes[1, 0]
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            learning_by_maze = agent_data.groupby('maze')['learning_improvement'].mean()
            ax3.plot(learning_by_maze.index, learning_by_maze.values, marker='^', label=agent, linewidth=2)
        
        ax3.set_xlabel('Maze')
        ax3.set_ylabel('Learning Improvement')
        ax3.set_title('Learning Improvement by Maze and Agent')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Learning improvement distribution by agent
        ax4 = axes[1, 1]
        learning_data = []
        labels = []
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            learning_data.append(agent_data['learning_improvement'].values)
            labels.append(agent)
        
        ax4.boxplot(learning_data, labels=labels)
        ax4.set_xlabel('Agent')
        ax4.set_ylabel('Learning Improvement')
        ax4.set_title('Learning Improvement Distribution by Agent')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'agent_learning_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_learning_report(self) -> Dict:
        """Generate a comprehensive learning trajectory analysis report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE LEARNING TRAJECTORY ANALYSIS")
        print("="*60)
        
        if not self.load_results():
            return {}
        
        # Run all learning analyses
        learning_curves = self.analyze_learning_curves()
        plateau_detection = self.detect_learning_plateaus()
        transfer_analysis = self.analyze_learning_transfer()
        evolution_analysis = self.analyze_learning_rate_evolution()
        
        # Create visualizations
        self.create_learning_visualizations()
        
        # Generate summary
        print("\n" + "="*60)
        print("LEARNING TRAJECTORY SUMMARY")
        print("="*60)
        
        # Best learners
        print("\nBEST LEARNING AGENTS:")
        agent_learning = self.results.groupby('agent')['learning_improvement'].mean()
        best_learner = agent_learning.idxmax()
        worst_learner = agent_learning.idxmin()
        print(f"  Best learner: {best_learner} ({agent_learning[best_learner]:.3f})")
        print(f"  Worst learner: {worst_learner} ({agent_learning[worst_learner]:.3f})")
        
        # Learning conditions
        print("\nBEST LEARNING CONDITIONS:")
        for condition in ['maze', 'reward_config', 'swap_prob', 'step_budget']:
            condition_learning = self.results.groupby(condition)['learning_improvement'].mean()
            best_condition = condition_learning.idxmax()
            print(f"  {condition}: {best_condition} ({condition_learning[best_condition]:.3f})")
        
        # Transfer learning summary
        print("\nTRANSFER LEARNING SUMMARY:")
        for agent, transfer_data in transfer_analysis.items():
            maze_transfer = transfer_data['maze_transfer_efficiency']
            reward_transfer = transfer_data['reward_transfer_efficiency']
            print(f"  {agent}: maze transfer={maze_transfer:.3f}, reward transfer={reward_transfer:.3f}")
        
        report = {
            'learning_curves': learning_curves,
            'plateau_detection': plateau_detection,
            'transfer_analysis': transfer_analysis,
            'evolution_analysis': evolution_analysis
        }
        
        return report 