#!/usr/bin/env python3
"""
Analysis Script for Existing CSV Results

This script loads the existing comprehensive_results.csv and generates all analysis
and visualizations without running experiments again.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy import stats

# Add analysis directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis'))

# Import advanced analysis modules
from sensitivity_analysis import SensitivityAnalyzer
from learning_trajectory_analysis import LearningTrajectoryAnalyzer
from behavioral_analysis import BehavioralAnalyzer
from enhanced_statistical_analysis import EnhancedStatisticalAnalyzer

class CSVAnalyzer:
    """Analyzes existing CSV results and generates visualizations."""
    
    def __init__(self):
        self.results_dir = os.path.join(os.path.dirname(__file__), 'results')
        self.csv_file = os.path.join(self.results_dir, 'comprehensive_results.csv')
        
        # Load existing results
        if os.path.exists(self.csv_file):
            self.df = pd.read_csv(self.csv_file)
            # Alias reward_improvement to learning_improvement if needed
            if 'learning_improvement' not in self.df.columns and 'reward_improvement' in self.df.columns:
                self.df['learning_improvement'] = self.df['reward_improvement']
            print(f"✅ Loaded {len(self.df)} results from {self.csv_file}")
            print(f"   Shape: {self.df.shape}")
        else:
            print(f"❌ CSV file not found: {self.csv_file}")
            sys.exit(1)
    
    def create_all_visualizations(self):
        """Create all visualizations from existing data."""
        print("\n" + "=" * 60)
        print("CREATING ALL VISUALIZATIONS FROM EXISTING DATA")
        print("=" * 60)
        
        # Set up the plotting style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. Agent Performance Comparison
        print("1. Creating agent performance comparison...")
        self._create_agent_performance_comparison()
        
        # 2. Maze Performance Analysis
        print("2. Creating maze performance analysis...")
        self._create_maze_performance_analysis()
        
        # 3. Reward Configuration Performance
        print("3. Creating reward configuration performance...")
        self._create_reward_config_performance()
        
        # 4. Agent-Maze Heatmap
        print("4. Creating agent-maze heatmap...")
        self._create_agent_maze_heatmap()
        
        # 5. Agent-Reward Heatmap
        print("5. Creating agent-reward heatmap...")
        self._create_agent_reward_heatmap()
        
        # 6. Composite Improvement Heatmap
        print("6. Creating composite improvement heatmap...")
        self._create_composite_improvement_heatmap()
        
        # 7. Step Budget Analysis
        print("7. Creating step budget analysis...")
        self._create_step_budget_analysis()
        
        # 8. Swap Probability Analysis
        print("8. Creating swap probability analysis...")
        self._create_swap_probability_analysis()
        
        # 9. Statistical Summary
        print("9. Creating statistical summary...")
        self._create_statistical_summary()
        
        # 10. Death Point Heatmaps
        print("10. Creating death point heatmaps...")
        self._create_death_point_heatmaps()
        
        print("✅ All visualizations completed!")
    
    def run_advanced_analysis(self):
        """Run all advanced analysis modules."""
        print("\n" + "=" * 60)
        print("RUNNING ADVANCED ANALYSIS MODULES")
        print("=" * 60)
        
        # Initialize advanced analyzers
        sensitivity_analyzer = SensitivityAnalyzer()
        learning_analyzer = LearningTrajectoryAnalyzer()
        behavioral_analyzer = BehavioralAnalyzer()
        statistical_analyzer = EnhancedStatisticalAnalyzer()
        
        # Load data for all analyzers
        sensitivity_analyzer.load_results("results/comprehensive_results.csv")
        learning_analyzer.load_results("results/comprehensive_results.csv")
        behavioral_analyzer.load_results("results/comprehensive_results.csv")
        statistical_analyzer.load_results("results/comprehensive_results.csv")
        
        # Run all advanced analyses
        print("\n1️⃣ SENSITIVITY ANALYSIS")
        print("-" * 40)
        sensitivity_results = sensitivity_analyzer.generate_sensitivity_report()
        
        print("\n2️⃣ LEARNING TRAJECTORY ANALYSIS")
        print("-" * 40)
        learning_results = learning_analyzer.generate_learning_report()
        
        print("\n3️⃣ BEHAVIORAL ANALYSIS")
        print("-" * 40)
        behavioral_results = behavioral_analyzer.generate_behavioral_report()
        
        print("\n4️⃣ ENHANCED STATISTICAL ANALYSIS")
        print("-" * 40)
        statistical_results = statistical_analyzer.generate_enhanced_statistical_report()
        
        print("✅ All advanced analysis modules completed!")
        
        return {
            'sensitivity': sensitivity_results,
            'learning': learning_results,
            'behavioral': behavioral_results,
            'statistical': statistical_results
        }
    
    def _create_agent_performance_comparison(self):
        """Create agent performance comparison plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement']
        titles = ['Average Reward', 'Exit Rate', 'Survival Rate', 'Composite Improvement']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            data = self.df.groupby('agent')[metric].mean().sort_values(ascending=False)
            
            bars = ax.bar(range(len(data)), data.values, color='skyblue', alpha=0.7)
            ax.set_title(f'{title} by Agent')
            ax.set_xlabel('Agent')
            ax.set_ylabel(title)
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, data.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'agent_performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Agent performance comparison saved")
    
    def _create_maze_performance_analysis(self):
        """Create maze performance analysis plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement']
        titles = ['Average Reward', 'Exit Rate', 'Survival Rate', 'Composite Improvement']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            data = self.df.groupby('maze')[metric].mean().sort_values(ascending=False)
            
            bars = ax.bar(range(len(data)), data.values, color='lightgreen', alpha=0.7)
            ax.set_title(f'{title} by Maze')
            ax.set_xlabel('Maze')
            ax.set_ylabel(title)
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, data.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'maze_performance_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Maze performance analysis saved")
    
    def _create_reward_config_performance(self):
        """Create reward configuration performance plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement']
        titles = ['Average Reward', 'Exit Rate', 'Survival Rate', 'Composite Improvement']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            data = self.df.groupby('reward_config')[metric].mean().sort_values(ascending=False)
            
            bars = ax.bar(range(len(data)), data.values, color='lightcoral', alpha=0.7)
            ax.set_title(f'{title} by Reward Configuration')
            ax.set_xlabel('Reward Configuration')
            ax.set_ylabel(title)
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, data.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'reward_config_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Reward configuration performance saved")
    
    def _create_agent_maze_heatmap(self):
        """Create agent-maze interaction heatmap."""
        pivot_data = self.df.pivot_table(
            values='avg_reward',
            index='agent',
            columns='maze',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
        plt.title('Agent-Maze Interaction Heatmap (Average Reward)')
        plt.xlabel('Maze')
        plt.ylabel('Agent')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'agent_maze_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Agent-maze heatmap saved")
    
    def _create_agent_reward_heatmap(self):
        """Create agent-reward interaction heatmap."""
        pivot_data = self.df.pivot_table(
            values='avg_reward',
            index='agent',
            columns='reward_config',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
        plt.title('Agent-Reward Configuration Interaction Heatmap')
        plt.xlabel('Reward Configuration')
        plt.ylabel('Agent')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'agent_reward_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Agent-reward heatmap saved")
    
    def _create_composite_improvement_heatmap(self):
        """Create composite improvement heatmap."""
        pivot_data = self.df.pivot_table(
            values='learning_improvement',
            index='agent',
            columns='reward_config',
            aggfunc='mean'
        )
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
        plt.title('Composite Improvement by Agent and Reward Configuration')
        plt.xlabel('Reward Configuration')
        plt.ylabel('Agent')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'composite_improvement_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Composite improvement heatmap saved")
    
    def _create_step_budget_analysis(self):
        """Create step budget analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement']
        titles = ['Average Reward', 'Exit Rate', 'Survival Rate', 'Composite Improvement']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            data = self.df.groupby('step_budget')[metric].mean()
            
            ax.plot(data.index, data.values, 'o-', linewidth=2, markersize=8)
            ax.set_title(f'{title} by Step Budget')
            ax.set_xlabel('Step Budget')
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'step_budget_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Step budget analysis saved")
    
    def _create_swap_probability_analysis(self):
        """Create swap probability analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement']
        titles = ['Average Reward', 'Exit Rate', 'Survival Rate', 'Composite Improvement']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            data = self.df.groupby('swap_prob')[metric].mean()
            
            ax.plot(data.index, data.values, 'o-', linewidth=2, markersize=8)
            ax.set_title(f'{title} by Swap Probability')
            ax.set_xlabel('Swap Probability')
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'swap_probability_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Swap probability analysis saved")
    
    def _create_statistical_summary(self):
        """Create statistical summary plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 1. Agent performance distribution
        ax1 = axes[0]
        for agent in self.df['agent'].unique():
            agent_data = self.df[self.df['agent'] == agent]['avg_reward']
            ax1.hist(agent_data, alpha=0.5, label=agent, bins=20)
        ax1.set_title('Distribution of Average Rewards by Agent')
        ax1.set_xlabel('Average Reward')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # 2. Maze performance distribution
        ax2 = axes[1]
        for maze in self.df['maze'].unique():
            maze_data = self.df[self.df['maze'] == maze]['avg_reward']
            ax2.hist(maze_data, alpha=0.5, label=maze, bins=20)
        ax2.set_title('Distribution of Average Rewards by Maze')
        ax2.set_xlabel('Average Reward')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # 3. Step budget vs reward
        ax3 = axes[2]
        ax3.scatter(self.df['step_budget'], self.df['avg_reward'], alpha=0.6)
        ax3.set_title('Step Budget vs Average Reward')
        ax3.set_xlabel('Step Budget')
        ax3.set_ylabel('Average Reward')
        
        # 4. Swap probability vs reward
        ax4 = axes[3]
        ax4.scatter(self.df['swap_prob'], self.df['avg_reward'], alpha=0.6)
        ax4.set_title('Swap Probability vs Average Reward')
        ax4.set_xlabel('Swap Probability')
        ax4.set_ylabel('Average Reward')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'statistical_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Statistical summary saved")
    
    def _create_death_point_heatmaps(self):
        """Create death point heatmaps for each maze using the new unified script."""
        import subprocess
        import os
        script_path = os.path.join(os.path.dirname(__file__), 'analysis', 'death_point_heatmap.py')
        subprocess.run(['python', script_path], check=True)
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE SUMMARY REPORT")
        print("=" * 60)
        
        # Agent performance summary
        print("\nAGENT PERFORMANCE SUMMARY:")
        print("-" * 30)
        agent_summary = self.df.groupby('agent').agg({
            'avg_reward': ['mean', 'std'],
            'exit_rate': ['mean', 'std'],
            'survival_rate': ['mean', 'std'],
            'avg_risk_adjusted_return': ['mean', 'std'],
            'learning_improvement': ['mean', 'std'],  # This is composite improvement
            'avg_collected_rewards': ['mean', 'std']
        }).round(3)
        
        print(agent_summary)
        
        # Composite improvement ranking
        print("\nCOMPOSITE IMPROVEMENT RANKING:")
        print("-" * 30)
        learning_ranking = self.df.groupby('agent')['learning_improvement'].mean().sort_values(ascending=False)
        for i, (agent, score) in enumerate(learning_ranking.items(), 1):
            print(f"  {i}. {agent}: {score:.2f}")
        
        # Reward configuration impact
        print("\nREWARD CONFIGURATION IMPACT:")
        print("-" * 35)
        reward_summary = self.df.groupby('reward_config').agg({
            'avg_reward': 'mean',
            'exit_rate': 'mean',
            'survival_rate': 'mean',
            'learning_improvement': 'mean',
            'avg_collected_rewards': 'mean'
        }).round(3)
        
        print(reward_summary)
        
        # Swap probability impact
        print("\nSWAP PROBABILITY IMPACT:")
        print("-" * 30)
        swap_summary = self.df.groupby('swap_prob').agg({
            'avg_reward': 'mean',
            'exit_rate': 'mean',
            'survival_rate': 'mean',
            'learning_improvement': 'mean'
        }).round(3)
        
        print(swap_summary)
        
        # Maze performance
        print("\nMAZE PERFORMANCE:")
        print("-" * 20)
        maze_summary = self.df.groupby('maze').agg({
            'avg_reward': 'mean',
            'exit_rate': 'mean',
            'survival_rate': 'mean',
            'learning_improvement': 'mean'
        }).round(3)
        
        print(maze_summary)
        
        # Step budget impact
        print("\nSTEP BUDGET IMPACT:")
        print("-" * 25)
        budget_summary = self.df.groupby('step_budget').agg({
            'avg_reward': 'mean',
            'exit_rate': 'mean',
            'survival_rate': 'mean',
            'learning_improvement': 'mean'
        }).round(3)
        
        print(budget_summary)
    
    def list_generated_files(self):
        """List all generated files."""
        print("\n" + "=" * 60)
        print("GENERATED FILES")
        print("=" * 60)
        
        generated_files = [
            'agent_performance_comparison.png',
            'maze_performance_analysis.png',
            'reward_config_performance.png',
            'agent_maze_heatmap.png',
            'agent_reward_heatmap.png',
            'composite_improvement_heatmap.png',
            'step_budget_analysis.png',
            'swap_probability_analysis.png',
            'statistical_summary.png'
        ]
        
        # Add death point heatmaps
        for maze in self.df['maze'].unique():
            generated_files.append(f'death_point_heatmap_{maze}.png')
        
        print("Generated files:")
        for file in generated_files:
            file_path = os.path.join(self.results_dir, file)
            if os.path.exists(file_path):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} (not found)")
        
        print("\nMain result files:")
        print("- agent_performance_comparison.png: Agent performance comparison")
        print("- maze_performance_analysis.png: Maze performance analysis")
        print("- reward_config_performance.png: Reward configuration analysis")
        print("- agent_maze_heatmap.png: Agent-maze interaction heatmap")
        print("- agent_reward_heatmap.png: Agent-reward interaction heatmap")
        print("- composite_improvement_heatmap.png: Composite improvement heatmap")
        print("- step_budget_analysis.png: Step budget analysis")
        print("- swap_probability_analysis.png: Swap probability analysis")
        print("- statistical_summary.png: Statistical summary plots")
        print(f"\nAll files are saved in: {self.results_dir}")

def main():
    """Main function - analyze existing CSV results."""
    print("CSV ANALYSIS SCRIPT")
    print("="*60)
    print("This script will:")
    print("1. Load experiment results (classic or lifelong/transfer protocol)")
    print("2. Generate all visualizations and analysis")
    print("3. Create performance reports and comparisons")
    print("4. Generate statistical summaries")
    print("5. Run advanced analysis modules (sensitivity, learning, behavioral, statistical)")
    print("="*60)

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    block_metrics_path = os.path.join(results_dir, 'block_metrics.csv')
    transfer_metrics_path = os.path.join(results_dir, 'transfer_metrics.csv')
    classic_path = os.path.join(results_dir, 'comprehensive_results.csv')

    if not os.path.exists(classic_path) and os.path.exists(block_metrics_path):
        # Synthesize a classic-style comprehensive_results.csv from block_metrics.csv
        print("\nSynthesizing comprehensive_results.csv from block_metrics.csv for classic analysis...")
        block_df = pd.read_csv(block_metrics_path)
        # Map block metrics to classic columns (fill missing with NaN or reasonable defaults)
        classic_cols = [
            'agent', 'maze', 'reward_config', 'step_budget', 'swap_prob',
            'avg_reward', 'exit_rate', 'survival_rate', 'avg_risk_adjusted_return',
            'avg_path_efficiency', 'avg_steps', 'avg_collected_rewards',
            'reward_improvement', 'survival_improvement', 'learning_improvement',
            'reward_slope', 'episodes', 'death_count', 'death_x', 'death_y', 'death_reason'
        ]
        # Use available columns, fill others with NaN or block values
        synth = pd.DataFrame()
        synth['agent'] = block_df['agent']
        synth['maze'] = block_df['maze'] if 'maze' in block_df else np.nan
        synth['reward_config'] = block_df['reward_config'] if 'reward_config' in block_df else np.nan
        synth['step_budget'] = block_df['step_budget'] if 'step_budget' in block_df else np.nan
        synth['swap_prob'] = block_df['swap_prob'] if 'swap_prob' in block_df else np.nan
        # Use final_reward as avg_reward, others as NaN or 0
        synth['avg_reward'] = block_df['final_reward'] if 'final_reward' in block_df else np.nan
        synth['exit_rate'] = np.nan
        synth['survival_rate'] = np.nan
        synth['avg_risk_adjusted_return'] = np.nan
        synth['avg_path_efficiency'] = np.nan
        synth['avg_steps'] = np.nan
        synth['avg_collected_rewards'] = np.nan
        synth['reward_improvement'] = block_df['slope'] if 'slope' in block_df else np.nan
        synth['survival_improvement'] = np.nan
        synth['learning_improvement'] = block_df['slope'] if 'slope' in block_df else np.nan
        synth['reward_slope'] = block_df['slope'] if 'slope' in block_df else np.nan
        synth['episodes'] = np.nan
        synth['death_count'] = np.nan
        synth['death_x'] = np.nan
        synth['death_y'] = np.nan
        synth['death_reason'] = np.nan
        synth.to_csv(classic_path, index=False)
        print(f"Classic-style CSV synthesized and saved to: {classic_path}")

    # Always run both analyses if block_metrics.csv exists
    if os.path.exists(block_metrics_path) and os.path.exists(transfer_metrics_path):
        print("\nDetected lifelong/transfer learning results. Running lifelong/transfer analysis...")
        # --- Lifelong/Transfer Learning Block Analysis ---
        block_df = pd.read_csv(block_metrics_path)
        # Barplot: initial slope per block (with error bars)
        plt.figure(figsize=(12, 6))
        for agent in block_df['agent'].unique():
            agent_df = block_df[block_df['agent'] == agent]
            plt.bar(agent_df['block_index'] + 0.1 * list(block_df['agent'].unique()).index(agent),
                    agent_df['slope'], width=0.1, label=agent)
        plt.xlabel('Block Index')
        plt.ylabel('Initial Slope (Learning Rate)')
        plt.title('Initial Slope per Block (by Agent)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'block_initial_slope_barplot.png'), dpi=200)
        # Line plot: cumulative AUC across blocks
        plt.figure(figsize=(12, 6))
        for agent in block_df['agent'].unique():
            agent_df = block_df[block_df['agent'] == agent].sort_values('block_index')
            cum_auc = np.cumsum(agent_df['auc'])
            plt.plot(agent_df['block_index'], cum_auc, label=agent)
        plt.xlabel('Block Index')
        plt.ylabel('Cumulative AUC')
        plt.title('Cumulative AUC Across Blocks (by Agent)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'cumulative_auc_lineplot.png'), dpi=200)
        if os.path.exists(transfer_metrics_path):
            transfer_df = pd.read_csv(transfer_metrics_path)
            # Line plot: test performance (transfer efficiency) across blocks
            plt.figure(figsize=(12, 6))
            for agent in transfer_df['agent'].unique():
                agent_df = transfer_df[transfer_df['agent'] == agent].sort_values('block_index')
                plt.plot(agent_df['block_index'], agent_df['test_mean'], label=agent)
                plt.fill_between(agent_df['block_index'],
                                 agent_df['test_mean'] - agent_df['test_std'],
                                 agent_df['test_mean'] + agent_df['test_std'],
                                 alpha=0.2)
            plt.xlabel('Block Index')
            plt.ylabel('Test Mean Reward')
            plt.title('Transfer Test Performance Across Blocks (by Agent)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'transfer_test_performance_lineplot.png'), dpi=200)
        print("\nLifelong/transfer learning analysis complete. Plots saved to results/ directory.")

    # --- New block_metrics and transfer_metrics plots ---
    if os.path.exists(block_metrics_path):
        block_df = pd.read_csv(block_metrics_path)
        # 1. avg_reward per block
        if 'final_reward' in block_df.columns:
            plt.figure(figsize=(12, 6))
            for agent in block_df['agent'].unique():
                agent_df = block_df[block_df['agent'] == agent]
                plt.plot(agent_df['block_index'], agent_df['final_reward'], label=agent)
            plt.xlabel('Block Index')
            plt.ylabel('Final Reward (per block)')
            plt.title('Final Reward per Block (by Agent)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'block_final_reward_lineplot.png'), dpi=200)
            print('✓ Block final reward lineplot saved')
        # 2. Slope per block
        if 'slope' in block_df.columns:
            plt.figure(figsize=(12, 6))
            for agent in block_df['agent'].unique():
                agent_df = block_df[block_df['agent'] == agent]
                plt.plot(agent_df['block_index'], agent_df['slope'], label=agent)
            plt.xlabel('Block Index')
            plt.ylabel('Slope (Learning Rate)')
            plt.title('Slope per Block (by Agent)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'block_slope_lineplot.png'), dpi=200)
            print('✓ Block slope lineplot saved')
        # 3. AUC per block
        if 'auc' in block_df.columns:
            plt.figure(figsize=(12, 6))
            for agent in block_df['agent'].unique():
                agent_df = block_df[block_df['agent'] == agent]
                plt.plot(agent_df['block_index'], agent_df['auc'], label=agent)
            plt.xlabel('Block Index')
            plt.ylabel('AUC (Area Under Reward Curve)')
            plt.title('AUC per Block (by Agent)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'block_auc_lineplot.png'), dpi=200)
            print('✓ Block AUC lineplot saved')
        # 4. Regret per block
        if 'regret' in block_df.columns:
            plt.figure(figsize=(12, 6))
            for agent in block_df['agent'].unique():
                agent_df = block_df[block_df['agent'] == agent]
                plt.plot(agent_df['block_index'], agent_df['regret'], label=agent)
            plt.xlabel('Block Index')
            plt.ylabel('Regret (per block)')
            plt.title('Regret per Block (by Agent)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'block_regret_lineplot.png'), dpi=200)
            print('✓ Block regret lineplot saved')
    if os.path.exists(transfer_metrics_path):
        transfer_df = pd.read_csv(transfer_metrics_path)
        # 5. Test mean per block (with std)
        if 'test_mean' in transfer_df.columns and 'test_std' in transfer_df.columns:
            plt.figure(figsize=(12, 6))
            for agent in transfer_df['agent'].unique():
                agent_df = transfer_df[transfer_df['agent'] == agent]
                plt.plot(agent_df['block_index'], agent_df['test_mean'], label=agent)
                plt.fill_between(agent_df['block_index'],
                                 agent_df['test_mean'] - agent_df['test_std'],
                                 agent_df['test_mean'] + agent_df['test_std'],
                                 alpha=0.2)
            plt.xlabel('Block Index')
            plt.ylabel('Test Mean Reward')
            plt.title('Transfer Test Performance per Block (by Agent)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'transfer_test_performance_per_block.png'), dpi=200)
            print('✓ Transfer test performance per block plot saved')

    # Always run classic analysis if classic_path exists (including if just synthesized)
    if os.path.exists(classic_path):
        print("\nDetected classic experiment results. Running classic analysis...")
        # Create analyzer
        analyzer = CSVAnalyzer()
        # Ensure learning_improvement column exists for all downstream code
        if 'learning_improvement' not in analyzer.df.columns:
            if 'reward_improvement' in analyzer.df.columns:
                analyzer.df['learning_improvement'] = analyzer.df['reward_improvement']
            else:
                raise KeyError("Neither 'learning_improvement' nor 'reward_improvement' exist in the CSV.")
        # Generate summary report
        analyzer.generate_summary_report()
        # Create all visualizations
        analyzer.create_all_visualizations()
        # Run advanced analysis modules
        advanced_results = analyzer.run_advanced_analysis()
        # List generated files
        analyzer.list_generated_files()
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("All visualizations and analysis generated from existing CSV data!")
        print("Check the generated PNG files for detailed results.")
        print("Advanced analysis modules completed successfully!")
    else:
        print("❌ No experiment results found. Please run experiments first.")
        exit(1)

if __name__ == "__main__":
    main() 