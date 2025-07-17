"""
Sensitivity analysis module for maze navigation experiments.

This module analyzes how agent performance changes with parameter variations
across different experimental conditions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from scipy import stats
from scipy.optimize import curve_fit
import os
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

class SensitivityAnalyzer:
    """Sensitivity analysis for parameter variations and their effects on performance."""
    
    def __init__(self):
        self.results = None
        self.sensitivity_data = {}
        
    def load_results(self, results_file: str = "results/comprehensive_results.csv"):
        """Load experiment results for analysis."""
        try:
            self.results = pd.read_csv(results_file)
            # — if the CSV has only reward_improvement, make sure learning_improvement exists —
            if 'learning_improvement' not in self.results.columns and 'reward_improvement' in self.results.columns:
                self.results['learning_improvement'] = self.results['reward_improvement']
            print(f"Loaded {len(self.results)} results for sensitivity analysis")
            return True
        except FileNotFoundError:
            print(f"Results file {results_file} not found")
            return False
    
    def analyze_parameter_sensitivity(self) -> Dict:
        """Analyze how performance varies with different parameters."""
        if self.results is None:
            return {}
        
        sensitivity_results = {}
        
        # Parameters to analyze
        parameters = ['step_budget', 'swap_prob', 'reward_config']
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement']
        
        for param in parameters:
            print(f"\n=== {param.upper()} SENSITIVITY ANALYSIS ===")
            param_sensitivity = {}
            
            for metric in metrics:
                # Group by parameter value and calculate statistics
                grouped = self.results.groupby(param)[metric]
                means = grouped.mean()
                stds = grouped.std()
                
                # Calculate coefficient of variation (CV) as sensitivity measure
                cv = (stds / means).abs()
                
                # Calculate correlation with parameter
                if param == 'step_budget':
                    x = self.results[param]
                    y = self.results[metric]
                    mask = (~np.isnan(x)) & (~np.isnan(y)) & np.isfinite(x) & np.isfinite(y)
                    if mask.sum() < 2:
                        correlation, p_value = 0, 1
                    else:
                        try:
                            correlation, p_value = stats.pearsonr(x[mask], y[mask])
                        except Exception as e:
                            print(f"[WARNING] Could not compute correlation for step_budget vs {metric}: {e}")
                            correlation, p_value = 0, 1
                elif param == 'swap_prob':
                    x = self.results[param]
                    y = self.results[metric]
                    mask = (~np.isnan(x)) & (~np.isnan(y)) & np.isfinite(x) & np.isfinite(y)
                    if mask.sum() < 2:
                        correlation, p_value = 0, 1
                    else:
                        try:
                            correlation, p_value = stats.pearsonr(x[mask], y[mask])
                        except Exception as e:
                            print(f"[WARNING] Could not compute correlation for swap_prob vs {metric}: {e}")
                            correlation, p_value = 0, 1
                else:
                    # For categorical parameters, use ANOVA
                    groups = [group[metric].values for name, group in self.results.groupby(param)]
                    if len(groups) > 1:
                        f_stat, p_value = stats.f_oneway(*groups)
                        correlation = np.sqrt(f_stat / (f_stat + len(self.results) - len(groups)))
                    else:
                        correlation, p_value = 0, 1
                
                param_sensitivity[metric] = {
                    'mean_by_param': means.to_dict(),
                    'std_by_param': stds.to_dict(),
                    'cv': cv.to_dict(),
                    'correlation': correlation,
                    'p_value': p_value,
                    'sensitivity_score': abs(correlation)  # Simple sensitivity score
                }
                
                print(f"{metric}: correlation={correlation:.3f}, p={p_value:.3f}")
                if p_value < 0.05:
                    print(f"  *** SIGNIFICANT SENSITIVITY TO {param} ***")
            
            sensitivity_results[param] = param_sensitivity
        
        return sensitivity_results
    
    def analyze_agent_parameter_interactions(self) -> Dict:
        """Analyze how different agents respond to parameter changes."""
        if self.results is None:
            return {}
        
        interaction_results = {}
        parameters = ['step_budget', 'swap_prob']
        metrics = ['avg_reward', 'exit_rate', 'survival_rate']
        
        for param in parameters:
            print(f"\n=== AGENT-{param.upper()} INTERACTIONS ===")
            param_interactions = {}
            
            for metric in metrics:
                # Calculate sensitivity for each agent
                agent_sensitivities = {}
                
                for agent in self.results['agent'].unique():
                    agent_data = self.results[self.results['agent'] == agent]
                    
                    if param == 'step_budget':
                        x = agent_data[param]
                        y = agent_data[metric]
                        mask = (~np.isnan(x)) & (~np.isnan(y)) & np.isfinite(x) & np.isfinite(y)
                        if mask.sum() < 2:
                            correlation, p_value = 0, 1
                        else:
                            try:
                                correlation, p_value = stats.pearsonr(x[mask], y[mask])
                            except Exception as e:
                                print(f"[WARNING] Could not compute correlation for {agent} step_budget vs {metric}: {e}")
                                correlation, p_value = 0, 1
                    else:  # swap_prob
                        x = agent_data[param]
                        y = agent_data[metric]
                        mask = (~np.isnan(x)) & (~np.isnan(y)) & np.isfinite(x) & np.isfinite(y)
                        if mask.sum() < 2:
                            correlation, p_value = 0, 1
                        else:
                            try:
                                correlation, p_value = stats.pearsonr(x[mask], y[mask])
                            except Exception as e:
                                print(f"[WARNING] Could not compute correlation for {agent} swap_prob vs {metric}: {e}")
                                correlation, p_value = 0, 1
                    
                    agent_sensitivities[agent] = {
                        'correlation': correlation,
                        'p_value': p_value,
                        'sensitivity': abs(correlation)
                    }
                
                # Find most and least sensitive agents
                sensitivities = [(agent, data['sensitivity']) for agent, data in agent_sensitivities.items()]
                sensitivities.sort(key=lambda x: x[1], reverse=True)
                
                most_sensitive = sensitivities[0]
                least_sensitive = sensitivities[-1]
                
                param_interactions[metric] = {
                    'agent_sensitivities': agent_sensitivities,
                    'most_sensitive': most_sensitive,
                    'least_sensitive': least_sensitive
                }
                
                print(f"{metric}:")
                print(f"  Most sensitive: {most_sensitive[0]} ({most_sensitive[1]:.3f})")
                print(f"  Least sensitive: {least_sensitive[0]} ({least_sensitive[1]:.3f})")
            
            interaction_results[param] = param_interactions
        
        return interaction_results
    
    def analyze_threshold_effects(self) -> Dict:
        """Analyze threshold effects where performance changes dramatically."""
        if self.results is None:
            return {}
        
        threshold_results = {}
        
        # Analyze step budget thresholds
        print("\n=== STEP BUDGET THRESHOLD ANALYSIS ===")
        step_thresholds = {}
        
        for metric in ['avg_reward', 'exit_rate', 'survival_rate']:
            # Group by step budget and calculate means
            step_performance = self.results.groupby('step_budget')[metric].mean()
            
            # Find inflection points (where rate of change changes)
            differences = step_performance.diff()
            second_differences = differences.diff()
            
            # Find step budgets where performance improves dramatically
            improvement_thresholds = []
            for i in range(1, len(step_performance)):
                if differences.iloc[i] > differences.iloc[i-1] * 1.5:  # 50% improvement
                    improvement_thresholds.append(step_performance.index[i])
            
            step_thresholds[metric] = {
                'performance_by_budget': step_performance.to_dict(),
                'improvement_thresholds': improvement_thresholds,
                'optimal_budget': step_performance.idxmax()
            }
            
            print(f"{metric}:")
            print(f"  Optimal budget: {step_performance.idxmax()}")
            print(f"  Improvement thresholds: {improvement_thresholds}")
        
        threshold_results['step_budget'] = step_thresholds
        
        # Analyze swap probability thresholds
        print("\n=== SWAP PROBABILITY THRESHOLD ANALYSIS ===")
        swap_thresholds = {}
        
        for metric in ['avg_reward', 'exit_rate', 'survival_rate']:
            swap_performance = self.results.groupby('swap_prob')[metric].mean()
            
            # Find where performance starts to decline
            decline_thresholds = []
            for i in range(1, len(swap_performance)):
                if swap_performance.iloc[i] < swap_performance.iloc[i-1] * 0.9:  # 10% decline
                    decline_thresholds.append(swap_performance.index[i])
            
            swap_thresholds[metric] = {
                'performance_by_swap': swap_performance.to_dict(),
                'decline_thresholds': decline_thresholds,
                'optimal_swap_prob': swap_performance.idxmax()
            }
            
            print(f"{metric}:")
            print(f"  Optimal swap prob: {swap_performance.idxmax():.1f}")
            print(f"  Decline thresholds: {decline_thresholds}")
        
        threshold_results['swap_prob'] = swap_thresholds
        
        return threshold_results
    
    def create_sensitivity_visualizations(self):
        """Create visualizations for sensitivity analysis."""
        if self.results is None:
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        # 1. Step budget sensitivity
        for i, metric in enumerate(['avg_reward', 'exit_rate', 'survival_rate']):
            ax = axes[0, i]
            
            # Plot mean performance by step budget
            step_performance = self.results.groupby('step_budget')[metric].mean()
            step_std = self.results.groupby('step_budget')[metric].std()
            
            ax.errorbar(step_performance.index, step_performance.values, 
                       yerr=step_std.values, marker='o', capsize=5)
            ax.set_xlabel('Step Budget')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} vs Step Budget')
            ax.grid(True, alpha=0.3)
        
        # 2. Swap probability sensitivity
        for i, metric in enumerate(['avg_reward', 'exit_rate', 'survival_rate']):
            ax = axes[1, i]
            
            # Plot mean performance by swap probability
            swap_performance = self.results.groupby('swap_prob')[metric].mean()
            swap_std = self.results.groupby('swap_prob')[metric].std()
            
            ax.errorbar(swap_performance.index, swap_performance.values, 
                       yerr=swap_std.values, marker='s', capsize=5)
            ax.set_xlabel('Swap Probability')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} vs Swap Probability')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create agent-specific sensitivity plots
        self._create_agent_sensitivity_plots()

        # --- NEW: Volatility signature plots ---
        plt.figure(figsize=(12, 8))
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            stds = agent_data.groupby('swap_prob')['avg_reward'].std()
            cvs = (agent_data.groupby('swap_prob')['avg_reward'].std() / agent_data.groupby('swap_prob')['avg_reward'].mean()).abs()
            plt.plot(stds.index, stds.values, marker='o', label=f'{agent} Std')
            plt.plot(cvs.index, cvs.values, marker='s', linestyle='--', label=f'{agent} CV')
        plt.xlabel('Swap Probability')
        plt.ylabel('Volatility (Std/CV) of Reward')
        plt.title('Volatility Signature: Std and CV of Reward vs. Swap Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'volatility_signature.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # (Optional) Repeat for steps if desired
        plt.figure(figsize=(12, 8))
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            stds = agent_data.groupby('swap_prob')['avg_steps'].std()
            cvs = (agent_data.groupby('swap_prob')['avg_steps'].std() / agent_data.groupby('swap_prob')['avg_steps'].mean()).abs()
            plt.plot(stds.index, stds.values, marker='o', label=f'{agent} Std')
            plt.plot(cvs.index, cvs.values, marker='s', linestyle='--', label=f'{agent} CV')
        plt.xlabel('Swap Probability')
        plt.ylabel('Volatility (Std/CV) of Steps')
        plt.title('Volatility Signature: Std and CV of Steps vs. Swap Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'volatility_signature_steps.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_agent_sensitivity_plots(self):
        """Create agent-specific sensitivity plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Agent-Specific Parameter Sensitivity', fontsize=16, fontweight='bold')
        
        # Step budget sensitivity by agent
        ax1 = axes[0, 0]
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            performance = agent_data.groupby('step_budget')['avg_reward'].mean()
            ax1.plot(performance.index, performance.values, marker='o', label=agent, linewidth=2)
        
        ax1.set_xlabel('Step Budget')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Step Budget Sensitivity by Agent')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Swap probability sensitivity by agent
        ax2 = axes[0, 1]
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            performance = agent_data.groupby('swap_prob')['avg_reward'].mean()
            ax2.plot(performance.index, performance.values, marker='s', label=agent, linewidth=2)
        
        ax2.set_xlabel('Swap Probability')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Swap Probability Sensitivity by Agent')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Survival rate sensitivity
        ax3 = axes[1, 0]
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            performance = agent_data.groupby('step_budget')['survival_rate'].mean()
            ax3.plot(performance.index, performance.values, marker='o', label=agent, linewidth=2)
        
        ax3.set_xlabel('Step Budget')
        ax3.set_ylabel('Survival Rate')
        ax3.set_title('Survival Rate vs Step Budget by Agent')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Exit rate sensitivity
        ax4 = axes[1, 1]
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            performance = agent_data.groupby('swap_prob')['exit_rate'].mean()
            ax4.plot(performance.index, performance.values, marker='s', label=agent, linewidth=2)
        
        ax4.set_xlabel('Swap Probability')
        ax4.set_ylabel('Exit Rate')
        ax4.set_title('Exit Rate vs Swap Probability by Agent')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'agent_sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_sensitivity_report(self) -> Dict:
        """Generate a comprehensive sensitivity analysis report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE SENSITIVITY ANALYSIS")
        print("="*60)
        
        if not self.load_results():
            return {}
        
        # Run all sensitivity analyses
        param_sensitivity = self.analyze_parameter_sensitivity()
        agent_interactions = self.analyze_agent_parameter_interactions()
        threshold_effects = self.analyze_threshold_effects()
        
        # Create visualizations
        self.create_sensitivity_visualizations()
        
        # Generate summary
        print("\n" + "="*60)
        print("SENSITIVITY ANALYSIS SUMMARY")
        print("="*60)
        
        # Most sensitive parameters
        print("\nMOST SENSITIVE PARAMETERS:")
        for param, sensitivity in param_sensitivity.items():
            avg_sensitivity = np.mean([data['sensitivity_score'] for data in sensitivity.values()])
            print(f"  {param}: {avg_sensitivity:.3f}")
        
        # Most sensitive agents
        print("\nMOST SENSITIVE AGENTS:")
        for param, interactions in agent_interactions.items():
            for metric, data in interactions.items():
                most_sensitive = data['most_sensitive']
                print(f"  {param} - {metric}: {most_sensitive[0]} ({most_sensitive[1]:.3f})")
        
        # Key thresholds
        print("\nKEY THRESHOLDS:")
        for param, thresholds in threshold_effects.items():
            for metric, data in thresholds.items():
                if 'optimal' in data:
                    print(f"  {param} - {metric}: optimal = {data['optimal']}")
        
        report = {
            'parameter_sensitivity': param_sensitivity,
            'agent_interactions': agent_interactions,
            'threshold_effects': threshold_effects
        }
        
        return report 