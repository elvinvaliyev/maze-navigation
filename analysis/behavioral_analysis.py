"""
Behavioral analysis module for maze navigation experiments.

This module analyzes agent behavior patterns, decision-making processes,
and navigation strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from scipy import stats
from collections import defaultdict
import os
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

class BehavioralAnalyzer:
    """Behavioral analysis for agent navigation patterns and decision-making."""
    
    def __init__(self):
        self.results = None
        self.behavioral_data = {}
        
    def load_results(self, results_file: str = "results/comprehensive_results.csv"):
        """Load experiment results for analysis."""
        try:
            self.results = pd.read_csv(results_file)
            # — if the CSV has only reward_improvement, make sure learning_improvement exists —
            if 'learning_improvement' not in self.results.columns and 'reward_improvement' in self.results.columns:
                self.results['learning_improvement'] = self.results['reward_improvement']
            print(f"Loaded {len(self.results)} results for behavioral analysis")
            return True
        except FileNotFoundError:
            print(f"Results file {results_file} not found")
            return False
    
    def analyze_decision_making_patterns(self) -> Dict:
        """Analyze decision-making patterns of agents."""
        if self.results is None:
            return {}
        
        decision_analysis = {}
        
        print("\n=== DECISION-MAKING PATTERN ANALYSIS ===")
        
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            
            # Analyze consistency in decision-making
            # Use survival rate and exit rate as proxies for decision quality
            decision_consistency = {}
            
            # Analyze decision consistency across different conditions
            for condition in ['maze', 'reward_config', 'swap_prob', 'step_budget']:
                condition_performance = agent_data.groupby(condition)[['exit_rate', 'survival_rate']].std()
                
                # Lower standard deviation indicates more consistent decisions
                exit_consistency = 1 - condition_performance['exit_rate'].mean()
                survival_consistency = 1 - condition_performance['survival_rate'].mean()
                
                decision_consistency[condition] = {
                    'exit_consistency': exit_consistency,
                    'survival_consistency': survival_consistency,
                    'overall_consistency': (exit_consistency + survival_consistency) / 2
                }
            
            # Analyze risk-taking behavior
            # High reward collection with low survival indicates risk-taking
            risk_taking = agent_data['avg_collected_rewards'].mean() / (agent_data['survival_rate'].mean() + 1e-6)
            
            # Analyze adaptability (how well agent performs across different conditions)
            adaptability_score = 1 - agent_data.groupby('maze')['avg_reward'].std().mean() / (agent_data['avg_reward'].mean() + 1e-6)
            
            decision_analysis[agent] = {
                'decision_consistency': decision_consistency,
                'risk_taking_score': risk_taking,
                'adaptability_score': adaptability_score,
                'avg_exit_rate': agent_data['exit_rate'].mean(),
                'avg_survival_rate': agent_data['survival_rate'].mean(),
                'avg_reward_collection': agent_data['avg_collected_rewards'].mean()
            }
            
            print(f"{agent}:")
            print(f"  Risk-taking score: {risk_taking:.3f}")
            print(f"  Adaptability score: {adaptability_score:.3f}")
            print(f"  Avg exit rate: {agent_data['exit_rate'].mean():.3f}")
            print(f"  Avg survival rate: {agent_data['survival_rate'].mean():.3f}")
        
        return decision_analysis
    
    def analyze_navigation_strategies(self) -> Dict:
        """Analyze navigation strategies used by different agents."""
        if self.results is None:
            return {}
        
        strategy_analysis = {}
        
        print("\n=== NAVIGATION STRATEGY ANALYSIS ===")
        
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            
            # Analyze strategy effectiveness across different conditions
            strategies = {}
            
            # Strategy 1: Reward-focused vs Exit-focused
            # High collected rewards relative to exit rate indicates reward-focused strategy
            reward_focus = agent_data['avg_collected_rewards'].mean() / (agent_data['exit_rate'].mean() + 1e-6)
            strategies['reward_focus'] = reward_focus
            
            # Strategy 2: Conservative vs Aggressive
            # High survival rate with low reward collection indicates conservative strategy
            conservative_score = agent_data['survival_rate'].mean() / (agent_data['avg_collected_rewards'].mean() + 1e-6)
            strategies['conservative_score'] = conservative_score
            
            # Strategy 3: Efficiency (exit rate per step budget)
            efficiency = agent_data['exit_rate'].mean() / (agent_data['step_budget'].mean() + 1e-6)
            strategies['efficiency'] = efficiency
            
            # Strategy 4: Learning effectiveness
            learning_effectiveness = agent_data['learning_improvement'].mean()
            strategies['learning_effectiveness'] = learning_effectiveness
            
            # Determine dominant strategy
            strategy_scores = {
                'Reward-focused': reward_focus,
                'Conservative': conservative_score,
                'Efficient': efficiency,
                'Learning-oriented': learning_effectiveness
            }
            dominant_strategy = max(strategy_scores, key=strategy_scores.get)
            
            strategy_analysis[agent] = {
                'strategies': strategies,
                'dominant_strategy': dominant_strategy,
                'strategy_scores': strategy_scores
            }
            
            print(f"{agent}:")
            print(f"  Dominant strategy: {dominant_strategy}")
            print(f"  Reward focus: {reward_focus:.3f}")
            print(f"  Conservative score: {conservative_score:.3f}")
            print(f"  Efficiency: {efficiency:.3f}")
            print(f"  Learning effectiveness: {learning_effectiveness:.3f}")
        
        return strategy_analysis
    
    def analyze_environmental_adaptation(self) -> Dict:
        """Analyze how agents adapt to different environmental conditions."""
        if self.results is None:
            return {}
        
        adaptation_analysis = {}
        
        print("\n=== ENVIRONMENTAL ADAPTATION ANALYSIS ===")
        
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            
            adaptations = {}
            
            # Adaptation to step budget changes
            budget_adaptation = agent_data.groupby('step_budget')['avg_reward'].mean()
            budget_sensitivity = budget_adaptation.std() / (budget_adaptation.mean() + 1e-6)
            adaptations['budget_adaptation'] = budget_sensitivity
            
            # Adaptation to swap probability changes
            swap_adaptation = agent_data.groupby('swap_prob')['avg_reward'].mean()
            swap_sensitivity = swap_adaptation.std() / (swap_adaptation.mean() + 1e-6)
            adaptations['swap_adaptation'] = swap_sensitivity
            
            # Adaptation to reward configuration changes
            reward_adaptation = agent_data.groupby('reward_config')['avg_reward'].mean()
            reward_sensitivity = reward_adaptation.std() / (reward_adaptation.mean() + 1e-6)
            adaptations['reward_adaptation'] = reward_sensitivity
            
            # Adaptation to maze changes
            maze_adaptation = agent_data.groupby('maze')['avg_reward'].mean()
            maze_sensitivity = maze_adaptation.std() / (maze_adaptation.mean() + 1e-6)
            adaptations['maze_adaptation'] = maze_sensitivity
            
            # Overall adaptation score (lower sensitivity = better adaptation)
            overall_adaptation = 1 - np.mean([budget_sensitivity, swap_sensitivity, reward_sensitivity, maze_sensitivity])
            
            adaptation_analysis[agent] = {
                'adaptations': adaptations,
                'overall_adaptation': overall_adaptation,
                'budget_sensitivity': budget_sensitivity,
                'swap_sensitivity': swap_sensitivity,
                'reward_sensitivity': reward_sensitivity,
                'maze_sensitivity': maze_sensitivity
            }
            
            print(f"{agent}:")
            print(f"  Overall adaptation: {overall_adaptation:.3f}")
            print(f"  Budget sensitivity: {budget_sensitivity:.3f}")
            print(f"  Swap sensitivity: {swap_sensitivity:.3f}")
            print(f"  Reward sensitivity: {reward_sensitivity:.3f}")
            print(f"  Maze sensitivity: {maze_sensitivity:.3f}")
        
        return adaptation_analysis
    
    def analyze_behavioral_clusters(self) -> Dict:
        """Cluster agents based on behavioral patterns."""
        if self.results is None:
            return {}
        
        print("\n=== BEHAVIORAL CLUSTERING ANALYSIS ===")
        
        # Create behavioral features for clustering
        behavioral_features = []
        agent_names = []
        
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            
            features = [
                agent_data['avg_reward'].mean(),
                agent_data['exit_rate'].mean(),
                agent_data['survival_rate'].mean(),
                agent_data['learning_improvement'].mean(),
                agent_data['avg_collected_rewards'].mean(),
                agent_data['avg_reward'].std(),  # Consistency
                agent_data['exit_rate'].std(),   # Consistency
                agent_data['survival_rate'].std() # Consistency
            ]
            
            behavioral_features.append(features)
            agent_names.append(agent)
        
        behavioral_features = np.array(behavioral_features)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(behavioral_features)
        
        # Perform clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(3, len(agent_names)), random_state=42)
        clusters = kmeans.fit_predict(features_normalized)
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(len(set(clusters))):
            cluster_agents = [agent_names[j] for j in range(len(agent_names)) if clusters[j] == i]
            cluster_features = behavioral_features[clusters == i]
            
            cluster_analysis[f'cluster_{i}'] = {
                'agents': cluster_agents,
                'avg_reward': np.mean(cluster_features[:, 0]),
                'avg_exit_rate': np.mean(cluster_features[:, 1]),
                'avg_survival_rate': np.mean(cluster_features[:, 2]),
                'avg_learning': np.mean(cluster_features[:, 3]),
                'avg_reward_collection': np.mean(cluster_features[:, 4]),
                'consistency': np.mean(cluster_features[:, 5:8])
            }
        
        print("Behavioral Clusters:")
        for cluster_name, cluster_data in cluster_analysis.items():
            print(f"  {cluster_name}: {cluster_data['agents']}")
            print(f"    Avg reward: {cluster_data['avg_reward']:.3f}")
            print(f"    Avg exit rate: {cluster_data['avg_exit_rate']:.3f}")
            print(f"    Avg survival rate: {cluster_data['avg_survival_rate']:.3f}")
            print(f"    Consistency: {cluster_data['consistency']:.3f}")
        
        return {
            'clusters': cluster_analysis,
            'cluster_assignments': dict(zip(agent_names, clusters))
        }
    
    def create_behavioral_visualizations(self):
        """Create visualizations for behavioral analysis."""
        if self.results is None:
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create behavioral analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Behavioral Analysis', fontsize=16, fontweight='bold')
        
        # 1. Risk-taking vs Reward collection
        ax1 = axes[0, 0]
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            risk_taking = agent_data['avg_collected_rewards'].mean() / (agent_data['survival_rate'].mean() + 1e-6)
            reward_collection = agent_data['avg_collected_rewards'].mean()
            ax1.scatter(risk_taking, reward_collection, s=100, label=agent, alpha=0.7)
        
        ax1.set_xlabel('Risk-taking Score')
        ax1.set_ylabel('Average Reward Collection')
        ax1.set_title('Risk-taking vs Reward Collection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Strategy effectiveness by agent
        ax2 = axes[0, 1]
        strategies = ['exit_rate', 'survival_rate', 'learning_improvement']
        x = np.arange(len(self.results['agent'].unique()))
        width = 0.25
        
        for i, strategy in enumerate(strategies):
            strategy_means = []
            for agent in self.results['agent'].unique():
                agent_data = self.results[self.results['agent'] == agent]
                strategy_means.append(agent_data[strategy].mean())
            
            ax2.bar(x + i*width, strategy_means, width, label=strategy.replace('_', ' ').title())
        
        ax2.set_xlabel('Agent')
        ax2.set_ylabel('Strategy Score')
        ax2.set_title('Strategy Effectiveness by Agent')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(self.results['agent'].unique(), rotation=45)
        ax2.legend()
        
        # 3. Environmental adaptation
        ax3 = axes[1, 0]
        adaptation_scores = []
        agent_names = []
        
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            
            # Calculate adaptation score (lower sensitivity = better adaptation)
            budget_sensitivity = agent_data.groupby('step_budget')['avg_reward'].mean().std()
            swap_sensitivity = agent_data.groupby('swap_prob')['avg_reward'].mean().std()
            reward_sensitivity = agent_data.groupby('reward_config')['avg_reward'].mean().std()
            maze_sensitivity = agent_data.groupby('maze')['avg_reward'].mean().std()
            
            overall_adaptation = 1 - np.mean([budget_sensitivity, swap_sensitivity, reward_sensitivity, maze_sensitivity])
            adaptation_scores.append(overall_adaptation)
            agent_names.append(agent)
        
        bars = ax3.bar(agent_names, adaptation_scores, alpha=0.7)
        ax3.set_xlabel('Agent')
        ax3.set_ylabel('Adaptation Score')
        ax3.set_title('Environmental Adaptation by Agent')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars, adaptation_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 4. Decision consistency
        ax4 = axes[1, 1]
        consistency_scores = []
        
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            
            # Calculate consistency (lower std = higher consistency)
            exit_consistency = 1 - agent_data.groupby('maze')['exit_rate'].std().mean()
            survival_consistency = 1 - agent_data.groupby('maze')['survival_rate'].std().mean()
            overall_consistency = (exit_consistency + survival_consistency) / 2
            
            consistency_scores.append(overall_consistency)
        
        bars = ax4.bar(agent_names, consistency_scores, alpha=0.7)
        ax4.set_xlabel('Agent')
        ax4.set_ylabel('Decision Consistency')
        ax4.set_title('Decision Consistency by Agent')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars, consistency_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # --- NEW: Risk threshold vs. swap_prob ---
        if 'risk_threshold' in self.results.columns:
            plt.figure(figsize=(10, 6))
            for agent in self.results['agent'].unique():
                agent_data = self.results[self.results['agent'] == agent]
                swap_means = agent_data.groupby('swap_prob')['risk_threshold'].mean()
                plt.plot(swap_means.index, swap_means.values, marker='o', label=agent)
            plt.xlabel('Swap Probability')
            plt.ylabel('Risk Threshold')
            plt.title('Risk Threshold vs. Swap Probability')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'risk_threshold_vs_swap_prob.png'), dpi=300, bbox_inches='tight')
            plt.close()
        # --- NEW: Fraction of aborts vs. swap_prob ---
        plt.figure(figsize=(10, 6))
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            aborts = 1 - agent_data.groupby('swap_prob')['exit_rate'].mean()
            plt.plot(aborts.index, aborts.values, marker='o', label=agent)
        plt.xlabel('Swap Probability')
        plt.ylabel('Fraction of Aborts (Did Not Reach Exit)')
        plt.title('Fraction of Aborts vs. Swap Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'fraction_aborts_vs_swap_prob.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # --- NEW: Average path length vs. swap_prob ---
        plt.figure(figsize=(10, 6))
        for agent in self.results['agent'].unique():
            agent_data = self.results[self.results['agent'] == agent]
            avg_path = agent_data.groupby('swap_prob')['avg_steps'].mean()
            plt.plot(avg_path.index, avg_path.values, marker='o', label=agent)
        plt.xlabel('Swap Probability')
        plt.ylabel('Average Path Length')
        plt.title('Average Path Length vs. Swap Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'avg_path_length_vs_swap_prob.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # --- (Optional) Fraction of safe path choices vs. swap_prob ---
        # Define "safe path" as survived and avg_steps > median for that maze/agent
        if 'survival_rate' in self.results.columns:
            plt.figure(figsize=(10, 6))
            for agent in self.results['agent'].unique():
                agent_data = self.results[self.results['agent'] == agent]
                safe_path_frac = []
                for swap_prob, group in agent_data.groupby('swap_prob'):
                    median_path = group['avg_steps'].median()
                    safe = ((group['survival_rate'] > 0.5) & (group['avg_steps'] > median_path)).mean()
                    safe_path_frac.append((swap_prob, safe))
                safe_path_frac = sorted(safe_path_frac)
                x, y = zip(*safe_path_frac) if safe_path_frac else ([],[])
                plt.plot(x, y, marker='o', label=agent)
            plt.xlabel('Swap Probability')
            plt.ylabel('Fraction of Safe Path Choices')
            plt.title('Fraction of Safe Path Choices vs. Swap Probability')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'fraction_safe_paths_vs_swap_prob.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'behavioral_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_behavioral_report(self) -> Dict:
        """Generate a comprehensive behavioral analysis report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE BEHAVIORAL ANALYSIS")
        print("="*60)
        
        if not self.load_results():
            return {}
        
        # Run all behavioral analyses
        decision_patterns = self.analyze_decision_making_patterns()
        navigation_strategies = self.analyze_navigation_strategies()
        environmental_adaptation = self.analyze_environmental_adaptation()
        behavioral_clusters = self.analyze_behavioral_clusters()
        
        # Create visualizations
        self.create_behavioral_visualizations()
        
        # Generate summary
        print("\n" + "="*60)
        print("BEHAVIORAL ANALYSIS SUMMARY")
        print("="*60)
        
        # Most adaptive agent
        adaptation_scores = {}
        for agent, data in environmental_adaptation.items():
            adaptation_scores[agent] = data['overall_adaptation']
        
        most_adaptive = max(adaptation_scores, key=adaptation_scores.get)
        least_adaptive = min(adaptation_scores, key=adaptation_scores.get)
        
        print(f"\nMOST ADAPTIVE AGENT: {most_adaptive} ({adaptation_scores[most_adaptive]:.3f})")
        print(f"LEAST ADAPTIVE AGENT: {least_adaptive} ({adaptation_scores[least_adaptive]:.3f})")
        
        # Most consistent decision maker
        consistency_scores = {}
        for agent, data in decision_patterns.items():
            avg_consistency = np.mean([v['overall_consistency'] for v in data['decision_consistency'].values()])
            consistency_scores[agent] = avg_consistency
        
        most_consistent = max(consistency_scores, key=consistency_scores.get)
        least_consistent = min(consistency_scores, key=consistency_scores.get)
        
        print(f"\nMOST CONSISTENT DECISION MAKER: {most_consistent} ({consistency_scores[most_consistent]:.3f})")
        print(f"LEAST CONSISTENT DECISION MAKER: {least_consistent} ({consistency_scores[least_consistent]:.3f})")
        
        # Dominant strategies
        print("\nDOMINANT STRATEGIES:")
        for agent, data in navigation_strategies.items():
            print(f"  {agent}: {data['dominant_strategy']}")
        
        report = {
            'decision_patterns': decision_patterns,
            'navigation_strategies': navigation_strategies,
            'environmental_adaptation': environmental_adaptation,
            'behavioral_clusters': behavioral_clusters
        }
        
        return report 