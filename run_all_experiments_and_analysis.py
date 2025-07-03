#!/usr/bin/env python3
"""
Unified Experiment and Analysis Runner

This single script:
1. Runs comprehensive experiments with learning agents across different conditions
2. Tests multiple reward configurations (small, medium, large, extreme, etc.)
3. Tracks learning progress over episodes, mazes, and conditions
4. Analyzes learning transfer between mazes
5. Immediately analyzes the results with statistics and visualizations
6. Generates all reports and plots

Everything in one file - no need for separate scripts!
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy import stats

# Add analysis directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis'))

from environments.maze_environment import MazeEnvironment
from agents.model_based_greedy_agent import ModelBasedGreedyAgent
from agents.model_based_survival_agent import ModelBasedSurvivalAgent
from agents.sr_greedy_agent import SuccessorRepresentationGreedyAgent
from agents.sr_reasonable_agent import SuccessorRepresentationReasonableAgent

class UnifiedExperimentAnalyzer:
    """Runs experiments and analyzes results in one unified class."""
    
    def __init__(self):
        self.results = []
        self.learning_progress = defaultdict(list)  # Track learning over time
        self.learning_transfer = defaultdict(dict)  # Track learning transfer between mazes
        
        # Initialize LEARNING agents (with learning capabilities)
        self.agents = {
            'Model-Based Greedy': ModelBasedGreedyAgent(learning_rate=0.1, exploration_rate=0.1),
            'Model-Based Survival': ModelBasedSurvivalAgent(step_budget=50, learning_rate=0.1, exploration_rate=0.05),
            'SR-Greedy': SuccessorRepresentationGreedyAgent(alpha=0.1, exploration_rate=0.1),
            'SR-Reasonable': SuccessorRepresentationReasonableAgent(alpha=0.1, beta=1.0, exploration_rate=0.05)
        }
        
        # Load maze configurations
        self.maze_configs = [cfg for cfg in self._load_mazes() if cfg['name'] != 'maze4']
        
        # Experiment conditions
        self.swap_probabilities = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
        self.episodes_per_condition = 100  
        
        # Enhanced reward configurations
        self.reward_configs = [
            ('small', [5, 3]),           # Small rewards: 5 and 3
            ('medium', [10, 6]),         # Medium rewards: 10 and 6  
            ('large', [20, 12]),         # Large rewards: 20 and 12
            ('extreme', [50, 30]),       # Extreme rewards: 50 and 30
            ('big_diff', [50, 1]),       # Big difference: 50 and 1
            ('small_diff', [10, 9]),     # Small difference: 10 and 9
            ('equal', [10, 10])          # Equal rewards: 10 and 10
        ]
        
    def _load_mazes(self) -> List[Dict]:
        """Load all maze configurations, including the new three-routes maze."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        maze_dir = os.path.join(base_dir, "environments", "maze_configs")
        configs = []
        for filename in os.listdir(maze_dir):
            if filename.endswith('.json'):
                with open(os.path.join(maze_dir, filename), 'r') as f:
                    config = json.load(f)
                    config['name'] = filename.replace('.json', '')
                    configs.append(config)
        # Ensure the new maze is included
        return configs
    
    def create_environment_with_rewards(self, maze_config: Dict, reward_config: Tuple[str, List[int]], swap_prob: float) -> MazeEnvironment:
        """Create environment with specific reward configuration."""
        reward_name, [high_val, low_val] = reward_config
        
        # Get original reward positions from maze config
        original_rewards = maze_config['rewards']
        if len(original_rewards) >= 2:
            reward_positions = [original_rewards[0][0], original_rewards[1][0]]
        else:
            # Fallback if not enough rewards in config
            reward_positions = [(1, 1), (2, 2)]
        
        return MazeEnvironment(
            grid=maze_config['grid'],
            start=tuple(maze_config['start']),
            exit=tuple(maze_config['exit']),
            rewards=[(tuple(reward_positions[0]), high_val), (tuple(reward_positions[1]), low_val)],
            swap_prob=swap_prob
        )
    
    def run_single_episode(self, agent, env, episode_num: int, condition_key: str) -> Dict:
        """Run a single episode with specified conditions and track learning."""
        # Run episode
        env.reset()
        
        # Start new episode for swap tracking
        if hasattr(agent, 'start_new_episode'):
            agent.start_new_episode()
        
        state = env
        total_reward = 0
        steps = 0
        collected_rewards = set()
        path = [env.get_position()]
        
        while steps < env.max_steps:
            # Call observe_state for swap detection
            if hasattr(agent, 'observe_state'):
                agent.observe_state(state)
            
            action = agent.select_action(state)
            next_pos, reward, done = env.step(action)
            agent.update(state, action, reward, env)
            
            total_reward += reward
            if reward > 0:
                collected_rewards.add(env.get_position())
            
            path.append(env.get_position())
            state = env
            steps += 1
            if done: break
        
        # End episode swap tracking
        if hasattr(agent, 'end_episode_swap_tracking'):
            agent.end_episode_swap_tracking()
        
        # Calculate metrics
        success = env.get_position() == env.exit
        survival = steps < env.max_steps  # Survived if didn't run out of steps
        
        # Death analysis
        death_reason = None
        if not success:
            if steps >= env.max_steps:
                death_reason = 'timeout'
            else:
                death_reason = 'other'
        
        # Risk-adjusted return (simplified)
        risk_adjusted_return = total_reward / max(steps, 1) if survival else 0
        
        # Path efficiency (reward per step)
        path_efficiency = total_reward / max(steps, 1)
        
        # Track learning progress
        self.learning_progress[condition_key].append({
            'episode': episode_num,
            'total_reward': total_reward,
            'success': success,
            'survival': survival,
            'steps': steps,
            'efficiency': path_efficiency,
            'collected_rewards': len(collected_rewards),
            'path_length': len(path)
        })
        
        # Track swap prediction accuracy if agent has swap learning
        swap_accuracy = None
        if hasattr(agent, 'swap_prediction_accuracy'):
            swap_accuracy = agent.swap_prediction_accuracy()
        
        return {
            'total_reward': total_reward,
            'success': success,
            'survival': survival,
            'steps': steps,
            'collected_rewards': len(collected_rewards),
            'risk_adjusted_return': risk_adjusted_return,
            'path_efficiency': path_efficiency,
            'max_steps': env.max_steps,
            'episode_num': episode_num,
            'path_length': len(path),
            'death_reason': death_reason,
            'swap_accuracy': swap_accuracy
        }
    
    def run_comprehensive_experiments(self):
        """Run experiments across all conditions with learning tracking."""
        print("=" * 60)
        print("RUNNING COMPREHENSIVE LEARNING EXPERIMENTS")
        print("=" * 60)
        print(f"Agents: {len(self.agents)} (with learning capabilities)")
        print(f"Mazes: {len(self.maze_configs)}")
        print(f"Reward configurations: {len(self.reward_configs)}")
        print(f"Swap probabilities: {len(self.swap_probabilities)}")
        print(f"Episodes per condition: {self.episodes_per_condition}")
        total_episodes = len(self.agents) * len(self.maze_configs) * len(self.reward_configs) * len(self.swap_probabilities) * self.episodes_per_condition
        print(f"Total episodes: {total_episodes:,}")
        
        # Track agent knowledge before and after each maze
        agent_knowledge_before = {}
        agent_knowledge_after = {}
        
        for agent_name, agent in self.agents.items():
            print(f"\nTesting {agent_name} (with learning)...")
            
            # Store initial knowledge
            agent_knowledge_before[agent_name] = self._get_agent_knowledge(agent)
            
            for maze_config in self.maze_configs:
                print(f"  Maze: {maze_config['name']}")
                
                # Reset environment but keep agent knowledge (learning transfer)
                # The agent's Q-values, SR matrix, and swap predictions carry over
                
                for reward_name, reward_values in self.reward_configs:
                    print(f"    Rewards: {reward_name} ({reward_values[0]}, {reward_values[1]})")
                    
                    for swap_prob in self.swap_probabilities:
                        print(f"      Swap prob: {swap_prob}")
                        
                        # Create environment with specific reward configuration
                        env = self.create_environment_with_rewards(maze_config, (reward_name, reward_values), swap_prob)
                        
                        # Run multiple episodes for this condition (tracking learning)
                        condition_results = []
                        condition_key = f"{agent_name}_{maze_config['name']}_{reward_name}_{swap_prob}"
                        
                        for episode in range(self.episodes_per_condition):
                            result = self.run_single_episode(agent, env, episode, condition_key)
                            
                            # Call end_episode for adaptive risk aversion
                            if hasattr(agent, 'end_episode'):
                                agent.end_episode(result['survival'], result['total_reward'])
                            
                            # Track risk threshold if agent has it
                            risk_threshold = getattr(agent, 'risk_threshold', None)
                            
                            # Add metadata
                            result.update({
                                'agent': agent_name,
                                'maze': maze_config['name'],
                                'reward_config': reward_name,
                                'high_reward': reward_values[0],
                                'low_reward': reward_values[1],
                                'swap_prob': swap_prob,
                                'episode': episode,
                                'risk_threshold': risk_threshold
                            })
                            
                            condition_results.append(result)
                        
                        # Calculate averages for this condition
                        avg_reward = np.mean([r['total_reward'] for r in condition_results])
                        exit_rate = np.mean([r['success'] for r in condition_results])
                        survival_rate = np.mean([r['survival'] for r in condition_results])
                        avg_risk_adjusted = np.mean([r['risk_adjusted_return'] for r in condition_results])
                        avg_path_efficiency = np.mean([r['path_efficiency'] for r in condition_results])
                        avg_steps = np.mean([r['steps'] for r in condition_results])
                        avg_collected = np.mean([r['collected_rewards'] for r in condition_results])
                        
                        # Calculate learning improvement (last 20 vs first 20 episodes)
                        if len(condition_results) >= 40:
                            first_20_rewards = [r['total_reward'] for r in condition_results[:20]]
                            last_20_rewards = [r['total_reward'] for r in condition_results[-20:]]
                            learning_improvement = np.mean(last_20_rewards) - np.mean(first_20_rewards)
                        else:
                            learning_improvement = 0
                        
                        # Add summary result
                        summary_result = {
                            'agent': agent_name,
                            'maze': maze_config['name'],
                            'reward_config': reward_name,
                            'high_reward': reward_values[0],
                            'low_reward': reward_values[1],
                            'swap_prob': swap_prob,
                            'avg_reward': avg_reward,
                            'exit_rate': exit_rate,
                            'survival_rate': survival_rate,
                            'avg_risk_adjusted_return': avg_risk_adjusted,
                            'avg_path_efficiency': avg_path_efficiency,
                            'avg_steps': avg_steps,
                            'avg_collected_rewards': avg_collected,
                            'learning_improvement': learning_improvement,
                            'episodes': self.episodes_per_condition
                        }
                        
                        self.results.append(summary_result)
                        
                        print(f"        Avg Reward: {avg_reward:.2f}, Exit Rate: {exit_rate:.2%}, Survival: {survival_rate:.2%}")
                        print(f"        Learning Improvement: {learning_improvement:.2f}")
                
                # Store knowledge after this maze
                agent_knowledge_after[f"{agent_name}_{maze_config['name']}"] = self._get_agent_knowledge(agent)
        
        # Analyze learning transfer
        self._analyze_learning_transfer(agent_knowledge_before, agent_knowledge_after)
        
        print(f"\nCompleted {total_episodes:,} episodes!")
        
        # Save results immediately
        self.save_results()
    
    def _get_agent_knowledge(self, agent) -> Dict:
        """Extract current knowledge state from agent."""
        knowledge = {}
        
        # Try to get Q-values if available
        if hasattr(agent, 'q_values'):
            knowledge['q_values'] = len(agent.q_values) if agent.q_values else 0
        
        # Try to get successor representation if available
        if hasattr(agent, 'sr_matrix'):
            knowledge['sr_entries'] = agent.sr_matrix.size if hasattr(agent.sr_matrix, 'size') else 0
        
        # Try to get model if available
        if hasattr(agent, 'model'):
            knowledge['model_entries'] = len(agent.model) if agent.model else 0
        
        return knowledge
    
    def _run_statistical_tests(self, df):
        """Run statistical significance tests on agent performance."""
        print("\n" + "=" * 60)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("=" * 60)
        
        # Compare agents on key metrics
        metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'learning_improvement']
        
        for metric in metrics:
            print(f"\n{metric.upper()} Comparison:")
            
            # Get data for each agent
            agent_data = {}
            for agent in df['agent'].unique():
                agent_data[agent] = df[df['agent'] == agent][metric].values
            
            # Run ANOVA if we have multiple agents
            if len(agent_data) > 2:
                try:
                    f_stat, p_value = stats.f_oneway(*agent_data.values())
                    print(f"  ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.3f}")
                    if p_value < 0.05:
                        print(f"  *** SIGNIFICANT DIFFERENCE (p < 0.05) ***")
                    else:
                        print(f"  No significant difference")
                except:
                    print(f"  Could not compute ANOVA")
            
            # Pairwise comparisons
            agents = list(agent_data.keys())
            for i in range(len(agents)):
                for j in range(i+1, len(agents)):
                    agent1, agent2 = agents[i], agents[j]
                    try:
                        t_stat, p_value = stats.ttest_ind(agent_data[agent1], agent_data[agent2])
                        mean1, mean2 = np.mean(agent_data[agent1]), np.mean(agent_data[agent2])
                        print(f"  {agent1} vs {agent2}: {mean1:.3f} vs {mean2:.3f}, p={p_value:.3f}")
                        if p_value < 0.05:
                            print(f"    *** SIGNIFICANT ***")
                    except:
                        print(f"  {agent1} vs {agent2}: Could not compute t-test")
    
    def _analyze_learning_transfer(self, knowledge_before: Dict, knowledge_after: Dict):
        """Analyze how learning transfers between mazes."""
        print("\n" + "=" * 60)
        print("LEARNING TRANSFER ANALYSIS")
        print("=" * 60)
        
        for agent_name in self.agents.keys():
            print(f"\n{agent_name}:")
            
            # Get initial knowledge
            initial_knowledge = knowledge_before.get(agent_name, {})
            print(f"  Initial knowledge: {initial_knowledge}")
            
            # Track knowledge growth across mazes
            for maze_config in self.maze_configs:
                maze_name = maze_config['name']
                key = f"{agent_name}_{maze_name}"
                after_knowledge = knowledge_after.get(key, {})
                
                print(f"  After {maze_name}: {after_knowledge}")
                
                # Calculate knowledge growth
                if 'q_values' in initial_knowledge and 'q_values' in after_knowledge:
                    q_growth = after_knowledge['q_values'] - initial_knowledge['q_values']
                    print(f"    Q-values growth: {q_growth}")
                
                if 'sr_entries' in initial_knowledge and 'sr_entries' in after_knowledge:
                    sr_growth = after_knowledge['sr_entries'] - initial_knowledge['sr_entries']
                    print(f"    SR entries growth: {sr_growth}")
    
    def save_results(self, output_file: str = "comprehensive_results.csv"):
        """Save results to CSV file."""
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(output_dir, output_file), index=False)
        print(f"\nResults saved to {os.path.join(output_dir, output_file)}")
        print(f"Shape: {df.shape}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total conditions tested: {len(self.results)}")
        print(f"Average reward across all conditions: {df['avg_reward'].mean():.2f}")
        print(f"Average exit rate: {df['exit_rate'].mean():.2%}")
        print(f"Average survival rate: {df['survival_rate'].mean():.2%}")
        print(f"Average learning improvement: {df['learning_improvement'].mean():.2f}")
        print(f"Average collected rewards: {df['avg_collected_rewards'].mean():.2f}")
        
        # Statistical significance testing
        self._run_statistical_tests(df)
        
        return df
    
    def create_learning_visualizations(self):
        """Create learning progress visualizations."""
        print("\nCreating learning progress visualizations...")
        
        # Ensure output directory exists for plots
        output_dir = os.path.join(os.path.dirname(__file__), 'analysis')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create learning curves for each agent-maze-reward-swap combination
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, (agent_name, agent) in enumerate(self.agents.items()):
            ax = axes[i]
            
            # Plot learning curves for different conditions
            for maze_config in self.maze_configs:
                for reward_name, _ in self.reward_configs[:3]:  # Show first 3 reward configs
                    for swap_prob in [0.0, 0.5, 1.0]:  # Show key swap probabilities
                        condition_key = f"{agent_name}_{maze_config['name']}_{reward_name}_{swap_prob}"
                        
                        if condition_key in self.learning_progress:
                            progress = self.learning_progress[condition_key]
                            episodes = [p['episode'] for p in progress]
                            rewards = [p['total_reward'] for p in progress]
                            
                            # Moving average for smoother curves
                            if len(rewards) >= 10:
                                window = min(10, len(rewards) // 4)
                                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                                ax.plot(episodes[window-1:], moving_avg, 
                                       label=f"{maze_config['name']}, {reward_name}, p={swap_prob}", alpha=0.7)
            
            ax.set_title(f'{agent_name} - Learning Progress')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward (Moving Average)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_progress_comprehensive.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create learning improvement heatmap by reward configuration
        df = pd.DataFrame(self.results)
        pivot_data = df.pivot_table(
            values='learning_improvement', 
            index='agent', 
            columns='reward_config', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        plt.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto')
        plt.colorbar(label='Learning Improvement')
        plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
        plt.yticks(range(len(pivot_data.index)), pivot_data.index)
        plt.xlabel('Reward Configuration')
        plt.ylabel('Agent')
        plt.title('Learning Improvement by Agent and Reward Configuration')
        
        # Add text annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                plt.text(j, i, f'{pivot_data.iloc[i, j]:.1f}', 
                        ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_improvement_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create risk threshold evolution plot for adaptive agents
        self._create_risk_threshold_plot()
        
        print("Learning visualizations created!")
    
    def _create_risk_threshold_plot(self):
        """Create a plot showing risk threshold evolution over time for adaptive agents."""
        plt.figure(figsize=(12, 8))
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'risk_threshold_history'):
                plt.plot(range(len(agent.risk_threshold_history)), agent.risk_threshold_history, label=agent_name)
        plt.title('Risk Threshold Evolution Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Risk Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'analysis', 'risk_threshold_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self):
        """Run all analysis components."""
        print("\n" + "=" * 60)
        print("RUNNING COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        
        # Create learning visualizations first
        self.create_learning_visualizations()
        
        # Import analysis modules
        try:
            from analysis.visualization_engine import VisualizationEngine
            from analysis.statistical_analysis import StatisticalAnalyzer
            from analysis.performance_metrics import PerformanceAnalyzer
            from analysis.comparative_analysis import ComparativeAnalyzer
        except ImportError as e:
            print(f"Error importing analysis modules: {e}")
            return False
        
        # Run each analysis component
        print("\n1. Running Statistical Analysis...")
        try:
            stat_analyzer = StatisticalAnalyzer()
            stat_results = stat_analyzer.generate_comprehensive_report()
            print("✓ Statistical analysis complete")
        except Exception as e:
            print(f"✗ Statistical analysis failed: {e}")
        
        print("\n2. Running Performance Metrics Analysis...")
        try:
            perf_analyzer = PerformanceAnalyzer()
            perf_results = perf_analyzer.generate_comprehensive_performance_report()
            print("✓ Performance metrics analysis complete")
        except Exception as e:
            print(f"✗ Performance metrics analysis failed: {e}")
        
        print("\n3. Running Comparative Analysis...")
        try:
            comp_analyzer = ComparativeAnalyzer()
            comp_results = comp_analyzer.generate_comprehensive_comparison_report()
            print("✓ Comparative analysis complete")
        except Exception as e:
            print(f"✗ Comparative analysis failed: {e}")
        
        print("\n4. Generating Visualizations...")
        try:
            viz_engine = VisualizationEngine()
            viz_engine.generate_all_visualizations()
            print("✓ Visualizations complete")
        except Exception as e:
            print(f"✗ Visualization generation failed: {e}")
        
        return True
    
    def generate_experiment_summary(self):
        """Generate a summary of the experiments including learning."""
        df = pd.DataFrame(self.results)
        
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY (WITH LEARNING)")
        print("=" * 60)
        
        # Agent performance summary
        print("\nAGENT PERFORMANCE SUMMARY:")
        print("-" * 30)
        agent_summary = df.groupby('agent').agg({
            'avg_reward': ['mean', 'std'],
            'exit_rate': ['mean', 'std'],
            'survival_rate': ['mean', 'std'],
            'avg_risk_adjusted_return': ['mean', 'std'],
            'learning_improvement': ['mean', 'std'],
            'avg_collected_rewards': ['mean', 'std']
        }).round(3)
        
        print(agent_summary)
        
        # Learning improvement ranking
        print("\nLEARNING IMPROVEMENT RANKING:")
        print("-" * 30)
        learning_ranking = df.groupby('agent')['learning_improvement'].mean().sort_values(ascending=False)
        for i, (agent, improvement) in enumerate(learning_ranking.items(), 1):
            print(f"  {i}. {agent}: {improvement:.2f}")
        
        # Reward configuration impact
        print("\nREWARD CONFIGURATION IMPACT:")
        print("-" * 35)
        reward_summary = df.groupby('reward_config').agg({
            'avg_reward': 'mean',
            'exit_rate': 'mean',
            'survival_rate': 'mean',
            'learning_improvement': 'mean',
            'avg_collected_rewards': 'mean'
        }).round(3)
        
        print(reward_summary)
        
        # Swap probability impact on learning
        print("\nSWAP PROBABILITY IMPACT ON LEARNING:")
        print("-" * 40)
        swap_learning = df.groupby('swap_prob').agg({
            'avg_reward': 'mean',
            'exit_rate': 'mean',
            'survival_rate': 'mean',
            'learning_improvement': 'mean'
        }).round(3)
        
        print(swap_learning)
        
        # Maze performance with learning
        print("\nMAZE PERFORMANCE (WITH LEARNING):")
        print("-" * 35)
        maze_summary = df.groupby('maze').agg({
            'avg_reward': 'mean',
            'exit_rate': 'mean',
            'survival_rate': 'mean',
            'learning_improvement': 'mean'
        }).round(3)
        
        print(maze_summary)
    
    def list_generated_files(self):
        """List all generated files."""
        print("\n" + "=" * 60)
        print("GENERATED FILES")
        print("=" * 60)
        # Only list files that are actually generated by the script
        generated_files = [
            'reward_configuration_analysis.png',
            'learning_progress_comprehensive.png',
            'learning_improvement_heatmap.png',
            'learning_improvement_by_rewards.png',
            'comprehensive_results.csv'
        ]
        print("Generated files:")
        for file in generated_files:
            if os.path.exists(file):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} (not found)")
        # User reference: these are the main result files to check as PNG/CSV
        print("\nMain result files:")
        print("- reward_configuration_analysis.png: Reward config analysis plot")
        print("- learning_progress_comprehensive.png: Learning progress curves")
        print("- learning_improvement_heatmap.png: Heatmap of learning improvement")
        print("- learning_improvement_by_rewards.png: Learning by reward config")
        print("- comprehensive_results.csv: All experiment results (CSV)")

def main():
    """Main function - runs everything in one go!"""
    print("UNIFIED LEARNING EXPERIMENT AND ANALYSIS RUNNER")
    print("=" * 60)
    print("This script will:")
    print("1. Run comprehensive experiments with LEARNING agents across all conditions")
    print("2. Test 7 different reward configurations (small, medium, large, extreme, etc.)")
    print("3. Track learning progress over 100 episodes per condition")
    print("4. Analyze learning transfer between mazes")
    print("5. Generate statistical analysis and visualizations")
    print("6. Create performance reports and learning comparisons")
    print("=" * 60)
    
    # Create unified analyzer
    analyzer = UnifiedExperimentAnalyzer()
    
    # Run experiments
    analyzer.run_comprehensive_experiments()
    
    # Generate experiment summary
    analyzer.generate_experiment_summary()
    
    # Run analysis
    success = analyzer.run_analysis()
    
    # List generated files
    analyzer.list_generated_files()
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print("All learning experiments and analysis completed in one run!")
    print("Check the generated PNG files and reports for detailed results.")
    print("Includes learning progress tracking, reward configuration analysis, and learning transfer!")

if __name__ == "__main__":
    main() 