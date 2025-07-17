#!/usr/bin/env python3
"""
Experiment and Analysis Runner

This script runs experiments with different agents and settings, then analyzes the results and generates charts.

CONFIGURATION:
- Fixed step budgets: 10, 20, 30, 45, 60, 80 steps
- Only mazes 5 and 6
- All swap probabilities
- 100 episodes per condition
- All reward configurations
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
logging.basicConfig(level=logging.INFO, filename='runs.log')
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from scipy import stats
import random
from itertools import product
from utils.logger import ExperimentLogger
import csv

# Add analysis directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis'))

from environments.maze_environment import MazeEnvironment
from agents.model_based_greedy_agent import ModelBasedGreedyAgent
from agents.model_based_survival_agent import ModelBasedSurvivalAgent
from agents.sr_greedy_agent import SuccessorRepresentationGreedyAgent
from agents.sr_reasonable_agent import SuccessorRepresentationReasonableAgent
from utils.schedule_swaps import schedule_fork_swap

import copy

class ExperimentAnalyzer:
    """Runs experiments and analyzes results with the current configuration."""
    
    def __init__(self):
        self.results_dir = os.path.join(os.path.dirname(__file__), 'results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.results = []
        self.learning_progress = defaultdict(list)
        self.learning_transfer = defaultdict(dict)

        self.agent_prototypes = {
            'Model-Based Greedy': ModelBasedGreedyAgent(learning_rate=0.1, exploration_rate=0.1),
            'Model-Based Survival': ModelBasedSurvivalAgent(step_budget=50, learning_rate=0.1, exploration_rate=0.05),
            'SR-Greedy': SuccessorRepresentationGreedyAgent(alpha=0.1, exploration_rate=0.1),
            'SR-Reasonable': SuccessorRepresentationReasonableAgent(alpha=0.1, beta=1.0, exploration_rate=0.05)
        }
        # we'll deep‑copy these prototypes for each new condition
        self.agents = {name: copy.deepcopy(proto) for name, proto in self.agent_prototypes.items()}
        
        # Global episode counter and per-agent learning log
        self.global_episode = 0
        self.learning_log = {name: [] for name in self.agents}
        
        # Load maze configurations - always include mazes 5 and 6
        self.maze_configs = self._load_mazes()
        
        # Configuration
        self.episodes_per_condition = 150
        self.swap_probabilities = [0,0.3,0.5,0.7,1.0]
             
        
        self.step_budgets = [10,20,30,50,80,120,200]
        
        # All reward configurations
        self.reward_configs = [
            ('small', [5, 3]),
            ('medium', [10, 6]),
            ('large', [20, 12]),
            ('extreme', [50, 30]),
            ('big_diff', [50, 1]),
            ('small_diff', [10, 9]),
            ('equal', [10, 10])
        ]
        
        # Calculate total episodes
        self.total_episodes = len(self.agents) * len(self.maze_configs) * len(self.reward_configs) * len(self.step_budgets) * len(self.swap_probabilities) * self.episodes_per_condition
        
        # Performance tracking
        self.start_time = None
        self.episodes_completed = 0
        
    def _load_mazes(self) -> List[Dict]:
        """Load only mazes 5 and 6 configurations."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        maze_dir = os.path.join(base_dir, "environments", "maze_configs")
        configs = []
        
        # Only load mazes 5 and 6
        target_mazes = ['maze5', 'maze6']
        
        for filename in os.listdir(maze_dir):
            if filename.endswith('.json'):
                maze_name = filename.replace('.json', '')
                if maze_name in target_mazes:
                    with open(os.path.join(maze_dir, filename), 'r') as f:
                        config = json.load(f)
                        config['name'] = maze_name
                        configs.append(config)
        
        # Ensure we found mazes 5 and 6
        maze_names = [config['name'] for config in configs]
        print(f"Loaded mazes: {maze_names}")
        
        if len(configs) != 2:
            print(f"Warning: Expected 2 mazes (5 and 6), but found {len(configs)}")
        
        return configs
    
    def create_environment_with_rewards(self, maze_config: Dict, reward_config: Tuple[str, List[int]], swap_prob: float, step_budget: int) -> MazeEnvironment:
        """Create environment with specific reward configuration and step budget."""
        reward_name, [high_val, low_val] = reward_config
        
        # Get original reward positions from maze config
        original_rewards = maze_config['rewards']
        if len(original_rewards) >= 2:
            reward_positions = [original_rewards[0][0], original_rewards[1][0]]
        else:
            # Fallback if not enough rewards in config
            reward_positions = [(1, 1), (2, 2)]
        
        # Create environment with specified step budget
        env = MazeEnvironment(
            grid=maze_config['grid'],
            start=tuple(maze_config['start']),
            exit=tuple(maze_config['exit']),
            rewards=[(tuple(reward_positions[0]), high_val), (tuple(reward_positions[1]), low_val)],
            swap_prob=swap_prob
        )
        
        # Set the step budget
        env.max_steps = step_budget
        
        # Use fork logic to schedule swaps (like in interactive visualization)
        if swap_prob > 0:  # Only schedule swaps if probability > 0
            env.swap_steps = schedule_fork_swap(
                grid=env.grid,
                start=env.start,
                reward_positions=[pos for pos, _ in env.initial_rewards],
                jitter=1
            )
            # If fork logic didn't find a swap point, use a default step
            if not env.swap_steps:
                env.swap_steps = [max(1, step_budget // 3)]  # Swap at 1/3 of the budget
        
        return env
    
    def _print_progress(self, current_episode, total_episodes, start_time):
        """Print progress with time estimates."""
        elapsed = time.time() - start_time
        if current_episode > 0:
            episodes_per_second = current_episode / elapsed
            remaining_episodes = total_episodes - current_episode
            eta_seconds = remaining_episodes / episodes_per_second
            eta_minutes = eta_seconds / 60
            
            progress = (current_episode / total_episodes) * 100
            print(f"\rProgress: {progress:.1f}% ({current_episode:,}/{total_episodes:,}) "
                  f"ETA: {eta_minutes:.1f} minutes", end='', flush=True)
    
    def run_single_episode(self, agent, env, episode_num: int, condition_key: str, maze_name: str) -> Dict:
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
        death_x = None
        death_y = None
        if not success:
            if steps >= env.max_steps:
                death_reason = 'timeout'
            else:
                death_reason = 'other'
            if death_reason:
                death_point = env.get_position()
                death_x = death_point[0]
                death_y = death_point[1]
        
        # Risk-adjusted return
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
            'path_length': len(path),
            'risk_adjusted_return': risk_adjusted_return,
            'path_efficiency': path_efficiency,
            'swap_accuracy': swap_accuracy,
            'death_reason': death_reason,
            'death_x': death_x,
            'death_y': death_y
        }
    
    def run_comprehensive_experiments(self):
        print("=" * 60)
        print("RUNNING LIFELONG/TRANSFER + CLASSIC EXPERIMENTS")
        print("=" * 60)
        print(f"Agents: {len(self.agent_prototypes)} (with lifelong learning)")
        print(f"Mazes: {len(self.maze_configs)}")
        print(f"Reward configurations: {len(self.reward_configs)}")
        print(f"Step budgets: {self.step_budgets}")
        print(f"Swap probabilities: {self.swap_probabilities}")
        print(f"Episodes per block: {self.episodes_per_condition}")

        all_blocks = list(product(
            self.maze_configs,
            self.reward_configs,
            self.step_budgets,
            self.swap_probabilities
        ))
        random.shuffle(all_blocks)

        logger = ExperimentLogger(self.results_dir)
        classic_results = []

        test_maze = self.maze_configs[0]
        test_reward_name, test_reward_values = self.reward_configs[0]
        test_step_budget = self.step_budgets[0]
        test_swap_prob = self.swap_probabilities[0]
        test_env = self.create_environment_with_rewards(test_maze, (test_reward_name, test_reward_values), test_swap_prob, test_step_budget)
        test_episodes = 10

        for agent_name, prototype in self.agent_prototypes.items():
            print(f"\nTesting {agent_name} (lifelong + classic)...")
            agent = copy.deepcopy(prototype)
            block_index = 0
            for maze_config, (reward_name, reward_values), step_budget, swap_prob in all_blocks:
                env = self.create_environment_with_rewards(maze_config, (reward_name, reward_values), swap_prob, step_budget)
                rewards = []
                successes = []
                survivals = []
                path_efficiencies = []
                steps_list = []
                collected_list = []
                risk_adjusted_list = []
                death_locations = []
                death_reasons = []
                for epi in range(self.episodes_per_condition):
                    start = time.time()
                    result = self.run_single_episode(agent, env, epi, f"{agent_name}_block{block_index}", maze_config['name'])
                    end = time.time()
                    rewards.append(result['total_reward'])
                    successes.append(result['success'])
                    survivals.append(result['survival'])
                    path_efficiencies.append(result['path_efficiency'])
                    steps_list.append(result['steps'])
                    collected_list.append(result['collected_rewards'])
                    risk_adjusted_list.append(result['risk_adjusted_return'])
                    if not result['survival']:
                        death_locations.append((result['death_x'], result['death_y']))
                        death_reasons.append(result['death_reason'])
                    # Per-episode timing log
                    logging.info(f"{agent_name},{maze_config['name']},{epi},{end-start:.4f}s")
                # Classic stats
                avg_reward = np.mean(rewards)
                exit_rate = np.mean(successes)
                survival_rate = np.mean(survivals)
                avg_path_efficiency = np.mean(path_efficiencies)
                avg_steps = np.mean(steps_list)
                avg_collected = np.mean(collected_list)
                avg_risk_adjusted = np.mean(risk_adjusted_list)
                # Learning improvement (windowed)
                N = len(rewards)
                window = max(1, int(self.episodes_per_condition * 0.2))
                if N >= 2 * window:
                    reward_improvement = np.mean(rewards[-window:]) - np.mean(rewards[:window])
                    survival_improvement = np.mean(survivals[-window:]) - np.mean(survivals[:window])
                else:
                    reward_improvement = np.nan
                    survival_improvement = np.nan
                # Slope
                if N > 1:
                    reward_slope, _ = np.polyfit(np.arange(N), rewards, 1)
                else:
                    reward_slope = np.nan
                # Death stats
                from collections import Counter
                if death_locations:
                    most_common_location = Counter(death_locations).most_common(1)[0][0]
                    death_x, death_y = most_common_location
                    death_reason = Counter(death_reasons).most_common(1)[0][0]
                else:
                    death_x, death_y, death_reason = np.nan, np.nan, np.nan
                death_count = len(death_locations)
                # Classic result row
                classic_results.append({
                    'agent': agent_name,
                    'maze': maze_config['name'],
                    'reward_config': reward_name,
                    'step_budget': step_budget,
                    'swap_prob': swap_prob,
                    'avg_reward': avg_reward,
                    'exit_rate': exit_rate,
                    'survival_rate': survival_rate,
                    'avg_risk_adjusted_return': avg_risk_adjusted,
                    'avg_path_efficiency': avg_path_efficiency,
                    'avg_steps': avg_steps,
                    'avg_collected_rewards': avg_collected,
                    'reward_improvement': reward_improvement,
                    'survival_improvement': survival_improvement,
                    'learning_improvement': reward_slope,
                    'reward_slope': reward_slope,
                    'episodes': self.episodes_per_condition,
                    'death_count': death_count,
                    'death_x': death_x,
                    'death_y': death_y,
                    'death_reason': death_reason
                })
                # Lifelong/transfer metrics
                k_initial = min(10, len(rewards))
                if k_initial >= 2:
                    slope = np.polyfit(range(k_initial), rewards[:k_initial], 1)[0]
                else:
                    slope = 0.0
                auc = np.trapz(rewards, dx=1)
                optimal_block_reward = max(rewards) if rewards else 0
                regret = sum(optimal_block_reward - r for r in rewards)
                final_reward = rewards[-1] if rewards else 0
                logger.record_block(agent_name, maze_config['name'], reward_name, step_budget, swap_prob, block_index, slope, auc, regret, final_reward)
                # Held-out test after each block
                test_rewards = []
                for _ in range(test_episodes):
                    test_result = self.run_single_episode(agent, test_env, 0, f"{agent_name}_test", test_maze['name'])
                    test_rewards.append(test_result['total_reward'])
                logger.record_transfer(agent_name, block_index, np.mean(test_rewards), np.std(test_rewards))
                block_index += 1
        # Save all results
        logger.save()
        # Save classic results
        classic_path = os.path.join(self.results_dir, 'comprehensive_results.csv')
        with open(classic_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=classic_results[0].keys())
            writer.writeheader()
            writer.writerows(classic_results)
        print(f"\n✅ Both classic and lifelong/transfer experiment results saved!")
    
    def _get_agent_knowledge(self, agent) -> Dict:
        """Extract current knowledge state from agent."""
        knowledge = {}
        # Try to get Q-values if available (case-insensitive)
        if hasattr(agent, 'Q_values'):
            knowledge['q_values'] = len(agent.Q_values)
        elif hasattr(agent, 'q_values'):
            knowledge['q_values'] = len(agent.q_values) if agent.q_values else 0
        # Try to get successor representation if available (case-insensitive)
        if hasattr(agent, 'M'):
            # successor-representation entries
            knowledge['sr_entries'] = sum(len(v) for v in agent.M.values())
        elif hasattr(agent, 'sr_matrix'):
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
                    try:
                        stat, p_val = stats.ttest_ind(agent_data[agents[i]], agent_data[agents[j]])
                        print(f"  {agents[i]} vs {agents[j]}: t={stat:.3f}, p={p_val:.3f}")
                        if p_val < 0.05:
                            print(f"    *** SIGNIFICANT ***")
                    except:
                        print(f"  {agents[i]} vs {agents[j]}: Could not compute")
    
    def _analyze_learning_transfer(self, knowledge_before: Dict, knowledge_after: Dict):
        """Analyze how agent knowledge transfers between mazes."""
        print("\n" + "=" * 60)
        print("LEARNING TRANSFER ANALYSIS")
        print("=" * 60)
        
        for agent_name in self.agents.keys():
            print(f"\n{agent_name}:")
            
            # Find knowledge before and after for this agent
            before = knowledge_before.get(agent_name, {})
            
            # Find all after-knowledge entries for this agent
            after_entries = {k: v for k, v in knowledge_after.items() if k.startswith(agent_name)}
            
            if before and after_entries:
                print(f"  Initial knowledge:")
                for key, value in before.items():
                    print(f"    {key}: {value}")
                
                print(f"  Knowledge after each maze:")
                for maze_key, after in after_entries.items():
                    maze_name = maze_key.split('_', 1)[1] if '_' in maze_key else 'unknown'
                    print(f"    {maze_name}:")
                    for key, value in after.items():
                        print(f"      {key}: {value}")
    
    def save_results(self, output_file: str = None):
        """Save results to CSV file."""
        if output_file is None:
            output_file = os.path.join(self.results_dir, 'comprehensive_results.csv')
        df = pd.DataFrame(self.results)
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")
        print(f"   Shape: {df.shape}")
    
    def list_generated_files(self):
        # REMOVE all print statements in this function
        pass

def main():
    """Main function - runs everything with the current configuration."""
    print("EXPERIMENT AND ANALYSIS RUNNER")
    print("=" * 60)
    print("This script will:")
    print("1. Run comprehensive experiments with LEARNING agents across all conditions")
    print("2. Test 7 different reward configurations")
    print("3. Use step budgets: 10, 20, 30, 45, 60, 80")
    print("4. Test only mazes 5 and 6")
    print("5. Test all swap probabilities")
    print("6. Run 100 episodes per condition")
    print("7. Generate statistical analysis and visualizations")
    print("8. Create performance reports and learning comparisons")
    print("=" * 60)
    
    # Create analyzer
    analyzer = ExperimentAnalyzer()
    
    # Run experiments
    analyzer.run_comprehensive_experiments()
    
    
    # List generated files
    analyzer.list_generated_files()
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print("All learning experiments and analysis completed!")
    print("Check the generated PNG files and reports for detailed results.")
    print("Includes learning progress tracking, reward configuration analysis, and learning transfer!")

if __name__ == "__main__":
    main() 