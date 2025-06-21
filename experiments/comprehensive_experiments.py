#!/usr/bin/env python3
import sys, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns
from scipy import stats

# Ensure project root is importable
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from environments.maze_environment import MazeEnvironment
from agents.model_based_greedy_agent import ModelBasedGreedyAgent
from agents.model_based_survival_agent import ModelBasedSurvivalAgent
from agents.sr_greedy_agent import SuccessorRepresentationGreedyAgent
from agents.sr_reasonable_agent import SuccessorRepresentationReasonableAgent
from utils.schedule_swaps import schedule_fork_swap

def run_experiment(env, agent, num_episodes=100, render=False):
    """Run agent for multiple episodes and collect detailed statistics."""
    episode_rewards = []
    episode_steps = []
    exits_reached = 0
    rewards_collected = []
    paths_taken = []
    survival_episodes = 0  # Reached exit without timeout
    optimal_path_lengths = []  # For efficiency calculation
    decision_consistency = []  # Track similar decisions across episodes
    
    # Create figure for visualization if rendering
    fig = None
    if render:
        fig = plt.figure(figsize=(12, 6))
        plt.ion()  # Turn on interactive mode
    
    for episode in range(num_episodes):
        state = env.reset()
        agent.set_position(state)
        agent.inform_swap_info(env.swap_prob)
        total_reward = 0
        steps = 0
        collected = set()
        path = [state]
        
        while True:
            if render and episode == num_episodes-1:  # Render only last episode
                visualize_state(env, agent, steps, total_reward, fig)
                plt.pause(0.1)
                
            action = agent.select_action(env)
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            path.append(next_state)
            
            if reward > 0:
                collected.add(next_state)
                
            agent.update(env, action, reward, next_state)
            agent.set_position(next_state)
            
            if done:
                if next_state == env.exit:
                    exits_reached += 1
                    if steps < env.max_steps:  # Reached exit without timeout
                        survival_episodes += 1
                break
                
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        rewards_collected.append(len(collected))
        paths_taken.append(path)
        
        # Calculate optimal path length (shortest path to highest reward + exit)
        optimal_length = calculate_optimal_path_length(env, state)
        optimal_path_lengths.append(optimal_length)
        
        if render and episode == num_episodes-1:
            plt.ioff()  # Turn off interactive mode
            plt.close(fig)  # Close the figure
            
    # Calculate additional metrics
    risk_adjusted_returns = [r/s if s > 0 else 0 for r, s in zip(episode_rewards, episode_steps)]
    path_efficiency = [o/a if a > 0 else 0 for o, a in zip(optimal_path_lengths, episode_steps)]
    survival_rate = survival_episodes / num_episodes
    
    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_steps': np.mean(episode_steps),
        'std_steps': np.std(episode_steps),
        'exit_rate': exits_reached / num_episodes,
        'survival_rate': survival_rate,
        'avg_rewards_collected': np.mean(rewards_collected),
        'std_rewards_collected': np.std(rewards_collected),
        'avg_risk_adjusted_return': np.mean(risk_adjusted_returns),
        'std_risk_adjusted_return': np.std(risk_adjusted_returns),
        'avg_path_efficiency': np.mean(path_efficiency),
        'std_path_efficiency': np.std(path_efficiency),
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'paths_taken': paths_taken,
        'risk_adjusted_returns': risk_adjusted_returns,
        'path_efficiency': path_efficiency
    }

def calculate_optimal_path_length(env, start_state):
    """Calculate optimal path length to highest reward then exit."""
    rewards = env.get_reward_positions()
    if not rewards:
        # No rewards left, just distance to exit
        return env._bfs_dist(start_state, env.exit)
    
    # Find highest reward
    high_reward_pos = max(rewards.items(), key=lambda kv: kv[1])[0]
    
    # Distance to highest reward + distance from reward to exit
    dist_to_reward = env._bfs_dist(start_state, high_reward_pos)
    dist_to_exit = env._bfs_dist(high_reward_pos, env.exit)
    
    return dist_to_reward + dist_to_exit

def visualize_state(env, agent, step, total_reward=0, fig=None):
    """Enhanced visualization with colored markers and stats."""
    if fig is None:
        fig = plt.figure(figsize=(12, 6))
    
    plt.figure(fig.number)
    plt.clf()
    
    # Create a figure with two subplots side by side
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    # Left subplot: Maze visualization
    ax1.imshow(env.grid, cmap='binary', interpolation='none')
    
    # Draw rewards with red and green colors
    rp = env.get_reward_positions()
    if rp:
        reward_positions = list(rp.items())
        if len(reward_positions) >= 2:
            # First reward: Green
            (r1, c1), v1 = reward_positions[0]
            ax1.scatter(c1, r1, s=100 + v1 * 10, c='green', alpha=0.8, label=f'Reward 1: {v1}')
            
            # Second reward: Red
            (r2, c2), v2 = reward_positions[1]
            ax1.scatter(c2, r2, s=100 + v2 * 10, c='red', alpha=0.8, label=f'Reward 2: {v2}')
        elif len(reward_positions) == 1:
            # Only one reward left
            (r, c), v = reward_positions[0]
            ax1.scatter(c, r, s=100 + v * 10, c='green', alpha=0.8, label=f'Reward: {v}')
    
    # Draw start, exit, agent with distinct shapes
    sr, sc = env.start
    er, ec = env.exit
    ar, ac = agent.get_position()
    
    ax1.scatter(sc, sr, marker='*', color='blue', s=150, label='Start')
    ax1.scatter(ec, er, marker='X', color='red', s=150, label='Exit')
    ax1.scatter(ac, ar, marker='o', color='black', s=120, label='Agent')
    
    ax1.set_title(f"{agent.name if hasattr(agent, 'name') else 'Agent'}\nStep: {step}")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right subplot: Statistics
    ax2.axis('off')
    
    # Display statistics
    stats_text = f"""
    AGENT STATISTICS
    
    Current Step: {step}
    Total Steps Available: {env.max_steps}
    Steps Remaining: {env.max_steps - step}
    
    Total Reward Collected: {total_reward}
    Agent Position: {agent.get_position()}
    
    REWARD INFORMATION:
    """
    
    # Add reward information
    if rp:
        for i, ((r, c), v) in enumerate(rp.items(), 1):
            stats_text += f"Reward {i}: {v} at position ({r}, {c})\n"
    else:
        stats_text += "All rewards collected!\n"
    
    # Add collected rewards info
    if hasattr(env, 'collected') and env.collected:
        stats_text += f"\nCOLLECTED REWARDS:\n"
        for pos in env.collected:
            # Find the original value for this position
            original_value = None
            for orig_pos, val in env.initial_rewards:
                if orig_pos == pos:
                    original_value = val
                    break
            if original_value is not None:
                stats_text += f"  {original_value} at ({pos[0]}, {pos[1]})\n"
    
    # Add exit status
    if agent.get_position() == env.exit:
        stats_text += "\n🎉 REACHED EXIT! 🎉"
    elif step >= env.max_steps:
        stats_text += "\n⏰ STEP BUDGET EXHAUSTED!"
    
    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()

def run_comprehensive_experiments(num_episodes=100, enable_visualization=False):
    """Run comprehensive experiments across all mazes, agents, and swap probabilities."""
    
    # Configuration
    mazes = ['maze1.json', 'maze2.json', 'maze3.json', 'maze4.json']
    swap_probs = [0.0, 0.25, 0.5, 0.75, 1.0]  # More granular swap probabilities
    
    # Enhanced reward configurations
    reward_configs = [
        ('small', [5, 3]),           # Small rewards: 5 and 3
        ('medium', [10, 6]),         # Medium rewards: 10 and 6  
        ('large', [20, 12]),         # Large rewards: 20 and 12
        ('extreme', [50, 30]),       # Extreme rewards: 50 and 30
        ('big_diff', [50, 1]),       # Big difference: 50 and 1
        ('small_diff', [10, 9]),     # Small difference: 10 and 9
        ('equal', [10, 10])          # Equal rewards: 10 and 10
    ]
    
    agent_configs = [
        ('Greedy', lambda: ModelBasedGreedyAgent(step_budget=100)),
        ('Survival', lambda: ModelBasedSurvivalAgent(100)),
        ('SR-Greedy', lambda: SuccessorRepresentationGreedyAgent(alpha=0.1)),
        ('SR-Reasonable', lambda: SuccessorRepresentationReasonableAgent(alpha=0.1, beta=1.0))
    ]
    
    all_results = []
    
    for maze_file in mazes:
        maze_path = os.path.join(project_root, "environments", "maze_configs", maze_file)
        base_env = MazeEnvironment.from_json(maze_path)
        
        for reward_name, [high_val, low_val] in reward_configs:
            print(f"\n=== Testing {maze_file} with {reward_name} rewards ({high_val}, {low_val}) ===")
            
            # Create environment with modified rewards
            env = MazeEnvironment(
                grid=base_env.grid,
                start=base_env.start,
                exit=base_env.exit,
                rewards=[(base_env.initial_rewards[0][0], high_val), 
                        (base_env.initial_rewards[1][0], low_val)],
                swap_steps=base_env.swap_steps,
                swap_prob=base_env.swap_prob
            )
            
            for swap_prob in swap_probs:
                print(f"  Swap probability: {swap_prob}")
                
                # Configure swap at fork points
                env.swap_steps = schedule_fork_swap(
                    grid=env.grid,
                    start=env.start,
                    reward_positions=[pos for pos,_ in env.initial_rewards],
                    jitter=1
                )
                env.swap_prob = swap_prob
                
                for agent_name, agent_factory in agent_configs:
                    print(f"    Running {agent_name}...")
                    
                    agent = agent_factory()
                    agent.name = agent_name
                    
                    # Run experiments with or without visualization
                    if enable_visualization and num_episodes > 1:
                        # Run all but the last episode without rendering
                        results = run_experiment(env, agent, num_episodes=num_episodes-1, render=False)
                        # Run the last episode with rendering
                        last_result = run_experiment(env, agent, num_episodes=1, render=True)
                        # Combine results
                        for k in ['episode_rewards', 'episode_steps', 'paths_taken']:
                            results[k] += last_result[k]
                        for k in ['avg_reward', 'std_reward', 'avg_steps', 'std_steps', 'exit_rate', 'avg_rewards_collected', 'std_rewards_collected']:
                            # Recompute averages including the last episode
                            all_values = results['episode_rewards'] if 'reward' in k else results['episode_steps']
                            if k.startswith('avg_'):
                                results[k] = np.mean(all_values)
                            elif k.startswith('std_'):
                                results[k] = np.std(all_values)
                        # Recompute exit_rate
                        results['exit_rate'] = (results['exit_rate'] * (num_episodes-1) + (1 if last_result['paths_taken'][-1][-1] == env.exit else 0)) / num_episodes
                    else:
                        results = run_experiment(env, agent, num_episodes=num_episodes, render=False)
                    
                    # Store results with metadata
                    result_entry = {
                        'maze': maze_file.replace('.json', ''),
                        'reward_config': reward_name,
                        'high_reward': high_val,
                        'low_reward': low_val,
                        'swap_prob': swap_prob,
                        'agent': agent_name,
                        **results
                    }
                    all_results.append(result_entry)
    
    return all_results

def create_comprehensive_plots(results):
    """Create comprehensive visualization plots."""
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Overall performance comparison
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    fig.suptitle('Comprehensive Agent Performance Analysis', fontsize=16, fontweight='bold')
    
    # Average reward by agent and reward config
    ax1 = axes[0, 0]
    pivot_reward = df.groupby(['agent', 'reward_config'])['avg_reward'].mean().unstack()
    pivot_reward.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Average Reward by Agent and Reward Configuration')
    ax1.set_ylabel('Average Reward')
    ax1.legend(title='Reward Config')
    ax1.tick_params(axis='x', rotation=45)
    
    # Exit rate by agent and swap probability
    ax2 = axes[0, 1]
    pivot_exit = df.groupby(['agent', 'swap_prob'])['exit_rate'].mean().unstack()
    pivot_exit.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Exit Rate by Agent and Swap Probability')
    ax2.set_ylabel('Exit Rate')
    ax2.legend(title='Swap Probability')
    ax2.tick_params(axis='x', rotation=45)
    
    # Risk-adjusted returns by agent and reward config
    ax3 = axes[1, 0]
    pivot_risk = df.groupby(['agent', 'reward_config'])['avg_risk_adjusted_return'].mean().unstack()
    pivot_risk.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_title('Risk-Adjusted Returns by Agent and Reward Configuration')
    ax3.set_ylabel('Risk-Adjusted Return')
    ax3.legend(title='Reward Config')
    ax3.tick_params(axis='x', rotation=45)
    
    # Path efficiency by agent and maze
    ax4 = axes[1, 1]
    pivot_efficiency = df.groupby(['agent', 'maze'])['avg_path_efficiency'].mean().unstack()
    pivot_efficiency.plot(kind='bar', ax=ax4, width=0.8)
    ax4.set_title('Path Efficiency by Agent and Maze')
    ax4.set_ylabel('Path Efficiency')
    ax4.legend(title='Maze')
    ax4.tick_params(axis='x', rotation=45)
    
    # Survival rate by agent and swap probability
    ax5 = axes[2, 0]
    pivot_survival = df.groupby(['agent', 'swap_prob'])['survival_rate'].mean().unstack()
    pivot_survival.plot(kind='bar', ax=ax5, width=0.8)
    ax5.set_title('Survival Rate by Agent and Swap Probability')
    ax5.set_ylabel('Survival Rate')
    ax5.legend(title='Swap Probability')
    ax5.tick_params(axis='x', rotation=45)
    
    # Average steps by agent and maze
    ax6 = axes[2, 1]
    pivot_steps = df.groupby(['agent', 'maze'])['avg_steps'].mean().unstack()
    pivot_steps.plot(kind='bar', ax=ax6, width=0.8)
    ax6.set_title('Average Steps by Agent and Maze')
    ax6.set_ylabel('Average Steps')
    ax6.legend(title='Maze')
    ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('comprehensive_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Enhanced heatmap visualizations
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    fig.suptitle('Performance Heatmaps', fontsize=16, fontweight='bold')
    
    # Reward heatmap by agent and reward config
    ax1 = axes[0, 0]
    reward_pivot = df.groupby(['agent', 'reward_config'])['avg_reward'].mean().unstack()
    sns.heatmap(reward_pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax1)
    ax1.set_title('Average Reward Heatmap')
    
    # Risk-adjusted returns heatmap
    ax2 = axes[0, 1]
    risk_pivot = df.groupby(['agent', 'reward_config'])['avg_risk_adjusted_return'].mean().unstack()
    sns.heatmap(risk_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2)
    ax2.set_title('Risk-Adjusted Returns Heatmap')
    
    # Exit rate heatmap
    ax3 = axes[1, 0]
    exit_pivot = df.groupby(['agent', 'swap_prob'])['exit_rate'].mean().unstack()
    sns.heatmap(exit_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax3)
    ax3.set_title('Exit Rate Heatmap')
    
    # Survival rate heatmap
    ax4 = axes[1, 1]
    survival_pivot = df.groupby(['agent', 'swap_prob'])['survival_rate'].mean().unstack()
    sns.heatmap(survival_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax4)
    ax4.set_title('Survival Rate Heatmap')
    
    # Path efficiency heatmap
    ax5 = axes[2, 0]
    efficiency_pivot = df.groupby(['agent', 'maze'])['avg_path_efficiency'].mean().unstack()
    sns.heatmap(efficiency_pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax5)
    ax5.set_title('Path Efficiency Heatmap')
    
    # Steps heatmap
    ax6 = axes[2, 1]
    steps_pivot = df.groupby(['agent', 'maze'])['avg_steps'].mean().unstack()
    sns.heatmap(steps_pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax6)
    ax6.set_title('Average Steps Heatmap')
    
    plt.tight_layout()
    plt.savefig('performance_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Statistical significance analysis
    create_statistical_analysis(df)

def create_statistical_analysis(df):
    """Perform statistical analysis and create significance plots."""
    
    # Compare agents across all conditions
    agents = df['agent'].unique()
    metrics = ['avg_reward', 'exit_rate', 'survival_rate', 'avg_steps', 'avg_rewards_collected', 
               'avg_risk_adjusted_return', 'avg_path_efficiency']
    
    fig, axes = plt.subplots(3, 3, figsize=(24, 20))
    fig.suptitle('Statistical Analysis: Agent Performance Distributions', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        ax = axes[i//3, i%3]
        
        # Create box plots
        data_to_plot = [df[df['agent'] == agent][metric].values for agent in agents]
        bp = ax.boxplot(data_to_plot, labels=agents, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.tick_params(axis='x', rotation=45)
        
        # Add statistical significance markers
        if len(agents) > 1:
            # Perform ANOVA
            groups = [df[df['agent'] == agent][metric].values for agent in agents]
            f_stat, p_value = stats.f_oneway(*groups)
            ax.text(0.02, 0.98, f'ANOVA p={p_value:.4f}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove unused subplots
    for i in range(len(metrics), 9):
        ax = axes[i//3, i%3]
        ax.remove()
    
    plt.tight_layout()
    plt.savefig('statistical_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_detailed_analysis(results):
    """Print comprehensive analysis of results."""
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EXPERIMENTAL ANALYSIS")
    print("="*80)
    
    # Overall agent rankings
    print("\n1. OVERALL AGENT RANKINGS (across all conditions):")
    print("-" * 50)
    
    for metric in ['avg_reward', 'exit_rate', 'survival_rate', 'avg_rewards_collected', 'avg_risk_adjusted_return']:
        print(f"\n{metric.replace('_', ' ').title()}:")
        ranking = df.groupby('agent')[metric].mean().sort_values(ascending=False)
        for i, (agent, value) in enumerate(ranking.items(), 1):
            print(f"  {i}. {agent}: {value:.3f}")
    
    print(f"\nAverage Steps (lower is better):")
    ranking = df.groupby('agent')['avg_steps'].mean().sort_values()
    for i, (agent, value) in enumerate(ranking.items(), 1):
        print(f"  {i}. {agent}: {value:.1f}")
    
    print(f"\nPath Efficiency (higher is better):")
    ranking = df.groupby('agent')['avg_path_efficiency'].mean().sort_values(ascending=False)
    for i, (agent, value) in enumerate(ranking.items(), 1):
        print(f"  {i}. {agent}: {value:.3f}")
    
    # Performance by reward configuration
    print("\n2. PERFORMANCE BY REWARD CONFIGURATION:")
    print("-" * 50)
    
    for reward_config in df['reward_config'].unique():
        print(f"\n{reward_config.upper()} REWARDS:")
        reward_data = df[df['reward_config'] == reward_config]
        
        # Best agent for each metric
        for metric in ['avg_reward', 'exit_rate', 'survival_rate', 'avg_rewards_collected', 'avg_risk_adjusted_return']:
            best_agent = reward_data.loc[reward_data[metric].idxmax(), 'agent']
            best_value = reward_data[metric].max()
            print(f"  Best {metric.replace('_', ' ')}: {best_agent} ({best_value:.3f})")
        
        best_steps_agent = reward_data.loc[reward_data['avg_steps'].idxmin(), 'agent']
        best_steps = reward_data['avg_steps'].min()
        print(f"  Best average steps: {best_steps_agent} ({best_steps:.1f})")
        
        best_efficiency_agent = reward_data.loc[reward_data['avg_path_efficiency'].idxmax(), 'agent']
        best_efficiency = reward_data['avg_path_efficiency'].max()
        print(f"  Best path efficiency: {best_efficiency_agent} ({best_efficiency:.3f})")
    
    # Performance by maze
    print("\n3. PERFORMANCE BY MAZE:")
    print("-" * 50)
    
    for maze in df['maze'].unique():
        print(f"\n{maze.upper()}:")
        maze_data = df[df['maze'] == maze]
        
        # Best agent for each metric
        for metric in ['avg_reward', 'exit_rate', 'survival_rate', 'avg_rewards_collected', 'avg_risk_adjusted_return']:
            best_agent = maze_data.loc[maze_data[metric].idxmax(), 'agent']
            best_value = maze_data[metric].max()
            print(f"  Best {metric.replace('_', ' ')}: {best_agent} ({best_value:.3f})")
        
        best_steps_agent = maze_data.loc[maze_data['avg_steps'].idxmin(), 'agent']
        best_steps = maze_data['avg_steps'].min()
        print(f"  Best average steps: {best_steps_agent} ({best_steps:.1f})")
        
        best_efficiency_agent = maze_data.loc[maze_data['avg_path_efficiency'].idxmax(), 'agent']
        best_efficiency = maze_data['avg_path_efficiency'].max()
        print(f"  Best path efficiency: {best_efficiency_agent} ({best_efficiency:.3f})")
    
    # Performance by swap probability
    print("\n4. PERFORMANCE BY SWAP PROBABILITY:")
    print("-" * 50)
    
    for swap_prob in df['swap_prob'].unique():
        print(f"\nSwap Probability {swap_prob}:")
        swap_data = df[df['swap_prob'] == swap_prob]
        
        for metric in ['avg_reward', 'exit_rate', 'survival_rate', 'avg_rewards_collected', 'avg_risk_adjusted_return']:
            best_agent = swap_data.loc[swap_data[metric].idxmax(), 'agent']
            best_value = swap_data[metric].max()
            print(f"  Best {metric.replace('_', ' ')}: {best_agent} ({best_value:.3f})")
        
        best_steps_agent = swap_data.loc[swap_data['avg_steps'].idxmin(), 'agent']
        best_steps = swap_data['avg_steps'].min()
        print(f"  Best average steps: {best_steps_agent} ({best_steps:.1f})")
        
        best_efficiency_agent = swap_data.loc[swap_data['avg_path_efficiency'].idxmax(), 'agent']
        best_efficiency = swap_data['avg_path_efficiency'].max()
        print(f"  Best path efficiency: {best_efficiency_agent} ({best_efficiency:.3f})")
    
    # Statistical significance
    print("\n5. STATISTICAL SIGNIFICANCE ANALYSIS:")
    print("-" * 50)
    
    agents = df['agent'].unique()
    for metric in ['avg_reward', 'exit_rate', 'survival_rate', 'avg_steps', 'avg_rewards_collected', 
                   'avg_risk_adjusted_return', 'avg_path_efficiency']:
        print(f"\n{metric.replace('_', ' ').title()}:")
        groups = [df[df['agent'] == agent][metric].values for agent in agents]
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"  ANOVA F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("  *** SIGNIFICANT DIFFERENCES DETECTED ***")
        else:
            print("  No significant differences detected")
    
    # Save detailed results to CSV
    df.to_csv('comprehensive_results.csv', index=False)
    print(f"\nDetailed results saved to 'comprehensive_results.csv'")

def main():
    print("Starting comprehensive experiments...")
    print("This will test all agents across 4 mazes, 7 reward configurations, and 5 swap probabilities")
    print("Running 100 episodes per condition for statistical significance...")
    print("Visualization is DISABLED for fast results.")
    print("Enhanced metrics: risk-adjusted returns, survival rate, path efficiency")
    
    # Run experiments without visualization for speed
    results = run_comprehensive_experiments(num_episodes=100, enable_visualization=False)
    
    # Create visualizations
    print("\nCreating comprehensive visualizations...")
    create_comprehensive_plots(results)
    
    # Print analysis
    print_detailed_analysis(results)
    
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETED!")
    print("Generated files:")
    print("- comprehensive_performance.png")
    print("- performance_heatmaps.png") 
    print("- statistical_analysis.png")
    print("- comprehensive_results.csv")
    print("="*80)
    print("\nTo enable visualization later, change enable_visualization=True in run_comprehensive_experiments()")

if __name__ == "__main__":
    main() 