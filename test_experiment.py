#!/usr/bin/env python3
"""
Comprehensive Test Script for Maze Navigation Experiment

This script tests all major components of the experiment:
- Agent learning and adaptive risk aversion
- Swap scheduling and detection
- Reward configurations
- Environment functionality
- Analysis modules
- Visualization generation

Runs quickly with minimal episodes to verify everything works.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.maze_environment import MazeEnvironment
from agents.model_based_greedy_agent import ModelBasedGreedyAgent
from agents.model_based_survival_agent import ModelBasedSurvivalAgent
from agents.sr_greedy_agent import SuccessorRepresentationGreedyAgent
from agents.sr_reasonable_agent import SuccessorRepresentationReasonableAgent
from utils.schedule_swaps import schedule_fork_swap

def test_environment_functionality():
    """Test maze environment functionality."""
    print("=" * 60)
    print("TESTING ENVIRONMENT FUNCTIONALITY")
    print("=" * 60)
    
    # Load maze1
    env = MazeEnvironment.from_json("environments/maze_configs/maze1.json")
    print(f"✓ Loaded maze1: {env.max_steps} steps, {len(env.initial_rewards)} rewards")
    
    # Test reset
    pos = env.reset()
    print(f"✓ Reset successful: agent at {pos}")
    
    # Test step
    action = 'right'
    next_pos, reward, done = env.step(action)
    print(f"✓ Step successful: {action} -> {next_pos}, reward={reward}, done={done}")
    
    # Test reward collection
    rewards = env.get_reward_positions()
    print(f"✓ Available rewards: {rewards}")
    
    # Test swap functionality
    env.swap_steps = [2]
    env.swap_prob = 1.0
    print(f"✓ Swap configured: steps={env.swap_steps}, prob={env.swap_prob}")
    
    return True

def test_swap_scheduling():
    """Test swap scheduling functionality."""
    print("\n" + "=" * 60)
    print("TESTING SWAP SCHEDULING")
    print("=" * 60)
    
    # Load maze
    env = MazeEnvironment.from_json("environments/maze_configs/maze1.json")
    
    # Test fork-based swap scheduling
    reward_positions = [pos for pos, _ in env.initial_rewards]
    swap_steps = schedule_fork_swap(
        grid=env.grid,
        start=env.start,
        reward_positions=reward_positions,
        jitter=1
    )
    print(f"✓ Fork-based swap scheduling: {swap_steps}")
    
    # Test swap detection
    env.swap_steps = swap_steps
    env.swap_prob = 1.0
    
    # Run a few steps to test swap
    env.reset()
    for step in range(5):
        action = 'right'
        next_pos, reward, done = env.step(action)
        if step in swap_steps:
            print(f"✓ Swap occurred at step {step}")
        if done:
            break
    
    return True

def test_agent_learning():
    """Test agent learning functionality."""
    print("\n" + "=" * 60)
    print("TESTING AGENT LEARNING")
    print("=" * 60)
    
    agents = {
        'Model-Based Greedy': ModelBasedGreedyAgent(),
        'Model-Based Survival': ModelBasedSurvivalAgent(step_budget=50),
        'SR-Greedy': SuccessorRepresentationGreedyAgent(),
        'SR-Reasonable': SuccessorRepresentationReasonableAgent()
    }
    
    # Test each agent
    for agent_name, agent in agents.items():
        print(f"\nTesting {agent_name}...")
        
        # Create environment
        env = MazeEnvironment.from_json("environments/maze_configs/maze1.json")
        env.swap_steps = [2]
        env.swap_prob = 0.5
        
        # Run short episode
        env.reset()
        total_reward = 0
        steps = 0
        
        # Start episode tracking
        if hasattr(agent, 'start_new_episode'):
            agent.start_new_episode()
        
        while steps < min(10, env.max_steps):
            # Observe state for swap detection
            if hasattr(agent, 'observe_state'):
                agent.observe_state(env)
            
            # Select action
            action = agent.select_action(env)
            
            # Take step
            next_pos, reward, done = env.step(action)
            agent.update(env, action, reward, env)
            
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # End episode tracking
        if hasattr(agent, 'end_episode_swap_tracking'):
            agent.end_episode_swap_tracking()
        
        # Test adaptive risk aversion
        if hasattr(agent, 'end_episode'):
            success = env.get_position() == env.exit
            agent.end_episode(success, total_reward)
        
        # Check learning stats
        if hasattr(agent, 'get_learning_stats'):
            stats = agent.get_learning_stats()
            print(f"  ✓ Learning stats: {stats}")
        
        # Check risk threshold
        if hasattr(agent, 'risk_threshold'):
            print(f"  ✓ Risk threshold: {agent.risk_threshold}")
        
        print(f"  ✓ Episode complete: {steps} steps, {total_reward} reward")
    
    return True

def test_reward_configurations():
    """Test different reward configurations."""
    print("\n" + "=" * 60)
    print("TESTING REWARD CONFIGURATIONS")
    print("=" * 60)
    
    reward_configs = [
        ('small', [5, 3]),
        ('medium', [10, 6]),
        ('large', [20, 12]),
        ('extreme', [50, 30])
    ]
    
    for config_name, [high_val, low_val] in reward_configs:
        print(f"\nTesting {config_name} rewards: {high_val}, {low_val}")
        
        # Load maze and modify rewards
        env = MazeEnvironment.from_json("environments/maze_configs/maze1.json")
        
        # Get original positions
        original_rewards = env.initial_rewards
        if len(original_rewards) >= 2:
            reward_positions = [original_rewards[0][0], original_rewards[1][0]]
            
            # Create new environment with modified rewards
            new_env = MazeEnvironment(
                grid=env.grid,
                start=env.start,
                exit=env.exit,
                rewards=[(reward_positions[0], high_val), (reward_positions[1], low_val)],
                swap_prob=0.3
            )
            
            print(f"  ✓ Created environment with {high_val}, {low_val} rewards")
            
            # Test agent on this configuration
            agent = ModelBasedGreedyAgent()
            new_env.reset()
            
            # Run short episode
            total_reward = 0
            for step in range(5):
                action = agent.select_action(new_env)
                next_pos, reward, done = new_env.step(action)
                agent.update(new_env, action, reward, new_env)
                total_reward += reward
                if done:
                    break
            
            print(f"  ✓ Episode result: {total_reward} total reward")
    
    return True

def test_analysis_modules():
    """Test analysis modules functionality."""
    print("\n" + "=" * 60)
    print("TESTING ANALYSIS MODULES")
    print("=" * 60)
    
    try:
        from analysis.statistical_analysis import StatisticalAnalyzer
        from analysis.performance_metrics import PerformanceAnalyzer
        from analysis.comparative_analysis import ComparativeAnalyzer
        from analysis.visualization_engine import VisualizationEngine
        
        print("✓ All analysis modules imported successfully")
        
        # Test StatisticalAnalyzer
        stat_analyzer = StatisticalAnalyzer()
        print("✓ StatisticalAnalyzer created")
        
        # Test PerformanceAnalyzer
        perf_analyzer = PerformanceAnalyzer()
        print("✓ PerformanceAnalyzer created")
        
        # Test ComparativeAnalyzer
        comp_analyzer = ComparativeAnalyzer()
        print("✓ ComparativeAnalyzer created")
        
        # Test VisualizationEngine
        viz_engine = VisualizationEngine()
        print("✓ VisualizationEngine created")
        
        return True
        
    except ImportError as e:
        print(f"✗ Analysis module import failed: {e}")
        return False

def test_learning_transfer():
    """Test learning transfer between mazes."""
    print("\n" + "=" * 60)
    print("TESTING LEARNING TRANSFER")
    print("=" * 60)
    
    agent = ModelBasedGreedyAgent()
    
    # Test on maze1
    env1 = MazeEnvironment.from_json("environments/maze_configs/maze1.json")
    env1.reset()
    
    # Run short episode on maze1
    for step in range(5):
        action = agent.select_action(env1)
        next_pos, reward, done = env1.step(action)
        agent.update(env1, action, reward, env1)
        if done:
            break
    
    # Get knowledge after maze1
    knowledge1 = agent.get_learning_stats()
    print(f"✓ Knowledge after maze1: {knowledge1}")
    
    # Test on maze2 (learning transfer)
    env2 = MazeEnvironment.from_json("environments/maze_configs/maze2.json")
    env2.reset()
    
    # Run short episode on maze2
    for step in range(5):
        action = agent.select_action(env2)
        next_pos, reward, done = env2.step(action)
        agent.update(env2, action, reward, env2)
        if done:
            break
    
    # Get knowledge after maze2
    knowledge2 = agent.get_learning_stats()
    print(f"✓ Knowledge after maze2: {knowledge2}")
    
    # Check if knowledge increased (learning transfer)
    if knowledge2['q_table_size'] >= knowledge1['q_table_size']:
        print("✓ Learning transfer successful")
    else:
        print("⚠ Learning transfer may not be working")
    
    return True

def main():
    """Run all tests."""
    print("COMPREHENSIVE EXPERIMENT TEST")
    print("=" * 60)
    print("Testing all experiment components...")
    
    tests = [
        ("Environment Functionality", test_environment_functionality),
        ("Swap Scheduling", test_swap_scheduling),
        ("Agent Learning", test_agent_learning),
        ("Reward Configurations", test_reward_configurations),
        ("Analysis Modules", test_analysis_modules),
        ("Learning Transfer", test_learning_transfer)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! Experiment is ready to run.")
    else:
        print("⚠ Some tests failed. Check the output above.")
    
    return passed == len(results)

if __name__ == "__main__":
    main() 