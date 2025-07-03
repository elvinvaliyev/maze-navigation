# Maze Navigation Experiment

A comprehensive study of different AI agent strategies for navigating mazes with dynamic reward environments and probabilistic reward swaps.

## Overview

This experiment compares four different AI agents as they learn to navigate mazes with:
- **Dynamic rewards** that can swap positions during episodes
- **Probabilistic swap schedules** based on maze topology
- **Adaptive risk aversion** that adjusts based on survival outcomes
- **Multiple maze configurations** with varying complexity

## Agents

### Model-Based Agents
- **Model-Based Greedy**: Uses Q-learning to maximize immediate rewards
- **Model-Based Survival**: Prioritizes survival and exit over reward collection

### Successor Representation (SR) Agents  
- **SR-Greedy**: Learns successor representations for efficient value estimation
- **SR-Reasonable**: Balances exploration and exploitation with SR learning

## Key Features

### 🎯 Dynamic Environment
- Rewards swap positions during episodes based on fork-based scheduling
- Swap probability and timing learned by agents
- Late exit penalties and proximity bonuses

### 🧠 Adaptive Learning
- All agents adapt their risk tolerance based on recent outcomes
- Swap schedule prediction and risk assessment
- Learning transfer between different maze configurations

### 📊 Comprehensive Analysis
- Statistical analysis of performance metrics
- Comparative analysis between agent strategies
- Interactive visualizations of agent behavior

## Quick Start

### Run Tests
```bash
python test_experiment.py
```

### Run Full Experiment
```bash
python run_all_experiments_and_analysis.py
```

### Interactive Visualization
```bash
python visualizations/interactive_visualization.py
```

## Project Structure

```
maze-navigation/
├── agents/                    # AI agent implementations
├── environments/             # Maze environment and configurations
├── analysis/                 # Statistical and comparative analysis
├── utils/                    # Utility functions (swap scheduling)
├── visualizations/           # Interactive visualization tools
├── test_experiment.py        # Comprehensive test script
└── run_all_experiments_and_analysis.py  # Main experiment runner
```

## Maze Configurations

- **maze1.json**: Simple linear path with two rewards
- **maze2.json**: T-junction with reward trade-off
- **maze3.json**: Complex branching with multiple paths
- **maze4.json**: Loop structure with strategic decisions
- **maze5.json**: Three-route fork with varying rewards
- **maze6.json**: Complex multi-fork structure

## Experiment Design

### Learning Process
1. Agents explore mazes and learn Q-values or successor representations
2. Swap schedules are predicted based on observed patterns
3. Risk aversion adapts based on survival/failure outcomes
4. Knowledge transfers between different maze configurations

### Performance Metrics
- **Success Rate**: Percentage of episodes reaching the exit
- **Average Reward**: Mean total reward per episode
- **Steps to Exit**: Efficiency of pathfinding
- **Swap Detection**: Accuracy of swap schedule prediction
- **Risk Adaptation**: Evolution of risk tolerance

## Key Findings

- **Model-based agents** generally perform better in complex environments
- **SR agents** show efficient learning but struggle with dynamic changes
- **Adaptive risk aversion** improves survival rates across all agents
- **Swap prediction** enhances performance in dynamic environments

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Pandas
- Seaborn

## Usage

1. **Test everything works**: `python test_experiment.py`
2. **Run experiments**: `python run_all_experiments_and_analysis.py`
3. **View results**: Check `comprehensive_results.csv` and generated plots
4. **Interactive exploration**: `python visualizations/interactive_visualization.py`

## Research Applications

This experiment provides insights into:
- **Reinforcement learning** in dynamic environments
- **Risk management** in AI decision-making
- **Transfer learning** between similar tasks
- **Successor representations** vs traditional Q-learning
- **Adaptive behavior** in uncertain environments 