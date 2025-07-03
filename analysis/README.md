# Analysis Package for Maze Navigation Experiments

This package provides comprehensive analysis capabilities for maze navigation experiments, including statistical analysis, performance metrics, comparative analysis, and visualization generation.

## Overview

The analysis package consists of four main modules:

1. **StatisticalAnalyzer** - Statistical significance testing and effect size analysis
2. **PerformanceAnalyzer** - Performance metrics and efficiency calculations
3. **ComparativeAnalyzer** - Agent and environment comparisons
4. **VisualizationEngine** - Comprehensive visualization generation

## Modules

### StatisticalAnalyzer

Provides statistical analysis capabilities including:

- **Agent Performance Comparison**: ANOVA and pairwise t-tests
- **Effect Size Analysis**: Cohen's d calculations
- **Confidence Intervals**: 95% confidence intervals for agent performance
- **Comprehensive Reports**: Detailed statistical summaries

**Key Methods:**
- `agent_performance_comparison()` - Compare agents using statistical tests
- `effect_size_analysis()` - Calculate effect sizes for comparisons
- `confidence_intervals()` - Generate confidence intervals
- `generate_comprehensive_report()` - Full statistical analysis

### PerformanceAnalyzer

Analyzes performance metrics and efficiency:

- **Efficiency Metrics**: Reward per step, collection rate, survival efficiency
- **Agent Rankings**: Performance rankings across different metrics
- **Environment Analysis**: Performance across mazes, reward configs, swap probabilities
- **Learning Curve Analysis**: Learning improvement patterns

**Key Methods:**
- `calculate_efficiency_metrics()` - Calculate various efficiency metrics
- `agent_rankings()` - Generate performance rankings
- `environment_analysis()` - Analyze performance across environments
- `learning_curve_analysis()` - Analyze learning patterns

### ComparativeAnalyzer

Provides detailed comparisons between agents and environments:

- **Comparison Matrix**: Comprehensive agent comparison matrix
- **Environment Comparison**: Performance across different environments
- **Agent-Environment Interactions**: How agents perform in different conditions
- **Best/Worst Conditions**: Optimal and challenging conditions for each agent

**Key Methods:**
- `agent_comparison_matrix()` - Create comparison matrix
- `environment_comparison()` - Compare performance across environments
- `agent_environment_interaction()` - Analyze interactions
- `best_performing_conditions()` - Find optimal conditions

### VisualizationEngine

Generates comprehensive visualizations:

- **Agent Performance Plots**: Box plots and comparison charts
- **Environment Analysis**: Performance across different environments
- **Heatmap Visualizations**: Agent-environment interaction heatmaps
- **Learning Curves**: Learning improvement visualizations
- **Statistical Summary Plots**: Confidence intervals and error bars

**Key Methods:**
- `create_agent_performance_plot()` - Agent performance visualizations
- `create_environment_analysis_plots()` - Environment analysis plots
- `create_heatmap_visualizations()` - Heatmap generation
- `create_learning_curves()` - Learning curve plots
- `generate_all_visualizations()` - Generate all visualization types

## Usage

### Basic Usage

```python
from analysis import StatisticalAnalyzer, PerformanceAnalyzer, ComparativeAnalyzer, VisualizationEngine

# Load and analyze results
stat_analyzer = StatisticalAnalyzer()
stat_results = stat_analyzer.generate_comprehensive_report()

perf_analyzer = PerformanceAnalyzer()
perf_results = perf_analyzer.generate_comprehensive_performance_report()

comp_analyzer = ComparativeAnalyzer()
comp_results = comp_analyzer.generate_comprehensive_comparison_report()

viz_engine = VisualizationEngine()
viz_engine.generate_all_visualizations()
```

### Advanced Usage

```python
# Custom statistical analysis
stat_analyzer = StatisticalAnalyzer()
comparisons = stat_analyzer.agent_performance_comparison()
effect_sizes = stat_analyzer.effect_size_analysis()

# Custom performance analysis
perf_analyzer = PerformanceAnalyzer()
efficiency = perf_analyzer.calculate_efficiency_metrics()
rankings = perf_analyzer.agent_rankings()

# Custom comparative analysis
comp_analyzer = ComparativeAnalyzer()
matrix = comp_analyzer.agent_comparison_matrix()
interactions = comp_analyzer.agent_environment_interaction()

# Custom visualizations
viz_engine = VisualizationEngine()
viz_engine.create_agent_performance_plot()
viz_engine.create_heatmap_visualizations()
```

## Output Files

The analysis package generates various output files:

### Statistical Analysis
- Statistical significance reports
- Effect size calculations
- Confidence interval summaries

### Performance Analysis
- Efficiency metrics reports
- Agent ranking summaries
- Environment performance analysis

### Comparative Analysis
- Agent comparison matrices
- Environment comparison reports
- Best/worst condition summaries

### Visualizations
- `agent_performance_comparison.png` - Agent performance box plots
- `maze_performance_analysis.png` - Maze performance analysis
- `reward_config_performance.png` - Reward configuration analysis
- `agent_maze_heatmap.png` - Agent-maze interaction heatmap
- `agent_reward_heatmap.png` - Agent-reward interaction heatmap
- `learning_curves.png` - Learning improvement curves
- `statistical_summary.png` - Statistical summary plots

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scipy

## Notes

- All analysis modules expect results in CSV format from the main experiment runner
- Visualizations are saved as high-resolution PNG files (300 DPI)
- Statistical tests use α = 0.05 significance level by default
- Effect sizes are interpreted using Cohen's d guidelines (small: <0.2, medium: 0.2-0.5, large: >0.5) 