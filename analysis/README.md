# Maze Navigation Analysis Package

This package provides comprehensive analysis tools for maze navigation experiments, comparing different agent types across various experimental conditions.

## Overview

The analysis package consists of four main modules:

1. **Visualization Engine** (`visualization_engine.py`) - Creates comprehensive visualizations
2. **Statistical Analysis** (`statistical_analysis.py`) - Performs statistical tests and analysis
3. **Performance Metrics** (`performance_metrics.py`) - Calculates advanced performance metrics
4. **Comparative Analysis** (`comparative_analysis.py`) - Compares agents across different dimensions

## Quick Start

To run the complete analysis:

```bash
cd analysis
python run_comprehensive_analysis.py
```

This will generate:
- 8 comprehensive visualization plots
- Detailed statistical analysis
- Performance rankings and metrics
- Agent comparison reports
- A summary report

## Module Details

### 1. Visualization Engine

**File:** `visualization_engine.py`

**Purpose:** Creates comprehensive visualizations of experimental results

**Key Features:**
- Comprehensive dashboard with all metrics
- Swap probability impact analysis
- Reward configuration analysis
- Agent comparison radar charts
- Statistical significance plots
- Maze complexity analysis
- Performance trend analysis

**Usage:**
```python
from visualization_engine import VisualizationEngine

viz_engine = VisualizationEngine()
viz_engine.generate_all_visualizations()
```

**Generated Files:**
- `comprehensive_dashboard.png` - Main dashboard with all metrics
- `swap_probability_analysis.png` - Impact of swap probabilities
- `reward_configuration_analysis.png` - Performance by reward config
- `agent_radar_comparison.png` - Radar chart comparison
- `statistical_significance.png` - Statistical significance heatmaps
- `maze_complexity_analysis.png` - Performance by maze complexity
- `performance_trends.png` - Trend analysis across conditions
- `comparative_analysis.png` - Comparative visualizations

### 2. Statistical Analysis

**File:** `statistical_analysis.py`

**Purpose:** Performs comprehensive statistical analysis

**Key Features:**
- Descriptive statistics for all metrics
- One-way ANOVA analysis
- Post-hoc Tukey HSD tests
- Effect size calculations (Cohen's d)
- Correlation analysis
- Confidence intervals
- Swap probability effect analysis

**Usage:**
```python
from statistical_analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()
results = analyzer.generate_comprehensive_report()
```

**Output:**
- Detailed statistical tables
- Significance testing results
- Effect size interpretations
- Correlation matrices

### 3. Performance Metrics

**File:** `performance_metrics.py`

**Purpose:** Calculates advanced performance metrics and rankings

**Key Features:**
- Composite performance scores
- Performance stability analysis
- Risk-adjusted metrics (Sharpe, Sortino, Calmar ratios)
- Agent specialization analysis
- Efficiency calculations
- Performance rankings
- Trend analysis

**Usage:**
```python
from performance_metrics import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
results = analyzer.generate_comprehensive_performance_report()
```

**Output:**
- Performance rankings across all metrics
- Risk-adjusted performance measures
- Efficiency metrics
- Stability analysis
- Specialization profiles

### 4. Comparative Analysis

**File:** `comparative_analysis.py`

**Purpose:** Compares agents across different dimensions and conditions

**Key Features:**
- Agent-to-agent comparison matrices
- Strengths and weaknesses identification
- Condition-specific performance analysis
- Detailed performance profiles
- Agent recommendation system
- Comparative visualizations

**Usage:**
```python
from comparative_analysis import ComparativeAnalyzer

analyzer = ComparativeAnalyzer()
results = analyzer.generate_comprehensive_comparison_report()
```

**Output:**
- Pairwise agent comparisons
- Agent strengths/weaknesses
- Performance under different conditions
- Usage recommendations

## Data Requirements

The analysis expects a CSV file named `comprehensive_results.csv` in the parent directory with the following columns:

- `agent` - Agent type (Model-Based Greedy, Model-Based Survival, SR-Greedy, SR-Reasonable)
- `maze` - Maze configuration name
- `reward_config` - Reward configuration description
- `swap_prob` - Swap probability value
- `avg_reward` - Average reward collected
- `exit_rate` - Rate of successful exits
- `survival_rate` - Rate of survival
- `avg_risk_adjusted_return` - Risk-adjusted return metric
- `avg_path_efficiency` - Path efficiency metric
- `avg_steps` - Average number of steps taken

## Output Structure

```
analysis/
├── __init__.py
├── README.md
├── run_comprehensive_analysis.py
├── visualization_engine.py
├── statistical_analysis.py
├── performance_metrics.py
├── comparative_analysis.py
├── comprehensive_dashboard.png
├── swap_probability_analysis.png
├── reward_configuration_analysis.png
├── agent_radar_comparison.png
├── statistical_significance.png
├── maze_complexity_analysis.png
├── performance_trends.png
├── comparative_analysis.png
├── reports/
│   └── comprehensive_summary_report.txt
├── visualizations/
└── tables/
```

## Key Metrics Analyzed

1. **Average Reward** - Total reward collected per episode
2. **Exit Rate** - Percentage of episodes ending in successful exit
3. **Survival Rate** - Percentage of episodes ending in survival (not death)
4. **Risk-Adjusted Return** - Reward adjusted for risk and uncertainty
5. **Path Efficiency** - Efficiency of navigation path taken

## Statistical Tests Performed

- **ANOVA** - Tests for significant differences between agents
- **Tukey HSD** - Post-hoc pairwise comparisons
- **Effect Sizes** - Cohen's d for practical significance
- **Correlation Analysis** - Relationships between metrics
- **Confidence Intervals** - Uncertainty quantification

## Performance Metrics Calculated

- **Composite Scores** - Weighted combination of all metrics
- **Risk-Adjusted Ratios** - Sharpe, Sortino, Calmar ratios
- **Efficiency Metrics** - Performance per step
- **Stability Measures** - Coefficient of variation, IQR
- **Specialization Profiles** - Agent strengths and weaknesses

## Agent Types Analyzed

1. **Model-Based Greedy** - Greedy agent with model-based planning
2. **Model-Based Survival** - Survival-focused agent with model-based planning
3. **SR-Greedy** - Greedy agent using successor representations
4. **SR-Reasonable** - Balanced agent using successor representations

## Experimental Conditions

- **Maze Configurations** - Different maze layouts and complexities
- **Reward Configurations** - Various reward structures (big difference, small difference, equal)
- **Swap Probabilities** - Different probabilities of reward swapping
- **Episode Counts** - Multiple episodes for statistical power

## Usage Examples

### Run Complete Analysis
```bash
python run_comprehensive_analysis.py
```

### Run Individual Modules
```python
# Just visualizations
from visualization_engine import VisualizationEngine
viz_engine = VisualizationEngine()
viz_engine.create_comprehensive_dashboard()

# Just statistical analysis
from statistical_analysis import StatisticalAnalyzer
analyzer = StatisticalAnalyzer()
analyzer.perform_anova_analysis()

# Just performance metrics
from performance_metrics import PerformanceAnalyzer
analyzer = PerformanceAnalyzer()
analyzer.calculate_composite_score()

# Just comparative analysis
from comparative_analysis import ComparativeAnalyzer
analyzer = ComparativeAnalyzer()
analyzer.analyze_agent_strengths_weaknesses()
```

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scipy

## Notes

- All visualizations are saved as high-resolution PNG files (300 DPI)
- Statistical analysis includes significance levels (*** p<0.001, ** p<0.01, * p<0.05)
- Performance metrics are normalized for fair comparison
- Analysis handles missing data gracefully
- Results are both printed to console and saved to files

## Troubleshooting

1. **Import Errors**: Ensure you're running from the correct directory
2. **File Not Found**: Check that `comprehensive_results.csv` exists in parent directory
3. **Memory Issues**: For large datasets, consider running modules individually
4. **Visualization Errors**: Ensure matplotlib backend is properly configured

## Contributing

To add new analysis features:
1. Create new module in the analysis package
2. Update `__init__.py` to include new module
3. Add new functionality to `run_comprehensive_analysis.py`
4. Update this README with new features 