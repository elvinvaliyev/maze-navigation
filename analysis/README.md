# Maze Navigation Analysis

This folder contains all the code for analyzing your maze experiment results. You can use these tools to create charts, compare strategies, and see how different settings affect performance. All output images and data are saved in the `results/` folder.

## What’s Included
- Tools to check how different settings (like step limit or reward swaps) affect results
- Charts showing how agents learn and improve
- Analysis of agent behavior and decision-making
- Advanced statistics and comparisons

## How to Use

### Run All Analysis
If you already have results (CSV file in `results/`), you can generate all charts and analysis by running:
```bash
python run_analysis_from_csv.py
```
- This will create all PNG charts and reports in the `results/` folder.

## Output Files
All output PNGs and CSVs are saved in `results/`:
- `comprehensive_results.csv`: All experiment results
- `agent_performance_comparison.png`: How each agent did overall
- `maze_performance_analysis.png`: Which mazes were harder or easier
- `reward_config_performance.png`: How reward settings changed results
- `sensitivity_analysis.png`, `agent_sensitivity_analysis.png`: How much each setting matters
- `learning_trajectory_analysis.png`, `agent_learning_analysis.png`: How agents learn over time
- `behavioral_analysis.png`: What strategies agents used
- `enhanced_statistical_analysis.png`: Advanced stats and comparisons
- And more (see the `results/` folder for all files)

## Main Analysis Tools
- **Sensitivity Analysis**: Shows which settings (like step limit or swap chance) have the biggest effect
- **Learning Trajectory**: Shows how agents get better (or not) over time
- **Behavioral Analysis**: Looks at what choices agents make and how they adapt
- **Enhanced Statistics**: Advanced comparisons and groupings

## Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn

## Notes
- All analysis uses the CSV file in `results/` (created by running experiments)
- All charts and reports are saved in `results/` for easy access
- You don’t need to know any statistics or coding to use these tools—just run the script and check the results folder! 