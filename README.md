# Maze Navigation Project

This project lets you run and analyze different strategies for solving mazes where rewards can move around. You can run experiments, see results, and explore how different settings affect performance. All results and charts are saved in the `results/` folder.

## What’s Included
- Several types of maze-solving agents (different strategies)
- Mazes with rewards that can swap places
- Experiments that test agents in different mazes and settings
- Automatic analysis and charts showing how well each strategy works
- **Lifelong/transfer learning protocol**: Agents accumulate knowledge across blocks, with transfer tests after each block
- **Classic protocol**: Standard per-condition evaluation
- **Composite improvement metric**: Weighted sum of reward, exit rate, survival, and efficiency improvements
- **Robust error handling** and advanced logging for reproducibility

## How to Use

### 1. Run All Experiments (Lifelong + Classic)
This will run all experiments (using all agent types, all mazes, all reward settings, all step limits, all swap probabilities) and then save results:
```bash
python run_experiments.py
```
- Results and logs will be saved in the `results/` folder.
- The `--seed` flag ensures full reproducibility (recommended).

### 2. Analyze Existing Results Only
If you already have results and just want to generate charts and analysis:
```bash
python run_analysis_from_csv.py
```
- This will use `results/comprehensive_results.csv` and create all charts and reports in `results/`.
- Both classic and lifelong/transfer plots will be generated from a single run.

## Project Structure
```
maze-navigation/
├── agents/                    # Different agent strategies
├── environments/              # Maze layouts and settings
├── analysis/                  # All analysis code
├── utils/                     # Helper code
├── visualizations/            # Interactive visualization tools
├── results/                   # All output CSVs and PNGs
├── run_experiments.py         # Main experiment runner (classic + lifelong)
├── run_analysis_from_csv.py   # Analysis script
└── README.md
```

## What You Get
- **comprehensive_results.csv**: All experiment results (classic and lifelong/transfer)
- **Charts and plots**: PNG files in `results/` showing agent performance, maze difficulty, reward settings, and more
- **Advanced analysis**: Sensitivity, learning progress, behavior, and statistics (all as PNGs in `results/`)
- **Death point heatmaps**: Visualize where agents most often fail in each maze
- **Per-episode timing logs**: For performance analysis

## Requirements
- Python 3.7+
- NumPy
- Matplotlib
- Pandas
- Seaborn
- SciPy
- scikit-learn

## Example Workflow
1. Run all experiments (with reproducibility):
   ```bash
   python run_experiments.py 
   ```
2. Or, just analyze existing results:
   ```bash
   python run_analysis_from_csv.py
   ```
3. Open the `results/` folder to see all your charts and data.

## Metrics and Interpretation
- **Composite improvement**: A weighted sum of reward, exit rate, survival rate, and path efficiency improvements (from start to end of each block). Negative values mean the agent got worse over time, even if its absolute performance is high.
- **Death point heatmaps**: Show where agents most frequently fail in each maze. If an agent's death point is at the start, it is actually dying there (not a logging bug).
- **All plots and tables labeled 'learning improvement' now use the composite metric.**

## Extra
- You can also run `python visualizations/interactive_visualization.py` to explore results interactively.

---
**This project is for anyone interested in comparing different maze-solving strategies and seeing how changes in the environment affect results.** 
