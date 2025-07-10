import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Set results directory
results_dir = os.path.join(os.path.dirname(__file__), '../results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Read the comprehensive results CSV
results_path = os.path.join(results_dir, 'comprehensive_results.csv')
df = pd.read_csv(results_path)

# Filter for a specific maze (e.g., maze5) and only rows with death points
maze_name = 'maze5'
maze_df = df[(df['maze'] == maze_name) & df['death_x'].notnull() & df['death_y'].notnull()]

# Maze config path
maze_config_path = os.path.join(os.path.dirname(__file__), f'../environments/maze_configs/{maze_name}.json')
with open(maze_config_path, 'r') as f:
    maze_config = json.load(f)

maze_layout = np.array(maze_config['grid'])
maze_height, maze_width = maze_layout.shape

# Create heatmap of death points
heatmap = np.zeros((maze_height, maze_width))
for _, row in maze_df.iterrows():
    x, y = int(row['death_x']), int(row['death_y'])
    if 0 <= x < maze_height and 0 <= y < maze_width:
        heatmap[x, y] += 1

# Plot maze background
plt.figure(figsize=(8, 8))
# Show maze: walls as black, paths as white
plt.imshow(maze_layout == 1, cmap='gray', interpolation='nearest', alpha=0.5)
# Overlay heatmap
plt.imshow(heatmap, cmap='hot', interpolation='nearest', alpha=0.7)
plt.colorbar(label='Death Count')
plt.title(f'Death Point Heatmap with Maze Background ({maze_name})')
plt.xlabel('Y')
plt.ylabel('X')
plt.gca().invert_yaxis()  # Display from top to bottom like the maze grid

# Save to results directory
output_path = os.path.join(results_dir, f'death_point_heatmap_{maze_name}.png')
plt.savefig(output_path)
plt.show()
print(f"Saved heatmap to {output_path}")
