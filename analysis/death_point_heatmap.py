import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import glob
from matplotlib.colors import ListedColormap, Normalize

# Set results directory
results_dir = os.path.join(os.path.dirname(__file__), '../results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Read the comprehensive results CSV
results_path = os.path.join(results_dir, 'comprehensive_results.csv')
df = pd.read_csv(results_path)

# List of maze configs
maze_config_dir = os.path.join(os.path.dirname(__file__), '../environments/maze_configs')
maze_files = sorted(glob.glob(os.path.join(maze_config_dir, 'maze*.json')))

# Assign unique colormaps for each agent
agent_colormaps = [
    ('Blues', 'SR-Reasonable'),
    ('Purples', 'SR-Greedy'),
    ('Reds', 'Model-Based Survival'),
    ('Greens', 'Model-Based Greedy'),
]

agents = list(df['agent'].unique())

for maze_file in maze_files:
    maze_name = os.path.splitext(os.path.basename(maze_file))[0]
    # Include rows where agent died (death_count > 0) even if death_x/death_y are empty
    maze_df = df[(df['maze'] == maze_name) & (df['death_count'] > 0)]
    if maze_df.empty:
        continue
    with open(maze_file, 'r') as f:
        maze_config = json.load(f)
    maze_layout = np.array(maze_config['grid'])
    maze_height, maze_width = maze_layout.shape

    plt.figure(figsize=(8, 8))
    plt.imshow(maze_layout, cmap=ListedColormap(['white', 'black']), interpolation='nearest', alpha=1.0)

    # --- Unified heatmap (all agents) ---
    agent_labels = []
    sorted_agents = [a for a in agents if a != 'Model-Based Survival'] + [a for a in agents if a == 'Model-Based Survival']
    for idx, agent in enumerate(sorted_agents):
        agent_df = maze_df[maze_df['agent'] == agent]
        if agent_df.empty:
            continue
        heatmap = np.zeros((maze_height, maze_width))
        for _, row in agent_df.iterrows():
            if pd.notna(row['death_x']) and pd.notna(row['death_y']):
                x, y = int(row['death_x']), int(row['death_y'])
                if 0 <= x < maze_height and 0 <= y < maze_width:
                    heatmap[x, y] += 1
        cmap_name, label = agent_colormaps[idx % len(agent_colormaps)]
        cmap = plt.get_cmap(cmap_name)
        norm = Normalize(vmin=0, vmax=np.max(heatmap) if np.max(heatmap) > 0 else 1)
        masked_heatmap = np.where(heatmap == 0, np.nan, heatmap)
        plt.imshow(masked_heatmap, cmap=cmap, interpolation='nearest', alpha=1.0, norm=norm)
        agent_labels.append((label, cmap(0.8)))
    plt.title(f'Death Point Heatmap by Agent with Maze Background ({maze_name})')
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.gca().invert_yaxis()
    for i, (label, color) in enumerate(agent_labels):
        plt.text(maze_width + 1.5, 2 + i * 2, label, fontsize=11, va='center', ha='left')
        plt.scatter([maze_width + 0.7], [2 + i * 2], s=100, color=color, edgecolors='black', zorder=20)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    output_path = os.path.join(results_dir, f'death_point_heatmap_{maze_name}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved heatmap to {output_path}")

    # --- Per-agent subplots ---
    import matplotlib.pyplot as plt2
    fig, axs = plt2.subplots(2, 2, figsize=(12, 12))
    for idx, (agent, (cmap_name, label)) in enumerate(zip(agents, agent_colormaps)):
        ax = axs[idx // 2, idx % 2]
        ax.imshow(maze_layout, cmap=ListedColormap(['white', 'black']), interpolation='nearest', alpha=1.0)
        agent_df = maze_df[maze_df['agent'] == agent]
        heatmap = np.zeros((maze_height, maze_width))
        for _, row in agent_df.iterrows():
            if pd.notna(row['death_x']) and pd.notna(row['death_y']):
                x, y = int(row['death_x']), int(row['death_y'])
                if 0 <= x < maze_height and 0 <= y < maze_width:
                    heatmap[x, y] += 1
        cmap = plt2.get_cmap(cmap_name)
        norm = Normalize(vmin=0, vmax=np.max(heatmap) if np.max(heatmap) > 0 else 1)
        masked_heatmap = np.where(heatmap == 0, np.nan, heatmap)
        ax.imshow(masked_heatmap, cmap=cmap, interpolation='nearest', alpha=1.0, norm=norm)
        ax.set_title(label)
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        ax.invert_yaxis()
    plt2.tight_layout()
    output_path_per_agent = os.path.join(results_dir, f'death_point_heatmap_{maze_name}_per_agent.png')
    plt2.savefig(output_path_per_agent)
    plt2.close()
    print(f"Saved per-agent heatmap to {output_path_per_agent}")
