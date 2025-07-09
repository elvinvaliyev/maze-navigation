import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file containing death points
df = pd.read_csv("death_points.csv", header=None,
                 names=["Agent", "Maze", "Episode", "X", "Y", "Reason"])

# Maze dimensions (assumed to be 20x20)
maze_height, maze_width = 20, 20
heatmap = np.zeros((maze_height, maze_width))

# Count the number of deaths at each (x, y) coordinate
for _, row in df.iterrows():
    x, y = int(row["X"]), int(row["Y"])
    heatmap[x, y] += 1

# Visualize the heatmap
plt.figure(figsize=(8, 8))
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.colorbar(label='Death Count')
plt.title('Death Point Heatmap (Maze 5)')
plt.xlabel('Y')
plt.ylabel('X')
plt.gca().invert_yaxis()  # Display from top to bottom like the maze grid
plt.savefig("death_point_heatmap_maze5.png")
plt.show()
