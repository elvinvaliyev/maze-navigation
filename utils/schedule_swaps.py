"""
Swap scheduling utilities for maze navigation experiments.

This module provides functions to schedule reward swaps at strategic
points in the maze, such as decision forks.
"""

from typing import List, Tuple
from collections import deque

def schedule_fork_swap(
    grid: List[List[int]], 
    start: Tuple[int, int], 
    reward_positions: List[Tuple[int, int]], 
    jitter: int = 1
) -> List[int]:
    """
    Schedule a swap at the decision fork between rewards.
    
    Args:
        grid: The maze grid
        start: Starting position
        reward_positions: List of reward positions
        jitter: Random jitter to add to the swap step
    
    Returns:
        List of steps when swaps should occur
    """
    if len(reward_positions) < 2:
        return []
    
    # Find the decision fork (point where paths to rewards diverge)
    fork_point = find_decision_fork(grid, start, reward_positions)
    
    if fork_point is None:
        return []
    
    # Calculate distance to fork
    dist_to_fork = bfs_shortest_dist(grid, start, fork_point)
    
    # Schedule swap at fork with some jitter
    import random
    swap_step = dist_to_fork + random.randint(-jitter, jitter)
    swap_step = max(1, swap_step)  # Ensure positive step
    
    return [swap_step]

def find_decision_fork(
    grid: List[List[int]], 
    start: Tuple[int, int], 
    reward_positions: List[Tuple[int, int]]
) -> Tuple[int, int]:
    """
    Find the decision fork where paths to different rewards diverge.
    
    Args:
        grid: The maze grid
        start: Starting position
        reward_positions: List of reward positions
    
    Returns:
        Position of the decision fork, or None if not found
    """
    if len(reward_positions) < 2:
        return None
    
    # Find shortest paths to each reward
    paths = []
    for reward_pos in reward_positions:
        path = bfs_shortest_path(grid, start, reward_pos)
        if path:
            paths.append(path)
    
    if len(paths) < 2:
        return None
    
    # Find the last common point in the paths
    common_points = set(paths[0])
    for path in paths[1:]:
        common_points = common_points.intersection(set(path))
    
    if not common_points:
        return None
    
    # Return the point furthest from start (closest to rewards)
    return max(common_points, key=lambda p: len(bfs_shortest_path(grid, start, p)))

def bfs_shortest_path(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Find shortest path using BFS.
    
    Args:
        grid: The maze grid
        start: Starting position
        goal: Goal position
    
    Returns:
        List of positions forming the shortest path
    """
    rows, cols = len(grid), len(grid[0])
    visited = {start}
    queue = deque([[start]])
    
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        path = queue.popleft()
        current = path[-1]
        
        if current == goal:
            return path
        
        for dr, dc in moves:
            nr, nc = current[0] + dr, current[1] + dc
            
            if (0 <= nr < rows and 0 <= nc < cols and 
                grid[nr][nc] == 0 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(path + [(nr, nc)])
    
    return []

def bfs_shortest_dist(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> int:
    """
    Find shortest distance using BFS.
    
    Args:
        grid: The maze grid
        start: Starting position
        goal: Goal position
    
    Returns:
        Shortest distance to goal
    """
    path = bfs_shortest_path(grid, start, goal)
    return len(path) - 1 if path else float('inf')

def schedule_multiple_swaps(
    grid: List[List[int]], 
    start: Tuple[int, int], 
    reward_positions: List[Tuple[int, int]], 
    num_swaps: int = 2
) -> List[int]:
    """
    Schedule multiple swaps throughout the maze.
    
    Args:
        grid: The maze grid
        start: Starting position
        reward_positions: List of reward positions
        num_swaps: Number of swaps to schedule
    
    Returns:
        List of steps when swaps should occur
    """
    if len(reward_positions) < 2:
        return []
    
    # Find the longest path to any reward
    max_dist = 0
    for reward_pos in reward_positions:
        dist = bfs_shortest_dist(grid, start, reward_pos)
        if dist != float('inf'):
            max_dist = max(max_dist, dist)
    
    if max_dist == 0:
        return []
    
    # Schedule swaps at regular intervals
    import random
    swap_steps = []
    for i in range(num_swaps):
        step = int(max_dist * (i + 1) / (num_swaps + 1))
        step += random.randint(-1, 1)  # Add jitter
        step = max(1, step)
        swap_steps.append(step)
    
    return sorted(swap_steps)

def schedule_adaptive_swaps(
    grid: List[List[int]], 
    start: Tuple[int, int], 
    reward_positions: List[Tuple[int, int]], 
    difficulty: str = "medium"
) -> List[int]:
    """
    Schedule swaps adaptively based on maze difficulty.
    
    Args:
        grid: The maze grid
        start: Starting position
        reward_positions: List of reward positions
        difficulty: Difficulty level ("easy", "medium", "hard")
    
    Returns:
        List of steps when swaps should occur
    """
    if len(reward_positions) < 2:
        return []
    
    # Find decision fork
    fork_point = find_decision_fork(grid, start, reward_positions)
    
    if fork_point is None:
        return []
    
    dist_to_fork = bfs_shortest_dist(grid, start, fork_point)
    
    # Adjust based on difficulty
    if difficulty == "easy":
        # Early swap, easy to adapt
        swap_step = max(1, dist_to_fork - 2)
    elif difficulty == "medium":
        # At the fork
        swap_step = dist_to_fork
    else:  # hard
        # Late swap, harder to adapt
        swap_step = dist_to_fork + 3
    
    return [swap_step] 