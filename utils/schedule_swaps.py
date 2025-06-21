import random
from collections import deque
from typing import List, Tuple

def shortest_path(
    grid: List[List[int]],
    start: Tuple[int,int],
    goal:  Tuple[int,int]
) -> List[Tuple[int,int]]:
    """
    Return one shortest path (as list of cells) from start to goal using BFS.
    """
    rows, cols = len(grid), len(grid[0])
    q = deque([[start]])
    seen = {start}
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    while q:
        path = q.popleft()
        r, c = path[-1]
        if (r, c) == goal:
            return path
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                and grid[nr][nc] == 0
                and (nr, nc) not in seen):
                seen.add((nr, nc))
                q.append(path + [(nr, nc)])
    return []

def find_all_decision_points(
    grid: List[List[int]],
    start: Tuple[int,int],
    reward_positions: List[Tuple[int,int]]
) -> List[Tuple[int,int]]:
    """Find all points where paths to different rewards diverge."""
    if len(reward_positions) != 2:
        return []
        
    # Get paths to both rewards
    path1 = shortest_path(grid, start, reward_positions[0])
    path2 = shortest_path(grid, start, reward_positions[1])
    
    # Find all points where paths diverge
    decision_points = []
    common_prefix_len = 0
    for p1, p2 in zip(path1, path2):
        if p1 == p2:
            common_prefix_len += 1
        else:
            break
            
    if common_prefix_len > 0:
        # Add the last common point
        decision_points.append(path1[common_prefix_len - 1])
        
    # Also check for decision points from each reward to the other
    path_between = shortest_path(grid, reward_positions[0], reward_positions[1])
    if len(path_between) > 2:  # If rewards aren't adjacent
        mid_point = path_between[len(path_between) // 2]
        if mid_point not in decision_points:
            decision_points.append(mid_point)
            
    return decision_points

def compute_optimal_swap_steps(
    grid: List[List[int]],
    start: Tuple[int,int],
    reward_positions: List[Tuple[int,int]],
    max_steps: int
) -> List[int]:
    """
    Schedule swaps at strategic decision points, considering:
    1. Points where paths to different rewards diverge
    2. Midpoints between rewards
    3. Time needed to reach these points
    """
    decision_points = find_all_decision_points(grid, start, reward_positions)
    if not decision_points:
        return []
        
    swap_steps = []
    for point in decision_points:
        # Calculate steps needed to reach this point
        path_to_point = shortest_path(grid, start, point)
        steps_to_point = len(path_to_point) - 1  # -1 because path includes start
        
        # Add some randomness to make it less predictable
        jitter = random.randint(-2, 2)
        swap_step = max(1, min(steps_to_point + jitter, max_steps - 1))
        
        if swap_step not in swap_steps:
            swap_steps.append(swap_step)
            
    return sorted(swap_steps)

def schedule_fork_swap(
    grid: List[List[int]],
    start: Tuple[int,int],
    reward_positions: List[Tuple[int,int]],
    jitter: int = 1
) -> List[int]:
    """
    Enhanced swap scheduling that works well for both simple and complex mazes.
    """
    # First try to find the optimal swap steps
    swap_steps = compute_optimal_swap_steps(grid, start, reward_positions, 100)  # Use large max_steps initially
    
    if not swap_steps:
        # Fallback to original method for simpler mazes
        if len(reward_positions) != 2:
            return []
            
        # Find last common node in paths to rewards
        p1 = shortest_path(grid, start, reward_positions[0])
        p2 = shortest_path(grid, start, reward_positions[1])
        
        lcn = start
        for u, v in zip(p1, p2):
            if u == v:
                lcn = u
            else:
                break
                
        # Distance to LCN
        try:
            d_fork = p1.index(lcn)
        except ValueError:
            return []
            
        # Compute swap step
        base = d_fork + 1
        # Apply jitter
        t = base + random.randint(-jitter, jitter)
        # Ensure at least 1
        t = max(1, t)
        return [t]
        
    return swap_steps
