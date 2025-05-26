import random
from collections import deque
from typing import List, Tuple

def shortest_path_length(
    grid: List[List[int]],
    start: Tuple[int,int],
    goal:  Tuple[int,int]
) -> int:
    rows, cols = len(grid), len(grid[0])
    q = deque([(start[0], start[1], 0)])
    seen = {start}
    moves = [(-1,0),(1,0),(0,-1),(0,1)]

    while q:
        r, c, d = q.popleft()
        if (r,c) == goal:
            return d
        for dr, dc in moves:
            nr, nc = r+dr, c+dc
            if (0 <= nr < rows and 0 <= nc < cols and
                grid[nr][nc]==0 and (nr,nc) not in seen):
                seen.add((nr,nc))
                q.append((nr, nc, d+1))
    return float('inf')

def schedule_reward_swaps(
    grid: List[List[int]],
    start: Tuple[int,int],
    reward_positions: List[Tuple[int,int]],
    exit: Tuple[int,int],
    total_budget: int,
    num_swaps: int = 3,
    jitter: int = 2
) -> List[int]:
    """
    Returns a sorted list of swap‐step indices that:
      - start just after the nearer reward’s distance
      - end before the remaining budget to get to exit
      - are evenly spaced with random jitter
    """
    dists = [shortest_path_length(grid, start, pos)
             for pos in reward_positions]
    d_to_exit = shortest_path_length(grid, start, exit)

    earliest = min(dists) + 1
    latest   = total_budget - d_to_exit - 1
    if latest <= earliest:
        return []

    window = latest - earliest
    swaps = []
    for k in range(num_swaps):
        frac = (k+1)/(num_swaps+1)
        base = earliest + frac*window
        t = int(round(base + random.randint(-jitter, jitter)))
        t = max(earliest, min(latest, t))
        swaps.append(t)
    return sorted(swaps)
