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

def schedule_fork_swap(
    grid: List[List[int]],
    start: Tuple[int,int],
    reward_positions: List[Tuple[int,int]],
    jitter: int = 1
) -> List[int]:
    """
    Schedule exactly one swap event at the decision fork:
      1) Compute shortest paths P1, P2 from start→each reward.
      2) Find last common node (LCN) in P1 & P2.
      3) Swap step = index_of_LCN + 1 ± jitter.
    Returns a single-element list [t] or [] if no fork found.
    """
    if len(reward_positions) != 2:
        return []
    # 1) get the two paths
    p1 = shortest_path(grid, start, reward_positions[0])
    p2 = shortest_path(grid, start, reward_positions[1])
    # 2) find LCN
    lcn = start
    for u, v in zip(p1, p2):
        if u == v:
            lcn = u
        else:
            break
    # distance to LCN
    try:
        d_fork = p1.index(lcn)
    except ValueError:
        return []
    # 3) compute swap step
    base = d_fork + 1
    # apply jitter
    t = base + random.randint(-jitter, jitter)
    # ensure at least 1
    t = max(1, t)
    return [t]
