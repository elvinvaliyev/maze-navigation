from collections import deque
from typing import Tuple, List, Dict
from agents.base_agent import BaseAgent

class ModelBasedGreedyAgent(BaseAgent):
    """
    Knows the full maze layout.
    Always greedily heads to the highest-value uncollected reward
    or, if none remain, heads to the exit via shortest path.
    """

    def __init__(self):
        super().__init__()

    def select_action(self, state) -> str:
        all_rewards: Dict[Tuple[int,int], int] = state.get_reward_positions()
        uncol = {
            pos: val for pos, val in all_rewards.items()
            if pos not in getattr(state, 'collected', set())
        }
        if uncol:
            target_pos = max(uncol.items(), key=lambda kv: kv[1])[0]
        else:
            target_pos = state.exit

        path = self._bfs(state.grid, state.get_position(), target_pos)
        if len(path) < 2:
            return state.available_actions()[0]
        next_cell = path[1]
        return self._delta_to_action(state.get_position(), next_cell)

    def update(self, state, action, reward, next_state) -> None:
        pass

    def _bfs(self, grid: List[List[int]], start, goal) -> List[Tuple[int,int]]:
        rows, cols = len(grid), len(grid[0])
        visited = {start}
        queue = deque([[start]])
        moves = [(-1,0),(1,0),(0,-1),(0,1)]

        while queue:
            path = queue.popleft()
            if path[-1] == goal:
                return path
            r, c = path[-1]
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and
                    grid[nr][nc] == 0 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(path + [(nr, nc)])
        return []

    def _delta_to_action(self, curr: Tuple[int,int], nxt: Tuple[int,int]) -> str:
        dr, dc = nxt[0] - curr[0], nxt[1] - curr[1]
        if (dr, dc) == (-1, 0): return 'up'
        if (dr, dc) == (1, 0):  return 'down'
        if (dr, dc) == (0, -1): return 'left'
        if (dr, dc) == (0, 1):  return 'right'
        raise ValueError(f"Cannot map delta {dr,dc} to action.")
