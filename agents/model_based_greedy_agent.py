from collections import deque
from typing import Tuple, List, Dict, Optional
from agents.base_agent import BaseAgent

class ModelBasedGreedyAgent(BaseAgent):
    """
    Greedy agent that:
      - Knows the full maze (grid, start/exit)
      - Knows the environment's max_steps budget and swap settings (for info)
      - ALWAYS targets the currently highest-value reward (or exit), ignoring survival.
    """

    def __init__(self, step_budget: Optional[int] = None):
        super().__init__()
        # Optional: store the budget so you can inspect remaining moves if you like
        self.step_budget = step_budget

    def select_action(self, state) -> str:
        # You could inspect remaining steps like this (but we ignore it):
        # remaining = state.max_steps - state.step_count

        # 1) Pick the position with the highest current reward value
        rewards: Dict[Tuple[int,int], int] = state.get_reward_positions()
        if rewards:
            # target the highest-value one
            target = max(rewards.items(), key=lambda kv: kv[1])[0]
        else:
            # no rewards left: head for exit
            target = state.exit

        # 2) Plan a shortest path there via BFS
        path = self._bfs(state.grid, state.get_position(), target)
        if len(path) < 2:
            # stuck or already there: just pick a legal move
            return state.available_actions()[0]

        # 3) Convert the first step of that path into an action
        return self._delta_to_action(state.get_position(), path[1])

    def update(self, state, action, reward, next_state) -> None:
        # Greedy agent doesn't learn
        pass

    def _bfs(
        self,
        grid: List[List[int]],
        start: Tuple[int,int],
        goal: Tuple[int,int]
    ) -> List[Tuple[int,int]]:
        """
        Simple BFS to return the list of cells from start→goal (inclusive).
        """
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
                if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr][nc] == 0
                    and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(path + [(nr, nc)])
        return []

    def _delta_to_action(
        self,
        curr: Tuple[int,int],
        nxt:  Tuple[int,int]
    ) -> str:
        dr, dc = (nxt[0] - curr[0], nxt[1] - curr[1])
        if (dr, dc) == (-1, 0): return 'up'
        if (dr, dc) == (1, 0):  return 'down'
        if (dr, dc) == (0, -1): return 'left'
        if (dr, dc) == (0, 1):  return 'right'
        raise ValueError(f"Unknown delta {dr,dc}")
