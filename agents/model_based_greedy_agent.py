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
        rewards: Dict[Tuple[int,int], int] = state.get_reward_positions()
        current_pos = state.get_position()
        exit_pos = state.exit
        grid = state.grid
        steps_remaining = state.max_steps - state.step_count

        if not rewards:
            target = exit_pos
        else:
            if len(rewards) == 2 and self.swap_prob is not None:
                reward_items = list(rewards.items())
                (pos1, val1), (pos2, val2) = reward_items
                dist1 = self.bfs_shortest_dist(grid, current_pos, pos1)
                dist2 = self.bfs_shortest_dist(grid, current_pos, pos2)
                exp_val1, exp_val2 = self.expected_value_with_swap(dist1, dist2, val1, val2, self.swap_prob)
                if exp_val1 >= exp_val2:
                    target = pos1
                else:
                    target = pos2
            else:
                target = max(rewards.items(), key=lambda kv: kv[1])[0]

        path = self.bfs_shortest_path(grid, current_pos, target)
        if len(path) < 2:
            return state.available_actions()[0]
        return self._delta_to_action(current_pos, path[1])

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
