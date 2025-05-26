from collections import deque
from typing import Tuple, List, Dict
from agents.base_agent import BaseAgent

class ModelBasedSurvivalAgent(BaseAgent):
    """
    Knows the maze and a fixed step budget.
    Targets the high-value reward only if it can collect it
    _and_ still reach the exit in time; otherwise heads to exit.
    """

    def __init__(self, step_budget: int):
        super().__init__()
        self.step_budget = step_budget

    def select_action(self, state) -> str:
        used = state.step_count
        remaining = self.step_budget - used

        all_rewards = state.get_reward_positions()
        uncol = {p:v for p,v in all_rewards.items()
                 if p not in getattr(state, 'collected', set())}

        def bfs(start, goal):
            rows, cols = len(state.grid), len(state.grid[0])
            moves = [(-1,0),(1,0),(0,-1),(0,1)]
            visited = {start}
            queue = deque([[start]])
            while queue:
                path = queue.popleft()
                if path[-1] == goal:
                    return path
                r, c = path[-1]
                for dr, dc in moves:
                    nr, nc = r+dr, c+dc
                    if (0 <= nr < rows and 0 <= nc < cols and
                        state.grid[nr][nc] == 0 and (nr,nc) not in visited):
                        visited.add((nr,nc))
                        queue.append(path + [(nr,nc)])
            return []

        target = state.exit
        if uncol:
            rp, _ = max(uncol.items(), key=lambda kv: kv[1])
            path1 = bfs(state.get_position(), rp)
            path2 = bfs(rp, state.exit)
            if path1 and path2 and (len(path1)-1 + len(path2)-1) <= remaining:
                target = rp

        path = bfs(state.get_position(), target)
        if len(path) < 2:
            return state.available_actions()[0]
        nxt = path[1]
        dr, dc = nxt[0]-state.get_position()[0], nxt[1]-state.get_position()[1]
        return {(-1,0):'up',(1,0):'down',(0,-1):'left',(0,1):'right'}[(dr,dc)]

    def update(self, state, action, reward, next_state) -> None:
        pass
