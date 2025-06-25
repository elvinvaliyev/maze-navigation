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

        grid = state.grid
        current_pos = state.get_position()
        exit_pos = state.exit

        target = exit_pos
        if uncol:
            if len(uncol) == 2 and self.swap_prob is not None:
                reward_items = list(uncol.items())
                (pos1, val1), (pos2, val2) = reward_items
                dist1 = self.bfs_shortest_dist(grid, current_pos, pos1)
                dist2 = self.bfs_shortest_dist(grid, current_pos, pos2)
                exp_val1, exp_val2 = self.expected_value_with_swap(dist1, dist2, val1, val2, self.swap_prob)
                # Only detour for a reward if can reach it and then exit in time
                path1 = self.bfs_shortest_path(grid, current_pos, pos1)
                path2 = self.bfs_shortest_path(grid, pos1, exit_pos)
                path3 = self.bfs_shortest_path(grid, current_pos, pos2)
                path4 = self.bfs_shortest_path(grid, pos2, exit_pos)
                if path1 and path2 and (len(path1)-1 + len(path2)-1) <= remaining and exp_val1 >= exp_val2:
                    target = pos1
                elif path3 and path4 and (len(path3)-1 + len(path4)-1) <= remaining:
                    target = pos2
            else:
                rp, _ = max(uncol.items(), key=lambda kv: kv[1])
                path1 = self.bfs_shortest_path(grid, current_pos, rp)
                path2 = self.bfs_shortest_path(grid, rp, exit_pos)
                if path1 and path2 and (len(path1)-1 + len(path2)-1) <= remaining:
                    target = rp

        path = self.bfs_shortest_path(grid, current_pos, target)
        if len(path) < 2:
            return state.available_actions()[0]
        nxt = path[1]
        dr, dc = nxt[0]-current_pos[0], nxt[1]-current_pos[1]
        return {(-1,0):'up',(1,0):'down',(0,-1):'left',(0,1):'right'}[(dr,dc)]

    def update(self, state, action, reward, next_state) -> None:
        pass
