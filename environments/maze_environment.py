import random
import json
from typing import List, Tuple, Dict, Optional, Any

class MazeEnvironment:
    """
    Static maze with two rewards that can swap.
    Load from JSON via from_json().
    """

    def __init__(
        self,
        grid: List[List[int]],
        start: Tuple[int, int],
        exit: Tuple[int, int],
        rewards: List[Tuple[Tuple[int, int], int]],
        swap_steps: Optional[List[int]] = None,
        swap_prob: float = 0.0,
    ):
        assert len(rewards) == 2, "Exactly two rewards required."
        self.grid = grid
        self.start = start
        self.exit = exit
        self.initial_rewards = list(rewards)
        self.swap_steps = swap_steps or []
        self.swap_prob = swap_prob
        self.reset()

    @classmethod
    def from_json(cls, path: str) -> "MazeEnvironment":
        data = json.load(open(path))
        grid       = data["grid"]
        start      = tuple(data["start"])
        exit       = tuple(data["exit"])
        rewards    = [(tuple(r[0]), r[1]) for r in data["rewards"]]
        swap_steps = data.get("swap_steps", [])
        swap_prob  = data.get("swap_prob", 0.0)
        return cls(grid, start, exit, rewards, swap_steps, swap_prob)

    def reset(self) -> Tuple[int, int]:
        self.agent_pos = self.start
        self.step_count = 0
        self.reward_positions: Dict[Tuple[int, int], int] = {
            pos: val for pos, val in self.initial_rewards
        }
        self.collected = set()
        return self.agent_pos

    def get_position(self) -> Tuple[int, int]:
        return self.agent_pos

    def step(self, action: str) -> Tuple[Tuple[int, int], int, bool]:
        moves = {'up':(-1,0),'down':(1,0),'left':(0,-1),'right':(0,1)}
        dr, dc = moves.get(action, (0,0))
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(self.grid) and 0 <= nc < len(self.grid[0]) and self.grid[nr][nc] == 0:
            self.agent_pos = (nr, nc)

        self.step_count += 1
        self._maybe_swap()

        reward = 0
        if self.agent_pos in self.reward_positions and self.agent_pos not in self.collected:
            reward = self.reward_positions[self.agent_pos]
            self.collected.add(self.agent_pos)

        done = (self.agent_pos == self.exit)
        return self.agent_pos, reward, done

    def _maybe_swap(self) -> None:
        if self.step_count in self.swap_steps and random.random() < self.swap_prob:
            if self.agent_pos not in self.reward_positions:
                items = list(self.reward_positions.items())
                (p1, v1), (p2, v2) = items
                self.reward_positions = {p1: v2, p2: v1}

    def available_actions(self) -> List[str]:
        actions = []
        for a, (dr, dc) in {'up':(-1,0),'down':(1,0),'left':(0,-1),'right':(0,1)}.items():
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if (0 <= nr < len(self.grid) and
                0 <= nc < len(self.grid[0]) and
                self.grid[nr][nc] == 0):
                actions.append(a)
        return actions

    def get_reward_positions(self) -> Dict[Tuple[int, int], int]:
        return dict(self.reward_positions)
