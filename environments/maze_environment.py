import random
import json
from collections import deque
from typing import List, Tuple, Dict, Optional

class MazeEnvironment:
    """
    Maze + two static rewards + probabilistic swaps + automatic step budget.

    max_steps = minimal moves to collect the HIGHER‐value reward then reach exit.
    """

    def __init__(
        self,
        grid: List[List[int]],
        start: Tuple[int,int],
        exit:  Tuple[int,int],
        rewards: List[Tuple[Tuple[int,int],int]],
        swap_steps: Optional[List[int]] = None,
        swap_prob: float = 0.0,
    ):
        assert len(rewards) == 2, "Need exactly two rewards."
        self.grid = grid
        self.start = start
        self.exit  = exit
        self.initial_rewards = list(rewards)
        self.swap_steps = swap_steps or []
        self.swap_prob  = swap_prob

        # NEW auto‐budget logic:
        self.max_steps = None  # Will be set manually later
        self.reset()

    @classmethod
    def from_json(cls, path: str) -> "MazeEnvironment":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        grid       = data["grid"]
        start      = tuple(data["start"])
        exit       = tuple(data["exit"])
        rewards    = [(tuple(r[0]), r[1]) for r in data["rewards"]]
        swap_steps = data.get("swap_steps", [])
        swap_prob  = data.get("swap_prob", 0.0)
        return cls(grid, start, exit, rewards, swap_steps, swap_prob)

    def _bfs_dist(self, a: Tuple[int,int], b: Tuple[int,int]) -> int:
        rows, cols = len(self.grid), len(self.grid[0])
        q = deque([(a[0],a[1],0)])
        seen = {a}
        moves = [(-1,0),(1,0),(0,-1),(0,1)]
        while q:
            r,c,d = q.popleft()
            if (r,c) == b:
                return d
            for dr,dc in moves:
                nr,nc = r+dr, c+dc
                if (0<=nr<rows and 0<=nc<cols and
                    self.grid[nr][nc]==0 and (nr,nc) not in seen):
                    seen.add((nr,nc))
                    q.append((nr,nc,d+1))
        return float('inf')

    def reset(self) -> Tuple[int,int]:
        self.agent_pos = self.start
        self.step_count = 0
        self.reward_positions = {pos:val for pos,val in self.initial_rewards}
        self.collected = set()
        # Track previous distance to exit for bonus
        self.prev_dist_to_exit = self._bfs_dist(self.agent_pos, self.exit)
        return self.agent_pos

    def get_position(self) -> Tuple[int,int]:
        return self.agent_pos
    
    def step(self, action: str) -> Tuple[Tuple[int,int],int,bool]:
        if self.max_steps is None:
            raise ValueError("max_steps is not set. Please set it manually after initializing the environment.")
        # 1) if already at or beyond budget → done
        if self.step_count >= self.max_steps:
            return self.agent_pos, 0, True

        # 2) move
        moves = {'up':(-1,0),'down':(1,0),'left':(0,-1),'right':(0,1)}
        dr,dc = moves.get(action,(0,0))
        r,c = self.agent_pos
        nr,nc = r+dr, c+dc
        if (0<=nr<len(self.grid) and 0<=nc<len(self.grid[0])
            and self.grid[nr][nc]==0):
            self.agent_pos = (nr,nc)

        # 3) increment step counter
        self.step_count += 1

        # 4) possibly swap (only if two remain)
        self._maybe_swap()

        # 5) collect reward once (and remove it)
        reward = 0
        if self.agent_pos in self.reward_positions:
            reward = self.reward_positions.pop(self.agent_pos)
            self.collected.add(self.agent_pos)

        # 6) done if exit or budget exhausted
        done = (self.agent_pos == self.exit) or (self.step_count >= self.max_steps)

        # Add exit reward if agent reaches exit successfully
        exit_reward = 100  # Increased reward for survival/exit
        if self.agent_pos == self.exit:
            reward += exit_reward

        # Add late exit penalty if agent reaches exit inefficiently
        late_exit_penalty = -10  # Slightly increased penalty for late exit
        if self.agent_pos == self.exit and self.step_count > int(self.max_steps * 0.9):
            reward += late_exit_penalty

        # Add near-exit bonus if agent is close to exit and close to deadline
        exit_r, exit_c = self.exit
        near_exit_cells = [
            (exit_r-1, exit_c), (exit_r+1, exit_c),
            (exit_r, exit_c-1), (exit_r, exit_c+1)
        ]
        near_exit_bonus = 2
        if self.agent_pos in near_exit_cells and self.step_count > int(self.max_steps * 0.8):
            reward += near_exit_bonus

        # Add bonus for getting closer to exit
        curr_dist_to_exit = self._bfs_dist(self.agent_pos, self.exit)
        if curr_dist_to_exit < self.prev_dist_to_exit:
            reward += 0.5  # Bonus for moving closer
        self.prev_dist_to_exit = curr_dist_to_exit

        # Per-step time penalty
        reward -= 0.1

        return self.agent_pos, reward, done

    def _maybe_swap(self):
        if len(self.reward_positions) != 2:
            return
        if (self.step_count in self.swap_steps
            and random.random() < self.swap_prob
            and self.agent_pos not in self.reward_positions):
            (p1,v1),(p2,v2) = list(self.reward_positions.items())
            self.reward_positions = {p1:v2, p2:v1}

    def available_actions(self) -> List[str]:
        acts = []
        for a,(dr,dc) in {'up':(-1,0),'down':(1,0),'left':(0,-1),'right':(0,1)}.items():
            nr,nc = self.agent_pos[0]+dr, self.agent_pos[1]+dc
            if 0<=nr<len(self.grid) and 0<=nc<len(self.grid[0]) and self.grid[nr][nc]==0:
                acts.append(a)
        return acts

    def get_reward_positions(self) -> Dict[Tuple[int,int],int]:
        return dict(self.reward_positions)
