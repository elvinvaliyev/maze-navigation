from abc import ABC, abstractmethod
from typing import Tuple, Any

class BaseAgent(ABC):
    """
    Abstract agent interface for maze navigation.
    """

    def __init__(self):
        self.position: Tuple[int, int] = (0, 0)
        self.total_reward: float = 0.0
        self.swap_prob = None
        self.swap_steps = None
        # Swap learning additions
        self.observed_swap_steps = []  # List of lists: each episode's swap steps
        self.swap_histogram = {}       # step -> count
        self.swap_histogram_total = 0  # total swaps observed

    def set_position(self, pos: Tuple[int, int]) -> None:
        self.position = pos

    def get_position(self) -> Tuple[int, int]:
        return self.position

    def add_reward(self, r: float) -> None:
        self.total_reward += r

    def get_total_reward(self) -> float:
        return self.total_reward

    def inform_swap_info(self, swap_prob):
        self.swap_prob = swap_prob

    @abstractmethod
    def select_action(self, state: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def update(self, state: Any, action: str, reward: float, next_state: Any) -> None:
        raise NotImplementedError

    @staticmethod
    def expected_value_with_swap(dist_to_r1, dist_to_r2, val1, val2, swap_prob):
        """
        Returns the expected value of going for reward 1 and reward 2,
        given the swap probability and distances.
        """
        def estimate_swap_prob(dist, swap_prob):
            if dist <= 0 or not swap_prob:
                return 0.0
            return 1.0 - (1.0 - swap_prob) ** dist

        Pswap1 = estimate_swap_prob(dist_to_r1, swap_prob)
        Pswap2 = estimate_swap_prob(dist_to_r2, swap_prob)
        exp_val1 = (1 - Pswap1) * val1 + Pswap1 * val2
        exp_val2 = (1 - Pswap2) * val2 + Pswap2 * val1
        return exp_val1, exp_val2

    @staticmethod
    def bfs_shortest_path(grid, start, goal):
        """
        Returns the shortest path from start to goal as a list of positions (inclusive), or [] if unreachable.
        """
        from collections import deque
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
                if (0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(path + [(nr, nc)])
        return []

    @staticmethod
    def bfs_shortest_dist(grid, start, goal):
        """
        Returns the shortest path distance from start to goal, or float('inf') if unreachable.
        """
        from collections import deque
        rows, cols = len(grid), len(grid[0])
        q = deque([(start[0], start[1], 0)])
        seen = {start}
        moves = [(-1,0),(1,0),(0,-1),(0,1)]
        while q:
            r, c, d = q.popleft()
            if (r, c) == goal:
                return d
            for dr, dc in moves:
                nr, nc = r+dr, c+dc
                if (0<=nr<rows and 0<=nc<cols and grid[nr][nc]==0 and (nr,nc) not in seen):
                    seen.add((nr,nc))
                    q.append((nr,nc,d+1))
        return float('inf')

    # --- Swap learning additions ---
    def record_swap(self, step: int):
        """Call this when a swap is observed at a given step."""
        if not self.observed_swap_steps or self.observed_swap_steps[-1] is None:
            self.observed_swap_steps.append([])
        self.observed_swap_steps[-1].append(step)
        self.swap_histogram[step] = self.swap_histogram.get(step, 0) + 1
        self.swap_histogram_total += 1

    def start_new_episode(self):
        """Call at the start of each episode to track swaps."""
        self.observed_swap_steps.append([])

    def end_episode_swap_tracking(self):
        """Call at the end of each episode to finalize swap tracking."""
        if self.observed_swap_steps and self.observed_swap_steps[-1] == []:
            self.observed_swap_steps[-1] = None  # Mark as no swaps

    def estimated_swap_likelihood(self, step: int) -> float:
        """Estimate likelihood of swap at a given step based on observed data."""
        if self.swap_histogram_total < 10:
            # Not enough data, fall back to global probability
            return self.swap_prob if self.swap_prob is not None else 0.0
        return self.swap_histogram.get(step, 0) / self.swap_histogram_total
