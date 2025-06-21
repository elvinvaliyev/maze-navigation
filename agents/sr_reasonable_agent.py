# File: agents/sr_reasonable_agent.py

from collections import defaultdict, deque
from typing import Tuple, Dict, List, Any
import random
from agents.base_agent import BaseAgent

class SuccessorRepresentationReasonableAgent(BaseAgent):
    """
    SR-Reasonable: Initialize with maze knowledge, then combine SR values with
    uncertainty bonus and risk-adjusted rewards. Uses maze layout to pre-compute
    initial SR values and considers swap probabilities.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.M = defaultdict(lambda: defaultdict(float))
        self.visit_counts = defaultdict(int)
        self.prev_state = None
        self.initialized = False

    def _initialize_sr_with_maze(self, state):
        """Pre-compute initial SR values using maze layout."""
        if self.initialized:
            return
            
        grid = state.grid
        rows, cols = len(grid), len(grid[0])
        
        # For each position in the maze
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:  # Wall
                    continue
                    
                pos = (r, c)
                # Initialize with nearby accessible positions
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0):
                        # Initialize with higher values for adjacent accessible cells
                        self.M[pos][(nr, nc)] = 0.5
                        
                # Also initialize some value for reaching rewards and exit
                if state.exit == pos:
                    self.M[pos][pos] = 1.0
                for reward_pos, _ in state.get_reward_positions().items():
                    if reward_pos == pos:
                        self.M[pos][pos] = 1.0
                        
        self.initialized = True

    def select_action(self, state: Any) -> str:
        # Initialize SR values if not done yet
        self._initialize_sr_with_maze(state)
        
        s = state.get_position()
        legal = state.available_actions()

        best_a, best_val = None, -float('inf')
        current_step = state.step_count

        # Pre-fetch reward positions and values
        rewards = state.get_reward_positions()
        sorted_rewards = sorted(rewards.items(), key=lambda kv: kv[1], reverse=True)
        
        for a in legal:
            # Simulate move → next_pos
            nr, nc = self._simulate_move(s, a, state.grid)
            next_pos = (nr, nc)

            # 1) SR-Value contribution: V_sr = sum_{s'} M[next_pos][s'] * R(s')
            V_sr = 0.0
            for (rpos, rval) in rewards.items():
                V_sr += self.M[next_pos].get(rpos, 0.0) * rval

            # Add value for reaching exit
            if len(rewards) == 0:  # If no rewards left, value the exit more
                exit_bonus = 5.0
            else:
                exit_bonus = 0.1
            V_sr += self.M[next_pos][state.exit] * exit_bonus

            # 2) Uncertainty bonus: bonus = beta / sqrt(visit_counts[next_pos] + 1)
            bonus = self.beta / ((self.visit_counts[next_pos] + 1) ** 0.5)

            # 3) Risk-adjusted reward for going to whichever door this path leads toward
            expected_reward = 0.0
            
            # Only consider swap probabilities if we have both rewards
            if len(sorted_rewards) == 2:
                P_swap = self._estimate_swap_probability(
                    state, next_pos, sorted_rewards, current_step
                )
                high_pos, high_val = sorted_rewards[0]
                low_pos, low_val = sorted_rewards[1]

                # If next_pos is on the shortest-path to high_pos:
                if self._is_on_path(next_pos, high_pos, state.grid):
                    expected_reward = (1 - P_swap) * high_val + P_swap * low_val
                # Else if next_pos is on path to low_pos first:
                elif self._is_on_path(next_pos, low_pos, state.grid):
                    expected_reward = (1 - P_swap) * low_val + P_swap * high_val
            # If only one reward left, no need for swap probability
            elif len(sorted_rewards) == 1:
                reward_pos, reward_val = sorted_rewards[0]
                if self._is_on_path(next_pos, reward_pos, state.grid):
                    expected_reward = reward_val
            # If no rewards left, head to exit
            elif len(sorted_rewards) == 0 and self._is_on_path(next_pos, state.exit, state.grid):
                expected_reward = 1.0  # Small bonus for heading to exit

            # Combine all factors
            val = V_sr + bonus + expected_reward
            
            # Add extra value for moves toward exit when close to step limit
            steps_remaining = state.max_steps - current_step
            if steps_remaining < 10:  # If close to step limit
                if self._is_on_path(next_pos, state.exit, state.grid):
                    val += (10 - steps_remaining) * 2  # Increasing urgency to reach exit

            if val > best_val:
                best_val = val
                best_a = a

        # If all actions seem equal, prefer unexplored directions
        if best_val <= 0:
            unexplored = [a for a in legal if self.visit_counts[self._simulate_move(s, a, state.grid)] == 0]
            if unexplored:
                return random.choice(unexplored)

        return best_a if best_a is not None else random.choice(legal)

    def update(self, state: Any, action: str, reward: float, next_state: Any) -> None:
        s = self.prev_state
        if s is not None:
            s_next = next_state  # next_state is already a position tuple
            # Ensure M[s] and M[s_next] exist
            _ = self.M[s][s]
            _ = self.M[s_next][s_next]
            # SR update
            for s_prime in set(self.M[s].keys()) | set(self.M[s_next].keys()):
                target = 1.0 if s_prime == s else 0.0
                target += self.M[s_next].get(s_prime, 0.0)
                self.M[s][s_prime] += self.alpha * (target - self.M[s].get(s_prime, 0.0))

        self.prev_state = next_state
        self.visit_counts[next_state] += 1

    def _simulate_move(self, pos: Tuple[int,int], action: str, grid: List[List[int]]) -> Tuple[int,int]:
        moves = {'up':(-1,0),'down':(1,0),'left':(0,-1),'right':(0,1)}
        dr, dc = moves[action]
        r, c = pos
        nr, nc = r+dr, c+dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == 0:
            return (nr, nc)
        return (r, c)  # if blocked, stay

    def _is_on_path(self, pos: Tuple[int,int], goal: Tuple[int,int], grid: List[List[int]]) -> bool:
        """
        Check if `pos` lies on any shortest path from start to `goal`.
        E.g., by BFS reconstructing a path or checking dist(start,pos)+dist(pos,goal) == dist(start,goal).
        """
        d_start_goal = self._bfs_dist(self.prev_state or pos, goal, grid)   # if prev_state is None, use pos
        d_start_pos  = self._bfs_dist(self.prev_state or pos, pos, grid)
        d_pos_goal   = self._bfs_dist(pos, goal, grid)
        return d_start_pos + d_pos_goal == d_start_goal

    def _estimate_swap_probability(
        self,
        state: Any,
        next_pos: Tuple[int,int],
        sorted_rewards: List[Tuple[Tuple[int,int],int]],
        current_step: int
    ) -> float:
        """
        Estimate P(swap before arriving at whichever reward branch next_pos leads to).
        Uses only the swap probability, not the actual swap schedule.
        """
        # If we don't have exactly 2 rewards, no swaps possible
        if len(sorted_rewards) != 2 or self.swap_prob is None:
            return 0.0

        p = self.swap_prob
        # Decide which reward branch this next_pos leads to:
        high_pos, high_val = sorted_rewards[0]
        low_pos,  low_val  = sorted_rewards[1]
        # Compute distances:
        d_to_high = self._bfs_dist(next_pos, high_pos, state.grid)
        d_to_low  = self._bfs_dist(next_pos, low_pos, state.grid)
        # If high is closer or equal, assume this branch is "toward high":
        if d_to_high <= d_to_low:
            target_dist = d_to_high
        else:
            target_dist = d_to_low

        # Assume a swap could happen at any step along the way
        # Probability that NO swap happens in target_dist steps: (1-p)^target_dist
        # Probability at least one swap: 1 - (1-p)^target_dist
        if target_dist <= 0 or p == 0.0:
            return 0.0
        return 1.0 - (1.0 - p) ** target_dist

    def _bfs_dist(self, a: Tuple[int,int], b: Tuple[int,int], grid: List[List[int]]) -> int:
        """Helper: compute shortest-path distance between two points."""
        rows, cols = len(grid), len(grid[0])
        q = deque([(a[0],a[1],0)])
        seen = {a}
        moves = [(-1,0),(1,0),(0,-1),(0,1)]
        while q:
            r,c,d = q.popleft()
            if (r,c) == b:
                return d
            for dr,dc in moves:
                nr,nc = r+dr, c+dc
                if (0<=nr<rows and 0<=nc<cols and grid[nr][nc]==0 and (nr,nc) not in seen):
                    seen.add((nr,nc))
                    q.append((nr,nc,d+1))
        return float('inf') 