# File: agents/sr_greedy_agent.py

from collections import defaultdict, deque
from typing import Tuple, Dict, List, Any
import random
from agents.base_agent import BaseAgent

class SuccessorRepresentationGreedyAgent(BaseAgent):
    """
    SR-Greedy: Initialize with maze knowledge, then use SR for value estimation.
    Uses maze layout to pre-compute initial SR values.
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha                    # SR learning rate
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
                        # Higher initial values for adjacent cells
                        self.M[pos][(nr, nc)] = 0.8
                        
                # Initialize values for rewards and exit using distance-based decay
                if state.exit == pos:
                    self.M[pos][pos] = 2.0  # Higher value for being at exit
                
                # Initialize reward values based on their magnitude
                for reward_pos, reward_val in state.get_reward_positions().items():
                    if reward_pos == pos:
                        self.M[pos][pos] = 1.0 + reward_val / 5.0  # Scale reward value
                    else:
                        # Add some initial value for paths leading to rewards
                        dist = self._bfs_dist(pos, reward_pos, grid)
                        if dist < float('inf'):
                            self.M[pos][reward_pos] = max(0.1, 1.0 / (dist + 1))
                        
        self.initialized = True

    def select_action(self, state: Any) -> str:
        # Initialize SR values if not done yet
        self._initialize_sr_with_maze(state)
        
        current = state.get_position()
        legal = state.available_actions()
        best_a, best_val = None, -float('inf')

        # Get current rewards and remaining steps
        rewards = state.get_reward_positions()
        steps_remaining = state.max_steps - state.step_count
        
        # For each action, simulate next state and compute V(next)
        for a in legal:
            # Simulate the next position without modifying environment
            nr, nc = self._simulate_move(current, a, state.grid)
            next_pos = (nr, nc)
            
            # Only avoid previous state if we have other options
            if next_pos == self.prev_state and len(legal) > 1:
                continue
            
            # Base value from SR predictions
            total = 0.0
            
            # Value from rewards, weighted by distance and remaining steps
            for rpos, rval in rewards.items():
                dist_to_reward = self._bfs_dist(next_pos, rpos, state.grid)
                if dist_to_reward < float('inf'):
                    can_reach = dist_to_reward < steps_remaining
                    # Higher value for closer rewards we can actually reach
                    sr_weight = self.M[next_pos][rpos]
                    dist_weight = 2.0 / (dist_to_reward + 1) if can_reach else 0.1
                    total += sr_weight * rval * dist_weight
            
            # Add exit value - increases as steps run out or rewards are collected
            dist_to_exit = self._bfs_dist(next_pos, state.exit, state.grid)
            if dist_to_exit < float('inf'):
                exit_urgency = 3.0 if len(rewards) == 0 else 0.5
                if steps_remaining < dist_to_exit + 5:  # Increase exit urgency when close to deadline
                    exit_urgency = 5.0
                total += self.M[next_pos][state.exit] * exit_urgency / (dist_to_exit + 1)
            
            # Add exploration bonus for less visited states
            explore_bonus = 0.5 / (self.visit_counts[next_pos] + 1)
            total += explore_bonus
            
            # Pick the action that maximizes value
            if total > best_val:
                best_val = total
                best_a = a

        # If no good moves found, pick least visited legal move
        if best_a is None:
            best_a = min(legal, key=lambda a: self.visit_counts[self._simulate_move(current, a, state.grid)])
            
        return best_a

    def update(self, state: Any, action: str, reward: float, next_state: Any) -> None:
        """Update M[prev][·] using the SR update whenever we have a valid prev_state."""
        if self.prev_state is not None:
            s = self.prev_state
            s_next = next_state  # next_state is already a position tuple

            # Ensure M[s] and M[s_next] exist
            _ = self.M[s][s]       # ensures M[s][s]=something
            _ = self.M[s_next][s_next]

            # SR update: M[s] ← M[s] + α [1_s + M[s_next] − M[s]]
            for s_prime in set(self.M[s].keys()) | set(self.M[s_next].keys()):
                target = 1.0 if s_prime == s else 0.0
                target += self.M[s_next].get(s_prime, 0.0)
                self.M[s][s_prime] += self.alpha * (target - self.M[s].get(s_prime, 0.0))

        # Move prev pointer forward
        self.prev_state = next_state  # next_state is already a position tuple

        # Track visits if you want to decay alpha over time
        self.visit_counts[next_state] += 1

    def _simulate_move(self, pos: Tuple[int,int], action: str, grid: List[List[int]]) -> Tuple[int,int]:
        """Simulate a move without modifying the environment state."""
        moves = {'up':(-1,0),'down':(1,0),'left':(0,-1),'right':(0,1)}
        dr, dc = moves[action]
        r, c = pos
        nr, nc = r+dr, c+dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == 0:
            return (nr, nc)
        return (r, c)  # if blocked, stay

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