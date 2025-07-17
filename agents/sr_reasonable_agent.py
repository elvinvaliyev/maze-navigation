# File: agents/sr_reasonable_agent.py

from collections import defaultdict, deque
from typing import Tuple, Dict, List, Any
import random
import numpy as np
from agents.base_agent import BaseAgent

class SuccessorRepresentationReasonableAgent(BaseAgent):
    """
    Enhanced SR-Reasonable: Initialize with maze knowledge, then use SR for value estimation.
    Uses maze layout to pre-compute initial SR values.
    Now includes learning tracking and performance monitoring.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 1.0, exploration_rate: float = 0.3, exploration_decay: float = 0.995):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = 0.01
        
        # SR learning
        self.M = defaultdict(lambda: defaultdict(float))  # M[s][s'] = expected future occupancy
        self.R = defaultdict(float)  # Reward vector for each state
        self.visit_counts = defaultdict(int)
        self.prev_state = None
        self.initialized = False
        
        # Performance tracking
        self.episode_rewards = []
        self.performance_history = []
        self.learning_progress = []
        
        # Swap-related
        self.swap_prob = None
        self.prev_reward_positions = None  # For swap detection
        self.swap_detected = False
        self.swap_exploration_boost = 0
        self.swap_exploration_boost_episodes = 5
        # --- Swap schedule learning additions ---
        self.swap_history = []  # List of swap steps in the current episode
        self.learned_swap_counts = defaultdict(int)  # step -> count
        self.total_episodes = 0
        self.risk_threshold = 0.7  # Start cautious
        self.episode_history = []
        self.risk_threshold_history = []  # Track risk threshold changes

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

    def _encode_state(self, state) -> str:
        pos = state.get_position()
        rewards = tuple(sorted(state.get_reward_positions().items()))
        return f"{pos}_{rewards}"

    def observe_swap(self, step):
        self.swap_history.append(step)
        self.learned_swap_counts[step] += 1

    def reset_swap_history(self):
        self.swap_history = []

    def get_learned_swap_prob(self, step):
        # Probability swap occurs at this step (empirical)
        if self.total_episodes == 0:
            return 0.0
        return self.learned_swap_counts[step] / self.total_episodes

    def estimated_swap_likelihood(self, step, dist):
        # Estimate probability of at least one swap in the next 'dist' steps
        if self.total_episodes == 0:
            return 0.0
        prob_no_swap = 1.0
        for s in range(step+1, step+dist+1):
            p = self.get_learned_swap_prob(s)
            prob_no_swap *= (1 - p)
        return 1.0 - prob_no_swap

    def select_action(self, state) -> str:
        if not self.initialized:
            self._initialize_sr_with_maze(state)
        legal_actions = state.available_actions()
        self.last_state = state
        if random.random() < self.exploration_rate:
            return random.choice(legal_actions)
        current_pos = state.get_position()
        grid = state.grid
        # Compute value-to-go for each action using SR and reward vector
        action_scores = {}
        for action in legal_actions:
            next_pos = self._simulate_move(current_pos, action, grid)
            # Value-to-go: sum over s' of M[next_pos][s'] * R[s']
            value = 0.0
            for s_prime in self.M[next_pos]:
                value += self.M[next_pos][s_prime] * self.R[s_prime]
            action_scores[action] = value
        if action_scores:
            best_action = max(action_scores, key=action_scores.get)
            return best_action
        else:
            return random.choice(legal_actions)

    def expected_value_with_swap(self, dist1, dist2, val1, val2, swap_likelihood1, swap_likelihood2):
        # Estimate expected value for each reward branch, considering swap risk
        # If swap is likely before arrival, expected value is lower
        exp_val1 = (1 - swap_likelihood1) * val1 + swap_likelihood1 * val2
        exp_val2 = (1 - swap_likelihood2) * val2 + swap_likelihood2 * val1
        return exp_val1, exp_val2

    def update(self, state: Any, action: str, reward: float, next_state: Any) -> None:
        s = self.prev_state
        if s is not None:
            s_next = next_state.get_position()  # next_state is env
            # Ensure M[s] and M[s_next] exist
            _ = self.M[s][s]
            _ = self.M[s_next][s_next]
            # SR update
            for s_prime in set(self.M[s].keys()) | set(self.M[s_next].keys()):
                target = 1.0 if s_prime == s else 0.0
                target += self.M[s_next].get(s_prime, 0.0)
                self.M[s][s_prime] += self.alpha * (target - self.M[s].get(s_prime, 0.0))

        # Update reward vector for the current state if reward is nonzero
        pos = next_state.get_position()
        if reward != 0:
            self.R[pos] = reward
        self.prev_state = next_state.get_position()  # next_state is env
        self.visit_counts[self.prev_state] += 1
        # --- Swap detection logic ---
        if self.prev_reward_positions is not None:
            if hasattr(self, 'last_state') and self.last_state is not None:
                current_rewards = self.last_state.get_reward_positions()
                if current_rewards != self.prev_reward_positions:
                    self.swap_detected = True
                    self.swap_exploration_boost = self.swap_exploration_boost_episodes
                    # Record swap step
                    step = getattr(state, 'step_count', 0)
                    self.observe_swap(step)
        self.prev_reward_positions = None
        if hasattr(self, 'last_state') and self.last_state is not None:
            self.prev_reward_positions = self.last_state.get_reward_positions()
        # --- End swap detection ---

    def end_episode(self, total_reward: float, success: bool, maze_name: str = "unknown"):
        """Called at the end of each episode for learning tracking."""
        self.episode_rewards.append(total_reward)
        
        # Track performance
        self.performance_history.append({
            'reward': total_reward,
            'success': success,
            'maze': maze_name,
            'episode': len(self.episode_rewards)
        })
        
        # Track learning progress
        if len(self.episode_rewards) >= 10:
            recent_avg = np.mean(self.episode_rewards[-10:])
            self.learning_progress.append(recent_avg)
            if len(self.episode_rewards) % 10 == 0:
                print(f"[SR-Reasonable] Episode {len(self.episode_rewards)} avg reward (last 10): {recent_avg}")
        
        # --- Swap learning: finalize episode ---
        self.total_episodes += 1
        self.reset_swap_history()
        # Decay exploration rate
        self.exploration_rate = max(0.01, self.exploration_rate * self.exploration_decay)
        # --- Swap learning: finalize episode ---
        self.end_episode_swap_tracking()

        self.episode_history.append((success, total_reward))
        if len(self.episode_history) > 20:
            self.episode_history.pop(0)
        deaths = sum(1 for s, _ in self.episode_history if not s)
        if deaths / len(self.episode_history) > 0.2:
            self.risk_threshold += 0.1  # Be more cautious
        elif deaths == 0 and len(self.episode_history) == 20:
            self.risk_threshold -= 0.1  # Be less cautious
        self.risk_threshold = min(max(self.risk_threshold, 0.1), 0.9)
        self.risk_threshold_history.append(self.risk_threshold)

    def get_learning_stats(self) -> Dict:
        """Get current learning statistics."""
        return {
            'episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'exploration_rate': self.exploration_rate,
            'sr_matrix_size': len(self.M),
            'total_visits': sum(self.visit_counts.values()),
            'learning_progress': self.learning_progress[-10:] if self.learning_progress else [],
            'recent_performance': self.episode_rewards[-10:] if self.episode_rewards else []
        }

    def save_learned_knowledge(self, filepath: str):
        """Save learned knowledge to file."""
        import pickle
        knowledge = {
            'M': dict(self.M),
            'performance_history': self.performance_history,
            'episode_rewards': self.episode_rewards,
            'learning_progress': self.learning_progress
        }
        with open(filepath, 'wb') as f:
            pickle.dump(knowledge, f)

    def load_learned_knowledge(self, filepath: str):
        """Load learned knowledge from file."""
        import pickle
        with open(filepath, 'rb') as f:
            knowledge = pickle.load(f)
        
        self.M = defaultdict(lambda: defaultdict(float), knowledge['M'])
        self.performance_history = knowledge['performance_history']
        self.episode_rewards = knowledge['episode_rewards']
        self.learning_progress = knowledge['learning_progress']

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
        d_start_goal = self.bfs_shortest_dist(grid, self.prev_state or pos, goal)
        d_start_pos  = self.bfs_shortest_dist(grid, self.prev_state or pos, pos)
        d_pos_goal   = self.bfs_shortest_dist(grid, pos, goal)
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

    def _delta_to_action(self, curr, nxt):
        dr, dc = (nxt[0] - curr[0], nxt[1] - curr[1])
        if (dr, dc) == (-1, 0): return 'up'
        if (dr, dc) == (1, 0): return 'down'
        if (dr, dc) == (0, -1): return 'left'
        if (dr, dc) == (0, 1): return 'right'
        raise ValueError(f"Unknown delta {dr,dc}") 