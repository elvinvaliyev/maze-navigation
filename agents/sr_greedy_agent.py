# File: agents/sr_greedy_agent.py

from collections import defaultdict, deque
from typing import Tuple, Dict, List, Any
import random
import numpy as np
from agents.base_agent import BaseAgent

class SuccessorRepresentationGreedyAgent(BaseAgent):
    """
    Enhanced SR-Greedy: Initialize with maze knowledge, then use SR for value estimation.
    Uses maze layout to pre-compute initial SR values.
    Now includes learning tracking and performance monitoring.
    """

    def __init__(self, alpha: float = 0.5, exploration_rate: float = 0.3, exploration_decay: float = 0.995):
        super().__init__()
        self.alpha = alpha                    # SR learning rate
        self.exploration_rate = exploration_rate  # Exploration rate
        self.exploration_decay = exploration_decay
        self.min_exploration = 0.01
        
        # SR learning
        self.M = defaultdict(lambda: defaultdict(float))  # M[s][s'] = expected future occupancy
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
        self.risk_threshold = 0.3  # Start moderately greedy
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
        encoded_state = self._encode_state(state)
        legal_actions = state.available_actions()
        if random.random() < self.exploration_rate:
            return random.choice(legal_actions)
        # Model-based plan
        rewards = state.get_reward_positions()
        current_pos = state.get_position()
        exit_pos = state.exit
        grid = state.grid
        if not rewards:
            target = exit_pos
        else:
            if len(rewards) == 2:
                reward_items = list(rewards.items())
                (pos1, val1), (pos2, val2) = reward_items
                dist1 = self.bfs_shortest_dist(grid, current_pos, pos1)
                dist2 = self.bfs_shortest_dist(grid, current_pos, pos2)
                step = getattr(state, 'step_count', 0)
                # Use learned swap schedule
                swap_likelihood1 = self.estimated_swap_likelihood(step, dist1)
                swap_likelihood2 = self.estimated_swap_likelihood(step, dist2)
                exp_val1, exp_val2 = self.expected_value_with_swap(dist1, dist2, val1, val2, swap_likelihood1, swap_likelihood2)
                if exp_val1 >= exp_val2:
                    target = pos1
                else:
                    target = pos2
            else:
                target = max(rewards.items(), key=lambda kv: kv[1])[0]
        path = self.bfs_shortest_path(grid, current_pos, target)
        if len(path) < 2:
            return legal_actions[0]
        optimal_action = self._delta_to_action(current_pos, path[1])
        # Weighted sum
        weights = 0.5
        action_scores = {}
        for action in legal_actions:
            sr_val = self.M[current_pos][self._simulate_move(current_pos, action, grid)]
            model_val = 1.0 if action == optimal_action else 0.0
            action_scores[action] = weights * model_val + (1 - weights) * sr_val
        best_action = max(action_scores, key=action_scores.get)
        return best_action

    def expected_value_with_swap(self, dist1, dist2, val1, val2, swap_likelihood1, swap_likelihood2):
        # Estimate expected value for each reward branch, considering swap risk
        # If swap is likely before arrival, expected value is lower
        exp_val1 = (1 - swap_likelihood1) * val1 + swap_likelihood1 * val2
        exp_val2 = (1 - swap_likelihood2) * val2 + swap_likelihood2 * val1
        return exp_val1, exp_val2

    def update(self, state: Any, action: str, reward: float, next_state: Any) -> None:
        """Update M[prev][·] using the SR update whenever we have a valid prev_state."""
        if self.prev_state is not None:
            s = self.prev_state
            s_next = next_state.get_position()  # next_state is env

            # Ensure M[s] and M[s_next] exist
            _ = self.M[s][s]       # ensures M[s][s]=something
            _ = self.M[s_next][s_next]

            # SR update: M[s] ← M[s] + α [1_s + M[s_next] − M[s]]
            for s_prime in set(self.M[s].keys()) | set(self.M[s_next].keys()):
                target = 1.0 if s_prime == s else 0.0
                target += self.M[s_next].get(s_prime, 0.0)
                self.M[s][s_prime] += self.alpha * (target - self.M[s].get(s_prime, 0.0))

        # Move prev pointer forward
        self.prev_state = next_state.get_position()  # next_state is env

        # Track visits if you want to decay alpha over time
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

    def end_episode(self, survived, total_reward):
        self.episode_history.append((survived, total_reward))
        if len(self.episode_history) > 20:
            self.episode_history.pop(0)
        deaths = sum(1 for s, _ in self.episode_history if not s)
        if deaths / len(self.episode_history) > 0.2:
            self.risk_threshold += 0.1  # Be more cautious
        elif deaths == 0 and len(self.episode_history) == 20:
            self.risk_threshold -= 0.1  # Be greedier
        self.risk_threshold = min(max(self.risk_threshold, 0.1), 0.9)
        self.risk_threshold_history.append(self.risk_threshold)
        self.episode_rewards.append(total_reward)
        
        # Track performance
        self.performance_history.append({
            'reward': total_reward,
            'success': survived,
            'maze': "unknown",
            'episode': len(self.episode_rewards)
        })
        
        # Track learning progress
        if len(self.episode_rewards) >= 10:
            recent_avg = np.mean(self.episode_rewards[-10:])
            self.learning_progress.append(recent_avg)
            if len(self.episode_rewards) % 10 == 0:
                print(f"[SR-Greedy] Episode {len(self.episode_rewards)} avg reward (last 10): {recent_avg}")
        
        # --- Swap learning: finalize episode ---
        self.total_episodes += 1
        self.reset_swap_history()
        # Decay exploration rate
        self.exploration_rate = max(0.01, self.exploration_rate * self.exploration_decay)
        # --- Swap learning: finalize episode ---
        self.end_episode_swap_tracking()

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

    def _delta_to_action(self, curr, nxt):
        dr, dc = (nxt[0] - curr[0], nxt[1] - curr[1])
        if (dr, dc) == (-1, 0): return 'up'
        if (dr, dc) == (1, 0): return 'down'
        if (dr, dc) == (0, -1): return 'left'
        if (dr, dc) == (0, 1): return 'right'
        raise ValueError(f"Unknown delta {dr,dc}") 