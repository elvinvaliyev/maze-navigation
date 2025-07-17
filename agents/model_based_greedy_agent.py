from collections import deque, defaultdict
from typing import Tuple, List, Dict, Optional
import random
import numpy as np
from agents.base_agent import BaseAgent

class ModelBasedGreedyAgent(BaseAgent):
    
    """
    Enhanced Greedy agent with learning capabilities:
      - Knows the full maze (grid, start/exit)
      - Learns from experience to improve decision making
      - Tracks performance and adapts strategy
      - Combines model-based planning with learned preferences
    """

    def __init__(self, step_budget: Optional[int] = None, 
                 learning_rate: float = 0.1,
                 exploration_rate: float = 0.1,
                 exploration_decay: float = 0.995):
        super().__init__()
        self.step_budget = step_budget
        
        # Learning components
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = 0.01
        
        # Experience tracking
        self.Q_values = defaultdict(lambda: defaultdict(float))  # Q(s, a)
        self.visit_counts = defaultdict(int)
        self.episode_rewards = []
        self.performance_history = []
        
        # State encoding
        self.state_cache = {}
        self.prev_reward_positions = None  # For swap detection
        self.swap_detected = False
        self.swap_exploration_boost = 0
        self.swap_exploration_boost_episodes = 5  # Number of episodes to boost exploration after swap
        # --- Swap schedule learning additions ---
        self.swap_history = []  # List of swap steps in the current episode
        self.learned_swap_counts = defaultdict(int)  # step -> count
        self.total_episodes = 0
        self.risk_threshold = 0.3  # Start moderately greedy
        self.episode_history = []  # Track (survived, reward)
        self.risk_threshold_history = []  # Track risk threshold changes

    def _encode_state(self, state) -> str:
        """Encode state for Q-learning (simplified)."""
        pos = state.get_position()
        rewards = tuple(sorted(state.get_reward_positions().items()))
        return f"{pos}_{rewards}"

    def select_action(self, state) -> str:
        encoded_state = self._encode_state(state)
        legal_actions = state.available_actions()
        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            return random.choice(legal_actions)
        # Hybrid: weighted sum of model-based and Q-values
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
            q_val = self.Q_values[encoded_state][action]
            model_val = 1.0 if action == optimal_action else 0.0
            action_scores[action] = weights * model_val + (1 - weights) * q_val
        best_action = max(action_scores, key=action_scores.get)
        return best_action

    def update(self, state, action, reward, next_state) -> None:
        """Update Q-values based on experience."""
        current_state = self._encode_state(state)
        next_state_encoded = self._encode_state(next_state)
        
        # Q-learning update
        current_q = self.Q_values[current_state][action]
        
        # Get max Q-value for next state
        next_actions = next_state.available_actions()  # Use next state's available actions
        if next_actions:
            max_next_q = max(self.Q_values[next_state_encoded][a] for a in next_actions)
        else:
            max_next_q = 0.0
        
        # Q-learning update rule: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        target = reward + 0.95 * max_next_q  # gamma = 0.95
        self.Q_values[current_state][action] += self.learning_rate * (target - current_q)
        
        # Update visit counts
        self.visit_counts[(current_state, action)] += 1
        # Track previous state as position
        self.prev_state = next_state.get_position()
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
        # >>> record for learning_stats
        self.episode_rewards.append(total_reward)
        if len(self.episode_history) > 20:
            self.episode_history.pop(0)
        deaths = sum(1 for s, _ in self.episode_history if not s)
        if deaths / len(self.episode_history) > 0.2:
            self.risk_threshold += 0.1  # Be more cautious
        elif deaths == 0 and len(self.episode_history) == 20:
            self.risk_threshold -= 0.1  # Be greedier
        self.risk_threshold = min(max(self.risk_threshold, 0.1), 0.9)
        self.risk_threshold_history.append(self.risk_threshold)
        self.risk_threshold_history.append(self.risk_threshold)

    def get_learning_stats(self) -> Dict:
        """Get current learning statistics."""
        return {
            'episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'exploration_rate': self.exploration_rate,
            'q_table_size': len(self.Q_values),
            'total_visits': sum(self.visit_counts.values()),
            'recent_performance': self.episode_rewards[-10:] if self.episode_rewards else []
        }

    def save_learned_knowledge(self, filepath: str):
        """Save learned knowledge to file."""
        import pickle
        knowledge = {
            'Q_values': dict(self.Q_values),
            'performance_history': self.performance_history,
            'episode_rewards': self.episode_rewards
        }
        with open(filepath, 'wb') as f:
            pickle.dump(knowledge, f)

    def load_learned_knowledge(self, filepath: str):
        """Load learned knowledge from file."""
        import pickle
        with open(filepath, 'rb') as f:
            knowledge = pickle.load(f)
        
        self.Q_values = defaultdict(lambda: defaultdict(float), knowledge['Q_values'])
        self.performance_history = knowledge['performance_history']
        self.episode_rewards = knowledge['episode_rewards']

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

    def observe_state(self, state):
        """Call this at each step to keep track of the last state for swap detection."""
        self.last_state = state

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

    def expected_value_with_swap(self, dist1, dist2, val1, val2, swap_likelihood1, swap_likelihood2):
        # Estimate expected value for each reward branch, considering swap risk
        # If swap is likely before arrival, expected value is lower
        exp_val1 = (1 - swap_likelihood1) * val1 + swap_likelihood1 * val2
        exp_val2 = (1 - swap_likelihood2) * val2 + swap_likelihood2 * val1
        return exp_val1, exp_val2
