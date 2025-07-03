from collections import deque, defaultdict
from typing import Tuple, List, Dict
import random
import numpy as np
from agents.base_agent import BaseAgent

class ModelBasedSurvivalAgent(BaseAgent):
    """
    Enhanced Survival agent with learning capabilities:
    - Knows the maze and step budget
    - Learns risk-aware strategies
    - Adapts survival probability estimation
    - Combines model-based planning with learned risk assessment
    """

    def __init__(self, step_budget: int,
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
        self.Q_values = defaultdict(lambda: defaultdict(float))
        self.survival_estimates = defaultdict(float)  # Track survival probability per state
        self.visit_counts = defaultdict(int)
        self.episode_rewards = []
        self.survival_history = []
        
        # Risk assessment
        self.risk_threshold = 0.7  # Start cautious
        self.prev_reward_positions = None  # For swap detection
        self.swap_detected = False
        self.swap_exploration_boost = 0
        self.swap_exploration_boost_episodes = 5  # Number of episodes to boost exploration after swap
        # --- Swap schedule learning additions ---
        self.swap_history = []  # List of swap steps in the current episode
        self.learned_swap_counts = defaultdict(int)  # step -> count
        self.total_episodes = 0
        self.episode_history = []
        self.risk_threshold_history = []  # Track risk threshold changes

    def _encode_state(self, state) -> str:
        """Encode state for Q-learning (simplified)."""
        pos = state.get_position()
        rewards = tuple(sorted(state.get_reward_positions().items()))
        return f"{pos}_{rewards}"

    def _estimate_survival_probability(self, state, target_pos) -> float:
        """Estimate probability of survival when going to target."""
        current_pos = state.get_position()
        exit_pos = state.exit
        steps_remaining = state.max_steps - state.step_count
        
        # Calculate path lengths
        dist_to_target = self.bfs_shortest_dist(state.grid, current_pos, target_pos)
        dist_to_exit = self.bfs_shortest_dist(state.grid, target_pos, exit_pos)
        
        if dist_to_target == float('inf') or dist_to_exit == float('inf'):
            return 0.0
        
        total_path_length = dist_to_target + dist_to_exit
        buffer_steps = steps_remaining - total_path_length
        
        # More conservative survival probability
        if buffer_steps < 0:
            return 0.0
        elif buffer_steps >= 8:
            return 0.95
        elif buffer_steps >= 5:
            return 0.8
        elif buffer_steps >= 3:
            return 0.6
        elif buffer_steps >= 1:
            return 0.3
        else:
            return 0.1

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
        
        # SURVIVAL-FIRST LOGIC
        current_pos = state.get_position()
        exit_pos = state.exit
        grid = state.grid
        rewards = state.get_reward_positions()
        
        # First priority: Can we reach exit safely?
        dist_to_exit = self.bfs_shortest_dist(grid, current_pos, exit_pos)
        steps_remaining = state.max_steps - state.step_count
        buffer_steps = steps_remaining - dist_to_exit
        
        # If we're cutting it close, go directly to exit
        if buffer_steps <= 2:
            target = exit_pos
        else:
            # We have some buffer, can we safely collect a reward?
            if rewards:
                if len(rewards) == 2:
                    reward_items = list(rewards.items())
                    (pos1, val1), (pos2, val2) = reward_items
                    
                    # Calculate survival probability for each reward path
                    surv_prob1 = self._estimate_survival_probability(state, pos1)
                    surv_prob2 = self._estimate_survival_probability(state, pos2)
                    
                    # Only go for reward if survival probability is high enough
                    if surv_prob1 >= self.risk_threshold and surv_prob1 > surv_prob2:
                        target = pos1
                    elif surv_prob2 >= self.risk_threshold:
                        target = pos2
                    else:
                        # Both paths too risky, go to exit
                        target = exit_pos
                else:
                    # Single reward - check if safe
                    reward_pos = list(rewards.keys())[0]
                    surv_prob = self._estimate_survival_probability(state, reward_pos)
                    if surv_prob >= self.risk_threshold:
                        target = reward_pos
                    else:
                        target = exit_pos
            else:
                # No rewards left, go to exit
                target = exit_pos
        
        # Plan path to target
        path = self.bfs_shortest_path(grid, current_pos, target)
        if len(path) < 2:
            return legal_actions[0]
        
        optimal_action = self._delta_to_action(current_pos, path[1])
        
        # Weighted sum of model-based and Q-values
        weights = 0.7  # Higher weight for model-based (survival-focused)
        action_scores = {}
        for action in legal_actions:
            q_val = self.Q_values[encoded_state][action]
            model_val = 1.0 if action == optimal_action else 0.0
            action_scores[action] = weights * model_val + (1 - weights) * q_val
        
        best_action = max(action_scores, key=action_scores.get)
        return best_action

    def expected_value_with_swap(self, dist1, dist2, val1, val2, swap_likelihood1, swap_likelihood2):
        # Estimate expected value for each reward branch, considering swap risk
        # If swap is likely before arrival, expected value is lower
        exp_val1 = (1 - swap_likelihood1) * val1 + swap_likelihood1 * val2
        exp_val2 = (1 - swap_likelihood2) * val2 + swap_likelihood2 * val1
        return exp_val1, exp_val2

    def update(self, state, action, reward, next_state) -> None:
        """Update Q-values and survival estimates based on experience."""
        current_state = self._encode_state(state)
        next_state_encoded = self._encode_state(next_state)
        
        # Q-learning update
        current_q = self.Q_values[current_state][action]
        
        # Get max Q-value for next state
        next_actions = state.available_actions()
        if next_actions:
            max_next_q = max(self.Q_values[next_state_encoded][a] for a in next_actions)
        else:
            max_next_q = 0.0
        
        # Q-learning update with survival bonus
        survival_bonus = 1.0 if reward > 0 else 0.0  # Bonus for successful reward collection
        target = reward + survival_bonus + 0.95 * max_next_q
        self.Q_values[current_state][action] += self.learning_rate * (target - current_q)
        
        # Update survival estimates
        if hasattr(state, 'collected'):
            # Update survival probability based on whether we successfully collected rewards
            collected_count = len(state.collected)
            if collected_count > 0:
                self.survival_estimates[current_state] = min(1.0, 
                    self.survival_estimates[current_state] + 0.1)
        
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

    def end_episode(self, total_reward: float, success: bool, maze_name: str = "unknown"):
        """Called at the end of each episode for learning tracking."""
        self.episode_rewards.append(total_reward)
        
        # Track survival history
        self.survival_history.append({
            'reward': total_reward,
            'success': success,
            'maze': maze_name,
            'episode': len(self.episode_rewards)
        })
        
        # Adjust risk threshold based on performance
        if len(self.episode_rewards) >= 10:
            recent_success_rate = np.mean([h['success'] for h in self.survival_history[-10:]])
            if recent_success_rate < 0.5:
                # Increase risk tolerance if survival rate is low
                self.risk_threshold = max(0.1, self.risk_threshold - 0.05)
            elif recent_success_rate > 0.8:
                # Decrease risk tolerance if survival rate is high
                self.risk_threshold = min(0.5, self.risk_threshold + 0.02)
        
        # --- Swap learning: finalize episode ---
        self.total_episodes += 1
        self.reset_swap_history()
        # Decay exploration rate, but boost if swap detected
        if self.swap_exploration_boost > 0:
            self.swap_exploration_boost -= 1
        else:
            self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
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
            'survival_rate': np.mean([h['success'] for h in self.survival_history]) if self.survival_history else 0,
            'exploration_rate': self.exploration_rate,
            'risk_threshold': self.risk_threshold,
            'q_table_size': len(self.Q_values),
            'total_visits': sum(self.visit_counts.values()),
            'recent_performance': self.episode_rewards[-10:] if self.episode_rewards else [],
            'risk_threshold_history': self.risk_threshold_history[-10:] if self.risk_threshold_history else []
        }

    def save_learned_knowledge(self, filepath: str):
        """Save learned knowledge to file."""
        import pickle
        knowledge = {
            'Q_values': dict(self.Q_values),
            'survival_estimates': dict(self.survival_estimates),
            'survival_history': self.survival_history,
            'episode_rewards': self.episode_rewards,
            'risk_threshold': self.risk_threshold,
            'risk_threshold_history': self.risk_threshold_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(knowledge, f)

    def load_learned_knowledge(self, filepath: str):
        """Load learned knowledge from file."""
        import pickle
        with open(filepath, 'rb') as f:
            knowledge = pickle.load(f)
        
        self.Q_values = defaultdict(lambda: defaultdict(float), knowledge['Q_values'])
        self.survival_estimates = defaultdict(float, knowledge['survival_estimates'])
        self.survival_history = knowledge['survival_history']
        self.episode_rewards = knowledge['episode_rewards']
        self.risk_threshold = knowledge.get('risk_threshold', 0.7)
        self.risk_threshold_history = knowledge.get('risk_threshold_history', [])

    def _simulate_move(self, pos: Tuple[int,int], action: str, grid: List[List[int]]) -> Tuple[int,int]:
        """Simulate a move without modifying the environment state."""
        moves = {'up':(-1,0),'down':(1,0),'left':(0,-1),'right':(0,1)}
        dr, dc = moves[action]
        r, c = pos
        nr, nc = r+dr, c+dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == 0:
            return (nr, nc)
        return (r, c)

    def _delta_to_action(self, current_pos, next_pos):
        """Convert position delta to action string."""
        dr = next_pos[0] - current_pos[0]
        dc = next_pos[1] - current_pos[1]
        if dr == -1: return 'up'
        if dr == 1: return 'down'
        if dc == -1: return 'left'
        if dc == 1: return 'right'
        return 'up'  # default

    def observe_state(self, state):
        """Call this at each step to keep track of the last state for swap detection."""
        self.last_state = state

    def bfs_shortest_dist(self, grid, start, goal):
        """BFS to find shortest distance between two points."""
        rows, cols = len(grid), len(grid[0])
        q = deque([(start[0], start[1], 0)])
        seen = {start}
        moves = [(-1,0), (1,0), (0,-1), (0,1)]
        
        while q:
            r, c, dist = q.popleft()
            if (r, c) == goal:
                return dist
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and 
                    grid[nr][nc] == 0 and (nr, nc) not in seen):
                    seen.add((nr, nc))
                    q.append((nr, nc, dist + 1))
        return float('inf')

    def bfs_shortest_path(self, grid, start, goal):
        """BFS to find shortest path between two points."""
        rows, cols = len(grid), len(grid[0])
        q = deque([[start]])
        seen = {start}
        moves = [(-1,0), (1,0), (0,-1), (0,1)]
        
        while q:
            path = q.popleft()
            if path[-1] == goal:
                return path
            r, c = path[-1]
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and 
                    grid[nr][nc] == 0 and (nr, nc) not in seen):
                    seen.add((nr, nc))
                    q.append(path + [(nr, nc)])
        return [] 