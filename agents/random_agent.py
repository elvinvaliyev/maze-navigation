import random
from agents.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """Picks uniformly random legal moves; no learning."""
    def select_action(self, state):
        return random.choice(state.available_actions())

    def update(self, state, action, reward, next_state):
        pass
