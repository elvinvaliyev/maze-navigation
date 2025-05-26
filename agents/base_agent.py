from abc import ABC, abstractmethod
from typing import Tuple, Any

class BaseAgent(ABC):
    """
    Abstract agent interface for maze navigation.
    """

    def __init__(self):
        self.position: Tuple[int, int] = (0, 0)
        self.total_reward: float = 0.0

    def set_position(self, pos: Tuple[int, int]) -> None:
        self.position = pos

    def get_position(self) -> Tuple[int, int]:
        return self.position

    def add_reward(self, r: float) -> None:
        self.total_reward += r

    def get_total_reward(self) -> float:
        return self.total_reward

    @abstractmethod
    def select_action(self, state: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def update(self, state: Any, action: str, reward: float, next_state: Any) -> None:
        raise NotImplementedError
