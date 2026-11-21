"""Tabular Q-learning agent with epsilon-greedy exploration.

The agent knows nothing about the environment beyond the list of action names.
States are any hashable value (in this project, ``(row, col)`` tuples).
"""

from __future__ import annotations

import random
from typing import Hashable

State = Hashable

DEFAULT_LEARNING_RATE: float = 0.1
DEFAULT_DISCOUNT_FACTOR: float = 0.9
DEFAULT_EPSILON: float = 1.0
DEFAULT_EPSILON_MIN: float = 0.05
DEFAULT_EPSILON_DECAY: float = 0.995


class QLearningAgent:
    """Tabular Q-learning agent.

    The Q-table is a dict ``{state: {action: value}}``; unseen states are
    added lazily with every action set to 0.0.
    """

    def __init__(
        self,
        actions: list[str],
        learning_rate: float = DEFAULT_LEARNING_RATE,
        discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
        epsilon: float = DEFAULT_EPSILON,
        epsilon_min: float = DEFAULT_EPSILON_MIN,
        epsilon_decay: float = DEFAULT_EPSILON_DECAY,
        seed: int | None = None,
    ) -> None:
        raise NotImplementedError

    def get_q_values(self, state: State) -> dict[str, float]:
        raise NotImplementedError

    def best_action(self, state: State) -> str:
        raise NotImplementedError

    def choose_action(self, state: State) -> str:
        raise NotImplementedError

    def update(
        self,
        state: State,
        action: str,
        reward: float,
        next_state: State,
        done: bool,
    ) -> float:
        raise NotImplementedError

    def decay_epsilon(self) -> float:
        raise NotImplementedError

    def greedy_policy(self) -> dict[State, str]:
        raise NotImplementedError
