# Backend package
from .game import SnakeGame, Snake, Position, Direction, GameState
from .ai import DQNAgent, DQN, PrioritizedReplayBuffer
from .training import train_dqn_agent, evaluate_agent, TrainingMonitor
from .api import app
from .config import game_config, ai_config, server_config

__all__ = [
    "SnakeGame", "Snake", "Position", "Direction", "GameState",
    "DQNAgent", "DQN", "PrioritizedReplayBuffer",
    "train_dqn_agent", "evaluate_agent", "TrainingMonitor",
    "app",
    "game_config", "ai_config", "server_config"
]
