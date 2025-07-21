import os
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class GameConfig:
    """Game configuration settings"""
    # Grid dimensions
    GRID_WIDTH: int = 15
    GRID_HEIGHT: int = 15
    
    # Visual settings
    CELL_SIZE: int = 20
    WINDOW_WIDTH: int = GRID_WIDTH * CELL_SIZE
    WINDOW_HEIGHT: int = GRID_HEIGHT * CELL_SIZE
    
    # Game mechanics
    INITIAL_SNAKE_LENGTH: int = 3
    MAX_STEPS_WITHOUT_FOOD: int = 250 # Increased steps to encourage exploration
    
    # Colors (RGB)
    COLOR_BACKGROUND: Tuple[int, int, int] = (30, 30, 30)
    COLOR_SNAKE_HEAD: Tuple[int, int, int] = (0, 255, 0)
    COLOR_SNAKE_BODY: Tuple[int, int, int] = (0, 200, 0)
    COLOR_FOOD: Tuple[int, int, int] = (255, 0, 0)
    COLOR_GRID: Tuple[int, int, int] = (50, 50, 50)

@dataclass
class AIConfig:
    """AI/DQN configuration settings"""
    # Network architecture
    HIDDEN_LAYER_SIZE: int = 256
    LEARNING_RATE: float = 0.0005  # Slower learning rate for stability
    
    # Training parameters
    BATCH_SIZE: int = 64  # Larger batch size for more stable updates
    MEMORY_SIZE: int = 10000
    GAMMA: float = 0.95  # Discount factor
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.01
    EPSILON_DECAY: int = 5000  # Slower decay for longer exploration
    TARGET_UPDATE_FREQUENCY: int = 100
    
    # Training episodes
    MAX_EPISODES: int = 2000
    MAX_STEPS_PER_EPISODE: int = 1000
    
    # Model saving
    SAVE_FREQUENCY: int = 100
    MODEL_SAVE_PATH: str = "models"
    
    # Input features (13-dimensional state space)
    STATE_SIZE: int = 13
    ACTION_SIZE: int = 4  # up, down, left, right

@dataclass
class ServerConfig:
    """Server configuration settings"""
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    CORS_ORIGINS: list = None
    
    def __post_init__(self):
        if self.CORS_ORIGINS is None:
            self.CORS_ORIGINS = [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
                "http://localhost:5174",  # Alternative Vite port
                "http://127.0.0.1:5174",
                "*"  # Allow all origins for development
            ]

# Global configuration instances
game_config = GameConfig()
ai_config = AIConfig()
server_config = ServerConfig()

# Environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "models/")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEVICE = os.getenv("DEVICE", "auto")  # auto, cpu, cuda
