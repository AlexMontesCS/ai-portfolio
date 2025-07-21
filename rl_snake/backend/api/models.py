from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ActionRequest(BaseModel):
    """Request model for game actions"""
    action: int  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

class GameStateResponse(BaseModel):
    """Response model for game state"""
    game_id: str
    snake_body: List[List[int]]
    food_position: Optional[List[int]]
    score: int
    steps: int
    game_state: str
    grid_size: List[int]
    direction: Optional[int]
    ai_analysis: Optional[Dict[str, Any]] = None

class ModelInfo(BaseModel):
    """AI model information"""
    model_loaded: bool
    training_episodes: int
    current_epsilon: float
    average_score: float
    memory_size: int

class TrainingStatus(BaseModel):
    """Training status information"""
    is_training: bool
    current_episode: int
    total_episodes: int
    current_score: float
    average_score: float
    epsilon: float
    recent_losses: List[float]

class AIAnalysis(BaseModel):
    """AI decision analysis"""
    q_values: List[float]
    action_probabilities: List[float]
    recommended_action: int
    action_names: List[str]
    confidence: float
