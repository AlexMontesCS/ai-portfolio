from enum import Enum
from typing import Tuple, List, Optional
from dataclasses import dataclass
import random
import numpy as np

class Direction(Enum):
    """Snake movement directions"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class GameState(Enum):
    """Game state enumeration"""
    PLAYING = "playing"
    GAME_OVER = "game_over"
    PAUSED = "paused"

@dataclass
class Position:
    """2D position with utility methods"""
    x: int
    y: int
    
    def __add__(self, other: 'Position') -> 'Position':
        return Position(self.x + other.x, self.y + other.y)
    
    def __eq__(self, other: 'Position') -> bool:
        return self.x == other.x and self.y == other.y
    
    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)

class Snake:
    """Snake entity with movement and collision logic"""
    
    def __init__(self, initial_position: Position, initial_length: int = 3):
        self.body: List[Position] = []
        self.direction = Direction.RIGHT
        self.next_direction = Direction.RIGHT
        
        # Initialize snake body
        for i in range(initial_length):
            self.body.append(Position(
                initial_position.x - i,
                initial_position.y
            ))
    
    @property
    def head(self) -> Position:
        """Get snake head position"""
        return self.body[0]
    
    @property
    def length(self) -> int:
        """Get snake length"""
        return len(self.body)
    
    def set_direction(self, direction: Direction) -> bool:
        """Set next direction for the snake."""
        self.next_direction = direction
        return True

    def get_next_head_position(self) -> Position:
        """Calculate the position of the snake's head after the next move."""
        direction_map = {
            Direction.UP: Position(0, -1),
            Direction.DOWN: Position(0, 1),
            Direction.LEFT: Position(-1, 0),
            Direction.RIGHT: Position(1, 0),
        }
        return self.head + direction_map[self.next_direction]
    
    def move(self, grow: bool = False) -> Position:
        """Move snake and return new head position"""
        self.direction = self.next_direction
        
        # Calculate new head position
        direction_map = {
            Direction.UP: Position(0, -1),
            Direction.DOWN: Position(0, 1),
            Direction.LEFT: Position(-1, 0),
            Direction.RIGHT: Position(1, 0)
        }
        
        new_head = self.head + direction_map[self.direction]
        self.body.insert(0, new_head)
        
        # Remove tail unless growing
        if not grow:
            removed_tail = self.body.pop()
            return removed_tail
        
        return None
    
    def check_self_collision(self) -> bool:
        """Check if snake collides with itself"""
        return self.head in self.body[1:]
    
    def get_body_positions(self) -> List[Tuple[int, int]]:
        """Get all body positions as tuples"""
        return [pos.to_tuple() for pos in self.body]
