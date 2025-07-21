import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .entities import Snake, Position, Direction, GameState
from backend.config import game_config

def _get_relative_directions(direction: Direction) -> Tuple[Tuple[int, int], ...]:
    """Get straight, right, and left direction vectors"""
    if direction == Direction.UP:
        return (0, -1), (1, 0), (-1, 0)
    if direction == Direction.DOWN:
        return (0, 1), (-1, 0), (1, 0)
    if direction == Direction.LEFT:
        return (-1, 0), (0, -1), (0, 1)
    # RIGHT
    return (1, 0), (0, 1), (0, -1)

class SnakeGame:
    """Core Snake game engine with RL environment interface"""
    
    def __init__(self):
        self.grid_width = game_config.GRID_WIDTH
        self.grid_height = game_config.GRID_HEIGHT
        self.max_steps_without_food = game_config.MAX_STEPS_WITHOUT_FOOD
        
        # Game state
        self.snake: Optional[Snake] = None
        self.food_position: Optional[Position] = None
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.game_state = GameState.PLAYING
        
        # Initialize game
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset game to initial state and return observation"""
        # Initialize snake at center
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        self.snake = Snake(Position(center_x, center_y), game_config.INITIAL_SNAKE_LENGTH)
        
        # Reset game state
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.game_state = GameState.PLAYING
        
        # Place initial food
        self._place_food()
        
        return self.get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one game step with the given action
        Returns: (next_state, reward, done, info)
        """
        if self.game_state != GameState.PLAYING:
            return self.get_state(), 0, True, {"reason": "game_over"}
        
        # Convert action to direction
        direction_map = {
            0: Direction.UP,
            1: Direction.DOWN,
            2: Direction.LEFT,
            3: Direction.RIGHT
        }
        
        direction = direction_map.get(action, self.snake.direction)
        
        # Prevent illegal moves
        if not self.snake.set_direction(direction):
            # Penalize and end game for illegal move attempt
            return self.get_state(), -10, True, {"reason": "illegal_move", "score": self.score}

        # Move snake
        old_head = self.snake.head
        
        # Check for food at the next position *before* moving
        next_head_position = self.snake.get_next_head_position()
        ate_food = next_head_position == self.food_position

        self.snake.move(grow=ate_food)
        self.steps += 1
        
        # Calculate reward
        reward = self._calculate_reward(old_head, ate_food)
        
        # Check game over conditions
        done = self._check_game_over()
        
        # Handle food consumption
        if ate_food:
            self.score += 1
            self.steps_without_food = 0
            self._place_food()
            if self.food_position is None:
                self.game_state = GameState.GAME_OVER
                done = True
        else:
            self.steps_without_food += 1
        
        # Prepare info
        info = {
            "score": self.score,
            "steps": self.steps,
            "snake_length": self.snake.length,
            "food_position": self.food_position.to_tuple() if self.food_position else None,
            "snake_head": self.snake.head.to_tuple(),
            "ate_food": ate_food
        }
        
        return self.get_state(), reward, done, info
    
    def get_state(self) -> np.ndarray:
        """
        Get current game state as a feature vector for the AI.
        State (13 features):
        - 3 danger indicators (straight, right, left)
        - 4 direction indicators (one-hot encoded)
        - 2 relative food coordinates
        - 4 food direction indicators (binary)
        """
        if not self.snake or not self.food_position:
            return np.zeros(13)
        
        head = self.snake.head
        direction = self.snake.direction
        
        straight, right, left = _get_relative_directions(direction)

        def is_dangerous(pos):
            """Check if a position is a wall or part of the snake's body."""
            return (pos.x < 0 or pos.x >= self.grid_width or
                    pos.y < 0 or pos.y >= self.grid_height or
                    pos in self.snake.body)

        # Check for danger in relative directions
        danger_straight = is_dangerous(Position(head.x + straight[0], head.y + straight[1]))
        danger_right = is_dangerous(Position(head.x + right[0], head.y + right[1]))
        danger_left = is_dangerous(Position(head.x + left[0], head.y + left[1]))

        # Current direction (one-hot)
        dir_left = direction == Direction.LEFT
        dir_right = direction == Direction.RIGHT
        dir_up = direction == Direction.UP
        dir_down = direction == Direction.DOWN
        
        # Relative food coordinates (normalized)
        food_dx = (self.food_position.x - head.x) / self.grid_width
        food_dy = (self.food_position.y - head.y) / self.grid_height

        state = np.array([
            danger_straight, danger_right, danger_left,
            dir_left, dir_right, dir_up, dir_down,
            food_dx, food_dy,
            1 if food_dx < 0 else 0, # Food left
            1 if food_dx > 0 else 0, # Food right
            1 if food_dy < 0 else 0, # Food up
            1 if food_dy > 0 else 0  # Food down
        ], dtype=np.float32)

        return state
    
    def _place_food(self):
        """Place food at random empty position"""
        empty_positions = []
        
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                pos = Position(x, y)
                if pos not in self.snake.body:
                    empty_positions.append(pos)
        
        if empty_positions:
            self.food_position = random.choice(empty_positions)
        else:
            # No empty positions (shouldn't happen in normal game)
            self.food_position = None
    
    def _calculate_reward(self, old_head: Position, ate_food: bool) -> float:
        """
        Calculate reward for the current step.
        This function provides a dense reward signal to guide the agent.
        """
        reward = 0
        if self._check_collision():
            return -10  # Penalty for dying

        if ate_food:
            return 10  # Reward for eating food

        # Reward/penalize moving closer/further from the food
        if self.food_position:
            new_distance = abs(self.snake.head.x - self.food_position.x) + abs(self.snake.head.y - self.food_position.y)
            old_distance = abs(old_head.x - self.food_position.x) + abs(old_head.y - self.food_position.y)
            if new_distance < old_distance:
                reward = 1  # Moved closer
            else:
                reward = -1 # Moved further away

        return reward
    
    def _check_game_over(self) -> bool:
        """Check if game should end"""
        if self._check_collision():
            self.game_state = GameState.GAME_OVER
            return True
        
        if self.steps_without_food >= self.max_steps_without_food:
            self.game_state = GameState.GAME_OVER
            return True
        
        return False
    
    def _check_collision(self) -> bool:
        """Check for wall or self-collision."""
        head = self.snake.head
        # Wall collision
        if (head.x < 0 or head.x >= self.grid_width or
            head.y < 0 or head.y >= self.grid_height):
            return True
        # Self-collision
        if self.snake.check_self_collision():
            return True
        return False
    
    def get_game_data(self) -> Dict[str, Any]:
        """Get complete game state for visualization"""
        return {
            "snake_body": self.snake.get_body_positions() if self.snake else [],
            "food_position": self.food_position.to_tuple() if self.food_position else None,
            "score": self.score,
            "steps": self.steps,
            "game_state": self.game_state.value,
            "grid_size": (self.grid_width, self.grid_height),
            "direction": self.snake.direction.value if self.snake else None
        }
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions (for UI hints)"""
        if not self.snake:
            return [0, 1, 2, 3]
        
        current_dir = self.snake.direction
        valid_actions = []
        
        # Check each direction
        for action, direction in enumerate([Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]):
            # Can't reverse direction
            if not ((current_dir == Direction.UP and direction == Direction.DOWN) or
                    (current_dir == Direction.DOWN and direction == Direction.UP) or
                    (current_dir == Direction.LEFT and direction == Direction.RIGHT) or
                    (current_dir == Direction.RIGHT and direction == Direction.LEFT)):
                valid_actions.append(action)
        
        return valid_actions

    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return self.game_state == GameState.GAME_OVER
