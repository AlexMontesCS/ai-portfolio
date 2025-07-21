import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from collections import deque, namedtuple
from typing import List, Tuple, Optional, Dict, Any
import math

from .dqn_model import DQN
from backend.config import ai_config
from .per_buffer import PrioritizedReplayBuffer

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNAgent:
    """DQN Agent for Snake game"""
    
    def __init__(self, state_size: int = None, action_size: int = None,
                 device: str = None, config: dict = None):
        
        # Use config or defaults
        self.state_size = state_size or ai_config.STATE_SIZE
        self.action_size = action_size or ai_config.ACTION_SIZE
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.batch_size = ai_config.BATCH_SIZE
        self.gamma = ai_config.GAMMA
        self.eps_start = ai_config.EPSILON_START
        self.eps_end = ai_config.EPSILON_END
        self.eps_decay = ai_config.EPSILON_DECAY
        self.target_update = ai_config.TARGET_UPDATE_FREQUENCY
        self.learning_rate = ai_config.LEARNING_RATE
        
        # Networks
        self.q_network = DQN(self.state_size, self.action_size, ai_config.HIDDEN_LAYER_SIZE).to(self.device)
        self.target_network = DQN(self.state_size, self.action_size, ai_config.HIDDEN_LAYER_SIZE).to(self.device)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.lr_scheduler = StepLR(self.optimizer, step_size=500, gamma=0.9)

        # Replay buffer
        self.memory = PrioritizedReplayBuffer(ai_config.MEMORY_SIZE)
        
        # Training state
        self.steps_done = 0
        self.episodes_done = 0
            
            # Performance tracking
        self.scores = deque(maxlen=100)
        self.losses = []
        self.q_values_history = []
            
            # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_epsilon(self) -> float:
        """Calculate current epsilon value based on steps_done"""
        return self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        epsilon = self.get_epsilon()
        if training and random.random() < epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.max(1)[1].item()
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self) -> Optional[float]:
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        experiences, indices, weights = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors efficiently
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.BoolTensor(np.array(batch.done)).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch)
        
        # Next Q values from target network
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
        
        # Compute loss
        td_error = torch.abs(current_q_values - target_q_values.unsqueeze(1)).squeeze()
        loss = (weights * td_error.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()

        # Update priorities
        self.memory.update_priorities(indices, td_error.detach().cpu().numpy())
        
        # Update target network
        if self.steps_done % self.target_update == 0:
            self.update_target_network()
        
        self.steps_done += 1
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        # Step the scheduler
        self.lr_scheduler.step()

        return loss_value
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for visualization"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities using softmax"""
        q_values = self.get_q_values(state)
        exp_values = np.exp(q_values - np.max(q_values))  # Subtract max for numerical stability
        probabilities = exp_values / np.sum(exp_values)
        return probabilities
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'epsilon': self.get_epsilon(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'scores': list(self.scores),
            'losses': self.losses[-1000:],  # Keep last 1000 losses
            'config': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'hidden_size': ai_config.HIDDEN_LAYER_SIZE,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)
        self.episodes_done = checkpoint.get('episodes_done', 0)
        if 'scores' in checkpoint:
            self.scores.extend(checkpoint['scores'])
        if 'losses' in checkpoint:
            self.losses.extend(checkpoint['losses'])
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics for monitoring"""
        return {
            'episodes': self.episodes_done,
            'steps': self.steps_done,
            'epsilon': self.get_epsilon(),
            'avg_score': np.mean(self.scores) if self.scores else 0,
            'recent_scores': list(self.scores)[-10:],
            'recent_losses': self.losses[-10:] if self.losses else [],
            'memory_size': len(self.memory)
        }
    
    def reset_episode(self, score: int):
        """Reset episode and update statistics"""
        self.episodes_done += 1
        self.scores.append(score)
    
    def is_ready_for_training(self) -> bool:
        """Check if agent has enough experiences for training"""
        return len(self.memory) >= self.batch_size
