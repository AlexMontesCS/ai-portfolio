import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class DQN(nn.Module):
    """Deep Q-Network for Snake AI"""
    
    def __init__(self, state_size: int = 13, action_size: int = 4, hidden_size: int = 256):
        super(DQN, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Neural network layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1) # Reduced dropout
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a given state (for visualization)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def get_best_action(self, state: np.ndarray) -> int:
        """Get the best action for a given state"""
        q_values = self.get_action_values(state)
        return np.argmax(q_values)
    
    def save_model(self, filepath: str):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_size': self.fc1.out_features
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu'):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(
            state_size=checkpoint['state_size'],
            action_size=checkpoint['action_size'],
            hidden_size=checkpoint['hidden_size']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
