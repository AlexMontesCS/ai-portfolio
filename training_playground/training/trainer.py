import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

class ModelTrainer:
    """
    A comprehensive trainer class for PyTorch models.
    """
    
    def __init__(self, model, train_loader, test_loader, learning_rate=0.001, 
                 optimizer_name='Adam', weight_decay=0.0001, device='cpu'):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            learning_rate: Learning rate for optimizer
            optimizer_name: Name of optimizer ('Adam', 'SGD', 'RMSprop')
            weight_decay: Weight decay for regularization
            device: Device to run training on ('cpu' or 'cuda')
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._get_optimizer(optimizer_name, learning_rate, weight_decay)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def _get_optimizer(self, optimizer_name, learning_rate, weight_decay):
        """Get optimizer based on name."""
        if optimizer_name == 'Adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            return optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_name == 'RMSprop':
            return optim.RMSprop(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def train_epoch(self, progress_callback=None):
        """
        Train the model for one epoch.
        
        Args:
            progress_callback: Optional callback function to report progress
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        num_batches = len(self.train_loader)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Report progress if callback provided (every 5% of batches, minimum every batch for small datasets)
            update_interval = max(1, min(num_batches // 20, 5))
            if progress_callback and batch_idx % update_interval == 0:
                current_loss = total_loss / (batch_idx + 1)
                current_acc = 100.0 * correct / total
                progress_callback(batch_idx, num_batches, current_loss, current_acc)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        # Final progress update to show completion
        if progress_callback:
            progress_callback(num_batches - 1, num_batches, avg_loss, accuracy)
        
        # Update history
        self.history['train_loss'].append(avg_loss)
        self.history['train_acc'].append(accuracy)
        
        return avg_loss, accuracy
    
    def validate(self):
        """
        Validate the model on test data.
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        
        # Update history
        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(accuracy)
        
        return avg_loss, accuracy
    
    def train(self, epochs):
        """
        Train the model for multiple epochs.
        
        Args:
            epochs: Number of epochs to train
            
        Returns:
            dict: Training history
        """
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Time: {epoch_time:.2f}s")
            print("-" * 50)
        
        return self.history
    
    def evaluate_model(self):
        """
        Evaluate the model and return detailed metrics.
        
        Returns:
            dict: Evaluation results including confusion matrix and classification report
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Compute metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        class_report = classification_report(all_targets, all_predictions)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def save_model(self, filepath):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)
    
    def load_model(self, filepath):
        """Load a saved model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
