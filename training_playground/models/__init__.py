from .cnn import SimpleCNN, CustomCNN
from .resnet import ResNet18
from .lstm import LSTMModel, BiLSTMModel, SimpleRNN

def get_model(model_type, num_classes, dropout_rate=0.2):
    """Factory function to create models based on type."""
    
    if model_type == "Simple CNN":
        return SimpleCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_type == "Custom CNN":
        return CustomCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_type == "ResNet-18":
        return ResNet18(num_classes=num_classes)
    elif model_type == "LSTM":
        return LSTMModel(vocab_size=10000, embedding_dim=128, hidden_dim=128, 
                        num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_type == "BiLSTM":
        return BiLSTMModel(vocab_size=10000, embedding_dim=128, hidden_dim=128, 
                          num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_type == "Simple RNN":
        return SimpleRNN(vocab_size=10000, embedding_dim=128, hidden_dim=128, 
                        num_classes=num_classes, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

__all__ = ['get_model', 'SimpleCNN', 'CustomCNN', 'ResNet18', 'LSTMModel', 'BiLSTMModel', 'SimpleRNN']
