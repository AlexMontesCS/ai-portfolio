import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    """LSTM model for text classification."""
    
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=128, 
                 num_classes=2, dropout_rate=0.2, num_layers=2):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state for classification
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Apply dropout and classification
        output = self.dropout(last_hidden)
        output = self.fc(output)
        
        return output

class BiLSTMModel(nn.Module):
    """Bidirectional LSTM model for text classification."""
    
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=128, 
                 num_classes=2, dropout_rate=0.2, num_layers=2):
        super(BiLSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM layer
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                             batch_first=True, bidirectional=True, 
                             dropout=dropout_rate if num_layers > 1 else 0)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head (hidden_dim * 2 for bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Bidirectional LSTM
        lstm_out, (hidden, cell) = self.bilstm(embedded)
        
        # Concatenate forward and backward hidden states
        # hidden shape: (num_layers * 2, batch_size, hidden_dim)
        forward_hidden = hidden[-2]  # Last forward layer
        backward_hidden = hidden[-1]  # Last backward layer
        last_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Apply dropout and classification
        output = self.dropout(last_hidden)
        output = self.fc(output)
        
        return output

class SimpleRNN(nn.Module):
    """Simple RNN model for text classification."""
    
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=128, 
                 num_classes=2, dropout_rate=0.2, num_layers=2):
        super(SimpleRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # RNN
        rnn_out, hidden = self.rnn(embedded)
        
        # Use the last hidden state for classification
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Apply dropout and classification
        output = self.dropout(last_hidden)
        output = self.fc(output)
        
        return output
