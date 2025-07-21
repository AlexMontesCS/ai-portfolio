import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_metrics(y_true, y_pred, average='weighted'):
    """
    Compute various classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for multi-class metrics
        
    Returns:
        dict: Dictionary containing various metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False):
    """
    Create a confusion matrix plot using Plotly.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        
    Returns:
        plotly.graph_objects.Figure: Confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        text_auto = '.2%'
    else:
        text_auto = True
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names,
        y=class_names,
        text_auto=text_auto,
        title="Confusion Matrix"
    )
    
    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        width=500,
        height=500
    )
    
    return fig

def plot_training_history(history):
    """
    Plot training history with loss and accuracy curves.
    
    Args:
        history: Dictionary containing training history
        
    Returns:
        plotly.graph_objects.Figure: Training history plot
    """
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Loss", "Accuracy"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history['train_loss'],
            name="Train Loss",
            line=dict(color="blue"),
            mode="lines+markers"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history['val_loss'],
            name="Val Loss",
            line=dict(color="red"),
            mode="lines+markers"
        ),
        row=1, col=1
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history['train_acc'],
            name="Train Acc",
            line=dict(color="green"),
            mode="lines+markers"
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history['val_acc'],
            name="Val Acc",
            line=dict(color="orange"),
            mode="lines+markers"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        title_text="Training Progress"
    )
    
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
    
    return fig

def plot_metrics_comparison(metrics_dict):
    """
    Create a bar plot comparing different metrics.
    
    Args:
        metrics_dict: Dictionary containing metrics
        
    Returns:
        plotly.graph_objects.Figure: Metrics comparison plot
    """
    metrics_names = list(metrics_dict.keys())
    metrics_values = [metrics_dict[key] * 100 if key != 'accuracy' else metrics_dict[key] * 100 
                     for key in metrics_names]
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics_names,
            y=metrics_values,
            text=[f'{val:.2f}%' for val in metrics_values],
            textposition='auto',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )
    ])
    
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Metrics",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 100]),
        showlegend=False
    )
    
    return fig

def create_learning_curve(train_scores, val_scores, train_sizes):
    """
    Create a learning curve plot.
    
    Args:
        train_scores: Training scores for different training set sizes
        val_scores: Validation scores for different training set sizes
        train_sizes: Different training set sizes
        
    Returns:
        plotly.graph_objects.Figure: Learning curve plot
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_scores,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=val_scores,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title="Learning Curve",
        xaxis_title="Training Set Size",
        yaxis_title="Score",
        showlegend=True
    )
    
    return fig

def plot_feature_importance(feature_names, importance_scores, top_n=20):
    """
    Plot feature importance scores.
    
    Args:
        feature_names: List of feature names
        importance_scores: Importance scores for each feature
        top_n: Number of top features to display
        
    Returns:
        plotly.graph_objects.Figure: Feature importance plot
    """
    # Sort features by importance
    sorted_indices = np.argsort(importance_scores)[::-1][:top_n]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_scores = [importance_scores[i] for i in sorted_indices]
    
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_scores,
            y=sorted_features,
            orientation='h',
            text=[f'{score:.3f}' for score in sorted_scores],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=f"Top {top_n} Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=max(400, top_n * 25)
    )
    
    return fig
