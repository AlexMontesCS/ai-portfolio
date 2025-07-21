import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import warnings
from datetime import datetime

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress torch class path warning
import logging
logging.getLogger("torch").setLevel(logging.ERROR)

# Try to import optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not installed. Using matplotlib for visualizations.")

# Import our custom modules with error handling
try:
    from models import get_model
    from datasets import get_dataset
    from training.trainer import ModelTrainer
    from utils.helpers import set_seed, format_time, create_model_summary
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"Error importing custom modules: {e}")
    st.info("Please make sure all dependencies are installed: pip install -r requirements.txt")

# Page configuration
st.set_page_config(
    page_title="AI Model Training Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize PyTorch properly to avoid warnings
@st.cache_resource
def init_torch():
    """Initialize PyTorch and suppress warnings."""
    torch.set_num_threads(1)  # Avoid threading issues
    return torch.cuda.is_available()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .stProgress .st-bo {
        background-color: #667eea;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    .config-header {
        color: #495057;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    .training-status {
        background-color: #e3f2fd;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize PyTorch
    cuda_available = init_torch()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Model Training Playground</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the AI Model Training Playground! This interactive application allows you to:
    - Train neural networks on different datasets
    - Experiment with various model architectures
    - Tune hyperparameter in real-time
    - Visualize training metrics live
    """)
    
    # Check if modules are available
    if not MODULES_AVAILABLE:
        st.error("⚠️ Required modules are not available. Please install dependencies and restart the app.")
        st.code("pip install -r requirements.txt")
        return
    
    # Initialize session state
    if 'start_training' not in st.session_state:
        st.session_state.start_training = False
    if 'stop_training' not in st.session_state:
        st.session_state.stop_training = False
    if 'save_model' not in st.session_state:
        st.session_state.save_model = False
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'model_info' not in st.session_state:
        st.session_state.model_info = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Dataset selection
        st.subheader("Dataset Selection")
        dataset_name = st.selectbox(
            "Choose Dataset:",
            ["MNIST", "CIFAR-10", "IMDB Sentiment"],
            help="Select the dataset for training"
        )
        
        # Model architecture selection
        st.subheader("Model Architecture")
        if dataset_name in ["MNIST", "CIFAR-10"]:
            model_type = st.selectbox(
                "Choose Model:",
                ["Simple CNN", "ResNet-18", "Custom CNN"],
                help="Select the neural network architecture"
            )
        else:  # IMDB
            model_type = st.selectbox(
                "Choose Model:",
                ["LSTM", "BiLSTM", "Simple RNN"],
                help="Select the neural network architecture"
            )
        
        # Hyperparameters
        st.subheader("Hyperparameters")
        
        learning_rate = st.slider(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.001,
            step=0.0001,
            format="%.4f"
        )
        
        batch_size = st.selectbox(
            "Batch Size:",
            [16, 32, 64, 128, 256],
            index=2
        )
        
        epochs = st.slider(
            "Number of Epochs:",
            min_value=1,
            max_value=50,
            value=10
        )
        
        optimizer_name = st.selectbox(
            "Optimizer:",
            ["Adam", "SGD", "RMSprop"]
        )
        
        # Advanced options
        with st.expander("Advanced Settings"):
            weight_decay = st.slider(
                "Weight Decay:",
                min_value=0.0,
                max_value=0.01,
                value=0.0001,
                step=0.0001,
                format="%.4f"
            )
            
            dropout_rate = st.slider(
                "Dropout Rate:",
                min_value=0.0,
                max_value=0.8,
                value=0.2,
                step=0.1
            )
            
            seed = st.number_input(
                "Random Seed:",
                min_value=0,
                max_value=9999,
                value=42
            )
        
        # Training controls
        st.subheader("Training Controls")
        
        if st.button("▶️ Start Training", type="primary"):
            st.session_state.start_training = True
        
        if st.button("⏹️ Stop Training"):
            st.session_state.stop_training = True
        
        # Save model button with status
        model_status = "✅ Model Ready" if st.session_state.trained_model is not None else "❌ No Model"
        save_button_disabled = st.session_state.trained_model is None
        
        if st.button(f"💾 Save Model ({model_status})", disabled=save_button_disabled):
            st.session_state.save_model = True
        
        # Show saved models
        if os.path.exists("saved_models"):
            saved_files = [f for f in os.listdir("saved_models") if f.endswith('.pth')]
            if saved_files:
                with st.expander(f"📁 Saved Models ({len(saved_files)})"):
                    for file in sorted(saved_files, reverse=True)[:5]:  # Show last 5
                        st.text(f"• {file}")
                    if len(saved_files) > 5:
                        st.text(f"... and {len(saved_files) - 5} more")
                        
                    # Load model option
                    st.subheader("Load Model")
                    selected_model = st.selectbox(
                        "Select model to load:",
                        [""] + sorted(saved_files, reverse=True),
                        help="Load a previously saved model"
                    )
                    
                    if selected_model and st.button("📂 Load Selected Model"):
                        try:
                            # Extract model info from filename
                            filename_parts = selected_model.replace('.pth', '').split('_')
                            if len(filename_parts) >= 3:
                                model_type_from_file = '_'.join(filename_parts[:-2])
                                dataset_name_from_file = filename_parts[-2]
                                
                                st.info(f"Loading {model_type_from_file} trained on {dataset_name_from_file}")
                                st.warning("Note: Make sure the model architecture matches your current selection!")
                            else:
                                st.warning("Could not parse model filename. Loading anyway...")
                                
                        except Exception as e:
                            st.error(f"Error loading model: {e}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Training Progress")
        
        # Training metrics placeholders
        metrics_container = st.container()
        with metrics_container:
            metric_cols = st.columns(4)
            with metric_cols[0]:
                epoch_placeholder = st.empty()
            with metric_cols[1]:
                loss_placeholder = st.empty()
            with metric_cols[2]:
                accuracy_placeholder = st.empty()
            with metric_cols[3]:
                time_placeholder = st.empty()
        
        # Progress bar
        progress_placeholder = st.empty()
        
        # Training plots
        plot_placeholder = st.empty()
        
        # Model evaluation
        evaluation_placeholder = st.empty()
    
    with col2:
        st.header("ℹ️ Model Information")
        
        # Dataset info
        with st.expander("**Dataset Information**", expanded=True):
            try:
                dataset_info = get_dataset_info(dataset_name)
                st.write(dataset_info)
            except Exception as e:
                st.error(f"Error loading dataset info: {e}")
        
        # Model architecture
        with st.expander("**Model Architecture**", expanded=True):
            try:
                model_info = get_model_info(model_type, dataset_name)
                st.write(model_info)
            except Exception as e:
                st.error(f"Error loading model info: {e}")
        
        # Training logs
        with st.expander("**Training Logs**"):
            log_placeholder = st.empty()
    
    # Training logic
    if hasattr(st.session_state, 'start_training') and st.session_state.start_training:
        try:
            # Set random seed
            set_seed(seed)
            
            # Initialize training
            with st.spinner("Initializing training..."):
                # Load dataset
                train_loader, test_loader, num_classes = get_dataset(
                    dataset_name, batch_size
                )
                
                # Create model
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = get_model(model_type, num_classes, dropout_rate)
                model = model.to(device)
                
                # Store model and info in session state
                st.session_state.trained_model = model
                st.session_state.model_info = {
                    'model_type': model_type,
                    'dataset_name': dataset_name,
                    'device': device
                }
                
                # Initialize trainer
                trainer = ModelTrainer(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    learning_rate=learning_rate,
                    optimizer_name=optimizer_name,
                    weight_decay=weight_decay,
                    device=device
                )
            
            # Training loop
            training_history = {"epochs": [], "train_loss": [], "train_acc": [], 
                              "val_loss": [], "val_acc": []}
            
            start_time = time.time()
            
            # Show initial training setup
            with metrics_container:
                with metric_cols[0]:
                    epoch_placeholder.metric("Epoch", "0/{}".format(epochs))
                with metric_cols[1]:
                    loss_placeholder.metric("Loss", "Initializing...")
                with metric_cols[2]:
                    accuracy_placeholder.metric("Accuracy", "Initializing...")
                with metric_cols[3]:
                    time_placeholder.metric("Time", "0s")
            
            # Show initial progress
            progress_placeholder.progress(0.0)
            
            # Show initial placeholder plot
            initial_fig = create_initial_plot()
            if PLOTLY_AVAILABLE:
                plot_placeholder.plotly_chart(initial_fig, use_container_width=True)
            else:
                plot_placeholder.pyplot(initial_fig)
            
            # Show initial log
            log_placeholder.text("Training initialized - Starting first epoch...")
            
            for epoch in range(epochs):
                if hasattr(st.session_state, 'stop_training') and st.session_state.stop_training:
                    st.warning("Training stopped by user.")
                    break
                
                epoch_start_time = time.time()
                
                # Update epoch indicator and initial status
                with metrics_container:
                    with metric_cols[0]:
                        epoch_placeholder.metric("Epoch", f"{epoch + 1}/{epochs}")
                    with metric_cols[1]:
                        loss_placeholder.metric("Loss", "Training...")
                    with metric_cols[2]:
                        accuracy_placeholder.metric("Accuracy", "Training...")
                    with metric_cols[3]:
                        elapsed_time = time.time() - start_time
                        time_placeholder.metric("Time", format_time(elapsed_time))
                
                # Show epoch progress
                epoch_progress = epoch / epochs
                progress_placeholder.progress(epoch_progress)
                
                # Update log for current epoch
                log_placeholder.text(f"Training epoch {epoch + 1}/{epochs}...")
                
                # Define progress callback for real-time updates
                def update_progress(batch_idx, num_batches, current_loss, current_acc):
                    # Update metrics during training
                    with metrics_container:
                        with metric_cols[1]:
                            loss_placeholder.metric("Loss", f"{current_loss:.4f}")
                        with metric_cols[2]:
                            accuracy_placeholder.metric("Accuracy", f"{current_acc:.2f}%")
                        with metric_cols[3]:
                            elapsed_time = time.time() - start_time
                            time_placeholder.metric("Time", format_time(elapsed_time))
                    
                    # Update progress bar with batch-level granularity
                    batch_progress = (batch_idx + 1) / num_batches
                    epoch_progress = (epoch + batch_progress) / epochs
                    progress_placeholder.progress(epoch_progress)
                    
                    # Update log with batch progress
                    log_placeholder.text(f"Training epoch {epoch + 1}/{epochs} - Batch {batch_idx + 1}/{num_batches} ({batch_progress:.1%})")
                    
                    # Small delay to make updates visible
                    time.sleep(0.01)
                
                # Train one epoch with progress updates
                train_loss, train_acc = trainer.train_epoch(progress_callback=update_progress)
                
                # Validation
                log_placeholder.text(f"Validating epoch {epoch + 1}/{epochs}...")
                val_loss, val_acc = trainer.validate()
                
                # Update history
                training_history["epochs"].append(epoch + 1)
                training_history["train_loss"].append(train_loss)
                training_history["train_acc"].append(train_acc)
                training_history["val_loss"].append(val_loss)
                training_history["val_acc"].append(val_acc)
                
                # Update final metrics and progress
                elapsed_time = time.time() - start_time
                
                with metrics_container:
                    with metric_cols[0]:
                        epoch_placeholder.metric("Epoch", f"{epoch + 1}/{epochs}")
                    with metric_cols[1]:
                        loss_placeholder.metric("Loss", f"{train_loss:.4f}")
                    with metric_cols[2]:
                        accuracy_placeholder.metric("Accuracy", f"{train_acc:.2f}%")
                    with metric_cols[3]:
                        time_placeholder.metric("Time", format_time(elapsed_time))
                
                # Update progress bar with completion of this epoch
                progress = (epoch + 1) / epochs
                progress_placeholder.progress(progress)
                
                # Update plots
                if len(training_history["epochs"]) > 0:
                    fig = create_training_plots(training_history)
                    if PLOTLY_AVAILABLE:
                        plot_placeholder.plotly_chart(fig, use_container_width=True)
                    else:
                        plot_placeholder.pyplot(fig)
                
                # Update logs with comprehensive information
                log_text = f"Epoch {epoch + 1}/{epochs} completed:\n"
                log_text += f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
                log_text += f"  Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%\n"
                log_text += f"  Time Elapsed: {format_time(elapsed_time)}"
                log_placeholder.text(log_text)
                
                # Small delay for real-time feel
                time.sleep(0.1)
            
            # Final evaluation
            st.success("Training completed!")
            
            # Model evaluation
            with st.spinner("Evaluating model..."):
                evaluation_results = trainer.evaluate_model()
                
                with evaluation_placeholder:
                    st.subheader("Model Evaluation")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Classification Report:**")
                        st.text(evaluation_results["classification_report"])
                    
                    with col2:
                        st.write("**Confusion Matrix:**")
                        if PLOTLY_AVAILABLE:
                            fig_cm = px.imshow(
                                evaluation_results["confusion_matrix"],
                                text_auto=True,
                                title="Confusion Matrix"
                            )
                            st.plotly_chart(fig_cm, use_container_width=True)
                        else:
                            fig_cm, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(evaluation_results["confusion_matrix"], annot=True, 
                                      fmt='d', cmap='Blues', ax=ax)
                            ax.set_title("Confusion Matrix")
                            ax.set_xlabel("Predicted")
                            ax.set_ylabel("Actual")
                            st.pyplot(fig_cm)
            
            # Reset training flag
            st.session_state.start_training = False
            if hasattr(st.session_state, 'stop_training'):
                st.session_state.stop_training = False
                
        except Exception as e:
            st.error(f"Training error: {e}")
            st.session_state.start_training = False
    
    # Save model logic
    if hasattr(st.session_state, 'save_model') and st.session_state.save_model:
        try:
            if st.session_state.trained_model is not None and st.session_state.model_info is not None:
                # Create saved_models directory if it doesn't exist
                os.makedirs("saved_models", exist_ok=True)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_type = st.session_state.model_info['model_type']
                dataset_name = st.session_state.model_info['dataset_name']
                filename = f"{model_type}_{dataset_name}_{timestamp}.pth"
                save_path = os.path.join("saved_models", filename)
                
                # Save the model state dict
                torch.save(st.session_state.trained_model.state_dict(), save_path)
                
                # Also save model metadata
                metadata = {
                    'model_type': model_type,
                    'dataset_name': dataset_name,
                    'timestamp': timestamp,
                    'model_state_dict_file': filename
                }
                metadata_path = os.path.join("saved_models", f"{model_type}_{dataset_name}_{timestamp}_metadata.json")
                
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                st.success(f"✅ Model saved successfully!")
                st.info(f"**Model file:** `{save_path}`")
                st.info(f"**Metadata file:** `{metadata_path}`")
                
            else:
                st.warning("⚠️ No trained model to save. Please train a model first.")
                
        except Exception as e:
            st.error(f"❌ Error saving model: {str(e)}")
            st.error("Please check that you have write permissions in the project directory.")
            
        finally:
            st.session_state.save_model = False

def get_dataset_info(dataset_name):
    """Get information about the selected dataset."""
    info = {
        "MNIST": {
            "Description": "Handwritten digit recognition",
            "Classes": 10,
            "Training samples": 60000,
            "Test samples": 10000,
            "Image size": "28x28 grayscale",
            "Task": "Multi-class classification"
        },
        "CIFAR-10": {
            "Description": "Object recognition in color images",
            "Classes": 10,
            "Training samples": 50000,
            "Test samples": 10000,
            "Image size": "32x32 RGB",
            "Task": "Multi-class classification"
        },
        "IMDB Sentiment": {
            "Description": "Movie review sentiment analysis",
            "Classes": 2,
            "Training samples": 25000,
            "Test samples": 25000,
            "Input type": "Text sequences",
            "Task": "Binary classification"
        }
    }
    return info[dataset_name]

def get_model_info(model_type, dataset_name):
    """Get information about the selected model."""
    if dataset_name in ["MNIST", "CIFAR-10"]:
        info = {
            "Simple CNN": {
                "Type": "Convolutional Neural Network",
                "Layers": "2 Conv + 2 FC",
                "Parameters": "~50K",
                "Best for": "Simple image classification"
            },
            "ResNet-18": {
                "Type": "Residual Network",
                "Layers": "18 layers with skip connections",
                "Parameters": "~11M",
                "Best for": "Complex image recognition"
            },
            "Custom CNN": {
                "Type": "Custom CNN Architecture",
                "Layers": "3 Conv + 3 FC",
                "Parameters": "~100K",
                "Best for": "Balanced performance"
            }
        }
    else:  # IMDB
        info = {
            "LSTM": {
                "Type": "Long Short-Term Memory",
                "Layers": "Embedding + LSTM + FC",
                "Parameters": "~1M",
                "Best for": "Sequential text processing"
            },
            "BiLSTM": {
                "Type": "Bidirectional LSTM",
                "Layers": "Embedding + BiLSTM + FC",
                "Parameters": "~2M",
                "Best for": "Context-aware text analysis"
            },
            "Simple RNN": {
                "Type": "Recurrent Neural Network",
                "Layers": "Embedding + RNN + FC",
                "Parameters": "~500K",
                "Best for": "Basic sequence modeling"
            }
        }
    return info[model_type]

def create_initial_plot():
    """Create initial placeholder plot before training starts."""
    if PLOTLY_AVAILABLE:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Loss", "Accuracy"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add placeholder text
        fig.add_annotation(
            text="First epoch in progress...<br>Loss curves will appear here",
            xref="x", yref="y",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray"),
            row=1, col=1
        )
        
        fig.add_annotation(
            text="First epoch in progress...<br>Accuracy curves will appear here",
            xref="x", yref="y",
            x=0.5, y=50, showarrow=False,
            font=dict(size=14, color="gray"),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Training Progress - Initializing..."
        )
        
        fig.update_xaxes(title_text="Epoch", range=[0, 1])
        fig.update_yaxes(title_text="Loss", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", range=[0, 100], row=1, col=2)
        
        return fig
    else:
        # Fallback to matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot placeholder
        ax1.text(0.5, 0.5, 'First epoch in progress...\nLoss curves will appear here', 
                ha='center', va='center', transform=ax1.transAxes, 
                fontsize=12, color='gray')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss - Initializing...")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot placeholder
        ax2.text(0.5, 0.5, 'First epoch in progress...\nAccuracy curves will appear here', 
                ha='center', va='center', transform=ax2.transAxes, 
                fontsize=12, color='gray')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Accuracy - Initializing...")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

def create_training_plots(history):
    """Create training progress plots."""
    if PLOTLY_AVAILABLE:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Loss", "Accuracy"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(
                x=history["epochs"],
                y=history["train_loss"],
                name="Train Loss",
                line=dict(color="blue"),
                mode="lines+markers"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=history["epochs"],
                y=history["val_loss"],
                name="Val Loss",
                line=dict(color="red"),
                mode="lines+markers"
            ),
            row=1, col=1
        )
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(
                x=history["epochs"],
                y=history["train_acc"],
                name="Train Acc",
                line=dict(color="green"),
                mode="lines+markers"
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=history["epochs"],
                y=history["val_acc"],
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
    else:
        # Fallback to matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history["epochs"], history["train_loss"], 'b-', label="Train Loss", marker='o')
        ax1.plot(history["epochs"], history["val_loss"], 'r-', label="Val Loss", marker='s')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history["epochs"], history["train_acc"], 'g-', label="Train Acc", marker='o')
        ax2.plot(history["epochs"], history["val_acc"], 'orange', label="Val Acc", marker='s')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Accuracy")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    main()
