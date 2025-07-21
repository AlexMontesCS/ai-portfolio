# AI Model Training Playground

A web application for training and evaluating machine learning models with interactive hyperparameter tuning and real-time visualization.

## Features

- **Multiple Datasets**: MNIST, CIFAR-10, and Sentiment Analysis (IMDB)
- **Various Model Architectures**: CNNs, ResNets, LSTMs, and Transformers
- **Interactive Hyperparameter Tuning**: Learning rate, batch size, epochs, etc.
- **Live Training Metrics**: Real-time loss and accuracy visualization
- **Model Evaluation**: Confusion matrices, classification reports
- **Model Persistence**: Save and load trained models
- **User-friendly Interface**: Clean Streamlit-based UI

## Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## Project Structure

```
training_playground/
├── app.py                 # Main Streamlit application
├── models/               # Model architectures
│   ├── __init__.py
│   ├── cnn.py           # CNN models
│   ├── resnet.py        # ResNet architectures
│   └── lstm.py          # LSTM/RNN models
├── datasets/            # Dataset handlers
│   ├── __init__.py
│   ├── mnist.py         # MNIST dataset
│   ├── cifar10.py       # CIFAR-10 dataset
│   └── sentiment.py     # Sentiment analysis dataset
├── training/            # Training utilities
│   ├── __init__.py
│   ├── trainer.py       # Main training loop
│   └── metrics.py       # Metrics and visualization
├── utils/               # Utility functions
│   ├── __init__.py
│   └── helpers.py       # Helper functions
└── saved_models/        # Directory for saved models
```

## Model Architectures

- **CNN**: Convolutional Neural Networks for image classification
- **ResNet**: Residual Networks for deep image recognition
- **LSTM**: Long Short-Term Memory networks for sequence data

## Datasets

- **MNIST**: Handwritten digit recognition (28x28 grayscale images)
- **CIFAR-10**: Object recognition (32x32 color images, 10 classes)
- **IMDB**: Sentiment analysis (movie reviews text classification)

## Hyperparameters

Tune the following parameters in real-time:
- Learning rate
- Batch size
- Number of epochs
- Optimizer (Adam, SGD, RMSprop)
- Loss function
- Model-specific parameters

## Getting Started

1. **Test Setup**: Run `python test_setup.py` to verify installation
2. **Quick Demo**: Start with MNIST dataset and Simple CNN model
3. **Experiment**: Try different hyperparameters and observe results
4. **Advanced**: Use ResNet-18 on CIFAR-10 for challenging image classification

## Technical Requirements

- Python 3.7+
- PyTorch 2.0+
- Streamlit 1.28+
- CUDA-compatible GPU (optional, for faster training)