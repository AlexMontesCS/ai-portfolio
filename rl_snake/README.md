# AI Snake: A Deep Reinforcement Learning Playground

This project is an implementation of the Snake game, powered by a Deep Q-Network (DQN) agent. It features a real-time, interactive frontend built with Vue.js that provides detailed insights into the AI's decision-making process.

This project is designed to be a comprehensive learning tool for anyone interested in reinforcement learning, as well as a robust foundation for developing more advanced AI agents.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   **Python 3.9+**: For the backend and AI agent.
*   **Node.js and npm**: For the frontend web application.

### Installation

1.  **Set up the backend:**
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

2.  **Set up the frontend:**
    ```bash
    cd frontend
    npm install
    ```

## Project Structure

The project is organized into a clean, modular architecture that separates the backend, frontend, and shared components.

```
ai-snake/
├── backend/         # Python AI, game engine, and API
│   ├── ai/          # Deep Q-Network implementation
│   ├── api/         # FastAPI web server
│   ├── game/        # Core game logic
│   ├── models/      # Saved AI models
│   └── training/    # Scripts for training the AI
├── frontend/        # Vue.js web application
│   ├── public/      # Static assets
│   └── src/         # Frontend source code
│       ├── assets/
│       ├── components/
│       ├── services/
│       ├── stores/
│       ├── views/
│       └── main.js
└── start_dev.bat    # Development startup script
```

## How to Use

The application is divided into two main sections: **Training** and **Playing**.

### Training the AI

1.  **Navigate to the Training Page:** Open the application and go to the "Train" tab.
2.  **Configure Training Parameters:** Adjust the learning rate, number of episodes, and other hyperparameters to customize the training process.
3.  **Start Training:** Click the "Start Training" button to begin the training session.
4.  **Monitor Progress:** Observe the real-time training graph to see the agent's average score improve over time.

### Playing the Game

1.  **Navigate to the Play Page:** Go to the "Play" tab to interact with the trained agent.
2.  **Select a Model:** Choose a trained model from the dropdown menu.
3.  **Watch the AI Play:** Observe the agent as it plays the game, and see its decision-making process in the AI Visualization panel.
4.  **Play Manually:** You can also play the game yourself to compare your skills against the AI.

##  Features

This project is more than just a game; it's a comprehensive platform for exploring and understanding reinforcement learning.

### The AI Agent

*   **Deep Q-Network (DQN):** A powerful and well-established reinforcement learning algorithm.
*   **Prioritized Experience Replay (PER):** A sophisticated technique that allows the agent to learn from its most significant experiences, leading to faster and more effective training.
*   **Configurable Hyperparameters:** Easily tune the agent's learning rate, exploration strategy, and other parameters to experiment with different training approaches.

### The User Interface

*   **Real-time Visualization:** See the agent's "thoughts" in real-time, including the Q-values for each possible action.
*   **Interactive Training:** A dedicated training page with a live-updating graph of the agent's performance.
*   **Model Management:** Easily switch between different trained models to compare their performance.