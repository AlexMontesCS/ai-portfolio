#!/usr/bin/env python3
"""
DQN Training Script for Snake AI

This script trains a Deep Q-Network agent to play Snake using reinforcement learning.
The training includes real-time monitoring, model checkpointing, and performance visualization.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ai import DQNAgent
from game import SnakeGame
from config import ai_config, game_config

class TrainingMonitor:
    """Monitor and visualize training progress"""
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training metrics
        self.episode_scores = []
        self.episode_lengths = []
        self.episode_losses = []
        self.episode_epsilons = []
        self.moving_averages = []
        
    def update(self, episode: int, score: int, steps: int, 
               avg_loss: float, epsilon: float):
        """Update training metrics"""
        self.episode_scores.append(score)
        self.episode_lengths.append(steps)
        self.episode_losses.append(avg_loss)
        self.episode_epsilons.append(epsilon)
        
        # Calculate moving average (last 100 episodes)
        recent_scores = self.episode_scores[-100:]
        moving_avg = np.mean(recent_scores)
        self.moving_averages.append(moving_avg)
    
    def plot_training_progress(self, save_path: str = None):
        """Create training progress plots"""
        if len(self.episode_scores) < 10:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = range(len(self.episode_scores))
        
        # Score progression
        ax1.plot(episodes, self.episode_scores, alpha=0.6, label='Episode Score')
        ax1.plot(episodes, self.moving_averages, 'r-', label='Moving Average (100)')
        ax1.set_title('Training Score Progress')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True)
        
        # Episode length
        ax2.plot(episodes, self.episode_lengths, 'g-', alpha=0.7)
        ax2.set_title('Episode Length (Steps)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True)
        
        # Loss progression
        if self.episode_losses:
            ax3.plot(episodes, self.episode_losses, 'orange', alpha=0.7)
            ax3.set_title('Training Loss')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Average Loss')
            ax3.grid(True)
        
        # Epsilon decay
        ax4.plot(episodes, self.episode_epsilons, 'purple', alpha=0.7)
        ax4.set_title('Epsilon Decay')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def save_metrics(self, filepath: str):
        """Save training metrics to JSON"""
        metrics = {
            'episode_scores': self.episode_scores,
            'episode_lengths': self.episode_lengths,
            'episode_losses': self.episode_losses,
            'episode_epsilons': self.episode_epsilons,
            'moving_averages': self.moving_averages
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

def train_dqn_agent(episodes: int = None, save_frequency: int = None, 
                   resume_training: bool = False,
                   verbose: bool = True) -> DQNAgent:
    """
    Train DQN agent to play Snake
    
    Args:
        episodes: Number of training episodes
        save_frequency: Save model every N episodes
        resume_training: Continue from existing model
        verbose: Print training progress
        
    Returns:
        Trained DQN agent
    """
    
    # Configuration
    episodes = episodes or ai_config.MAX_EPISODES
    save_frequency = save_frequency or ai_config.SAVE_FREQUENCY
    model_dir = Path(ai_config.MODEL_SAVE_PATH)  # Always use config value
    model_dir.mkdir(parents=True, exist_ok=True)

    # Enable PyTorch anomaly detection for gradient debugging
    import torch
    torch.autograd.set_detect_anomaly(True)

    # Initialize environment and agent
    env = SnakeGame()
    agent = DQNAgent()

    # Training monitor
    monitor = TrainingMonitor(model_dir / "training_logs")

    # Resume training if requested
    if resume_training:
        model_path = model_dir / "snake_dqn_latest.pth"
        if model_path.exists():
            if verbose:
                print(f"Resuming training from {model_path}")
            agent.load_model(str(model_path))
        else:
            print("No existing model found. Starting fresh training.")

    # Training statistics
    best_avg_score = -float('inf')
    episode_losses = []

    if verbose:
        print(f"Starting DQN training for {episodes} episodes")
        print(f"Device: {agent.device}")
        print(f"Model will be saved to: {model_dir}")
        print("-" * 50)

    try:
        for episode in tqdm(range(episodes), desc="Training Episodes"):
            # Reset environment
            state = env.reset()
            total_reward = 0
            steps = 0
            episode_loss = []
            
            while True:
                # Choose action
                action = agent.act(state, training=True)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                total_reward += reward
                steps += 1
                
                # Train agent
                if agent.is_ready_for_training():
                    loss = agent.replay()
                    if loss is not None:
                        episode_loss.append(loss)
                
                if done:
                    break
            
            # Update episode statistics
            agent.reset_episode(info['score'])
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            episode_losses.append(avg_loss)
            
            # Update monitor
            monitor.update(episode, info['score'], steps, avg_loss, agent.get_epsilon())
            
            # Print progress
            if verbose and episode % 100 == 0:
                stats = agent.get_training_stats()
                print(f"Episode {episode}: Score={info['score']}, "
                    f"Avg Score={stats['avg_score']:.2f}, "
                    f"Epsilon={agent.get_epsilon():.3f}, "
                    f"Loss={avg_loss:.4f}")
            
            # Save model periodically
            if episode % save_frequency == 0 and episode > 0:
                # Save latest model
                latest_path = model_dir / "snake_dqn_latest.pth"
                agent.save_model(str(latest_path))
                
                # Save checkpoint
                checkpoint_path = model_dir / f"snake_dqn_episode_{episode}.pth"
                agent.save_model(str(checkpoint_path))
                
                # Save best model if improved
                current_avg = np.mean(monitor.episode_scores[-100:])
                if current_avg > best_avg_score:
                    best_avg_score = current_avg
                    best_path = model_dir / "snake_dqn_best.pth"
                    agent.save_model(str(best_path))
                    if verbose:
                        print(f"New best average score: {best_avg_score:.2f}")
                
                # Save training plots
                plot_path = model_dir / "training_logs" / f"progress_episode_{episode}.png"
                monitor.plot_training_progress(str(plot_path))
                
                # Save metrics
                metrics_path = model_dir / "training_logs" / "training_metrics.json"
                monitor.save_metrics(str(metrics_path))
    
    except KeyboardInterrupt:
        if verbose:
            print("\nTraining interrupted by user")
    
    # Final save
    final_path = model_dir / "snake_dqn_final.pth"
    agent.save_model(str(final_path))
    
    # Final plots and metrics
    monitor.plot_training_progress(str(model_dir / "training_logs" / "final_progress.png"))
    monitor.save_metrics(str(model_dir / "training_logs" / "final_metrics.json"))
    
    if verbose:
        stats = agent.get_training_stats()
        print(f"\nTraining completed!")
        print(f"Final average score: {stats['avg_score']:.2f}")
        print(f"Total episodes: {stats['episodes']}")
        print(f"Total steps: {stats['steps']}")
        print(f"Model saved to: {final_path}")
    
    return agent

def evaluate_agent(agent: DQNAgent, episodes: int = 10, render: bool = False) -> Dict[str, float]:
    """Evaluate trained agent performance"""
    env = SnakeGame()
    scores = []
    steps_list = []
    
    print(f"Evaluating agent for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        total_steps = 0
        
        while True:
            action = agent.act(state, training=False)  # No exploration
            state, _, done, info = env.step(action)
            total_steps += 1
            
            if render and episode == 0:  # Render first episode
                print(f"Step {total_steps}: Score={info['score']}, Action={action}")
            
            if done:
                scores.append(info['score'])
                steps_list.append(total_steps)
                break
    
    # Calculate statistics
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    max_score = np.max(scores)
    avg_steps = np.mean(steps_list)
    
    results = {
        'average_score': avg_score,
        'std_score': std_score,
        'max_score': max_score,
        'average_steps': avg_steps,
        'scores': scores
    }
    
    print(f"Evaluation Results:")
    print(f"  Average Score: {avg_score:.2f} ± {std_score:.2f}")
    print(f"  Max Score: {max_score}")
    print(f"  Average Steps: {avg_steps:.1f}")
    
    return results

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train DQN agent for Snake game')
    parser.add_argument('--episodes', type=int, default=ai_config.MAX_EPISODES,
                       help='Number of training episodes')
    parser.add_argument('--save-freq', type=int, default=ai_config.SAVE_FREQUENCY,
                       help='Save model every N episodes')
    parser.add_argument('--model-dir', type=str, default=ai_config.MODEL_SAVE_PATH,
                       help='Directory to save models')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from existing model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate existing model instead of training')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    if args.evaluate:
        # Load and evaluate existing model
        model_path = Path(args.model_dir) / "snake_dqn_best.pth"
        if not model_path.exists():
            model_path = Path(args.model_dir) / "snake_dqn_latest.pth"
        
        if not model_path.exists():
            print(f"No trained model found in {args.model_dir}")
            return
        
        agent = DQNAgent()
        agent.load_model(str(model_path))
        evaluate_agent(agent, args.eval_episodes, render=not args.quiet)
    else:
        # Train new agent
        agent = train_dqn_agent(
            episodes=args.episodes,
            save_frequency=args.save_freq,
            resume_training=args.resume,
            verbose=not args.quiet
        )
        
        # Quick evaluation
        print("\nQuick evaluation of trained agent:")
        evaluate_agent(agent, episodes=5)

def start_training_session(agent: DQNAgent, config: dict, callback=None, stop_event=None):
    """
    Simplified training session for API calls
    
    Args:
        agent: DQN agent to train
        config: Training configuration dictionary
        callback: Optional callback function for progress updates
        stop_event: A threading.Event to signal when to stop training
    """
    episodes = config.get('episodes', 1000)
    learning_rate = config.get('learning_rate', 0.001)
    batch_size = config.get('batch_size', 32)
    memory_size = config.get('memory_size', 10000)
    
    # Update agent configuration
    agent.learning_rate = learning_rate
    agent.batch_size = batch_size
    agent.memory.maxlen = memory_size
    
    # Simple training loop
    game = SnakeGame()
    scores = []
    
    model_dir = Path(ai_config.MODEL_SAVE_PATH)
    model_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(episodes):
        if stop_event and stop_event.is_set():
            print("Training stopped by user.")
            break
        state = game.reset()
        episode_score = 0
        step_count = 0

        while not game.is_game_over() and step_count < 1000:  # Prevent infinite loops
            action = agent.act(state, training=True)
            next_state, reward, done, info = game.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            step_count += 1
            if done:
                episode_score = info['score']
                break
            
            # Train after each step
            if len(agent.memory) > batch_size:
                agent.replay()

        scores.append(episode_score)
        agent.reset_episode(episode_score)

        # Calculate average score, loss, and other stats for the episode
        loss = np.mean(agent.losses[-step_count:]) if step_count > 0 else 0
        avg_score = np.mean(scores[-100:]) if scores else 0

        # Call callback for progress updates
        if callback:
            try:
                callback(episode + 1, episode_score, avg_score, agent.get_epsilon(), loss)
            except:
                pass  # Don't let callback errors stop training

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Score: {episode_score}, Avg Score: {avg_score:.2f}, Epsilon: {agent.get_epsilon():.3f}")

        # Save model periodically
        save_frequency = config.get('save_frequency', 100)
        if (episode + 1) % save_frequency == 0:
            try:
                agent.save_model(str(model_dir / f"snake_dqn_episode_{episode + 1}.pth"))
            except:
                pass  # Don't let save errors stop training

    # Save final model
    try:
        agent.save_model(str(model_dir / "snake_dqn_final.pth"))
        print(f"Training completed! Final model saved.")
    except:
        print("Training completed but failed to save final model.")

if __name__ == "__main__":
    main()
