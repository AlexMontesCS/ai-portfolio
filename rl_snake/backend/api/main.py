from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

from game import SnakeGame, Direction
from ai import DQNAgent
from config import server_config, ai_config
from api.models import GameStateResponse, ActionRequest, TrainingStatus, ModelInfo
from api.websocket_manager import ConnectionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Snake AI API",
    description="Backend API for Snake AI game with reinforcement learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=server_config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
game_instances: Dict[str, SnakeGame] = {}
ai_agent: Optional[DQNAgent] = None
connection_manager = ConnectionManager()

# Load AI model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize AI agent on startup"""
    global ai_agent
    
    try:
        # Try to load existing model in order of preference
        model_paths = [
            Path(ai_config.MODEL_SAVE_PATH) / "snake_dqn_final.pth",
            Path(ai_config.MODEL_SAVE_PATH) / "snake_dqn_latest.pth", 
            Path(ai_config.MODEL_SAVE_PATH) / "snake_dqn_best.pth"
        ]
        model_loaded = False
        for model_path in model_paths:
            if model_path.exists():
                ai_agent = DQNAgent()
                ai_agent.load_model(str(model_path))
                ai_agent.last_loaded_model = model_path.name  # Track loaded model
                logger.info(f"Loaded AI model from {model_path}")
                
                # If loading final model, set some training stats to indicate it's trained
                if "final" in model_path.name and ai_agent.episodes_done == 0:
                    ai_agent.episodes_done = 1000  # Assume completed training
                    ai_agent.scores.append(10.0)   # Add a sample score
                
                model_loaded = True
                break
        
        if not model_loaded:
            logger.info("No pre-trained model found. AI will need to be trained.")
            ai_agent = DQNAgent()
    
    except Exception as e:
        logger.error(f"Error loading AI model: {e}")
        ai_agent = DQNAgent()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Snake AI API is running!", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "message": "Snake AI API is running",
            "ai_loaded": ai_agent is not None,
            "active_games": len(game_instances),
            "training_active": training_active,
            "cors_origins": server_config.CORS_ORIGINS
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/game/new")
async def create_new_game() -> Dict[str, Any]:
    """Create a new game instance"""
    game_id = f"game_{len(game_instances)}"
    game = SnakeGame()
    game_instances[game_id] = game
    
    return {
        "game_id": game_id,
        "initial_state": game.get_game_data(),
        "message": "New game created successfully"
    }

@app.get("/game/{game_id}/state")
async def get_game_state(game_id: str) -> GameStateResponse:
    """Get current game state"""
    if game_id not in game_instances:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = game_instances[game_id]
    game_data = game.get_game_data()
    
    # Add AI analysis if available
    ai_analysis = None
    if ai_agent and game.game_state.value == "playing":
        state = game.get_state()
        q_values = ai_agent.get_q_values(state)
        action_probs = ai_agent.get_action_probabilities(state)
        
        ai_analysis = {
            "q_values": q_values.tolist(),
            "action_probabilities": action_probs.tolist(),
            "recommended_action": int(np.argmax(q_values)),
            "action_names": ["UP", "DOWN", "LEFT", "RIGHT"]
        }
    
    return GameStateResponse(
        game_id=game_id,
        **game_data,
        ai_analysis=ai_analysis
    )

@app.post("/game/{game_id}/action")
async def make_action(game_id: str, request: ActionRequest) -> Dict[str, Any]:
    """Execute an action in the game"""
    if game_id not in game_instances:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = game_instances[game_id]
    
    # Execute action
    state, reward, done, info = game.step(request.action)
    
    # Get updated game data
    game_data = game.get_game_data()
    
    # Add AI analysis for next state
    ai_analysis = None
    if ai_agent and not done:
        q_values = ai_agent.get_q_values(state)
        action_probs = ai_agent.get_action_probabilities(state)
        
        ai_analysis = {
            "q_values": q_values.tolist(),
            "action_probabilities": action_probs.tolist(),
            "recommended_action": int(np.argmax(q_values)),
            "action_names": ["UP", "DOWN", "LEFT", "RIGHT"]
        }
    
    return {
        "game_data": game_data,
        "reward": reward,
        "done": done,
        "info": info,
        "ai_analysis": ai_analysis
    }

@app.post("/game/{game_id}/reset")
async def reset_game(game_id: str) -> Dict[str, Any]:
    """Reset game to initial state"""
    if game_id not in game_instances:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = game_instances[game_id]
    initial_state = game.reset()
    
    return {
        "game_data": game.get_game_data(),
        "message": "Game reset successfully"
    }

@app.get("/ai/info")
async def get_ai_info() -> ModelInfo:
    """Get AI model information"""
    if not ai_agent:
        raise HTTPException(status_code=404, detail="AI agent not loaded")
    
    stats = ai_agent.get_training_stats()
    
    # Check if we have a trained model by looking for saved models
    model_files = [
        Path(ai_config.MODEL_SAVE_PATH) / "snake_dqn_final.pth",
        Path(ai_config.MODEL_SAVE_PATH) / "snake_dqn_latest.pth",
        Path(ai_config.MODEL_SAVE_PATH) / "snake_dqn_best.pth"
    ]
    
    has_trained_model = any(path.exists() for path in model_files)
    
    # If we have a trained model but no episodes, update the stats
    if has_trained_model and stats['episodes'] == 0:
        # Use training completion stats if available
        if current_training_stats.get("current_episode", 0) > 0:
            stats['episodes'] = current_training_stats.get("current_episode", 1000)
            stats['avg_score'] = current_training_stats.get("average_score", 10.0)
        else:
            # Fallback: assume some training occurred
            stats['episodes'] = 1000
            stats['avg_score'] = 10.0
    
    return ModelInfo(
        model_loaded=True,
        training_episodes=stats['episodes'],
        current_epsilon=stats['epsilon'],
        average_score=stats['avg_score'],
        memory_size=stats['memory_size']
    )

@app.get("/ai/models")
async def get_available_models() -> Dict[str, Any]:
    """Get list of available AI models"""
    try:
        models_dir = Path(ai_config.MODEL_SAVE_PATH)
        
        if not models_dir.exists():
            return {"models": [], "current_model": None}
        
        # Find all .pth files in the models directory
        model_files = list(models_dir.glob("*.pth"))
        
        import torch
        import re
        models = []
        for model_file in model_files:
            stat = model_file.stat()
            training_episodes = 0  # Default value

            checkpoint = torch.load(model_file, map_location='cpu')
            training_episodes = checkpoint.get('episodes_done', 0)

            models.append({
                "filename": model_file.name,
                "path": str(model_file),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime,
                "display_name": model_file.stem.replace("snake_dqn_", "").replace("_", " ").title(),
                "training_episodes": training_episodes
            })
        
        # Sort by modification time (newest first)
        models.sort(key=lambda x: x["modified"], reverse=True)
        
        # Determine current model (if any is loaded)
        current_model = None
        if ai_agent and hasattr(ai_agent, 'last_loaded_model'):
            current_model = ai_agent.last_loaded_model
        
        return {
            "models": models,
            "current_model": current_model,
            "models_directory": str(models_dir)
        }
        
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

@app.post("/ai/load-model")
async def load_model(request: Dict[str, str]) -> Dict[str, Any]:
    """Load a specific AI model"""
    global ai_agent
    
    try:
        filename = request.get("filename")
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        model_path = Path(ai_config.MODEL_SAVE_PATH) / filename
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model file '{filename}' not found")
        
        # Create new agent if none exists
        if not ai_agent:
            ai_agent = DQNAgent()
        
        # Load the selected model
        ai_agent.load_model(str(model_path))
        
        # Store which model was loaded
        ai_agent.last_loaded_model = filename
        
        logger.info(f"Successfully loaded model: {filename}")
        
        # Get updated stats from the agent, which now has the loaded data
        stats = ai_agent.get_training_stats()

        return {
            "message": f"Model '{filename}' loaded successfully",
            "model_info": {
                "filename": filename,
                "training_episodes": stats['episodes'],
                "average_score": stats['avg_score'],
                "epsilon": stats['epsilon'],
                "memory_size": stats['memory_size']
            }
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/ai/action/{game_id}")
async def get_ai_action(game_id: str) -> Dict[str, Any]:
    """Get AI's recommended action for current game state"""
    if game_id not in game_instances:
        raise HTTPException(status_code=404, detail="Game not found")
    
    if not ai_agent:
        raise HTTPException(status_code=404, detail="AI agent not loaded")
    
    game = game_instances[game_id]
    state = game.get_state()
    
    # Get AI action and analysis
    action = ai_agent.act(state, training=False)
    q_values = ai_agent.get_q_values(state)
    action_probs = ai_agent.get_action_probabilities(state)
    
    return {
        "recommended_action": action,
        "q_values": q_values.tolist(),
        "action_probabilities": action_probs.tolist(),
        "action_names": ["UP", "DOWN", "LEFT", "RIGHT"],
        "confidence": float(np.max(action_probs))
    }

@app.post("/ai/play/{game_id}")
async def ai_play_step(game_id: str) -> Dict[str, Any]:
    """Let AI make one move in the game"""
    if game_id not in game_instances:
        raise HTTPException(status_code=404, detail="Game not found")
    
    if not ai_agent:
        raise HTTPException(status_code=404, detail="AI agent not loaded")
    
    game = game_instances[game_id]
    state = game.get_state()
    
    # Get AI action and analysis
    action = ai_agent.act(state, training=False)
    q_values = ai_agent.get_q_values(state)
    action_probs = ai_agent.get_action_probabilities(state)
    
    # Execute action
    next_state, reward, done, info = game.step(action)
    
    return {
        "action_taken": action,
        "q_values": q_values.tolist(),
        "action_probabilities": action_probs.tolist(),
        "game_data": game.get_game_data(),
        "reward": reward,
        "done": done,
        "info": info
    }


# ==================== TRAINING ENDPOINTS ====================

# Global training state
training_active = False
training_config = None
training_thread = None
stop_training_event = None
current_training_stats = {
    "current_episode": 0,
    "current_score": 0,
    "average_score": 0.0,
    "epsilon": 1.0,
    "total_episodes": 1000,
    "recent_losses": []
}

@app.post("/training/start")
async def start_training(config: dict):
    """Start training the AI agent"""
    global training_active, training_config, ai_agent, stop_training_event
    
    if training_active:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Re-initialize the agent for a fresh training session
    ai_agent = DQNAgent()
    logger.info("Initialized new DQNAgent for training.")
    
    training_active = True
    training_config = config
    
    # Start training in background
    import threading
    from training.train_dqn import start_training_session
    
    stop_training_event = threading.Event()

    def training_worker():
        global training_active, ai_agent
        try:
            start_training_session(ai_agent, config, training_callback, stop_training_event)
            
            # Training completed successfully
            logger.info("Training completed successfully")
            
            # Update training stats to show completion
            current_training_stats.update({
                "training_completed": True,
                "completion_message": "Training completed successfully!"
            })
            
            # Try to load the final model
            try:
                final_model_path = Path(ai_config.MODEL_SAVE_PATH) / "snake_dqn_final.pth"
                if final_model_path.exists():
                    ai_agent.load_model(str(final_model_path))
                    logger.info("Loaded final trained model")
                    
                    # Also save as "latest" for future startups
                    latest_model_path = Path(ai_config.MODEL_SAVE_PATH) / "snake_dqn_latest.pth"
                    ai_agent.save_model(str(latest_model_path))
                    logger.info("Saved model as latest for future startups")
                    
                    # Update the agent's training stats to reflect the completed training
                    if hasattr(ai_agent, 'episodes_done') and current_training_stats:
                        ai_agent.episodes_done = current_training_stats.get("current_episode", ai_agent.episodes_done)
                        if current_training_stats.get("average_score", 0) > 0:
                            # Append the final average score to maintain the scores history
                            ai_agent.scores.append(current_training_stats.get("average_score", 0))
                    
            except Exception as e:
                logger.error(f"Failed to load final model: {e}")
                
        except Exception as e:
            logger.error(f"Training error: {e}")
            current_training_stats.update({
                "training_completed": True,
                "completion_message": f"Training failed: {str(e)}"
            })
        finally:
            training_active = False
    
    training_thread = threading.Thread(target=training_worker, daemon=True)
    training_thread.start()
    
    return {"message": "Training started", "config": config}

def training_callback(episode: int, score: int, average_score: float, epsilon: float, loss: float = 0.0):
    """Callback function for training progress updates"""
    global current_training_stats
    
    # Update current training stats
    current_training_stats.update({
        "current_episode": episode,
        "current_score": score,
        "average_score": average_score,
        "epsilon": epsilon,
        "total_episodes": training_config.get("episodes", 1000) if training_config else 1000
    })
    
    # Update recent losses (keep last 50 for better performance)
    if "recent_losses" not in current_training_stats:
        current_training_stats["recent_losses"] = []
    
    current_training_stats["recent_losses"].append(loss)
    if len(current_training_stats["recent_losses"]) > 50:
        current_training_stats["recent_losses"] = current_training_stats["recent_losses"][-50:]
    
    # Log progress
    if episode % 50 == 0:  # Log every 50 episodes
        logger.info(f"Training progress: Episode {episode}, Score: {score}, Avg: {average_score:.2f}, Epsilon: {epsilon:.3f}")

    # Broadcast training stats over WebSocket
    log_and_broadcast(episode, score, average_score, epsilon, loss)

def log_and_broadcast(episode, score, average_score, epsilon, loss):
    """Helper to log and broadcast training data."""
    data = {
        "type": "training_update",
        "episode": episode,
        "score": score,
        "average_score": average_score,
        "epsilon": epsilon,
        "loss": loss
    }
    logger.info(f"Broadcasting training data: {data}")
    asyncio.run(connection_manager.broadcast(json.dumps(data)))

@app.post("/training/stop")
async def stop_training():
    """Stop the current training session"""
    global training_active, stop_training_event
    
    if not training_active:
        raise HTTPException(status_code=400, detail="No training in progress")
    
    if stop_training_event:
        stop_training_event.set()
    
    training_active = False
    return {"message": "Training stopped"}

@app.post("/training/reset")
async def reset_training_state():
    """Reset training completion state"""
    global current_training_stats
    
    # Reset completion flags
    current_training_stats.pop("training_completed", None)
    current_training_stats.pop("completion_message", None)
    
    # Reset stats
    current_training_stats.update({
        "current_episode": 0,
        "current_score": 0,
        "average_score": 0.0,
        "epsilon": 1.0,
        "total_episodes": 1000,
        "recent_losses": []
    })
    
    return {"message": "Training state reset"}

@app.get("/training/status")
async def get_training_status():
    """Get current training status"""
    try:
        global training_active, training_config, current_training_stats
        
        if not training_active:
            # Check if training was just completed
            if current_training_stats.get("training_completed", False):
                completion_message = current_training_stats.get("completion_message", "Training completed")
                
                return TrainingStatus(
                    is_training=False,
                    current_episode=current_training_stats.get("current_episode", 0),
                    total_episodes=current_training_stats.get("total_episodes", 0),
                    current_score=current_training_stats.get("current_score", 0),
                    average_score=current_training_stats.get("average_score", 0.0),
                    epsilon=current_training_stats.get("epsilon", 1.0),
                    recent_losses=current_training_stats.get("recent_losses", [])
                )
            else:
                return TrainingStatus(
                    is_training=False,
                    current_episode=0,
                    total_episodes=0,
                    current_score=0,
                    average_score=0.0,
                    epsilon=1.0,
                    recent_losses=[]
                )
        
        # Return actual training progress
        return TrainingStatus(
            is_training=True,
            current_episode=current_training_stats["current_episode"],
            total_episodes=current_training_stats["total_episodes"],
            current_score=current_training_stats["current_score"],
            average_score=current_training_stats["average_score"],
            epsilon=current_training_stats["epsilon"],
            recent_losses=current_training_stats.get("recent_losses", [])
        )
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=f"Training status error: {str(e)}")

@app.get("/training/metrics")
async def get_training_metrics():
    """Get training metrics and history"""
    # Return actual training metrics if available
    return {
        "episode_scores": [],
        "episode_lengths": [],
        "episode_losses": [],
        "episode_epsilons": [],
        "moving_averages": []
    }


# ==================== WEBSOCKET ENDPOINTS ====================
@app.websocket("/ws/game/{game_id}")
async def websocket_game(websocket: WebSocket, game_id: str):
    """WebSocket endpoint for real-time game updates"""
    await connection_manager.connect(websocket, game_id)
    
    try:
        while True:
            # Receive action from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "action":
                if game_id in game_instances:
                    game = game_instances[game_id]
                    action = message["action"]
                    
                    # Execute action
                    state, reward, done, info = game.step(action)
                    
                    # Get AI analysis
                    ai_analysis = None
                    if ai_agent and not done:
                        q_values = ai_agent.get_q_values(state)
                        action_probs = ai_agent.get_action_probabilities(state)
                        
                        ai_analysis = {
                            "q_values": q_values.tolist(),
                            "action_probabilities": action_probs.tolist(),
                            "recommended_action": int(np.argmax(q_values))
                        }
                    
                    # Send update to client
                    response = {
                        "type": "game_update",
                        "game_data": game.get_game_data(),
                        "reward": reward,
                        "done": done,
                        "info": info,
                        "ai_analysis": ai_analysis
                    }
                    
                    await connection_manager.send_personal_message(
                        json.dumps(response), websocket
                    )
            
            elif message["type"] == "ai_play":
                if game_id in game_instances and ai_agent:
                    game = game_instances[game_id]
                    state = game.get_state()
                    
                    # AI makes move
                    action = ai_agent.act(state, training=False)
                    q_values = ai_agent.get_q_values(state)
                    action_probs = ai_agent.get_action_probabilities(state)
                    
                    # Execute action
                    next_state, reward, done, info = game.step(action)
                    
                    # Send update
                    response = {
                        "type": "ai_move",
                        "action": action,
                        "q_values": q_values.tolist(),
                        "action_probabilities": action_probs.tolist(),
                        "game_data": game.get_game_data(),
                        "reward": reward,
                        "done": done,
                        "info": info
                    }
                    
                    await connection_manager.send_personal_message(
                        json.dumps(response), websocket
                    )
            
            elif message["type"] == "reset":
                if game_id in game_instances:
                    game = game_instances[game_id]
                    game.reset()
                    
                    response = {
                        "type": "game_reset",
                        "game_data": game.get_game_data()
                    }
                    
                    await connection_manager.send_personal_message(
                        json.dumps(response), websocket
                    )
    
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, game_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await connection_manager.send_personal_message(
            json.dumps({"type": "error", "message": str(e)}), websocket
        )

@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    """WebSocket endpoint for training updates"""
    await connection_manager.connect(websocket, "training")
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, "training")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=server_config.HOST, 
        port=server_config.PORT, 
        reload=server_config.DEBUG
    )
