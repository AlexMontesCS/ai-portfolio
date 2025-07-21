from fastapi import WebSocket
from typing import Dict, List
import json

class ConnectionManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, game_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        if game_id not in self.active_connections:
            self.active_connections[game_id] = []
        
        self.active_connections[game_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, game_id: str):
        """Remove WebSocket connection"""
        if game_id in self.active_connections:
            if websocket in self.active_connections[game_id]:
                self.active_connections[game_id].remove(websocket)
            
            # Clean up empty game rooms
            if not self.active_connections[game_id]:
                del self.active_connections[game_id]
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_text(message)
        except:
            # Connection closed, ignore error
            pass
    
    async def broadcast_to_game(self, message: str, game_id: str):
        """Broadcast message to all connections for a specific game"""
        if game_id in self.active_connections:
            disconnected = []
            
            for websocket in self.active_connections[game_id]:
                try:
                    await websocket.send_text(message)
                except:
                    disconnected.append(websocket)
            
            # Remove disconnected WebSockets
            for websocket in disconnected:
                self.active_connections[game_id].remove(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast a message to all connected clients."""
        for connections in self.active_connections.values():
            for connection in connections:
                await connection.send_text(message)

    async def broadcast_to_all(self, message: str):
        """Broadcast message to all active connections"""
        for game_id in list(self.active_connections.keys()):
            await self.broadcast_to_game(message, game_id)
    
    def get_connection_count(self, game_id: str = None) -> int:
        """Get number of active connections"""
        if game_id:
            return len(self.active_connections.get(game_id, []))
        
        return sum(len(connections) for connections in self.active_connections.values())
    
    def get_active_games(self) -> List[str]:
        """Get list of games with active connections"""
        return list(self.active_connections.keys())
