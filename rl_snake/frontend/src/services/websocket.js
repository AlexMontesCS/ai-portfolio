import { io } from 'socket.io-client'

class WebSocketService {
  constructor() {
    this.socket = null
    this.gameId = null
    this.isConnected = false
    this.messageHandlers = []
    this.connectHandlers = []
    this.disconnectHandlers = []
    this.errorHandlers = []
  }

  connect(gameId) {
    if (this.socket) {
      this.disconnect();
    }
    this.gameId = gameId;
    const baseWsUrl = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/^http/, 'ws');
    const wsUrl = `${baseWsUrl}/ws/game/${gameId}`;
    this.socket = new WebSocket(wsUrl);
    this.socket.onopen = () => {
      this.isConnected = true
      console.log('WebSocket connected')
      this.connectHandlers.forEach(handler => handler())
    }
    
    this.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        this.messageHandlers.forEach(handler => handler(data))
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }
    
    this.socket.onclose = () => {
      this.isConnected = false
      console.log('WebSocket disconnected')
      this.disconnectHandlers.forEach(handler => handler())
    }
    
    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error)
      this.errorHandlers.forEach(handler => handler(error))
    }
  }

  connectToTraining() {
    if (this.socket) {
      this.disconnect();
    }
    const baseWsUrl = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/^http/, 'ws');
    const wsUrl = `${baseWsUrl}/ws/training`;
    this.socket = new WebSocket(wsUrl);
    this.socket.onopen = () => {
      this.isConnected = true;
      console.log('Training WebSocket connected');
      this.connectHandlers.forEach(handler => handler());
    };
    this.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.messageHandlers.forEach(handler => handler(data));
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    this.socket.onclose = () => {
      this.isConnected = false;
      console.log('Training WebSocket disconnected');
      this.disconnectHandlers.forEach(handler => handler());
    };
    this.socket.onerror = (error) => {
      console.error('Training WebSocket error:', error);
      this.errorHandlers.forEach(handler => handler(error));
    };
  }

  disconnect() {
    if (this.socket) {
      this.socket.close()
      this.socket = null
      this.isConnected = false
      this.gameId = null
    }
  }

  sendMessage(message) {
    if (this.socket && this.isConnected) {
      this.socket.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket not connected')
    }
  }

  sendAction(action) {
    this.sendMessage({
      type: 'action',
      action: action
    })
  }

  sendAiPlay() {
    this.sendMessage({
      type: 'ai_play'
    })
  }

  sendReset() {
    this.sendMessage({
      type: 'reset'
    })
  }

  // Event handlers
  onMessage(handler) {
    this.messageHandlers.push(handler)
  }

  onConnect(handler) {
    this.connectHandlers.push(handler)
  }

  onDisconnect(handler) {
    this.disconnectHandlers.push(handler)
  }

  onError(handler) {
    this.errorHandlers.push(handler)
  }

  // Remove event handlers
  removeMessageHandler(handler) {
    const index = this.messageHandlers.indexOf(handler)
    if (index > -1) {
      this.messageHandlers.splice(index, 1)
    }
  }

  removeConnectHandler(handler) {
    const index = this.connectHandlers.indexOf(handler)
    if (index > -1) {
      this.connectHandlers.splice(index, 1)
    }
  }

  removeDisconnectHandler(handler) {
    const index = this.disconnectHandlers.indexOf(handler)
    if (index > -1) {
      this.disconnectHandlers.splice(index, 1)
    }
  }

  removeErrorHandler(handler) {
    const index = this.errorHandlers.indexOf(handler)
    if (index > -1) {
      this.errorHandlers.splice(index, 1)
    }
  }
}

// Create singleton instance
export const websocketService = new WebSocketService()

export default websocketService
