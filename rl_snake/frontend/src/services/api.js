import axios from 'axios'

// Create axios instance with base configuration
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add any auth headers here if needed
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    const message = error.response?.data?.detail || error.message || 'An error occurred'
    console.error('API Error:', message)
    return Promise.reject(new Error(message))
  }
)

// Game API endpoints
export const gameApi = {
  // Create new game
  createGame: () => api.post('/game/new'),
  
  // Get game state
  getGameState: (gameId) => api.get(`/game/${gameId}/state`),
  
  // Make action
  makeAction: (gameId, action) => api.post(`/game/${gameId}/action`, { action }),
  
  // Reset game
  resetGame: (gameId) => api.post(`/game/${gameId}/reset`),
  
  // Get AI info
  getAiInfo: () => api.get('/ai/info'),
  
  // Get available models
  getAvailableModels: () => api.get('/ai/models'),
  
  // Load specific model
  loadModel: (filename) => api.post('/ai/load-model', { filename }),
  
  // Get AI action
  getAiAction: (gameId) => api.post(`/ai/action/${gameId}`),
  
  // AI play step
  aiPlayStep: (gameId) => api.post(`/ai/play/${gameId}`)
}

// Health check API
export const healthApi = {
  // Health check
  healthCheck: () => api.get('/health'),
  
  // Test connection
  testConnection: async () => {
    try {
      const response = await api.get('/health')
      console.log('Backend health check:', response)
      return response
    } catch (error) {
      console.error('Backend connection failed:', error)
      throw error
    }
  }
}


// Training API endpoints (when implemented)
export const trainingApi = {
  // Start training
  startTraining: (config) => api.post('/training/start', config),
  
  // Stop training
  stopTraining: () => api.post('/training/stop'),
  
  // Reset training state
  resetTraining: () => api.post('/training/reset'),
  
  // Get training status
  getTrainingStatus: () => api.get('/training/status'),
  
  // Get training metrics
  getTrainingMetrics: () => api.get('/training/metrics'),
  
  // Save model
  saveModel: (name) => api.post('/training/save', { name }),
  
  // Load model
  loadModel: (name) => api.post('/training/load', { name })
}

export default api
