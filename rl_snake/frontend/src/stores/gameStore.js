import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { gameApi } from '../services/api'
import { websocketService } from '../services/websocket'

export const useGameStore = defineStore('game', () => {
  // State
  const gameId = ref(null)
  const gameState = ref({
    snake_body: [],
    food_position: null,
    score: 0,
    steps: 0,
    game_state: 'playing',
    grid_size: [15, 15],
    direction: null
  })
  
  const aiAnalysis = ref({
    q_values: [],
    action_probabilities: [],
    recommended_action: 0,
    action_names: ['UP', 'DOWN', 'LEFT', 'RIGHT'],
    confidence: 0
  })
  
  const isLoading = ref(false)
  const loadingMessage = ref('')
  const error = ref(null)
  const isConnected = ref(false)
  const isAiPlaying = ref(false)
  const gameHistory = ref([])

  // Computed
  const isGameOver = computed(() => gameState.value.game_state === 'game_over')
  const isPlaying = computed(() => gameState.value.game_state === 'playing')
  const gridWidth = computed(() => gameState.value.grid_size[0])
  const gridHeight = computed(() => gameState.value.grid_size[1])
  
  const actionName = computed(() => {
    if (aiAnalysis.value.recommended_action !== undefined) {
      return aiAnalysis.value.action_names[aiAnalysis.value.recommended_action]
    }
    return 'NONE'
  })

  // Actions
  const setLoading = (loading, message = '') => {
    isLoading.value = loading
    loadingMessage.value = message
  }

  const setError = (errorMessage) => {
    error.value = errorMessage
    setTimeout(() => {
      error.value = null
    }, 5000)
  }

  const clearError = () => {
    error.value = null
  }

  const createNewGame = async () => {
    try {
      setLoading(true, 'Creating new game...')
      const response = await gameApi.createGame()
      gameId.value = response.game_id
      gameState.value = response.initial_state
      gameHistory.value = []
      setLoading(false)
      return response.game_id
    } catch (err) {
      setError('Failed to create new game')
      setLoading(false)
      throw err
    }
  }

  const connectWebSocket = (gameIdToConnect) => {
    try {
      websocketService.connect(gameIdToConnect)
      
      websocketService.onMessage((data) => {
        handleWebSocketMessage(data)
      })
      
      websocketService.onConnect(() => {
        isConnected.value = true
      })
      
      websocketService.onDisconnect(() => {
        isConnected.value = false
      })
      
      websocketService.onError((error) => {
        setError(`WebSocket error: ${error}`)
      })
      
    } catch (err) {
      setError('Failed to connect to game server')
    }
  }

  const disconnectWebSocket = () => {
    websocketService.disconnect()
    isConnected.value = false
  }

  const handleWebSocketMessage = (data) => {
    switch (data.type) {
      case 'game_update':
        updateGameState(data.game_data, data.ai_analysis)
        addToHistory({
          type: 'player_action',
          gameData: data.game_data,
          reward: data.reward,
          info: data.info,
          timestamp: Date.now()
        })
        break
        
      case 'ai_move':
        updateGameState(data.game_data, { 
          q_values: data.q_values,
          action_probabilities: data.action_probabilities,
          recommended_action: data.action 
        })
        addToHistory({
          type: 'ai_action',
          action: data.action,
          qValues: data.q_values,
          actionProbabilities: data.action_probabilities,
          gameData: data.game_data,
          reward: data.reward,
          info: data.info,
          timestamp: Date.now()
        })
        break
        
      case 'game_reset':
        gameState.value = data.game_data
        gameHistory.value = []
        break
        
      case 'error':
        setError(data.message)
        break
    }
  }

  const updateGameState = (newGameState, newAiAnalysis = null) => {
    gameState.value = { ...gameState.value, ...newGameState }
    
    if (newAiAnalysis) {
      aiAnalysis.value = { ...aiAnalysis.value, ...newAiAnalysis }
    }
  }

  const addToHistory = (entry) => {
    gameHistory.value.push(entry)
    // Keep only last 1000 entries
    if (gameHistory.value.length > 1000) {
      gameHistory.value = gameHistory.value.slice(-1000)
    }
  }

  const makeAction = async (action) => {
    if (!gameId.value || !isConnected.value) return
    
    try {
      websocketService.sendAction(action)
    } catch (err) {
      setError('Failed to send action')
    }
  }

  const makeAiAction = async () => {
    if (!gameId.value || !isConnected.value) return
    
    try {
      websocketService.sendAiPlay()
    } catch (err) {
      setError('Failed to get AI action')
    }
  }

  const resetGame = async () => {
    if (!gameId.value || !isConnected.value) return
    
    try {
      websocketService.sendReset()
    } catch (err) {
      setError('Failed to reset game')
    }
  }

  const startAiAutoPlay = () => {
    isAiPlaying.value = true
  }

  const stopAiAutoPlay = () => {
    isAiPlaying.value = false
  }

  const getGameState = async (gameIdToGet) => {
    try {
      const response = await gameApi.getGameState(gameIdToGet)
      updateGameState(response, response.ai_analysis)
      return response
    } catch (err) {
      setError('Failed to get game state')
      throw err
    }
  }

  const getAiInfo = async () => {
    try {
      return await gameApi.getAiInfo()
    } catch (err) {
      setError('Failed to get AI information')
      throw err
    }
  }

  return {
    // State
    gameId,
    gameState,
    aiAnalysis,
    isLoading,
    loadingMessage,
    error,
    isConnected,
    isAiPlaying,
    gameHistory,
    
    // Computed
    isGameOver,
    isPlaying,
    gridWidth,
    gridHeight,
    actionName,
    
    // Actions
    setLoading,
    setError,
    clearError,
    createNewGame,
    connectWebSocket,
    disconnectWebSocket,
    makeAction,
    makeAiAction,
    resetGame,
    startAiAutoPlay,
    stopAiAutoPlay,
    getGameState,
    getAiInfo,
    updateGameState,
    addToHistory
  }
})
