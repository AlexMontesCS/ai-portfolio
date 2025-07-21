<template>
  <div class="play-page">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Header -->
      <div class="text-center mb-8">
        <h1 class="text-4xl font-bold text-gradient mb-4">AI Snake Playground</h1>
        <p class="text-xl text-gray-400">
          Watch the AI play Snake and explore its decision-making process in real-time
        </p>
      </div>

      <!-- Connection Status -->
      <div class="mb-6">
        <div class="flex items-center justify-center space-x-6">
          <div class="flex items-center space-x-2">
            <div 
              class="w-3 h-3 rounded-full"
              :class="gameStore.isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'"
            ></div>
            <span class="text-sm text-gray-400">
              {{ gameStore.isConnected ? 'Connected to Game Server' : 'Disconnected' }}
            </span>
          </div>

          <!-- Model Information -->
          <div v-if="displayModelInfo" class="flex items-center space-x-2">
            <div class="w-3 h-3 rounded-full bg-neural-blue"></div>
            <span class="text-sm text-gray-400">
              AI Model: {{ displayModelInfo.training_episodes > 0 ? `Trained (${displayModelInfo.training_episodes.toLocaleString()} episodes)` : 'Untrained' }}
            </span>
          </div>
          <div v-else-if="modelLoading" class="flex items-center space-x-2">
            <div class="w-3 h-3 rounded-full bg-gray-400 animate-pulse"></div>
            <span class="text-sm text-gray-400">Loading model info...</span>
          </div>

          <!-- Model Selection Dropdown -->
          <div class="flex items-center space-x-2">
            <select 
              v-model="selectedModel" 
              @change="loadSelectedModel"
              :disabled="modelChanging"
              class="px-3 py-1 text-sm bg-gray-800 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-neural-blue focus:border-transparent"
            >
              <option value="">Select Model...</option>
              <option 
                v-for="model in availableModels" 
                :key="model.filename" 
                :value="model.filename"
              >
                {{ model.display_name }} ({{ model.size_mb }}MB)
              </option>
            </select>
            <button 
              @click="refreshModels" 
              :disabled="loadingModels"
              class="p-1 text-gray-400 hover:text-white transition-colors"
              title="Refresh model list"
            >
              <svg class="w-4 h-4" :class="{ 'animate-spin': loadingModels }" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          </div>
          
          <div v-if="!gameStore.gameId" class="flex space-x-2">
            <button @click="startNewGame" class="btn-primary" :disabled="gameStore.isLoading">
              {{ gameStore.isLoading ? 'Starting...' : 'Start New Game' }}
            </button>
          </div>
          
          <div v-else class="flex space-x-2">
            <button @click="resetGame" class="btn-secondary">
              Reset Game
            </button>
          </div>
        </div>
      </div>

      <!-- Main Content -->
      <div v-if="gameStore.gameId" class="grid grid-cols-1 xl:grid-cols-3 gap-8">
        <!-- Game Area -->
        <div class="xl:col-span-2">
          <div class="card p-6">
            <div class="flex items-center justify-between mb-6">
              <h2 class="text-2xl font-semibold text-white">Game Board</h2>
              <div class="flex items-center space-x-4">
                <div class="status-indicator" :class="getStatusClass()">
                  {{ gameStore.gameState.game_state.replace('_', ' ').toUpperCase() }}
                </div>
                <div class="text-sm text-gray-400">
                  Score: <span class="font-mono text-snake-green font-bold">{{ gameStore.gameState.score }}</span>
                </div>
              </div>
            </div>
            
            <SnakeGame @reset="resetGame" @action="onPlayerAction" />
          </div>
        </div>

        <!-- AI Visualization Sidebar -->
        <div class="xl:col-span-1">
          <AIVisualization />
        </div>
      </div>

      <!-- Loading State -->
      <div v-else-if="gameStore.isLoading" class="text-center py-16">
        <div class="animate-spin rounded-full h-16 w-16 border-b-2 border-snake-green mx-auto mb-4"></div>
        <p class="text-gray-400">{{ gameStore.loadingMessage }}</p>
      </div>

      <!-- Welcome State -->
      <div v-else class="text-center py-16">
        <div class="max-w-md mx-auto">
          <div class="w-20 h-20 bg-gradient-to-br from-snake-green to-neural-blue rounded-full flex items-center justify-center mb-6 mx-auto">
            <svg class="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clip-rule="evenodd" />
            </svg>
          </div>
          <h2 class="text-2xl font-bold text-white mb-4">Ready to Play?</h2>
          <p class="text-gray-400 mb-6">
            Start a new game to watch the AI play Snake and see how it makes decisions using Deep Q-Learning.
          </p>
          <button @click="startNewGame" class="btn-primary text-lg px-8 py-3">
            Start New Game
          </button>
        </div>
      </div>

      <!-- Game Instructions -->
      <div v-if="gameStore.gameId" class="mt-12">
        <div class="card p-6">
          <h3 class="text-lg font-semibold text-white mb-4">How to Play</h3>
          <div class="grid md:grid-cols-3 gap-6 text-sm text-gray-400">
            <div>
              <h4 class="font-semibold text-gray-300 mb-2">Manual Control</h4>
              <ul class="space-y-1">
                <li>• Arrow keys or WASD to move</li>
                <li>• Click direction buttons</li>
                <li>• Guide the snake to food</li>
              </ul>
            </div>
            <div>
              <h4 class="font-semibold text-gray-300 mb-2">AI Control</h4>
              <ul class="space-y-1">
                <li>• Space bar for single AI step</li>
                <li>• 'P' key to toggle auto-play</li>
                <li>• Watch AI decision process</li>
              </ul>
            </div>
            <div>
              <h4 class="font-semibold text-gray-300 mb-2">Visualization</h4>
              <ul class="space-y-1">
                <li>• Q-values show action strength</li>
                <li>• Probabilities show confidence</li>
                <li>• Neural network shows activity</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <!-- Performance Metrics -->
      <div v-if="gameStore.gameId && gameStore.gameHistory.length > 0" class="mt-8">
        <div class="card p-6">
          <h3 class="text-lg font-semibold text-white mb-4">Session Performance</h3>
          <div class="grid md:grid-cols-4 gap-4">
            <div class="text-center">
              <div class="text-2xl font-bold text-snake-green">{{ totalActions }}</div>
              <div class="text-sm text-gray-400">Total Actions</div>
            </div>
            <div class="text-center">
              <div class="text-2xl font-bold text-neural-blue">{{ aiActions }}</div>
              <div class="text-sm text-gray-400">AI Actions</div>
            </div>
            <div class="text-center">
              <div class="text-2xl font-bold text-purple-400">{{ humanActions }}</div>
              <div class="text-sm text-gray-400">Human Actions</div>
            </div>
            <div class="text-center">
              <div class="text-2xl font-bold text-yellow-400">{{ avgReward.toFixed(2) }}</div>
              <div class="text-sm text-gray-400">Avg Reward</div>
            </div>
          </div>
        </div>
      </div>

      <!-- AI Model Information -->
      <div v-if="modelInfo || availableModels.length > 0" class="mt-8">
        <div class="card p-6">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-semibold text-white">AI Model Information</h3>
            <div v-if="selectedModel" class="text-sm text-gray-400">
              Current: <span class="text-neural-blue font-medium">{{ selectedModel }}</span>
            </div>
          </div>
          
          <div v-if="modelInfo" class="grid md:grid-cols-4 gap-4">
            <div class="text-center">
              <div class="text-2xl font-bold text-neural-blue">{{ modelInfo.training_episodes }}</div>
              <div class="text-sm text-gray-400">Training Episodes</div>
            </div>
            <div class="text-center">
              <div class="text-2xl font-bold text-purple-400">{{ modelInfo.average_score.toFixed(2) }}</div>
              <div class="text-sm text-gray-400">Average Score</div>
            </div>
            <div class="text-center">
              <div class="text-2xl font-bold text-yellow-400">{{ (modelInfo.current_epsilon * 100).toFixed(1) }}%</div>
              <div class="text-sm text-gray-400">Exploration Rate</div>
            </div>
            <div class="text-center">
              <div class="text-2xl font-bold text-green-400">{{ modelInfo.memory_size }}</div>
              <div class="text-sm text-gray-400">Memory Size</div>
            </div>
          </div>
          
          <div v-else class="text-center text-gray-400 py-8">
            No model information available. Please load a model using the dropdown above.
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useGameStore } from '../stores/gameStore'
import { useTrainingStore } from '../stores/trainingStore'
import SnakeGame from '../components/SnakeGame.vue'
import AIVisualization from '../components/AIVisualization.vue'
import { gameApi } from '../services/api'

const gameStore = useGameStore()
const trainingStore = useTrainingStore()

// Model information
const modelInfo = computed(() => trainingStore.modelInfo)
const modelLoading = ref(false)
const availableModels = ref([])
const selectedModel = ref('')
const loadingModels = ref(false)
const modelChanging = ref(false)

// Computed: selected model info from availableModels
const selectedModelInfo = computed(() => {
  if (!selectedModel.value) return null
  return availableModels.value.find(m => m.filename === selectedModel.value) || null
})

const displayModelInfo = computed(() => {
  // Prioritize the detailed modelInfo from the API call, but fall back to the list version
  return modelInfo.value || selectedModelInfo.value
})

// Load available models
const loadAvailableModels = async () => {
  try {
    loadingModels.value = true
    const response = await gameApi.getAvailableModels()
    availableModels.value = response.models
    selectedModel.value = response.current_model || ''
  } catch (error) {
    console.error('Failed to load available models:', error)
  } finally {
    loadingModels.value = false
  }
}

// Refresh models list
const refreshModels = async () => {
  await loadAvailableModels()
}

// Load selected model
const loadSelectedModel = async () => {
  if (!selectedModel.value) return
  
  try {
    modelChanging.value = true
    const response = await gameApi.loadModel(selectedModel.value)
    console.log('Model loaded:', response)
    
    // Refresh model info to show updated stats
    if (response.model_info) {
      trainingStore.updateModelInfo(response.model_info)
    } else {
      await loadModelInfo()
    }
    
    // Show success message (you can add a toast notification here)
    console.log(`Successfully loaded model: ${selectedModel.value}`)
    
  } catch (error) {
    console.error('Failed to load model:', error)
    // Reset selection on error
    selectedModel.value = ''
  } finally {
    modelChanging.value = false
  }
}

// Load model information
const loadModelInfo = async () => {
  try {
    modelLoading.value = true
    const info = await gameApi.getAiInfo()
    trainingStore.updateModelInfo(info)
  } catch (error) {
    console.error('Failed to load model info:', error)
  } finally {
    modelLoading.value = false
  }
}

// Computed properties
const totalActions = computed(() => gameStore.gameHistory.length)

const aiActions = computed(() => 
  gameStore.gameHistory.filter(entry => entry.type === 'ai_action').length
)

const humanActions = computed(() => 
  gameStore.gameHistory.filter(entry => entry.type === 'player_action').length
)

const avgReward = computed(() => {
  const rewards = gameStore.gameHistory
    .filter(entry => entry.reward !== undefined)
    .map(entry => entry.reward)
  
  if (rewards.length === 0) return 0
  return rewards.reduce((sum, reward) => sum + reward, 0) / rewards.length
})

// Methods
const startNewGame = async () => {
  try {
    const gameId = await gameStore.createNewGame()
    gameStore.connectWebSocket(gameId)
  } catch (error) {
    console.error('Failed to start new game:', error)
  }
}

const resetGame = () => {
  gameStore.resetGame()
}

const onPlayerAction = (action) => {
  // Track player actions for analytics
  console.log('Player action:', action)
}

const getStatusClass = () => {
  const state = gameStore.gameState.game_state
  return {
    'status-playing': state === 'playing',
    'status-game-over': state === 'game_over',
    'status-paused': state === 'paused'
  }
}

// Lifecycle
onMounted(async () => {
  // Load model information and available models
  await Promise.all([
    loadModelInfo(),
    loadAvailableModels()
  ])
  
  // Auto-start a game if none exists
  if (!gameStore.gameId) {
    startNewGame()
  }
})

onUnmounted(() => {
  // Clean up WebSocket connection
  gameStore.disconnectWebSocket()
})
</script>

<style scoped>
.play-page {
  min-height: 100vh;
  background-color: #1a202c; /* Tailwind bg-gray-900 */
}

.text-gradient {
  background: linear-gradient(45deg, #10b981, #3b82f6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.status-playing {
  background-color: #d1fae5; /* Tailwind bg-green-100 */
  color: #065f46; /* Tailwind text-green-800 */
}
.status-game-over {
  background-color: #fee2e2; /* Tailwind bg-red-100 */
  color: #991b1b; /* Tailwind text-red-800 */
}
.status-paused {
  background-color: #fef3c7; /* Tailwind bg-yellow-100 */
  color: #92400e; /* Tailwind text-yellow-800 */
}

/* Dark mode for status indicators */
@media (prefers-color-scheme: dark) {
  .status-playing {
    background-color: #065f46; /* Tailwind bg-green-900 */
    color: #bbf7d0; /* Tailwind text-green-200 */
  }
  .status-game-over {
    background-color: #991b1b; /* Tailwind bg-red-900 */
    color: #fecaca; /* Tailwind text-red-200 */
  }
  .status-paused {
    background-color: #92400e; /* Tailwind bg-yellow-900 */
    color: #fde68a; /* Tailwind text-yellow-200 */
  }
}
</style>
