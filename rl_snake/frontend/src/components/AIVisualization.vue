<template>
  <div class="ai-visualization">
    <!-- AI Decision Panel -->
    <div class="card">
      <div class="card-header">
        <h3 class="text-lg font-semibold text-white flex items-center space-x-2">
          <svg class="w-5 h-5 text-neural-blue" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clip-rule="evenodd" />
          </svg>
          <span>AI Decision Analysis</span>
        </h3>
      </div>
      <div class="card-body">
        <!-- Current Recommendation -->
        <div class="mb-6">
          <div class="flex items-center justify-between mb-2">
            <span class="text-sm text-gray-400">Recommended Action</span>
            <span class="text-lg font-mono font-bold text-snake-green">
              {{ gameStore.actionName }}
            </span>
          </div>
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-400">Confidence</span>
            <span class="text-lg font-mono font-bold text-neural-blue">
              {{ (confidence * 100).toFixed(1) }}%
            </span>
          </div>
        </div>

        <!-- Q-Values Visualization -->
        <div class="mb-6">
          <h4 class="text-sm font-semibold text-gray-300 mb-3">Q-Values (Action Strength)</h4>
          <div class="space-y-2">
            <div
              v-for="(value, index) in gameStore.aiAnalysis.q_values"
              :key="index"
              class="flex items-center space-x-3"
            >
              <span class="w-12 text-sm font-mono text-gray-400">
                {{ actionNames[index] }}
              </span>
              <div class="flex-1 relative">
                <div class="h-6 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    class="h-full transition-all duration-300 rounded-full"
                    :class="getQValueBarClass(index)"
                    :style="{ width: getQValueWidth(value) + '%' }"
                  ></div>
                </div>
                <span class="absolute right-2 top-0 text-xs text-gray-300 leading-6">
                  {{ value.toFixed(2) }}
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- Action Probabilities -->
        <div class="mb-6">
          <h4 class="text-sm font-semibold text-gray-300 mb-3">Action Probabilities</h4>
          <div class="space-y-2">
            <div
              v-for="(prob, index) in gameStore.aiAnalysis.action_probabilities"
              :key="index"
              class="flex items-center space-x-3"
            >
              <span class="w-12 text-sm font-mono text-gray-400">
                {{ actionNames[index] }}
              </span>
              <div class="flex-1 relative">
                <div class="h-4 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    class="h-full bg-gradient-to-r from-purple-500 to-purple-400 transition-all duration-300 rounded-full"
                    :style="{ width: (prob * 100) + '%' }"
                  ></div>
                </div>
                <span class="absolute right-2 top-0 text-xs text-gray-300 leading-4">
                  {{ (prob * 100).toFixed(1) }}%
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- Neural Network Visualization -->
        <div>
          <h4 class="text-sm font-semibold text-gray-300 mb-3">Neural Network Activity</h4>
          <div class="neural-network-container">
            <svg class="neural-network-svg" viewBox="0 0 300 200">
              <!-- Input layer -->
              <g class="input-layer">
                <text x="10" y="15" class="layer-label">Input</text>
                <circle
                  v-for="(node, i) in inputNodes"
                  :key="'input-' + i"
                  :cx="30"
                  :cy="30 + i * 15"
                  :r="4"
                  :class="getNodeClass(node.value)"
                />
              </g>

              <!-- Hidden layer -->
              <g class="hidden-layer">
                <text x="110" y="15" class="layer-label">Hidden</text>
                <circle
                  v-for="(node, i) in hiddenNodes"
                  :key="'hidden-' + i"
                  :cx="130"
                  :cy="40 + i * 12"
                  :r="3"
                  :class="getNodeClass(node.value)"
                />
              </g>

              <!-- Output layer -->
              <g class="output-layer">
                <text x="210" y="15" class="layer-label">Output</text>
                <circle
                  v-for="(node, i) in outputNodes"
                  :key="'output-' + i"
                  :cx="230"
                  :cy="50 + i * 25"
                  :r="6"
                  :class="getNodeClass(node.value)"
                />
                <text
                  v-for="(action, i) in actionNames"
                  :key="'label-' + i"
                  :x="245"
                  :y="55 + i * 25"
                  class="action-label"
                >
                  {{ action }}
                </text>
              </g>

              <!-- Connections (simplified) -->
              <g class="connections" opacity="0.3">
                <line
                  v-for="connection in connections"
                  :key="connection.id"
                  :x1="connection.x1"
                  :y1="connection.y1"
                  :x2="connection.x2"
                  :y2="connection.y2"
                  :stroke="connection.color"
                  :stroke-width="connection.width"
                />
              </g>
            </svg>
          </div>
        </div>
      </div>
    </div>

    <!-- Game Statistics -->
    <div class="card">
      <div class="card-header">
        <h3 class="text-lg font-semibold text-white">Game Statistics</h3>
      </div>
      <div class="card-body">
        <div class="grid grid-cols-2 gap-4">
          <div class="stat-item">
            <span class="stat-label">Score</span>
            <span class="stat-value text-snake-green">{{ gameStore.gameState.score }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Steps</span>
            <span class="stat-value text-neural-blue">{{ gameStore.gameState.steps }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Snake Length</span>
            <span class="stat-value text-purple-400">{{ gameStore.gameState.snake_body.length }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Game State</span>
            <span class="stat-value" :class="getGameStateClass()">
              {{ gameStore.gameState.game_state.replace('_', ' ').toUpperCase() }}
            </span>
          </div>
        </div>
      </div>
    </div>

    <!-- Recent Actions History -->
    <div class="card">
      <div class="card-header">
        <h3 class="text-lg font-semibold text-white">Action History</h3>
      </div>
      <div class="card-body">
        <div class="history-container">
          <div
            v-for="(entry, index) in recentHistory"
            :key="index"
            class="history-entry"
            :class="{ 'ai-action': entry.type === 'ai_action' }"
          >
            <div class="flex items-center justify-between">
              <span class="text-sm font-mono">
                {{ entry.type === 'ai_action' ? '🤖' : '👤' }}
                {{ actionNames[entry.action] || 'Unknown' }}
              </span>
              <span class="text-xs text-gray-500">
                {{ formatTime(entry.timestamp) }}
              </span>
            </div>
            <div v-if="entry.reward !== undefined" class="text-xs text-gray-400">
              Reward: {{ entry.reward.toFixed(2) }}
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useGameStore } from '../stores/gameStore'

const gameStore = useGameStore()

// Computed properties
const actionNames = computed(() => ['UP', 'DOWN', 'LEFT', 'RIGHT'])

const confidence = computed(() => {
  if (!gameStore.aiAnalysis.action_probabilities?.length) return 0
  return Math.max(...gameStore.aiAnalysis.action_probabilities)
})

const recentHistory = computed(() => {
  return gameStore.gameHistory.slice(-10).reverse()
})

// Neural network visualization data
const inputNodes = computed(() => {
  // Simplified representation of the 11-dimensional input
  const features = [
    'Danger Straight', 'Danger Right', 'Danger Left',
    'Dir Left', 'Dir Right', 'Dir Up', 'Dir Down',
    'Food Left', 'Food Right', 'Food Up', 'Food Down'
  ]
  
  return features.map((name, i) => ({
    name,
    value: Math.random() * 0.8 + 0.2 // Simulated activation
  }))
})

const hiddenNodes = computed(() => {
  return Array(8).fill().map(() => ({
    value: Math.random() * 0.9 + 0.1
  }))
})

const outputNodes = computed(() => {
  return gameStore.aiAnalysis.q_values.map(value => ({
    value: Math.max(0, Math.min(1, (value + 5) / 10)) // Normalize for visualization
  }))
})

const connections = computed(() => {
  // Generate some sample connections for visualization
  const conns = []
  for (let i = 0; i < 20; i++) {
    conns.push({
      id: i,
      x1: Math.random() > 0.5 ? 30 : 130,
      y1: 30 + Math.random() * 140,
      x2: Math.random() > 0.5 ? 130 : 230,
      y2: 40 + Math.random() * 120,
      color: Math.random() > 0.5 ? '#10b981' : '#3b82f6',
      width: Math.random() * 2 + 0.5
    })
  }
  return conns
})

// Methods
const getQValueWidth = (value) => {
  const maxValue = Math.max(...gameStore.aiAnalysis.q_values)
  const minValue = Math.min(...gameStore.aiAnalysis.q_values)
  const range = maxValue - minValue || 1
  return Math.max(5, ((value - minValue) / range) * 100)
}

const getQValueBarClass = (index) => {
  const isRecommended = index === gameStore.aiAnalysis.recommended_action
  return isRecommended 
    ? 'bg-gradient-to-r from-snake-green to-green-400' 
    : 'bg-gradient-to-r from-gray-500 to-gray-400'
}

const getNodeClass = (value) => {
  const intensity = Math.floor(value * 4)
  const classes = [
    'fill-gray-600',
    'fill-blue-700',
    'fill-blue-500',
    'fill-blue-400',
    'fill-blue-200'
  ]
  return classes[intensity] || classes[0]
}

const getGameStateClass = () => {
  const state = gameStore.gameState.game_state
  return {
    'text-green-400': state === 'playing',
    'text-red-400': state === 'game_over',
    'text-yellow-400': state === 'paused'
  }
}

const formatTime = (timestamp) => {
  const now = Date.now()
  const diff = now - timestamp
  if (diff < 1000) return 'now'
  if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`
  return `${Math.floor(diff / 60000)}m ago`
}
</script>

<style scoped>
.ai-visualization {
  @apply space-y-6;
}

.stat-item {
  @apply flex flex-col items-center p-3 bg-gray-700 rounded-lg;
}

.stat-label {
  @apply text-xs text-gray-400 uppercase tracking-wide;
}

.stat-value {
  @apply text-xl font-bold font-mono mt-1;
}

.neural-network-container {
  @apply bg-gray-900 rounded-lg p-4;
}

.neural-network-svg {
  @apply w-full h-32;
}

.layer-label {
  @apply text-xs fill-gray-400;
}

.action-label {
  @apply text-xs fill-gray-300;
}

.history-container {
  @apply space-y-2 max-h-48 overflow-y-auto;
}

.history-entry {
  @apply p-2 bg-gray-700 rounded border-l-2 border-gray-600;
}

.history-entry.ai-action {
  @apply border-l-neural-blue bg-gray-700;
}

.history-entry:not(.ai-action) {
  @apply border-l-snake-green;
}

/* Custom scrollbar for history */
.history-container::-webkit-scrollbar {
  @apply w-2;
}

.history-container::-webkit-scrollbar-track {
  @apply bg-gray-800 rounded;
}

.history-container::-webkit-scrollbar-thumb {
  @apply bg-gray-600 rounded;
}
</style>
