<template>
  <div class="snake-game-container">
    <!-- Game Grid -->
    <div class="game-grid-wrapper">
      <div 
        class="game-grid"
        :style="gridStyle"
        @keydown="handleKeyPress"
        tabindex="0"
        ref="gameGrid"
      >
        <div
          v-for="(cell, index) in flatGrid"
          :key="index"
          class="grid-cell"
          :class="getCellClass(index)"
          :style="getCellStyle(index)"
        >
          <!-- Snake head with direction indicator -->
          <div v-if="cell === 'head'" class="snake-head-indicator">
            <svg 
              class="w-4 h-4 text-white transform"
              :class="getHeadRotation()"
              fill="currentColor" 
              viewBox="0 0 20 20"
            >
              <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
          </div>
          
          <!-- Food with animation -->
          <div v-if="cell === 'food'" class="food-indicator">
            <div class="w-3 h-3 bg-red-400 rounded-full animate-pulse"></div>
          </div>
        </div>
      </div>
      
      <!-- Game overlay for game over state -->
      <div v-if="gameStore.isGameOver" class="game-overlay">
        <div class="game-over-content">
          <h2 class="text-3xl font-bold text-white mb-4">Game Over!</h2>
          <p class="text-xl text-gray-300 mb-2">Final Score: {{ gameStore.gameState.score }}</p>
          <p class="text-lg text-gray-400 mb-6">Steps: {{ gameStore.gameState.steps }}</p>
          <button @click="$emit('reset')" class="btn-primary">
            Play Again
          </button>
        </div>
      </div>
    </div>
    
    <!-- Game Controls -->
    <div class="game-controls">
      <div class="control-section">
        <h3 class="text-sm font-semibold text-gray-400 mb-2">Manual Controls</h3>
        <div class="control-grid">
          <div></div>
          <button @click="makeMove(0)" class="control-btn" :disabled="!canMove">
            ↑
          </button>
          <div></div>
          <button @click="makeMove(2)" class="control-btn" :disabled="!canMove">
            ←
          </button>
          <button @click="makeMove(1)" class="control-btn" :disabled="!canMove">
            ↓
          </button>
          <button @click="makeMove(3)" class="control-btn" :disabled="!canMove">
            →
          </button>
        </div>
      </div>
      
      <div class="control-section">
        <h3 class="text-sm font-semibold text-gray-400 mb-2">AI Controls</h3>
        <div class="space-y-2">
          <button 
            @click="makeAiMove" 
            class="btn-secondary w-full"
            :disabled="!canMove"
          >
            AI Step
          </button>
          <button 
            @click="toggleAutoPlay" 
            class="w-full"
            :class="gameStore.isAiPlaying ? 'btn-danger' : 'btn-primary'"
            :disabled="!canMove"
          >
            {{ gameStore.isAiPlaying ? 'Stop AI' : 'AI Auto-Play' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useGameStore } from '../stores/gameStore'

const gameStore = useGameStore()
const gameGrid = ref(null)
const autoPlayInterval = ref(null)

// Props and emits
const emit = defineEmits(['reset', 'action'])

// Computed properties
const canMove = computed(() => {
  return gameStore.isPlaying && gameStore.isConnected
})

const gridStyle = computed(() => ({
  gridTemplateColumns: `repeat(${gameStore.gridWidth}, 1fr)`,
  gridTemplateRows: `repeat(${gameStore.gridHeight}, 1fr)`
}))

const grid = computed(() => {
  const newGrid = Array(gameStore.gridHeight).fill().map(() => 
    Array(gameStore.gridWidth).fill('empty')
  )
  
  // Place snake
  gameStore.gameState.snake_body.forEach((segment, index) => {
    const [x, y] = segment
    if (x >= 0 && x < gameStore.gridWidth && y >= 0 && y < gameStore.gridHeight) {
      newGrid[y][x] = index === 0 ? 'head' : 'body'
    }
  })
  
  // Place food
  if (gameStore.gameState.food_position) {
    const [x, y] = gameStore.gameState.food_position
    if (x >= 0 && x < gameStore.gridWidth && y >= 0 && y < gameStore.gridHeight) {
      newGrid[y][x] = 'food'
    }
  }
  
  return newGrid
})

const flatGrid = computed(() => {
  return grid.value.flat()
})

// Methods
const getCellClass = (index) => {
  const cellType = flatGrid.value[index]
  return {
    'snake-head': cellType === 'head',
    'snake-body': cellType === 'body',
    'food-cell': cellType === 'food',
    'empty-cell': cellType === 'empty'
  }
}

const getCellStyle = (index) => {
  const y = Math.floor(index / gameStore.gridWidth)
  const x = index % gameStore.gridWidth
  const cellType = flatGrid.value[index]
  
  // Add subtle glow effect for AI attention (if available)
  if (gameStore.aiAnalysis.attention_map && gameStore.aiAnalysis.attention_map[y] && gameStore.aiAnalysis.attention_map[y][x]) {
    const attention = gameStore.aiAnalysis.attention_map[y][x]
    return {
      boxShadow: `inset 0 0 ${attention * 10}px rgba(59, 130, 246, ${attention * 0.5})`
    }
  }
  
  return {}
}

const getHeadRotation = () => {
  const direction = gameStore.gameState.direction
  const rotations = {
    0: 'rotate-180',  // UP
    1: 'rotate-0',    // DOWN
    2: 'rotate-90',   // LEFT
    3: '-rotate-90'   // RIGHT
  }
  return rotations[direction] || 'rotate-0'
}

const makeMove = (action) => {
  if (!canMove.value) return
  gameStore.makeAction(action)
  emit('action', action)
}

const makeAiMove = () => {
  if (!canMove.value) return
  gameStore.makeAiAction()
}

const toggleAutoPlay = () => {
  if (gameStore.isAiPlaying) {
    stopAutoPlay()
  } else {
    startAutoPlay()
  }
}

const startAutoPlay = () => {
  if (!canMove.value) return
  
  gameStore.startAiAutoPlay()
  autoPlayInterval.value = setInterval(() => {
    if (gameStore.isPlaying && gameStore.isConnected) {
      gameStore.makeAiAction()
    } else {
      stopAutoPlay()
    }
  }, 200) // AI moves every 200ms
}

const stopAutoPlay = () => {
  gameStore.stopAiAutoPlay()
  if (autoPlayInterval.value) {
    clearInterval(autoPlayInterval.value)
    autoPlayInterval.value = null
  }
}

const handleKeyPress = (event) => {
  if (!canMove.value) return
  
  const keyMap = {
    'ArrowUp': 0,
    'ArrowDown': 1,
    'ArrowLeft': 2,
    'ArrowRight': 3,
    'w': 0,
    's': 1,
    'a': 2,
    'd': 3
  }
  
  const action = keyMap[event.key]
  if (action !== undefined) {
    event.preventDefault()
    makeMove(action)
  }
  
  // AI control keys
  if (event.key === ' ') {
    event.preventDefault()
    makeAiMove()
  }
  
  if (event.key.toLowerCase() === 'p') {
    event.preventDefault()
    toggleAutoPlay()
  }
}

// Watchers
watch(() => gameStore.isGameOver, (isGameOver) => {
  if (isGameOver) {
    stopAutoPlay()
  }
})

// Lifecycle
onMounted(() => {
  if (gameGrid.value) {
    gameGrid.value.focus()
  }
})

onUnmounted(() => {
  stopAutoPlay()
})
</script>

<style scoped>
.snake-game-container {
  @apply flex flex-col lg:flex-row gap-6 items-start;
}

.game-grid-wrapper {
  @apply relative bg-gray-900 rounded-lg border-2 border-gray-700 p-4;
  background-image: radial-gradient(circle, rgba(59, 130, 246, 0.1) 1px, transparent 1px);
  background-size: 20px 20px;
}

.game-grid {
  @apply grid gap-0.5 bg-gray-800 p-2 rounded focus:outline-none focus:ring-2 focus:ring-snake-green;
  width: 400px;
  height: 400px;
}

.grid-cell {
  @apply relative flex items-center justify-center transition-all duration-150;
  aspect-ratio: 1;
}

.snake-head {
  @apply bg-gradient-to-br from-green-400 to-green-600 rounded-sm shadow-lg;
}

.snake-body {
  @apply bg-gradient-to-br from-green-500 to-green-700 rounded-sm;
}

.food-cell {
  @apply bg-gradient-to-br from-red-400 to-red-600 rounded-sm;
}

.empty-cell {
  @apply bg-gray-700 rounded-sm;
}

.snake-head-indicator {
  @apply absolute inset-0 flex items-center justify-center;
}

.food-indicator {
  @apply absolute inset-0 flex items-center justify-center;
}

.game-overlay {
  @apply absolute inset-0 bg-black bg-opacity-75 flex items-center justify-center rounded-lg;
}

.game-over-content {
  @apply text-center p-8 bg-gray-800 rounded-lg border border-gray-600;
}

.game-controls {
  @apply space-y-6;
}

.control-section {
  @apply card p-4;
}

.control-grid {
  @apply grid grid-cols-3 gap-2 w-32 mx-auto;
}

.control-btn {
  @apply w-10 h-10 bg-gray-700 hover:bg-gray-600 text-white rounded border border-gray-600 
         flex items-center justify-center font-mono text-lg transition-colors duration-200
         disabled:opacity-50 disabled:cursor-not-allowed;
}

.control-btn:hover:not(:disabled) {
  @apply bg-snake-green border-snake-green;
}

.control-btn:active:not(:disabled) {
  @apply transform scale-95;
}
</style>
