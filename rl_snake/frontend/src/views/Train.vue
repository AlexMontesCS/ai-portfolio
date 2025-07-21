<template>
  <div class="train-page">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Header -->
      <div class="text-center mb-8">
        <h1 class="text-4xl font-bold text-gradient mb-4">Train Your AI</h1>
        <p class="text-xl text-gray-400">
          Configure and train a Deep Q-Network to play Snake with custom parameters
        </p>
      </div>

      <!-- Training Status -->
      <div class="mb-8">
        <div class="card p-6">
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-xl font-semibold text-white">Training Status</h2>
            <div class="status-indicator" :class="trainingStore.isTraining ? 'status-training' : 'status-idle'">
              {{ trainingStore.isTraining ? 'Training in Progress' : 'Ready to Train' }}
            </div>
          </div>

          <div v-if="trainingStore.isTraining" class="space-y-6">
            <!-- Overall Progress Bar -->
            <div>
              <div class="flex justify-between text-sm text-gray-400 mb-2">
                <span>Episode {{ trainingStore.trainingProgress.current_episode }} / {{ trainingStore.trainingProgress.total_episodes }}</span>
                <span>{{ trainingStore.trainingProgressPercent.toFixed(1) }}%</span>
              </div>
              <div class="w-full bg-gray-700 rounded-full h-3 relative overflow-hidden">
                <div 
                  class="bg-gradient-to-r from-snake-green to-neural-blue h-full rounded-full transition-all duration-500 ease-out relative"
                  :style="{ width: trainingStore.trainingProgressPercent + '%' }"
                >
                  <div class="absolute inset-0 bg-white opacity-20 animate-pulse"></div>
                </div>
              </div>
            </div>

            <!-- Detailed Progress Metrics -->
            <div class="space-y-3">
              <!-- Score Progress -->
              <div>
                <div class="flex justify-between text-sm mb-1">
                  <span class="text-gray-400">Current Episode Score</span>
                  <span class="text-snake-green font-mono">{{ trainingStore.trainingProgress.current_score }}</span>
                </div>
                <div class="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    class="bg-gradient-to-r from-green-600 to-green-400 h-full rounded-full transition-all duration-300"
                    :style="{ width: Math.min(100, trainingStore.trainingProgress.current_score) + '%' }"
                  ></div>
                </div>
              </div>

              <!-- Average Score Progress -->
              <div>
                <div class="flex justify-between text-sm mb-1">
                  <span class="text-gray-400">Average Score (Last 100)</span>
                  <span class="text-neural-blue font-mono">{{ trainingStore.trainingProgress.average_score.toFixed(2) }}</span>
                </div>
                <div class="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    class="bg-gradient-to-r from-blue-600 to-blue-400 h-full rounded-full transition-all duration-300"
                    :style="{ width: Math.min(100, (trainingStore.trainingProgress.average_score / 30) * 100) + '%' }"
                  ></div>
                </div>
              </div>

              <!-- Exploration Rate (Epsilon) -->
              <div>
                <div class="flex justify-between text-sm mb-1">
                  <span class="text-gray-400">Exploration Rate</span>
                  <span class="text-purple-400 font-mono">{{ (trainingStore.trainingProgress.epsilon * 100).toFixed(1) }}%</span>
                </div>
                <div class="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    class="bg-gradient-to-r from-purple-600 to-purple-400 h-full rounded-full transition-all duration-300"
                    :style="{ width: (trainingStore.trainingProgress.epsilon * 100) + '%' }"
                  ></div>
                </div>
              </div>
            </div>

            <!-- Live Statistics Grid -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div class="bg-gray-800 rounded-lg p-4 text-center border-l-4 border-snake-green">
                <div class="text-2xl font-bold text-snake-green transition-all duration-300">
                  {{ trainingStore.trainingProgress.current_score }}
                </div>
                <div class="text-sm text-gray-400">Current Score</div>
                <div class="text-xs text-gray-500 mt-1">
                  {{ trainingStore.trainingProgress.current_score > lastScore ? '↗' : trainingStore.trainingProgress.current_score < lastScore ? '↘' : '→' }}
                </div>
              </div>
              <div class="bg-gray-800 rounded-lg p-4 text-center border-l-4 border-neural-blue">
                <div class="text-2xl font-bold text-neural-blue transition-all duration-300">
                  {{ trainingStore.trainingProgress.average_score.toFixed(1) }}
                </div>
                <div class="text-sm text-gray-400">Average Score</div>
                <div class="text-xs text-gray-500 mt-1">Last 100 episodes</div>
              </div>
              <div class="bg-gray-800 rounded-lg p-4 text-center border-l-4 border-purple-400">
                <div class="text-2xl font-bold text-purple-400 transition-all duration-300">
                  {{ (trainingStore.trainingProgress.epsilon * 100).toFixed(1) }}%
                </div>
                <div class="text-sm text-gray-400">Exploration</div>
                <div class="text-xs text-gray-500 mt-1">
                  {{ trainingStore.trainingProgress.epsilon > 0.1 ? 'Exploring' : 'Exploiting' }}
                </div>
              </div>
              <div class="bg-gray-800 rounded-lg p-4 text-center border-l-4 border-yellow-400">
                <div class="text-2xl font-bold text-yellow-400 transition-all duration-300">
                  {{ trainingTime }}
                </div>
                <div class="text-sm text-gray-400">Training Time</div>
                <div class="text-xs text-gray-500 mt-1">HH:MM:SS</div>
              </div>
            </div>

            <!-- Training Speed Indicator -->
            <div class="bg-gray-800 rounded-lg p-4">
              <div class="flex items-center justify-between mb-2">
                <span class="text-sm font-medium text-gray-300">Training Speed</span>
                <span class="text-sm text-gray-400">{{ episodesPerMinute.toFixed(1) }} episodes/min</span>
              </div>
              <div class="flex items-center space-x-2">
                <div class="flex-1 bg-gray-700 rounded-full h-2">
                  <div 
                    class="bg-gradient-to-r from-yellow-600 to-yellow-400 h-full rounded-full transition-all duration-500"
                    :style="{ width: Math.min(100, (episodesPerMinute / 10) * 100) + '%' }"
                  ></div>
                </div>
                <div class="w-3 h-3 rounded-full animate-pulse" 
                     :class="episodesPerMinute > 5 ? 'bg-green-400' : episodesPerMinute > 2 ? 'bg-yellow-400' : 'bg-red-400'">
                </div>
              </div>
            </div>

            <!-- Live Training Log -->
            <div class="bg-gray-800 rounded-lg p-4">
              <h4 class="text-sm font-medium text-gray-300 mb-3">Training Log</h4>
              <div class="space-y-1 max-h-32 overflow-y-auto text-xs font-mono">
                <div v-for="(log, index) in trainingLogs" :key="index" 
                     class="text-gray-400 animate-fade-in"
                     :class="getLogColor(log.type)">
                  <span class="text-gray-500">{{ formatTime(log.timestamp) }}</span>
                  {{ log.message }}
                </div>
              </div>
            </div>

            <!-- Control Buttons -->
            <div class="flex justify-center space-x-4">
              <button @click="pauseTraining" class="btn-secondary">
                <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd" />
                </svg>
                Pause Training
              </button>
              <button @click="stopTraining" class="btn-danger">
                <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clip-rule="evenodd" />
                </svg>
                Stop Training
              </button>
            </div>
          </div>

          <div v-else class="text-center py-8">
            <div class="mb-6">
              <div class="w-16 h-16 bg-gradient-to-br from-neural-blue to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg class="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clip-rule="evenodd" />
                </svg>
              </div>
              <h3 class="text-xl font-semibold text-white mb-2">Ready to Train</h3>
              <p class="text-gray-400">Configure your training parameters and start the learning process</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Real-time Training Chart -->
      <div v-if="trainingStore.isTraining" class="card mb-8">
        <div class="card-header">
          <h3 class="text-lg font-semibold text-white">Real-time Training Progress</h3>
        </div>
        <div class="card-body">
          <TrainingChart :chart-data="trainingChartData" />
        </div>
      </div>

      <!-- Training Configuration -->
      <div v-if="!trainingStore.isTraining" class="grid lg:grid-cols-2 gap-8 mb-8">
        <!-- Basic Configuration -->
        <div class="card">
          <div class="card-header">
            <h3 class="text-lg font-semibold text-white">Basic Configuration</h3>
          </div>
          <div class="card-body space-y-6">
            <div>
              <label class="block text-sm font-medium text-gray-300 mb-2">Number of Episodes</label>
              <input 
                v-model.number="trainingConfig.episodes" 
                type="number" 
                min="100" 
                max="10000" 
                step="100"
                class="input-field"
              >
              <p class="text-xs text-gray-500 mt-1">How many games the AI will play during training</p>
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-300 mb-2">Learning Rate</label>
              <select v-model="trainingConfig.learning_rate" class="input-field">
                <option value="0.0001">0.0001 (Very Slow)</option>
                <option value="0.001">0.001 (Recommended)</option>
                <option value="0.01">0.01 (Fast)</option>
                <option value="0.1">0.1 (Very Fast)</option>
              </select>
              <p class="text-xs text-gray-500 mt-1">How quickly the AI learns from each experience</p>
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-300 mb-2">Batch Size</label>
              <select v-model.number="trainingConfig.batch_size" class="input-field">
                <option value="16">16 (Small)</option>
                <option value="32">32 (Recommended)</option>
                <option value="64">64 (Large)</option>
                <option value="128">128 (Very Large)</option>
              </select>
              <p class="text-xs text-gray-500 mt-1">Number of experiences processed together</p>
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-300 mb-2">Memory Size</label>
              <select v-model.number="trainingConfig.memory_size" class="input-field">
                <option value="5000">5,000 (Small)</option>
                <option value="10000">10,000 (Recommended)</option>
                <option value="20000">20,000 (Large)</option>
                <option value="50000">50,000 (Very Large)</option>
              </select>
              <p class="text-xs text-gray-500 mt-1">How many past experiences to remember</p>
            </div>
          </div>
        </div>

        <!-- Advanced Configuration -->
        <div class="card">
          <div class="card-header">
            <h3 class="text-lg font-semibold text-white">Advanced Configuration</h3>
          </div>
          <div class="card-body space-y-6">
            <div>
              <label class="block text-sm font-medium text-gray-300 mb-2">Network Architecture</label>
              <select v-model.number="trainingConfig.hidden_size" class="input-field">
                <option value="128">128 neurons (Small)</option>
                <option value="256">256 neurons (Recommended)</option>
                <option value="512">512 neurons (Large)</option>
                <option value="1024">1024 neurons (Very Large)</option>
              </select>
              <p class="text-xs text-gray-500 mt-1">Size of the neural network hidden layers</p>
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-300 mb-2">Discount Factor (γ)</label>
              <input 
                v-model.number="trainingConfig.gamma" 
                type="number" 
                min="0.8" 
                max="0.99" 
                step="0.01"
                class="input-field"
              >
              <p class="text-xs text-gray-500 mt-1">How much future rewards matter (0.95 recommended)</p>
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-300 mb-2">Exploration Strategy</label>
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="text-xs text-gray-400">Start Epsilon</label>
                  <input 
                    v-model.number="trainingConfig.epsilon_start" 
                    type="number" 
                    min="0.5" 
                    max="1.0" 
                    step="0.1"
                    class="input-field mt-1"
                  >
                </div>
                <div>
                  <label class="text-xs text-gray-400">End Epsilon</label>
                  <input 
                    v-model.number="trainingConfig.epsilon_end" 
                    type="number" 
                    min="0.01" 
                    max="0.2" 
                    step="0.01"
                    class="input-field mt-1"
                  >
                </div>
              </div>
              <p class="text-xs text-gray-500 mt-1">Exploration rate decay from start to end</p>
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-300 mb-2">Save Frequency</label>
              <input 
                v-model.number="trainingConfig.save_frequency" 
                type="number" 
                min="50" 
                max="500" 
                step="50"
                class="input-field"
              >
              <p class="text-xs text-gray-500 mt-1">Save model checkpoint every N episodes</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Start Training Button -->
      <div v-if="!trainingStore.isTraining" class="text-center mb-8">
        <button @click="startTraining" class="btn-primary text-lg px-12 py-4" :disabled="!isConfigValid">
          <svg class="w-5 h-5 mr-2 inline" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clip-rule="evenodd" />
          </svg>
          Start Training
        </button>
        <p class="text-sm text-gray-500 mt-2">This will train a new AI model from scratch</p>
      </div>

      <!-- Training History Chart -->
      <div v-if="trainingStore.trainingHistory.length > 0 || (scoreData.length > 0 && !trainingStore.isTraining)" class="card">
        <div class="card-header">
          <h3 class="text-lg font-semibold text-white">Training Progress History</h3>
        </div>
        <div class="card-body">
          <div class="h-64 relative">
            <canvas ref="historyChart" class="w-full h-full"></canvas>
          </div>
          <div class="mt-4 text-center text-sm text-gray-400">
            {{ scoreData.length > 0 ? `${scoreData.length} episodes completed` : `${trainingStore.trainingHistory.length} data points collected` }}
          </div>
        </div>
      </div>

      <!-- Model Information -->
      <div class="grid md:grid-cols-2 gap-8">
        <!-- Current Model Info -->
        <div class="card">
          <div class="card-header">
            <h3 class="text-lg font-semibold text-white">Current Model</h3>
          </div>
          <div class="card-body">
            <div v-if="trainingStore.isModelTrained || (scoreData.length > 0 && !trainingStore.isTraining)" class="space-y-4">
              <div class="flex justify-between">
                <span class="text-gray-400">Training Episodes:</span>
                <span class="font-mono text-white">{{ 
                  trainingStore.modelInfo?.training_episodes?.toLocaleString() || 
                  Math.max(trainingStore.trainingProgress.current_episode, scoreData.length).toLocaleString()
                }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Average Score:</span>
                <span class="font-mono text-snake-green">{{ 
                  (trainingStore.modelInfo?.average_score || trainingStore.trainingProgress.average_score || 0).toFixed(2) 
                }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Current Epsilon:</span>
                <span class="font-mono text-neural-blue">{{ 
                  ((trainingStore.modelInfo?.current_epsilon || trainingStore.trainingProgress.epsilon || 0.1) * 100).toFixed(1) 
                }}%</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Memory Size:</span>
                <span class="font-mono text-purple-400">{{ 
                  (trainingStore.modelInfo?.memory_size || trainingConfig.memory_size).toLocaleString() 
                }}</span>
              </div>
              <div v-if="scoreData.length > 0" class="flex justify-between">
                <span class="text-gray-400">Best Score:</span>
                <span class="font-mono text-yellow-400">{{ 
                  Math.max(...scoreData.map(d => d.current), 0)
                }}</span>
              </div>
            </div>
            <div v-else class="text-center py-8 text-gray-500">
              <svg class="w-12 h-12 mx-auto mb-4" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd" />
              </svg>
              <p>No trained model available</p>
              <p class="text-xs mt-1">Start training to create a new model</p>
            </div>
          </div>
        </div>

        <!-- Training Tips -->
        <div class="card">
          <div class="card-header">
            <h3 class="text-lg font-semibold text-white">Training Tips</h3>
          </div>
          <div class="card-body">
            <div class="space-y-4 text-sm text-gray-400">
              <div class="flex items-start space-x-3">
                <div class="w-2 h-2 bg-snake-green rounded-full mt-2 flex-shrink-0"></div>
                <div>
                  <p class="font-medium text-gray-300">Start with default parameters</p>
                  <p>The recommended settings work well for most cases</p>
                </div>
              </div>
              <div class="flex items-start space-x-3">
                <div class="w-2 h-2 bg-neural-blue rounded-full mt-2 flex-shrink-0"></div>
                <div>
                  <p class="font-medium text-gray-300">Monitor the average score</p>
                  <p>Look for steady improvement over episodes</p>
                </div>
              </div>
              <div class="flex items-start space-x-3">
                <div class="w-2 h-2 bg-purple-400 rounded-full mt-2 flex-shrink-0"></div>
                <div>
                  <p class="font-medium text-gray-300">Be patient</p>
                  <p>Training can take several hours for best results</p>
                </div>
              </div>
              <div class="flex items-start space-x-3">
                <div class="w-2 h-2 bg-yellow-400 rounded-full mt-2 flex-shrink-0"></div>
                <div>
                  <p class="font-medium text-gray-300">Save frequently</p>
                  <p>Checkpoints help prevent losing progress</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useTrainingStore } from '../stores/trainingStore'
import { trainingApi, healthApi, gameApi } from '../services/api'
import websocketService from '../services/websocket'
import TrainingChart from '../components/TrainingChart.vue'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  LineController,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  LineController,
  Title,
  Tooltip,
  Legend,
  Filler
)

const trainingStore = useTrainingStore()

// Reactive data for enhanced UI
const lastScore = ref(0)
const trainingStartTime = ref(null)
const trainingTime = ref('00:00:00')
const episodesPerMinute = ref(0)
const trainingLogs = ref([])
const scoreChart = ref(null)
const epsilonChart = ref(null)
const historyChart = ref(null)
let scoreChartInstance = null
let epsilonChartInstance = null
let historyChartInstance = null

// Chart data
const scoreData = ref([])
const epsilonData = ref([])
const episodeLabels = ref([])
const trainingChartData = ref({ labels: [], data: [] })

// Training configuration
const trainingConfig = ref({
  episodes: 1000,
  learning_rate: 0.001,
  batch_size: 32,
  memory_size: 10000,
  hidden_size: 256,
  gamma: 0.95,
  epsilon_start: 1.0,
  epsilon_end: 0.01,
  save_frequency: 100
})

// Model selection logic
const availableModels = ref([])
const selectedModel = ref('')
const loadingModels = ref(false)
const modelChanging = ref(false)

// Computed properties
const isConfigValid = computed(() => {
  return trainingConfig.value.episodes > 0 &&
         trainingConfig.value.learning_rate > 0 &&
         trainingConfig.value.batch_size > 0 &&
         trainingConfig.value.memory_size > 0
})

// Methods
const initializeCharts = async () => {
  await nextTick()

  // Destroy previous chart instances if they exist
  if (scoreChartInstance) {
    scoreChartInstance.destroy()
    scoreChartInstance = null
  }
  if (epsilonChartInstance) {
    epsilonChartInstance.destroy()
    epsilonChartInstance = null
  }
  if (historyChartInstance) {
    historyChartInstance.destroy()
    historyChartInstance = null
  }

  if (scoreChart.value) {
    scoreChartInstance = new ChartJS(scoreChart.value, {
      type: 'line',
      data: {
        labels: episodeLabels.value,
        datasets: [
          {
            label: 'Current Score',
            data: scoreData.value.map(d => d.current),
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            tension: 0.4,
            pointRadius: 2
          },
          {
            label: 'Average Score',
            data: scoreData.value.map(d => d.average),
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            tension: 0.4,
            pointRadius: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#d1d5db' }
          }
        },
        scales: {
          x: {
            ticks: { color: '#9ca3af' },
            grid: { color: 'rgba(156, 163, 175, 0.1)' }
          },
          y: {
            ticks: { color: '#9ca3af' },
            grid: { color: 'rgba(156, 163, 175, 0.1)' }
          }
        },
        animation: {
          duration: 750,
          easing: 'easeInOutQuart'
        }
      }
    })
  }

  if (epsilonChart.value) {
    epsilonChartInstance = new ChartJS(epsilonChart.value, {
      type: 'line',
      data: {
        labels: episodeLabels.value,
        datasets: [
          {
            label: 'Exploration Rate',
            data: epsilonData.value,
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            tension: 0.4,
            pointRadius: 2,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#d1d5db' }
          }
        },
        scales: {
          x: {
            ticks: { color: '#9ca3af' },
            grid: { color: 'rgba(156, 163, 175, 0.1)' }
          },
          y: {
            min: 0,
            max: 1,
            ticks: { 
              color: '#9ca3af',
              callback: function(value) {
                return (value * 100).toFixed(0) + '%'
              }
            },
            grid: { color: 'rgba(156, 163, 175, 0.1)' }
          }
        },
        animation: {
          duration: 750,
          easing: 'easeInOutQuart'
        }
      }
    })
  }

  // Initialize history chart if we have data
  if (historyChart.value && (scoreData.value.length > 0 || trainingStore.trainingHistory.length > 0)) {
    const historyLabels = scoreData.value.length > 0 ? episodeLabels.value : 
      Array.from({length: trainingStore.trainingHistory.length}, (_, i) => i.toString())
    
    const historyScores = scoreData.value.length > 0 ? scoreData.value.map(d => d.current) :
      trainingStore.trainingHistory.map(h => h.score || 0)
    
    const historyAvgScores = scoreData.value.length > 0 ? scoreData.value.map(d => d.average) :
      trainingStore.trainingHistory.map(h => h.averageScore || 0)

    historyChartInstance = new ChartJS(historyChart.value, {
      type: 'line',
      data: {
        labels: historyLabels,
        datasets: [
          {
            label: 'Episode Score',
            data: historyScores,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            tension: 0.4,
            pointRadius: 1,
            fill: false
          },
          {
            label: 'Average Score',
            data: historyAvgScores,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            tension: 0.4,
            pointRadius: 1,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#d1d5db' }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Episode',
              color: '#9ca3af'
            },
            ticks: { color: '#9ca3af' },
            grid: { color: 'rgba(156, 163, 175, 0.1)' }
          },
          y: {
            title: {
              display: true,
              text: 'Score',
              color: '#9ca3af'
            },
            ticks: { color: '#9ca3af' },
            grid: { color: 'rgba(156, 163, 175, 0.1)' }
          }
        },
        animation: {
          duration: 750,
          easing: 'easeInOutQuart'
        }
      }
    })
  }
}

const updateCharts = () => {
  const progress = trainingStore.trainingProgress
  const episode = progress.current_episode

  // Update data arrays
  if (episode > 0) {
    scoreData.value.push({
      current: progress.current_score,
      average: progress.average_score
    })
    epsilonData.value.push(progress.epsilon)
    episodeLabels.value.push(episode.toString())

    // Keep only last 100 data points for performance
    if (scoreData.value.length > 100) {
      scoreData.value.shift()
      epsilonData.value.shift()
      episodeLabels.value.shift()
    }

    // Update charts
    if (scoreChartInstance) {
      scoreChartInstance.data.labels = episodeLabels.value
      scoreChartInstance.data.datasets[0].data = scoreData.value.map(d => d.current)
      scoreChartInstance.data.datasets[1].data = scoreData.value.map(d => d.average)
      scoreChartInstance.update('none')
    }

    if (epsilonChartInstance) {
      epsilonChartInstance.data.labels = episodeLabels.value
      epsilonChartInstance.data.datasets[0].data = epsilonData.value
      epsilonChartInstance.update('none')
    }

    // Update history chart for live progress
    if (historyChartInstance) {
      historyChartInstance.data.labels = episodeLabels.value
      historyChartInstance.data.datasets[0].data = scoreData.value.map(d => d.current)
      historyChartInstance.data.datasets[1].data = scoreData.value.map(d => d.average)
      historyChartInstance.update('none')
    }
  }
}

const updateTrainingTime = () => {
  if (trainingStartTime.value && trainingStore.isTraining) {
    const elapsed = Date.now() - trainingStartTime.value
    const hours = Math.floor(elapsed / 3600000)
    const minutes = Math.floor((elapsed % 3600000) / 60000)
    const seconds = Math.floor((elapsed % 60000) / 1000)
    trainingTime.value = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
    
    // Calculate episodes per minute
    const totalMinutes = elapsed / 60000
    if (totalMinutes > 0) {
      episodesPerMinute.value = trainingStore.trainingProgress.current_episode / totalMinutes
    }
  }
}

const addTrainingLog = (message, type = 'info') => {
  trainingLogs.value.push({
    message,
    type,
    timestamp: Date.now()
  })
  
  // Keep only last 50 logs
  if (trainingLogs.value.length > 50) {
    trainingLogs.value.shift()
  }
}

const getLogColor = (type) => {
  switch (type) {
    case 'success': return 'text-green-400'
    case 'warning': return 'text-yellow-400'
    case 'error': return 'text-red-400'
    default: return 'text-gray-400'
  }
}

const formatTime = (timestamp) => {
  const date = new Date(timestamp)
  return date.toLocaleTimeString('en-US', { 
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}
const startTraining = async () => {
  try {
    // Reset any previous training completion state
    try {
      await trainingApi.resetTraining()
    } catch (resetError) {
      console.log('No previous training state to reset')
    }
    
    trainingStore.startTraining(trainingConfig.value)
    trainingStartTime.value = Date.now()
    
    // Initialize charts
    await initializeCharts()
    
    // Clear previous data
    scoreData.value = []
    epsilonData.value = []
    episodeLabels.value = []
    trainingLogs.value = []
    
    addTrainingLog('Training session started', 'success')
    
    // Call backend API to start training
    const response = await trainingApi.startTraining(trainingConfig.value)
    console.log('Training started:', response)
    
    addTrainingLog(`Training started with ${trainingConfig.value.episodes} episodes`, 'info')
    
    // Start polling for training status
    startTrainingPolling()
  } catch (error) {
    console.error('Failed to start training:', error)
    trainingStore.stopTraining()
    addTrainingLog('Failed to start training: ' + error.message, 'error')
    alert('Failed to start training: ' + error.message)
  }
}

const startTrainingPolling = () => {
  // Clear any existing polling interval first
  if (trainingStore.pollingInterval) {
    clearInterval(trainingStore.pollingInterval)
    trainingStore.pollingInterval = null
  }
  
  let consecutiveErrors = 0
  const maxErrors = 3
  
  // Poll training status every 2 seconds to reduce load
  const pollInterval = setInterval(async () => {
    try {
      // Safety check - if not training and no polling needed, stop
      if (!trainingStore.isTraining && consecutiveErrors > 0) {
        clearInterval(pollInterval)
        return
      }
      
      const status = await trainingApi.getTrainingStatus()
      
      if (status.is_training) {
        // Reset error counter on success
        consecutiveErrors = 0
        
        // Track score changes
        lastScore.value = trainingStore.trainingProgress.current_score
        
        // Update training progress with rate limiting
        trainingStore.updateTrainingProgress({
          current_episode: status.current_episode,
          total_episodes: status.total_episodes,
          current_score: status.current_score,
          average_score: status.average_score,
          epsilon: status.epsilon,
          recent_losses: status.recent_losses || []
        })
        
  // Update charts and time on every poll for live updates
  updateCharts()
  updateTrainingTime()
        
        // Add milestone logs
        if (status.current_episode % 100 === 0 && status.current_episode > 0) {
          addTrainingLog(`Completed ${status.current_episode} episodes (Avg Score: ${status.average_score.toFixed(2)})`, 'success')
        }
        
      } else {
        // Training completed
        trainingStore.stopTraining()
        
        // Show completion message
        if (status.current_episode > 0) {
          addTrainingLog(`Training completed! Final results: ${status.current_episode} episodes, Average Score: ${status.average_score.toFixed(2)}`, 'success')
          addTrainingLog('Model has been saved and is ready to use', 'success')
        } else {
          addTrainingLog('Training session completed', 'success')
        }
        
        clearInterval(pollInterval)
        
        // Optional: Load updated model info after a delay
        setTimeout(async () => {
          try {
            await loadModelInfo()
            // Initialize history chart after training completion
            if (historyChart.value && !historyChartInstance) {
              initializeCharts()
            }
          } catch (e) {
            console.log('Failed to load model info after training completion:', e)
          }
        }, 2000)
      }
    } catch (error) {
      consecutiveErrors++
      console.error('Failed to get training status:', error)
      
      // If too many consecutive errors, stop polling
      if (consecutiveErrors >= maxErrors) {
        trainingStore.stopTraining()
        addTrainingLog('Connection lost to training server', 'error')
        clearInterval(pollInterval)
      }
    }
  }, 2000) // Poll every 2 seconds instead of 1 second
  
  // Store interval ID to clear it later
  trainingStore.pollingInterval = pollInterval
}

const pauseTraining = async () => {
  try {
    await trainingApi.stopTraining()
    trainingStore.stopTraining()
    console.log('Training paused')
  } catch (error) {
    console.error('Failed to pause training:', error)
  }
}

const stopTraining = async () => {
  try {
    await trainingApi.stopTraining()
    trainingStore.stopTraining()
    
    // Clear polling interval
    if (trainingStore.pollingInterval) {
      clearInterval(trainingStore.pollingInterval)
      trainingStore.pollingInterval = null
    }
    
    addTrainingLog('Training stopped by user', 'warning')
    console.log('Training stopped')
  } catch (error) {
    console.error('Failed to stop training:', error)
    addTrainingLog('Failed to stop training: ' + error.message, 'error')
  }
}

// Load available models for selection
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


// Refresh model info from backend and update store
const refreshModelInfo = async () => {
  try {
    const response = await gameApi.getAiInfo();
    console.log('refreshModelInfo response:', response.data);
    trainingStore.updateModelInfo(response.data);
  } catch (error) {
    console.error('Failed to refresh model info:', error);
  }
}

// Load the selected model
const loadSelectedModel = async () => {
  if (!selectedModel.value) return
  try {
    modelChanging.value = true
    await gameApi.loadModel(selectedModel.value)
    await refreshModelInfo()
  } catch (error) {
    console.error('Failed to load model:', error)
    selectedModel.value = ''
  } finally {
    modelChanging.value = false
  }
}

// Initialize model selection on component mount
onMounted(async () => {
  await loadAvailableModels()
  websocketService.connectToTraining()
  websocketService.onMessage((data) => {
    console.log('Received training data:', data)
    if (data.type === 'training_update') {
      const { episode, average_score } = data;
      const newLabels = [...trainingChartData.value.labels, episode];
      const newData = [...trainingChartData.value.data, average_score];

      if (newLabels.length > 200) { // Keep more history
        newLabels.shift();
        newData.shift();
      }

      trainingChartData.value = {
        labels: newLabels,
        data: newData,
      };
    }
  })
})

// Watchers
watch(() => trainingStore.isTraining, (isTraining) => {
  if (isTraining) {
    // Start time tracking
    const timeInterval = setInterval(updateTrainingTime, 1000)
    trainingStore.timeInterval = timeInterval
  } else {
    // Clean up intervals
    if (trainingStore.timeInterval) {
      clearInterval(trainingStore.timeInterval)
      trainingStore.timeInterval = null
    }
  }
})

onUnmounted(() => {
  // Clear any polling intervals when component is destroyed
  if (trainingStore.pollingInterval) {
    clearInterval(trainingStore.pollingInterval)
    trainingStore.pollingInterval = null
  }
  
  if (trainingStore.timeInterval) {
    clearInterval(trainingStore.timeInterval)
    trainingStore.timeInterval = null
  }
  
  // Destroy chart instances
  if (scoreChartInstance) {
    scoreChartInstance.destroy()
  }
  if (epsilonChartInstance) {
    epsilonChartInstance.destroy()
  }
  if (historyChartInstance) {
    historyChartInstance.destroy()
  }
})

const loadModelInfo = async () => {
  try {
    // First test the health endpoint
    console.log('Testing backend connection...')
    const healthResponse = await healthApi.testConnection()
    console.log('Backend connection successful:', healthResponse)
    
    const status = await trainingApi.getTrainingStatus()
    if (status.is_training) {
      trainingStore.startTraining()
      startTrainingPolling()
    }
  } catch (error) {
    console.error('Failed to load training status:', error)
    addTrainingLog('Failed to connect to backend: ' + error.message, 'error')
    
    // Show user-friendly error message
    if (error.message.includes('Network Error') || error.message.includes('CORS')) {
      addTrainingLog('Backend server may not be running on http://localhost:8000', 'error')
      addTrainingLog('Please start the backend server: py -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000', 'warning')
    }
  }
}
</script>

<style scoped>
.train-page {
  min-height: 100vh;
  background-color: #111827;
}

.text-gradient {
  background: linear-gradient(45deg, #10b981, #3b82f6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.input-field {
  width: 100%;
  padding: 0.75rem 0.75rem;
  background-color: #374151;
  border: 1px solid #4b5563;
  border-radius: 0.5rem;
  color: #fff;
}
.input-field::placeholder {
  color: #9ca3af;
}
.input-field:focus {
  outline: none;
  box-shadow: 0 0 0 2px #10b981;
  border-color: transparent;
}

.status-training {
  background-color: #dbeafe;
  color: #1e40af;
}
.status-idle {
  background-color: #f3f4f6;
  color: #1f2937;
}

/* Animations */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.animate-fade-in {
  animation: fadeIn 0.3s ease-out;
}

/* Progress bar animations */
.progress-bar {
  transition: width 0.5s ease-out;
}

/* Pulse effect for active training */
.training-pulse {
  animation: trainingPulse 2s ease-in-out infinite;
}

@keyframes trainingPulse {
  0%, 100% { 
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4);
  }
  50% { 
    box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
  }
}

/* Chart container */
.chart-container {
  position: relative;
  height: 300px;
}

/* Dark mode for status indicators */
@media (prefers-color-scheme: dark) {
  .status-training {
    background-color: #1e3a8a;
    color: #bfdbfe;
  }
  .status-idle {
    background-color: #374151;
    color: #e5e7eb;
  }
}

/* Responsive text sizes */
@media (max-width: 640px) {
  .text-2xl {
    font-size: 1.25rem;
    line-height: 1.75rem;
  }
}
</style>
