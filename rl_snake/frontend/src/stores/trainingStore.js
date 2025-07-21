import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useTrainingStore = defineStore('training', () => {
  // State
  const isTraining = ref(false)
  const pollingInterval = ref(null)
  const timeInterval = ref(null)
  const trainingProgress = ref({
    current_episode: 0,
    total_episodes: 0,
    current_score: 0,
    average_score: 0,
    epsilon: 1.0,
    recent_losses: [],
    recent_scores: []
  })
  
  const trainingHistory = ref([])
  const modelInfo = ref({
    model_loaded: false,
    training_episodes: 0,
    current_epsilon: 1.0,
    average_score: 0,
    memory_size: 0
  })
  
  const trainingMetrics = ref({
    episode_scores: [],
    episode_lengths: [],
    episode_losses: [],
    episode_epsilons: [],
    moving_averages: []
  })

  // Computed
  const trainingProgressPercent = computed(() => {
    if (trainingProgress.value.total_episodes === 0) return 0
    return (trainingProgress.value.current_episode / trainingProgress.value.total_episodes) * 100
  })
  
  const averageLoss = computed(() => {
    const losses = trainingProgress.value.recent_losses
    if (!losses || !Array.isArray(losses) || losses.length === 0) return 0
    
    // Filter out invalid values to prevent recursion
    const validLosses = losses.filter(loss => typeof loss === 'number' && !isNaN(loss))
    if (validLosses.length === 0) return 0
    
    return validLosses.reduce((sum, loss) => sum + loss, 0) / validLosses.length
  })
  
  const isModelTrained = computed(() => {
    return modelInfo.value.model_loaded && modelInfo.value.training_episodes > 0
  })

  // Actions
  const updateTrainingProgress = (progress) => {
    // Prevent updates if training is not active (avoids recursive updates after completion)
    if (!isTraining.value) {
      return
    }
    
    // Validate progress object
    if (!progress || typeof progress !== 'object') {
      console.warn('Invalid progress object:', progress)
      return
    }
    
    // Update individual properties to avoid reactivity issues
    if (typeof progress.current_episode === 'number') {
      trainingProgress.value.current_episode = progress.current_episode
    }
    if (typeof progress.total_episodes === 'number') {
      trainingProgress.value.total_episodes = progress.total_episodes
    }
    if (typeof progress.current_score === 'number') {
      trainingProgress.value.current_score = progress.current_score
    }
    if (typeof progress.average_score === 'number') {
      trainingProgress.value.average_score = progress.average_score
    }
    if (typeof progress.epsilon === 'number') {
      trainingProgress.value.epsilon = progress.epsilon
    }
    
    // Handle recent_losses array carefully to avoid recursion
    if (progress.recent_losses && Array.isArray(progress.recent_losses)) {
      // Only update if the array has actually changed (avoid unnecessary reactivity)
      const currentLosses = trainingProgress.value.recent_losses
      if (!currentLosses || currentLosses.length !== progress.recent_losses.length || 
          !progress.recent_losses.every((loss, index) => loss === currentLosses[index])) {
        // Create a frozen copy to prevent further modifications
        trainingProgress.value.recent_losses = Object.freeze([...progress.recent_losses])
      }
    }
    
    // Add to history (limit frequency to avoid performance issues)
    if (progress.current_episode && progress.current_episode % 5 === 0) {
      const historyEntry = {
        episode: progress.current_episode,
        score: progress.current_score,
        average_score: progress.average_score,
        epsilon: progress.epsilon,
        timestamp: Date.now()
      }
      
      trainingHistory.value.push(historyEntry)
      
      // Keep only last 1000 entries
      if (trainingHistory.value.length > 1000) {
        trainingHistory.value = trainingHistory.value.slice(-1000)
      }
    }
  }
  
  const updateModelInfo = (info) => {
    modelInfo.value = { ...modelInfo.value, ...info }
  }
  
  const updateTrainingMetrics = (metrics) => {
    trainingMetrics.value = { ...trainingMetrics.value, ...metrics }
  }
  
  const startTraining = (config = {}) => {
    isTraining.value = true
    
    // Reset and unfreeze the progress object for new training
    trainingProgress.value = {
      current_episode: 0,
      total_episodes: config.episodes || 1000,
      current_score: 0,
      average_score: 0,
      epsilon: 1.0,
      recent_losses: [],
      recent_scores: []
    }
  }
  
  const stopTraining = () => {
    // Set training flag to false first to prevent any more updates
    isTraining.value = false
    
    // Clear polling interval if it exists
    if (pollingInterval.value) {
      clearInterval(pollingInterval.value)
      pollingInterval.value = null
    }
    
    // Clear time interval if it exists
    if (timeInterval.value) {
      clearInterval(timeInterval.value)
      timeInterval.value = null
    }
    
    // Prevent any further reactive updates by freezing the progress object
    // This helps avoid recursion after training completion
    trainingProgress.value = Object.freeze({
      ...trainingProgress.value
    })
  }
  
  const resetTrainingData = () => {
    trainingHistory.value = []
    trainingMetrics.value = {
      episode_scores: [],
      episode_lengths: [],
      episode_losses: [],
      episode_epsilons: [],
      moving_averages: []
    }
  }

  return {
    // State
    isTraining,
    pollingInterval,
    timeInterval,
    trainingProgress,
    trainingHistory,
    modelInfo,
    trainingMetrics,
    
    // Computed
    trainingProgressPercent,
    averageLoss,
    isModelTrained,
    
    // Actions
    updateTrainingProgress,
    updateModelInfo,
    updateTrainingMetrics,
    startTraining,
    stopTraining,
    resetTrainingData
  }
})
