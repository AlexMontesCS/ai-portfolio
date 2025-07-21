<template>
  <div id="app" class="min-h-screen bg-gray-900">
    <nav class="bg-gray-800 border-b border-gray-700">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex items-center">
            <router-link to="/" class="flex items-center space-x-3">
              <div class="w-8 h-8 bg-gradient-to-br from-snake-green to-neural-blue rounded-lg flex items-center justify-center">
                <svg class="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M3 3a1 1 0 000 2v8a2 2 0 002 2h2.586l-1.293 1.293a1 1 0 101.414 1.414L10 15.414l2.293 2.293a1 1 0 001.414-1.414L12.414 15H15a2 2 0 002-2V5a1 1 0 100-2H3zm11.707 4.707a1 1 0 00-1.414-1.414L10 9.586 6.707 6.293a1 1 0 00-1.414 1.414l4 4a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
                </svg>
              </div>
              <h1 class="text-xl font-bold text-gradient">Snake AI</h1>
            </router-link>
          </div>
          
          <div class="flex items-center space-x-4">
            <router-link
              v-for="route in navRoutes"
              :key="route.path"
              :to="route.path"
              class="px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200"
              :class="$route.path === route.path 
                ? 'bg-snake-green text-white' 
                : 'text-gray-300 hover:text-white hover:bg-gray-700'"
            >
              {{ route.name }}
            </router-link>
          </div>
        </div>
      </div>
    </nav>

    <main class="flex-1">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>

    <!-- Global Loading Overlay -->
    <div v-if="gameStore.isLoading" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div class="bg-gray-800 rounded-lg p-6 max-w-sm w-full mx-4">
        <div class="flex items-center space-x-3">
          <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-snake-green"></div>
          <span class="text-gray-200">{{ gameStore.loadingMessage }}</span>
        </div>
      </div>
    </div>

    <!-- Global Error Toast -->
    <transition name="slide-up">
      <div v-if="gameStore.error" class="fixed bottom-4 right-4 bg-red-600 text-white px-6 py-4 rounded-lg shadow-lg z-40 max-w-md">
        <div class="flex justify-between items-start">
          <div>
            <h3 class="font-medium">Error</h3>
            <p class="text-sm mt-1">{{ gameStore.error }}</p>
          </div>
          <button @click="gameStore.clearError()" class="ml-4 text-red-200 hover:text-white">
            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
          </button>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useGameStore } from './stores/gameStore'

const gameStore = useGameStore()

const navRoutes = computed(() => [
  { path: '/', name: 'Home' },
  { path: '/play', name: 'Play' },
  { path: '/train', name: 'Train' }
])
</script>

<style scoped>
.text-gradient {
  background: linear-gradient(45deg, #10b981, #3b82f6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
</style>
