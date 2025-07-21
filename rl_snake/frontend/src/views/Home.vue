<template>
  <div class="home-page">
    <!-- Hero Section -->
    <section class="hero-section">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div class="text-center">
          <h1 class="text-5xl md:text-7xl font-bold text-gradient mb-6">
            Snake AI
          </h1>
          <p class="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto">
            Watch an artificial intelligence learn to play Snake using Deep Q-Learning 
            with real-time decision visualization and neural network insights.
          </p>
          <div class="flex flex-col sm:flex-row gap-4 justify-center">
            <router-link to="/play" class="btn-primary text-lg px-8 py-3">
              Watch AI Play
            </router-link>
            <router-link to="/train" class="btn-secondary text-lg px-8 py-3">
              Train New Model
            </router-link>
          </div>
        </div>
      </div>
    </section>

    <!-- Features Section -->
    <section class="features-section">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div class="text-center mb-16">
          <h2 class="text-4xl font-bold text-white mb-4">
            Reinforcement Learning in Action
          </h2>
          <p class="text-xl text-gray-400 max-w-2xl mx-auto">
            Explore the fascinating world of AI decision-making through interactive visualization
          </p>
        </div>

        <div class="grid md:grid-cols-3 gap-8">
          <!-- Deep Q-Learning -->
          <div class="feature-card">
            <div class="feature-icon bg-neural-blue">
              <svg class="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clip-rule="evenodd" />
              </svg>
            </div>
            <h3 class="text-xl font-semibold text-white mb-3">Deep Q-Network</h3>
            <p class="text-gray-400 mb-4">
              Advanced neural network architecture with experience replay and target networks for stable learning.
            </p>
            <ul class="text-sm text-gray-500 space-y-1">
              <li>• 4-layer neural network</li>
              <li>• Experience replay buffer</li>
              <li>• Epsilon-greedy exploration</li>
            </ul>
          </div>

          <!-- Real-time Visualization -->
          <div class="feature-card">
            <div class="feature-icon bg-snake-green">
              <svg class="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clip-rule="evenodd" />
              </svg>
            </div>
            <h3 class="text-xl font-semibold text-white mb-3">Real-time Visualization</h3>
            <p class="text-gray-400 mb-4">
              See exactly how the AI thinks with Q-values, action probabilities, and neural network activity.
            </p>
            <ul class="text-sm text-gray-500 space-y-1">
              <li>• Live Q-value charts</li>
              <li>• Neural network visualization</li>
              <li>• Decision confidence metrics</li>
            </ul>
          </div>

          <!-- Interactive Training -->
          <div class="feature-card">
            <div class="feature-icon bg-purple-600">
              <svg class="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clip-rule="evenodd" />
              </svg>
            </div>
            <h3 class="text-xl font-semibold text-white mb-3">Interactive Training</h3>
            <p class="text-gray-400 mb-4">
              Train your own AI agent and watch its performance improve over thousands of episodes.
            </p>
            <ul class="text-sm text-gray-500 space-y-1">
              <li>• Live training metrics</li>
              <li>• Customizable parameters</li>
              <li>• Model checkpointing</li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <!-- Demo Section -->
    <section class="demo-section">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div class="text-center mb-12">
          <h2 class="text-3xl font-bold text-white mb-4">See It in Action</h2>
          <p class="text-gray-400 mb-8">Experience the AI playing Snake with live decision analysis</p>
        </div>

        <div class="demo-container">
          <div class="demo-video-placeholder">
            <div class="flex items-center justify-center h-full">
              <div class="text-center">
                <div class="w-20 h-20 bg-gradient-to-br from-snake-green to-neural-blue rounded-full flex items-center justify-center mb-4 mx-auto">
                  <svg class="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clip-rule="evenodd" />
                  </svg>
                </div>
                <h3 class="text-xl font-semibold text-white mb-2">Interactive Demo</h3>
                <p class="text-gray-400 mb-4">
                  Watch the AI make decisions in real-time and explore the neural network visualization.
                </p>
                <router-link to="/play" class="btn-primary">
                  Start Demo
                </router-link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- CTA Section -->
    <section class="cta-section">
      <div class="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8 py-16">
        <h2 class="text-4xl font-bold text-white mb-4">
          Ready to Explore AI Decision-Making?
        </h2>
        <p class="text-xl text-gray-300 mb-8">
          Dive deep into reinforcement learning with our interactive Snake AI visualization
        </p>
        <div class="flex flex-col sm:flex-row gap-4 justify-center">
          <router-link to="/play" class="btn-primary text-lg px-8 py-3">
            Watch AI Play
          </router-link>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
</script>

<style scoped>
.home-page {
  @apply min-h-screen;
}

.hero-section {
  @apply bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900;
  background-image: 
    radial-gradient(circle at 25% 25%, rgba(16, 185, 129, 0.1) 0%, transparent 50%),
    radial-gradient(circle at 75% 75%, rgba(59, 130, 246, 0.1) 0%, transparent 50%);
}

.text-gradient {
  background: linear-gradient(45deg, #10b981, #3b82f6, #8b5cf6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: gradient-shift 3s ease-in-out infinite;
}

@keyframes gradient-shift {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}

.features-section {
  @apply bg-gray-800;
}

.feature-card {
  @apply card p-8 text-center hover:transform hover:scale-105 transition-all duration-300;
}

.feature-icon {
  @apply w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6;
}

.demo-section {
  @apply bg-gray-800;
}

.demo-container {
  @apply max-w-4xl mx-auto;
}

.demo-video-placeholder {
  @apply bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl border-2 border-gray-700 h-96;
}

.cta-section {
  @apply bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900;
}

/* Animations */
.feature-card:hover {
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
}

.feature-card:hover .feature-icon {
  @apply transform scale-110;
}
</style>
