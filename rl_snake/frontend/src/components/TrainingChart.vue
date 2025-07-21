<template>
  <div class="chart-container">
    <canvas ref="chart"></canvas>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { Chart, registerables } from 'chart.js'
Chart.register(...registerables)

const props = defineProps({
  chartData: {
    type: Object,
    required: true
  }
})

const chart = ref(null)
let chartInstance = null

onMounted(() => {
  if (chart.value) {
    const ctx = chart.value.getContext('2d')
    chartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Average Score',
            data: [],
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            tension: 0.4,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: {
              display: true,
              text: 'Episode'
            }
          },
          y: {
            title: {
              display: true,
              text: 'Average Score'
            }
          }
        }
      }
    })
  }
})

watch(() => props.chartData, (newData) => {
  if (chartInstance && newData) {
    chartInstance.data.labels = newData.labels
    chartInstance.data.datasets[0].data = newData.data
    chartInstance.update()
  }
})

onUnmounted(() => {
  if (chartInstance) {
    chartInstance.destroy()
  }
})
</script>

<style scoped>
.chart-container {
  position: relative;
  height: 400px;
  width: 100%;
}
</style>