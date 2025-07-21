import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Play from '../views/Play.vue'
import Train from '../views/Train.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'Home',
      component: Home
    },
    {
      path: '/play',
      name: 'Play',
      component: Play
    },
    {
      path: '/train',
      name: 'Train',
      component: Train
    },
  ]
})

export default router
