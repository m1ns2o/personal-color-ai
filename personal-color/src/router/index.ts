import { createRouter, createWebHistory } from 'vue-router'
import EntryView from '../views/EntryView.vue'
import HomeView from '../views/HomeView.vue'
import FaceShapeView from '../views/FaceShapeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'entry',
      component: EntryView,
    },
    {
      path: '/personal-color',
      name: 'personal-color',
      component: HomeView,
    },
    {
      path: '/face-shape',
      name: 'face-shape',
      component: FaceShapeView,
    },
  ],
})

export default router
