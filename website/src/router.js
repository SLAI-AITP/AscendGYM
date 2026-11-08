import { createRouter, createWebHashHistory } from 'vue-router';
import LeaderboardPage from './pages/LeaderboardPage.vue';
import MethodologyPage from './pages/MethodologyPage.vue';
import ImportsPage from './pages/ImportsPage.vue';

// Hash history keeps deep links working on any static host without
// server-side rewrite rules.
export default createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', name: 'leaderboard', component: LeaderboardPage },
    // Methodology and Imports are standalone pages: they are reading material /
    // tools rather than ranking views, so they leave the long-scroll to keep the
    // leaderboard focused on results.
    // Keep the primary navigation destinations in the app bundle so a rebuild
    // cannot leave an open homepage pointing at removed lazy page chunks.
    { path: '/methodology', name: 'methodology', component: MethodologyPage },
    // Data imports live on their own page: they are a tool, not a ranking view.
    { path: '/imports', name: 'imports', component: ImportsPage },
    { path: '/design-system', name: 'design-system', component: () => import('./pages/DesignSystemPage.vue') }
  ],
  scrollBehavior(to, from, savedPosition) {
    // Resolve section anchors on both the leaderboard and standalone pages.
    if (to.hash) {
      return { el: to.hash, behavior: 'smooth' };
    }
    return savedPosition || { top: 0 };
  }
});
