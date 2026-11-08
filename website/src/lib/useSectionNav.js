// ============================================================================
// Section / page navigation composable.
//
// Content sections live on the leaderboard route; the Imports tool lives on its
// own route. Navigating from one to the other needs a route change *before* the
// scroll, otherwise `scrollIntoView` runs against a page that is not mounted yet
// and silently does nothing.
//
// This lives outside `store/ui.js` on purpose: importing the router into that
// module would create a cycle (router → LeaderboardPage → store/ui → router).
// ============================================================================
import { computed } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { MAIN_SECTIONS } from './sections';
import { activeDataset } from '../services/api';

const LEADERBOARD_ROUTE = 'leaderboard';
const SECTION_IDS = new Set(MAIN_SECTIONS.map(section => section.id));

export function useSectionNav() {
  const router = useRouter();
  const route = useRoute();

  const onLeaderboardPage = computed(() => route.name === LEADERBOARD_ROUTE);

  /**
   * Carry the selected dataset across routes. Switching to /imports would
   * otherwise drop the query string and silently reset the UI to the default
   * (prototype) bundle.
   */
  function preservedQuery() {
    const dataset = activeDataset();
    return dataset ? { dataset } : {};
  }

  function pageLocation(name) {
    return { name, query: preservedQuery() };
  }

  /** Jump to a content section, routing back if we are on another page. */
  function goToSection(id) {
    const scroll = () => {
      const element = document.getElementById(id);
      element?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };
    if (onLeaderboardPage.value || !SECTION_IDS.has(id)) {
      scroll();
      return;
    }
    router.push({ name: LEADERBOARD_ROUTE, query: preservedQuery() }).then(() => {
      // Two frames: let the leaderboard page mount and settle its layout.
      requestAnimationFrame(() => requestAnimationFrame(scroll));
    });
  }

  /** Jump to a standalone page; re-clicking scrolls to top. */
  function goToPage(name) {
    if (route.name === name) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      return;
    }
    return router.push(pageLocation(name));
  }

  return { goToSection, goToPage, pageLocation, onLeaderboardPage, currentRouteName: computed(() => route.name) };
}
