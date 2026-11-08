import { createApp } from 'vue';
import App from './App.vue';
import router from './router';
import './assets/styles.css';
import './assets/styles-v2.css';
import './assets/design-system.css';
import './assets/desktop-typography.css';
import './assets/readability.css';
import './assets/display-scale.css';

createApp(App).use(router).mount('#app');
