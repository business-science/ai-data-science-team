import { createApp } from 'vue'
import App from './App.vue'
import MateChat from '@matechat/core'
import '@devui-design/icons/icomoon/devui-icon.css'

createApp(App)
  .use(MateChat)
  .mount('#app')


