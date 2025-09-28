<template>
  
  <McLayout class="container">

    <McIntroduction
      :logoImg="'https://matechat.gitcode.com/logo.svg'"
      :title="'MateChat'"
      :subTitle="'Hi，欢迎使用 EDA agent'"
      :description="description"
    ></McIntroduction>

    <McHeader :title="'Pandas Data Analyst'">
      <template #operationArea>
        <div class="ops">
        </div>
      </template>
    </McHeader>

    <McLayoutContent class="content-container">
      <div v-if="uploadMsg" class="hint">{{ uploadMsg }}</div>

      <div v-if="preview.columns.length" class="panel">
        <div class="panel-title">Data Preview</div>
        <div class="table-wrap">
          <table cellpadding="6" cellspacing="0" class="table">
            <thead>
              <tr>
                <th v-for="c in preview.columns" :key="c">{{ c }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(r, idx) in preview.rows" :key="idx">
                <td v-for="c in preview.columns" :key="c">{{ r[c] }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div v-for="m in messages" :key="m.id">
        <template v-if="m.from==='user'">
          <McBubble :align="'right'" :avatarConfig="{ imgSrc: 'https://matechat.gitcode.com/png/demo/userAvatar.svg' }" class="user-bubble">
            <template #default>
              <div v-if="m.type==='text'" v-html="renderMarkdown(m.text)"></div>
            </template>
          </McBubble>
        </template>
        <template v-else>
          <McBubble v-if="m.type!=='plotly'" :align="'left'" :avatarConfig="{ imgSrc: 'https://matechat.gitcode.com/logo.svg' }" class="ai-bubble">
            <template #default>
              <div v-if="m.type==='text'" v-html="renderMarkdown(m.text)"></div>
              <div v-else-if="m.type==='table'" class="panel">
                <div class="panel-title">Data Table</div>
                <div class="table-wrap">
                  <table cellpadding="6" cellspacing="0" class="table">
                    <thead>
                      <tr>
                        <th v-for="c in m.payload.columns" :key="c">{{ c }}</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="(r, idx2) in m.payload.rows" :key="idx2">
                        <td v-for="c in m.payload.columns" :key="c">{{ r[c] }}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </template>
          </McBubble>
          <div v-else class="panel full">
            <div class="panel-title">Chart</div>
            <div :ref="el => setPlotlyRef(m.id, el)" class="plotly-full"></div>
          </div>
        </template>
      </div>
    </McLayoutContent>



    <div class="input-foot" style="justify-content: flex-start; margin-bottom: 12px;">
      <button
        @click="createSession" :disabled="creating"
        style="padding: 6px 16px; border-radius: 6px; border: 1px solid #eee; background: #f5f5f5; cursor: pointer;"
      >
        新建会话
      </button>
    </div>

    <McLayoutSender>
      <McInput :value="question" :maxLength="2000" @change="e => question = e" @submit="onSubmit">
        <template #extra>
        
          <div class="input-foot">

            <div class="input-foot-left">
              <input type="file" @change="onFileChange" accept=".csv,.xlsx,.xls" />
              <button @click="upload" :disabled="!sessionId || !file || uploading">{{ uploading ? '上传中…' : '上传' }}</button>
              <span class="count">{{ question.length }}/2000</span>
            </div>
            
          </div>
        </template>
      </McInput>
    </McLayoutSender>
  </McLayout>
</template>

<script setup>
import { ref, nextTick, onMounted, onBeforeUnmount } from 'vue'
import axios from 'axios'
import MarkdownIt from 'markdown-it'
import DOMPurify from 'dompurify'

const md = new MarkdownIt({ linkify: true, breaks: true })
function renderMarkdown(text) {
  try { return DOMPurify.sanitize(md.render(text || '')) } catch (_) { return text || '' }
}

const sessionId = ref('')
const creating = ref(false)

const file = ref(null)
const uploading = ref(false)
const uploadMsg = ref('')

const apiKey = ref('')
const validating = ref(false)
const keyValid = ref(null)

let msgSeq = 0
const messages = ref([])
const question = ref('')
const asking = ref(false)

const preview = ref({ columns: [], rows: [] })

const plotlyDivMap = new Map()

const description = [
  'Welcome to the Pandas Data Analyst AI. Upload a CSV or Excel file and ask questions about the data.',  
  'The AI agent will analyze your dataset and return either data tables or interactive charts.',
];

function setPlotlyRef(id, el) {
  if (!id) return
  if (el) plotlyDivMap.set(id, el); else plotlyDivMap.delete(id)
}

function onFileChange(e) {
  const files = e.target.files
  file.value = files && files[0] ? files[0] : null
}

async function createSession() {
  creating.value = true
  try {
    const { data } = await axios.post('/api/session')
    sessionId.value = data.session_id
  } finally { creating.value = false }
}

async function upload() {
  if (!sessionId.value || !file.value) return
  uploading.value = true
  uploadMsg.value = ''
  try {
    const form = new FormData()
    form.append('session_id', sessionId.value)
    form.append('file', file.value)
    const { data } = await axios.post('/api/upload', form, { headers: { 'Content-Type': 'multipart/form-data' } })
    uploadMsg.value = `已上传：${data.rows} 行 × ${data.cols} 列`
    if (data.preview) preview.value = data.preview
  } catch (e) { uploadMsg.value = '上传失败' }
  finally { uploading.value = false }
}

async function loadDemo() {
  if (!sessionId.value) return
  uploading.value = true
  uploadMsg.value = ''
  try {
    const form = new FormData()
    form.append('session_id', sessionId.value)
    form.append('name', 'bikes')
    const { data } = await axios.post('/api/demo-data', form)
    uploadMsg.value = `已加载示例数据：${data.rows} 行 × ${data.cols} 列`
    if (data.preview) preview.value = data.preview
  } catch (e) { uploadMsg.value = '加载示例数据失败' }
  finally { uploading.value = false }
}

function pushText(from, text) {
  messages.value.push({ id: `m${++msgSeq}`, from, type: 'text', text, avatarConfig: from==='user'? { name: 'user' } : { name: 'model' } })
}
function pushTable(payload) { messages.value.push({ id: `m${++msgSeq}`, from: 'model', type: 'table', payload, avatarConfig: { name: 'model' } }) }
function pushPlotly(fig) {
  const id = `m${++msgSeq}`
  messages.value.push({ id, from: 'model', type: 'plotly', payload: fig, avatarConfig: { name: 'model' } })
  nextTick(async () => {
    const el = plotlyDivMap.get(id)
    if (el && window.Plotly) {
      try {
        const layout = { ...(fig.layout || {}), autosize: true }
        if (!layout.margin) layout.margin = { l: 24, r: 24, t: 40, b: 40 }
        await window.Plotly.react(el, fig.data || [], layout, { responsive: true })
        await window.Plotly.Plots.resize(el)
      } catch (_) {}
    }
  })
}

function rerenderAllPlots() {
  nextTick(async () => {
    for (const m of messages.value) {
      if (m.type === 'plotly') {
        const el = plotlyDivMap.get(m.id)
        if (el && window.Plotly) {
          try { await window.Plotly.Plots.resize(el) } catch (_) {}
        }
      }
    }
  })
}

function onResize() { rerenderAllPlots() }

onMounted(() => { window.addEventListener('resize', onResize) })
onBeforeUnmount(() => { window.removeEventListener('resize', onResize) })

async function onSubmit(val) {
  if (typeof val === 'string') question.value = val
  if (!sessionId.value || !question.value || asking.value) return
  const q = question.value
  question.value = ''
  pushText('user', q)
  asking.value = true
  try {
    const { data } = await axios.post('/api/chat', {
      session_id: sessionId.value,
      question: q,
      api_key: apiKey.value || undefined,
    })
    if (data.ai_message) pushText('model', data.ai_message)
    if (data.dataframe && data.dataframe.columns && data.dataframe.rows) pushTable(data.dataframe)
    if (data.plotly_figure) pushPlotly(data.plotly_figure)
  } finally { asking.value = false }
}

async function validateKey() {
  if (!apiKey.value) return
  validating.value = true
  keyValid.value = null
  try {
    const form = new FormData()
    form.append('api_key', apiKey.value)
    await axios.post('/api/validate-key', form)
    keyValid.value = true
  } catch (_) { keyValid.value = false }
  finally { validating.value = false }
}
</script>

<style>
.container { width: 1000px; margin: 0 auto; min-height: 100vh; background: #fff; }
.ops { display: flex; gap: 8px; align-items: center; }
.ok { color: #2e7d32; font-size: 12px; }
.err { color: #c62828; font-size: 12px; }
.content-container { display:flex; flex-direction:column; gap:8px; }
.hint { color:#666; font-size:12px; }
.panel { border:1px solid #eee; border-radius:8px; padding:12px; box-sizing: border-box; }
.panel-title { font-weight:600; margin-bottom:8px; }
.table { width:100%; border-collapse:collapse; }
.table th, .table td { border:1px solid #eee; text-align:left; }
.table-wrap { overflow:auto; max-height:420px; }
.full { width: 100%; max-width: 100%; overflow-x: hidden; }
.plotly-full { width: 100%; height: 520px; overflow: hidden; }
.user-bubble :deep(.mc-bubble-content) { background: #e6f4ff; }
.ai-bubble :deep(.mc-bubble-content) { background: #f7f7f8; }
.ai-full { width: 100%; display: block; }
.input-foot { 
  display: flex; 
  align-items: center; 
  width: 100%; 
  gap: 16px; /* 增加元素间距 */
}
.input-foot input[type="file"] {
  margin-right: 8px;
}
.input-foot button {
  margin-right: 8px;
}
.count { font-size:12px; color:#71757f; }
</style>



