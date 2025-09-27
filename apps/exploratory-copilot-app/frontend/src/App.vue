<template>
  <div style="display:flex; min-height:100vh; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;">
    <aside style="width: 320px; padding: 16px; border-right: 1px solid #eee; background:#fafbfc;">
      <h3 style="margin-top:0;">EDA Copilot: Data Upload/Selection</h3>
      <div style="margin-top:16px;">
        <h4>1) 创建会话</h4>
        <button @click="createSession" :disabled="creating" style="width:100%;">{{ creating ? '创建中...' : (sessionId ? '重新创建 Session' : '创建 Session') }}</button>
        <div v-if="sessionId" style="font-size:12px; color:#666; margin-top:6px;">{{ sessionId }}</div>
      </div>

      <div style="margin-top:16px;">
        <h4>Upload Data (CSV or Excel)</h4>
        <input type="file" @change="onFileChange" accept=".csv,.xlsx,.xls" style="width:100%;"/>
        <div style="display:flex; gap:8px; margin-top:8px;">
          <button @click="upload" :disabled="!sessionId || !file || uploading" style="flex:1;">{{ uploading ? '上传中...' : '上传' }}</button>
          <button @click="loadDemo" :disabled="!sessionId || uploading" style="flex:1;">Use demo data</button>
        </div>
        <div v-if="uploadMsg" style="margin-top:6px; font-size:12px; color:#666;">{{ uploadMsg }}</div>
      </div>

      <div style="margin-top:16px;">
        <h4>Enter your OpenAI API Key</h4>
        <input v-model="apiKey" type="password" placeholder="API Key" style="width:100%; padding:6px;" />
        <div style="display:flex; gap:8px; margin-top:8px;">
          <button @click="validateKey" :disabled="!apiKey || validating" style="flex:1;">{{ validating ? '校验中...' : '校验' }}</button>
          <span v-if="keyValid===true" style="color:#2e7d32; align-self:center;">API Key is valid!</span>
          <span v-if="keyValid===false" style="color:#c62828; align-self:center;">Invalid API Key</span>
        </div>
      </div>
    </aside>

    <main style="flex:1; padding: 16px;">
      <div v-if="preview.columns.length" style="margin-bottom:12px;">
        <div style="font-weight:600; margin-bottom:6px;">Data Preview</div>
        <div style="overflow:auto; max-height:360px; border:1px solid #eee;">
          <table cellpadding="6" cellspacing="0" style="width:100%; border-collapse:collapse;">
            <thead>
              <tr>
                <th v-for="c in preview.columns" :key="c" style="border-bottom:1px solid #eee; text-align:left;">{{ c }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(r, idx) in preview.rows" :key="idx">
                <td v-for="c in preview.columns" :key="c" style="border-bottom:1px solid #f3f3f3;">{{ r[c] }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div style="margin-bottom:12px;">
        <div style="font-weight:600; margin-bottom:6px;">How can I help you?</div>
        <div style="display:flex; gap:8px;">
          <input v-model="question" placeholder="Enter your question here" style="flex:1; padding:8px; border:1px solid #ddd; border-radius:6px;" />
          <button @click="ask" :disabled="!sessionId || !question || asking" style="width:120px;">{{ asking ? '思考中...' : '发送' }}</button>
        </div>
      </div>

      <div v-if="aiMessage" style="white-space: pre-wrap; margin-bottom:12px;">{{ aiMessage }}</div>

      <div v-if="table.rows.length" style="margin: 12px 0;">
        <h4>Data Table</h4>
        <div style="overflow:auto; max-height:420px; border:1px solid #eee;">
          <table cellpadding="6" cellspacing="0" style="width:100%; border-collapse:collapse;">
            <thead>
              <tr>
                <th v-for="c in table.columns" :key="c" style="border-bottom:1px solid #eee; text-align:left;">{{ c }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(r, idx) in table.rows" :key="idx">
                <td v-for="c in table.columns" :key="c" style="border-bottom:1px solid #f3f3f3;">{{ r[c] }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div v-if="plotlyFigure" style="margin: 12px 0;">
        <h4>Chart</h4>
        <div ref="plotlyDiv" style="width:100%; height:480px;"></div>
      </div>

      <div v-if="reportUrl" style="margin: 12px 0;">
        <h4>Sweetviz Report</h4>
        <a :href="reportUrl" target="_blank" rel="noreferrer">Open report</a>
      </div>

      <div v-if="dtaleUrl" style="margin: 12px 0;">
        <h4>Dtale</h4>
        <a :href="dtaleUrl" target="_blank" rel="noreferrer">Open Dtale</a>
      </div>

      <div v-if="missingMatrixUrl || missingBarUrl || missingHeatmapUrl" style="margin: 12px 0;">
        <h4>Missingness Visualization</h4>
        <div style="display:flex; gap: 12px; flex-wrap: wrap;">
          <img v-if="missingMatrixUrl" :src="missingMatrixUrl" alt="missing-matrix" style="max-width: 31%; border: 1px solid #eee;" />
          <img v-if="missingBarUrl" :src="missingBarUrl" alt="missing-bar" style="max-width: 31%; border: 1px solid #eee;" />
          <img v-if="missingHeatmapUrl" :src="missingHeatmapUrl" alt="missing-heatmap" style="max-width: 31%; border: 1px solid #eee;" />
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'
import axios from 'axios'

const sessionId = ref('')
const creating = ref(false)

const file = ref(null)
const uploading = ref(false)
const uploadMsg = ref('')

const apiKey = ref('')
const validating = ref(false)
const keyValid = ref(null)

const question = ref('')
const asking = ref(false)
const aiMessage = ref('')
const tool = ref('')

const preview = ref({ columns: [], rows: [] })
const table = ref({ columns: [], rows: [] })
const plotlyFigure = ref(null)
const reportUrl = ref('')
const dtaleUrl = ref('')
const missingMatrixUrl = ref('')
const missingBarUrl = ref('')
const missingHeatmapUrl = ref('')

const plotlyDiv = ref(null)

function onFileChange(e) {
  const files = e.target.files
  file.value = files && files[0] ? files[0] : null
}

async function createSession() {
  creating.value = true
  try {
    const { data } = await axios.post('/api/session')
    sessionId.value = data.session_id
  } finally {
    creating.value = false
  }
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
    if (data.preview) {
      preview.value = data.preview
    }
  } catch (e) {
    uploadMsg.value = '上传失败'
  } finally {
    uploading.value = false
  }
}

async function loadDemo() {
  if (!sessionId.value) return
  uploading.value = true
  uploadMsg.value = ''
  try {
    const form = new FormData()
    form.append('session_id', sessionId.value)
    form.append('name', 'churn')
    const { data } = await axios.post('/api/demo-data', form)
    uploadMsg.value = `已加载示例数据：${data.rows} 行 × ${data.cols} 列`
    if (data.preview) {
      preview.value = data.preview
    }
  } catch (e) {
    uploadMsg.value = '加载示例数据失败'
  } finally {
    uploading.value = false
  }
}

async function ask() {
  if (!sessionId.value || !question.value) return
  asking.value = true
  aiMessage.value = ''
  tool.value = ''
  table.value = { columns: [], rows: [] }
  plotlyFigure.value = null
  reportUrl.value = ''
  dtaleUrl.value = ''
  missingMatrixUrl.value = ''
  missingBarUrl.value = ''
  missingHeatmapUrl.value = ''
  try {
    const { data } = await axios.post('/api/chat', {
      session_id: sessionId.value,
      question: question.value,
      api_key: apiKey.value || undefined,
    })
    aiMessage.value = data.ai_message || ''
    tool.value = data.tool || ''
    if (data.dataframe && data.dataframe.columns && data.dataframe.rows) {
      table.value = { columns: data.dataframe.columns, rows: data.dataframe.rows }
    }
    if (data.plotly_figure) {
      plotlyFigure.value = data.plotly_figure
    }
    if (data.report_url) {
      reportUrl.value = data.report_url
    }
    if (data.dtale_url) {
      dtaleUrl.value = data.dtale_url
    }
    if (data.missing_matrix_url) missingMatrixUrl.value = data.missing_matrix_url
    if (data.missing_bar_url) missingBarUrl.value = data.missing_bar_url
    if (data.missing_heatmap_url) missingHeatmapUrl.value = data.missing_heatmap_url
  } finally {
    asking.value = false
  }
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
  } catch (_) {
    keyValid.value = false
  } finally {
    validating.value = false
  }
}

watch(plotlyFigure, async (fig) => {
  if (!fig) return
  await nextTick()
  if (window.Plotly && plotlyDiv.value) {
    try {
      const layout = fig.layout || {}
      await window.Plotly.react(plotlyDiv.value, fig.data || [], layout, { responsive: true })
    } catch (e) {
      // ignore
    }
  }
})
</script>

<style>
button {
  padding: 6px 12px;
  border: 1px solid #ddd;
  background: #f8f9fa;
  border-radius: 6px;
  cursor: pointer;
}
button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
textarea {
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 6px;
}
table {
  border-collapse: collapse;
}
th, td {
  border: 1px solid #eee;
}
</style>


