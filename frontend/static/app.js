const state = {
  sessionId: `uq-study-${crypto.randomUUID()}`,
  busy: false,
  files: [],
  course: '',
  mode: 'rag',
};

const welcomeMarkup = document.querySelector('#chat-messages').innerHTML;
const searchInput = document.querySelector('#file-search');
const courseFilters = document.querySelector('.course-filters');
function courseOf(filename) {
  return filename.toUpperCase().match(/(?:^|[^A-Z0-9])([A-Z]{4}\d{4})(?=[^A-Z0-9]|$)/)?.[1] || '';
}
function renderCourses() {
  const courses = [...new Set(state.files.map(file => courseOf(file.filename)).filter(Boolean))].sort();
  if (!courses.includes(state.course)) state.course = '';
  courseFilters.innerHTML = ['', ...courses].map(course => {
    const count = course ? state.files.filter(file => courseOf(file.filename) === course).length : state.files.length;
    return `<button class="${state.course === course ? 'active' : ''}" data-course="${course}" aria-pressed="${state.course === course}">${course || '全部'} <span>${count}</span></button>`;
  }).join('');
  const cards = document.querySelector('#course-cards');
  const subjects = {DECO6500:'系统思维与设计', INFS7203:'数据挖掘', INFS7410:'信息检索', REIT6811:'研究方法'};
  if (cards) cards.innerHTML = courses.map((course, index) => `<button class="course-card tone-${index % 4}" data-open-course="${course}"><span class="course-number">0${index + 1} <span>↗</span></span><strong>${course}</strong><span class="course-subject">${subjects[course] || '课程学习资料'}</span><small>${state.files.filter(file => courseOf(file.filename) === course).length} 份资料 <span>查看资料 →</span></small></button>`).join('');
}
function fileTitle(filename) {
  return filename.replace(/\.(md|txt)$/i, '').replace(/^[A-Za-z]{4}\d{4}[_ -]*/, '').replace(/(Week|Lecture)(\d+)/g, '$1 $2 ·').replaceAll('_', ' ');
}
function renderFiles() {
  const term = searchInput.value.trim().toLowerCase();
  const files = state.files.filter(file => file.filename.toLowerCase().includes(term) && (!state.course || courseOf(file.filename) === state.course));
  elements.fileList.innerHTML = files.map(file => `<div class="file-row"><span class="file-icon">${file.filename.endsWith('.md') ? 'MD' : 'TXT'}</span><div class="file-info"><div class="file-name" title="${escapeHtml(file.filename)}">${escapeHtml(fileTitle(file.filename))}</div><div class="file-meta">${courseOf(file.filename)} · ${formatSize(file.file_size)}</div></div><button class="delete-file" type="button" data-file-id="${escapeHtml(file.file_id)}" data-filename="${escapeHtml(file.filename)}" aria-label="删除 ${escapeHtml(file.filename)}">×</button></div>`).join('') || '<p class="muted">没有匹配的资料。试试其他课程或文件名。</p>';
}

const elements = {
  fileInput: document.querySelector('#file-input'),
  dropZone: document.querySelector('#drop-zone'),
  fileList: document.querySelector('#file-list'),
  fileCount: document.querySelector('#file-count'),
  status: document.querySelector('#service-status'),
  messages: document.querySelector('#chat-messages'),
  form: document.querySelector('#chat-form'),
  question: document.querySelector('#question-input'),
  send: document.querySelector('#send-button'),
  toast: document.querySelector('#toast'),
  newChat: document.querySelector('#new-chat'),
};

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add('show');
  window.setTimeout(() => elements.toast.classList.remove('show'), 3000);
}

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#039;', '"': '&quot;',
  }[character]));
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

async function checkHealth() {
  try {
    const [ragResponse, agentResponse] = await Promise.all([fetch('/health'), fetch('/api/agent/health')]);
    if (!ragResponse.ok) throw new Error();
    const agentOnline = agentResponse.ok;
    elements.status.textContent = agentOnline ? 'RAG 与 Agent 均已连接' : 'RAG 正常 · Agent 离线';
    document.querySelector('#connection-chip').textContent = agentOnline ? '双服务已连接' : 'RAG 已连接';
    document.querySelector('#connection-chip').classList.add('connected');
    document.querySelector('.sidebar-footer').classList.add('connected');
  } catch {
    elements.status.textContent = '服务暂时无法连接';
    document.querySelector('#connection-chip').textContent = '连接异常';
  }
}

async function loadFiles() {
  try {
    const files = [];
    let page = 1;
    let total = 0;
    do {
      const response = await fetch(`/api/knowledge/files?page=${page}&page_size=100`);
      if (!response.ok) throw new Error('无法读取文件列表');
      const data = await response.json();
      total = data.total;
      files.push(...data.items);
      if (!data.items.length) break;
      page += 1;
    } while (files.length < total);
    elements.fileCount.textContent = files.length;
    state.files = files;
    renderCourses();
    renderFiles();
  } catch (error) {
    elements.fileList.innerHTML = '<p class="muted">文件列表加载失败，请检查后端服务。</p>';
  }
}

async function uploadFile(file) {
  if (!file) return;
  const extension = file.name.split('.').pop().toLowerCase();
  if (!['md', 'txt'].includes(extension)) {
    showToast('当前仅支持 .md 和 .txt 文件');
    return;
  }
  const data = new FormData();
  data.append('file', file);
  elements.dropZone.classList.add('drag-over');
  try {
    const response = await fetch('/api/knowledge/upload', { method: 'POST', body: data });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || '上传失败');
    showToast(result.message);
    await loadFiles();
  } catch (error) {
    showToast(error.message);
  } finally {
    elements.dropZone.classList.remove('drag-over');
    elements.fileInput.value = '';
  }
}

function addMessage(role, text, sources = []) {
  const article = document.createElement('article');
  article.className = `message ${role}`;
  const sourceMarkup = role === 'assistant' && sources.length
    ? `<div class="source-list"><strong>↗ 参考资料 · ${sources.length} 个来源</strong>${sources.map((source) => `<span class="source-chip">${escapeHtml(source.filename)}</span>`).join('')}</div>`
    : '';
  article.innerHTML = `<span class="avatar">${role === 'user' ? '你' : 'UQ'}</span><div class="bubble"><div class="message-text">${escapeHtml(text)}</div>${sourceMarkup}</div>`;
  elements.messages.append(article);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return article;
}

function setMode(mode) {
  if (state.busy || !['rag', 'agent'].includes(mode)) return;
  state.mode = mode;
  document.body.classList.toggle('agent-mode', mode === 'agent');
  document.querySelectorAll('[data-mode]').forEach(button => {
    const active = button.dataset.mode === mode;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  document.querySelectorAll('[data-set-mode]').forEach(button => button.classList.toggle('active', button.dataset.setMode === mode));
  document.querySelector('#active-mode-label').textContent = mode === 'agent' ? 'Agent 任务执行' : 'RAG 知识问答';
  document.querySelector('#composer-mode-title').textContent = mode === 'agent' ? 'Agent 任务执行' : '知识库问答';
  document.querySelector('#composer-mode-hint').textContent = mode === 'agent' ? '规划任务并自主选择工具' : '从已上传资料中寻找答案';
  document.querySelector('#send-label').textContent = mode === 'agent' ? '执行' : '发送';
  elements.question.placeholder = mode === 'agent'
    ? '描述一个任务，例如：为 INFS7410 制定复习计划…'
    : '今天想弄懂什么？问问你的课程资料…';
  elements.question.focus();
}

function addAgentRun() {
  const article = document.createElement('article');
  article.className = 'message assistant agent-message';
  article.innerHTML = '<span class="avatar agent-avatar">A</span><div class="bubble"><div class="agent-run-title"><span>AGENT RUN</span><b>执行中</b></div><div class="agent-progress" aria-live="polite"></div><div class="message-text"></div><div class="agent-meta"></div></div>';
  elements.messages.append(article);
  return article;
}

async function runAgentTask(question) {
  const article = addAgentRun();
  const progress = article.querySelector('.agent-progress');
  const answer = article.querySelector('.message-text');
  const meta = article.querySelector('.agent-meta');
  const runState = article.querySelector('.agent-run-title b');
  const statuses = new Set();
  const toolLabels = {search_course_knowledge: '课程资料检索', create_or_update_study_plan: '创建学习计划', get_study_plan: '读取学习计划', list_study_plans: '列出学习计划', save_learning_preferences: '保存学习偏好'};
  const addStatus = content => {
    if (!content || statuses.has(content)) return;
    statuses.add(content);
    progress.insertAdjacentHTML('beforeend', `<span><i></i>${escapeHtml(content)}</span>`);
    elements.messages.scrollTop = elements.messages.scrollHeight;
  };
  addStatus('Agent 正在分析任务');
  const response = await fetch('/api/agent/chat/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question, session_id: state.sessionId}),
  });
  if (!response.ok || !response.body) throw new Error('Agent 服务连接失败');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalEvent = null;
  while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || '';
    for (const block of blocks) {
      const raw = block.split(/\r?\n/).filter(line => line.startsWith('data:')).map(line => line.slice(5).trim()).join('');
      if (!raw) continue;
      const event = JSON.parse(raw);
      if (event.type === 'status') addStatus(event.content);
      if (event.type === 'content') {
        answer.textContent += event.content || '';
        elements.messages.scrollTop = elements.messages.scrollHeight;
      }
      if (event.type === 'error') throw new Error(event.content || 'Agent 执行失败');
      if (event.type === 'done') finalEvent = event;
    }
    if (done) break;
  }
  runState.textContent = '已完成';
  runState.classList.add('done');
  if (!answer.textContent.trim()) answer.textContent = '任务已执行完成。';
  if (finalEvent) {
    const tools = (finalEvent.tools_called || []).map(tool => toolLabels[tool] || tool);
    const sources = finalEvent.sources || [];
    meta.innerHTML = `${tools.length ? `<div><strong>调用工具</strong>${tools.map(tool => `<span class="tool-chip">${escapeHtml(tool)}</span>`).join('')}</div>` : ''}${sources.length ? `<div><strong>参考资料</strong>${sources.map(source => `<span class="source-chip">${escapeHtml(source.filename)}</span>`).join('')}</div>` : ''}${finalEvent.trace_id ? `<small>TRACE · ${escapeHtml(finalEvent.trace_id.slice(0, 12))}</small>` : ''}`;
  }
}

function newConversationMarkup() {
  return `
    <article class="welcome-card new-conversation-card">
      <div class="hero-orb" aria-hidden="true"><span>✦</span></div>
      <p class="eyebrow">NEW CONVERSATION</p>
      <h3>新的学习对话，<em>从这里开始。</em></h3>
      <p>新的会话会使用新的历史记录。你可以继续围绕同一门课程提问，或切换到另一份资料。</p>
      <div class="capabilities">
        <span>⌕ 课程资料检索</span>
        <span>◌ 独立会话历史</span>
      </div>
    </article>`;
}

async function sendQuestion(question) {
  if (!question || state.busy) return;
  state.busy = true;
  elements.messages.querySelector('.welcome-card')?.remove();
  elements.newChat.disabled = true;
  elements.send.disabled = true;
  addMessage('user', question);
  elements.question.value = '';
  let typing = null;
  try {
    if (state.mode === 'agent') {
      await runAgentTask(question);
    } else {
      typing = document.createElement('p');
      typing.className = 'typing';
      typing.textContent = '正在检索你的知识库…';
      elements.messages.append(typing);
      const response = await fetch('/api/rag/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, session_id: state.sessionId }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || '问答请求失败');
      addMessage('assistant', result.answer, result.sources || []);
    }
  } catch (error) {
    addMessage('assistant', `暂时无法完成：${error.message}`);
  } finally {
    typing?.remove();
    state.busy = false;
    elements.send.disabled = false;
    elements.newChat.disabled = false;
    elements.question.focus();
  }
}

elements.fileInput.addEventListener('change', (event) => uploadFile(event.target.files[0]));
['dragenter', 'dragover'].forEach((eventName) => elements.dropZone.addEventListener(eventName, (event) => { event.preventDefault(); elements.dropZone.classList.add('drag-over'); }));
['dragleave', 'drop'].forEach((eventName) => elements.dropZone.addEventListener(eventName, (event) => { event.preventDefault(); elements.dropZone.classList.remove('drag-over'); }));
elements.dropZone.addEventListener('drop', (event) => uploadFile(event.dataTransfer.files[0]));
elements.fileList.addEventListener('click', async (event) => {
  const button = event.target.closest('.delete-file');
  if (!button || !window.confirm(`确定删除「${button.dataset.filename}」吗？`)) return;
  try {
    const response = await fetch(`/api/knowledge/files/${button.dataset.fileId}`, { method: 'DELETE' });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || '删除失败');
    showToast(`${result.filename} 已删除`);
    await loadFiles();
  } catch (error) { showToast(error.message); }
});
elements.form.addEventListener('submit', (event) => { event.preventDefault(); sendQuestion(elements.question.value.trim()); });
elements.question.addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); elements.form.requestSubmit(); } });
elements.messages.addEventListener('click', event => { const button = event.target.closest('.suggestion'); if (button) { setMode(button.dataset.suggestionMode || 'rag'); elements.question.value = button.dataset.question; elements.question.focus(); } });
elements.newChat.addEventListener('click', () => {
  state.sessionId = `uq-study-${crypto.randomUUID()}`;
  if (state.busy) return;
  elements.messages.innerHTML = welcomeMarkup;
  renderCourses();
  setMode(state.mode);
  elements.messages.scrollTop = 0;
  elements.question.focus();
  showToast('已开启新的学习会话');
});

checkHealth();
loadFiles();
searchInput.addEventListener('input', renderFiles);
document.querySelector('.mode-switch').addEventListener('click', event => { const button = event.target.closest('[data-mode]'); if (button) setMode(button.dataset.mode); });
elements.messages.addEventListener('click', event => { const button = event.target.closest('[data-set-mode]'); if (button) setMode(button.dataset.setMode); });
elements.messages.addEventListener('click', event => {
  const card = event.target.closest('[data-open-course]');
  if (!card) return;
  state.course = card.dataset.openCourse;
  searchInput.value = '';
  renderCourses();
  renderFiles();
  if (window.innerWidth <= 760) toggleLibrary(true);
});
courseFilters.addEventListener('click', event => {
  const button = event.target.closest('[data-course]');
  if (!button) return;
  state.course = button.dataset.course;
  document.querySelectorAll('[data-course]').forEach(item => { item.classList.toggle('active', item === button); item.setAttribute('aria-pressed', String(item === button)); });
  renderFiles();
});
function toggleLibrary(open) {
  document.body.classList.toggle('library-open', open);
  document.querySelector('#library-backdrop').hidden = !open;
  document.querySelector('#open-library').setAttribute('aria-expanded', String(open));
  document.querySelector('.sidebar').inert = !open && window.innerWidth <= 760;
  if (open) searchInput.focus();
  else document.querySelector('#open-library').focus();
}
document.querySelector('#open-library').addEventListener('click', () => toggleLibrary(true));
document.querySelector('#close-library').addEventListener('click', () => toggleLibrary(false));
document.querySelector('#library-backdrop').addEventListener('click', () => toggleLibrary(false));
document.addEventListener('keydown', event => { if (event.key === 'Escape' && document.body.classList.contains('library-open')) toggleLibrary(false); });
window.addEventListener('resize', () => { document.querySelector('.sidebar').inert = window.innerWidth <= 760 && !document.body.classList.contains('library-open'); });
document.querySelector('.sidebar').inert = window.innerWidth <= 760;
