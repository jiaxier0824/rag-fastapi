const state = {
  sessionId: `uq-study-${crypto.randomUUID()}`,
  busy: false,
};

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
    const response = await fetch('/health');
    if (!response.ok) throw new Error();
    elements.status.textContent = '知识库服务运行正常';
  } catch {
    elements.status.textContent = '服务暂时无法连接';
  }
}

async function loadFiles() {
  try {
    const response = await fetch('/api/knowledge/files?page=1&page_size=100');
    if (!response.ok) throw new Error('无法读取文件列表');
    const data = await response.json();
    elements.fileCount.textContent = data.total;
    if (!data.items.length) {
      elements.fileList.innerHTML = '<p class="muted">还没有资料。先上传一份课程笔记吧。</p>';
      return;
    }
    elements.fileList.innerHTML = data.items.map((file) => `
      <div class="file-row">
        <span class="file-icon">${file.filename.endsWith('.md') ? 'MD' : 'TXT'}</span>
        <div class="file-info"><div class="file-name" title="${escapeHtml(file.filename)}">${escapeHtml(file.filename)}</div>
        <div class="file-meta">${formatSize(file.file_size)} · ${file.chunk_count} 个切片</div></div>
        <button class="delete-file" type="button" data-file-id="${file.file_id}" data-filename="${escapeHtml(file.filename)}" aria-label="删除 ${escapeHtml(file.filename)}">×</button>
      </div>`).join('');
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

function addMessage(role, text) {
  const article = document.createElement('article');
  article.className = `message ${role}`;
  article.innerHTML = `<span class="avatar">${role === 'user' ? '你' : 'UQ'}</span><div class="bubble">${escapeHtml(text)}</div>`;
  elements.messages.append(article);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return article;
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
  elements.send.disabled = true;
  addMessage('user', question);
  elements.question.value = '';
  const typing = document.createElement('p');
  typing.className = 'typing';
  typing.textContent = '正在检索你的知识库…';
  elements.messages.append(typing);
  try {
    const response = await fetch('/api/rag/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, session_id: state.sessionId }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || '问答请求失败');
    addMessage('assistant', result.answer);
  } catch (error) {
    addMessage('assistant', `暂时无法回答：${error.message}`);
  } finally {
    typing.remove();
    state.busy = false;
    elements.send.disabled = false;
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
elements.question.addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); elements.form.requestSubmit(); } });
document.querySelectorAll('.suggestion').forEach((button) => button.addEventListener('click', () => { elements.question.value = button.textContent; elements.question.focus(); }));
elements.newChat.addEventListener('click', () => {
  state.sessionId = `uq-study-${crypto.randomUUID()}`;
  elements.messages.innerHTML = newConversationMarkup();
  elements.question.focus();
  showToast('已开启新的学习会话');
});

checkHealth();
loadFiles();
