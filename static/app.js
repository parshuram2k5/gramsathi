/**
 * GramSathi (ग्राम साथी) - Welfare RAG Chatbot Client Application
 * Features:
 * - Conversational RAG with 5,061 Indian Government Schemes
 * - Deep AI Judge Tournament vs Express Citizen Mitr modes
 * - Real-time Pipeline Progress Tracker
 * - Speech-to-Text (STT) Voice Input & Text-to-Speech (TTS) Voice Output
 * - Interactive AI Judge Scorecard Modal & 5 Multi-LLM Persona Viewer
 * - Full 5,061 Schemes Explorer Directory with instant search
 * - Persistent Conversation History in LocalStorage
 * - Theme switcher (Dark / Light)
 */

// Application State
const state = {
  currentSessionId: null,
  conversations: {},
  activeState: 'All',
  activeMode: 'deep', // 'deep' or 'express'
  isGenerating: false,
  indianStates: [],
  pipelineTimer: null,
  speechSynth: window.speechSynthesis || null,
  currentUtterance: null,
  activeSpeechBtn: null,
  recognition: null,
  isRecording: false,
  lastQueryResult: null,
  explorerOffset: 0,
  explorerLimit: 18,
  debounceTimer: null
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initSpeechRecognition();
  loadStoredPreferences();
  fetchStatesList();
  loadConversationHistory();
  
  // Set up Enter key listener
  const inputEl = document.getElementById('chatInput');
  if (inputEl) {
    inputEl.focus();
  }

  // Handle escape key to close modals
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      document.querySelectorAll('.modal-backdrop.active').forEach(m => m.classList.remove('active'));
    }
  });

  // Check health and display in navbar
  checkSystemHealth();
});

// ============================================================================
// Theme Management
// ============================================================================

function initTheme() {
  const savedTheme = localStorage.getItem('gramsathi_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('gramsathi_theme', next);
  updateThemeIcon(next);
}

function updateThemeIcon(theme) {
  const icon = document.getElementById('themeIcon');
  if (icon) {
    icon.innerText = theme === 'dark' ? '🌙' : '☀️';
  }
}

// ============================================================================
// System Health & State List
// ============================================================================

async function checkSystemHealth() {
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const data = await res.json();
      const statusText = document.getElementById('navStatusText');
      if (statusText && data.indexed_schemes) {
        statusText.innerText = `${data.indexed_schemes.toLocaleString()} Central & State Schemes Ready`;
      }
    }
  } catch (err) {
    console.warn("Could not reach health check:", err);
  }
}

async function fetchStatesList() {
  try {
    const res = await fetch('/api/states');
    if (!res.ok) return;
    const data = await res.json();
    state.indianStates = data.states || [];

    const stateSelect = document.getElementById('stateSelect');
    const explorerState = document.getElementById('explorerStateSelect');

    state.indianStates.forEach(st => {
      const opt = document.createElement('option');
      opt.value = st;
      opt.textContent = st;
      if (stateSelect) stateSelect.appendChild(opt.cloneNode(true));
      if (explorerState) explorerState.appendChild(opt);
    });

    if (state.activeState && stateSelect) {
      stateSelect.value = state.activeState;
    }
  } catch (err) {
    console.error("Error loading states:", err);
  }
}

function loadStoredPreferences() {
  const savedState = localStorage.getItem('gramsathi_state');
  if (savedState) {
    state.activeState = savedState;
    updateActiveStateBadge(savedState);
  }
  const savedMode = localStorage.getItem('gramsathi_mode');
  if (savedMode) {
    setMode(savedMode, false);
  }
}

function onStateChange() {
  const select = document.getElementById('stateSelect');
  if (!select) return;
  state.activeState = select.value;
  localStorage.setItem('gramsathi_state', select.value);
  updateActiveStateBadge(select.value);
  showToast(`State preference set to: ${select.value}`);
}

function updateActiveStateBadge(val) {
  const label = document.getElementById('activeStateLabel');
  const tag = document.getElementById('activeStateTag');
  const text = val === 'All' ? 'All India' : val;
  if (label) label.innerText = text;
  if (tag) tag.innerText = `📍 ${text}`;
}

function setMode(mode, save = true) {
  state.activeMode = mode;
  if (save) localStorage.setItem('gramsathi_mode', mode);

  const deepBtn = document.getElementById('modeDeepBtn');
  const expressBtn = document.getElementById('modeExpressBtn');
  const tag = document.getElementById('activeModeTag');

  if (mode === 'deep') {
    if (deepBtn) deepBtn.classList.add('active');
    if (expressBtn) expressBtn.classList.remove('active');
    if (tag) tag.innerText = '🏆 Deep AI Judge';
  } else {
    if (deepBtn) deepBtn.classList.remove('active');
    if (expressBtn) expressBtn.classList.add('active');
    if (tag) tag.innerText = '⚡ Express Mitr';
  }
}

// ============================================================================
// Sidebar Management
// ============================================================================

function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  if (sidebar) sidebar.classList.toggle('open');
}

const closeBtn = document.getElementById('sidebarCloseBtn');
if (closeBtn) {
  closeBtn.addEventListener('click', () => {
    const sidebar = document.getElementById('sidebar');
    if (sidebar) sidebar.classList.remove('open');
  });
}

// ============================================================================
// Conversation Management & LocalStorage
// ============================================================================

function loadConversationHistory() {
  try {
    const stored = localStorage.getItem('gramsathi_conversations');
    state.conversations = stored ? JSON.parse(stored) : {};
    renderHistoryList();

    // Check if there are sessions
    const keys = Object.keys(state.conversations);
    if (keys.length > 0) {
      // Load latest session
      const lastSession = keys[keys.length - 1];
      selectSession(lastSession);
    } else {
      startNewChat();
    }
  } catch (err) {
    console.error("Error loading chat history:", err);
    state.conversations = {};
    startNewChat();
  }
}

function renderHistoryList() {
  const listEl = document.getElementById('historyList');
  if (!listEl) return;
  listEl.innerHTML = '';

  const sessions = Object.values(state.conversations).sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));

  if (sessions.length === 0) {
    listEl.innerHTML = '<div style="color: var(--text-muted); font-size: 0.78rem; padding: 0.5rem 0.25rem;">No past chats yet.</div>';
    return;
  }

  sessions.forEach(session => {
    const item = document.createElement('div');
    item.className = 'history-item' + (session.id === state.currentSessionId ? ' active' : '');
    item.onclick = (e) => {
      if (!e.target.classList.contains('history-del-btn')) {
        selectSession(session.id);
      }
    };

    const titleSpan = document.createElement('span');
    titleSpan.innerText = session.title || 'Untitled Conversation';

    const delBtn = document.createElement('button');
    delBtn.className = 'history-del-btn';
    delBtn.innerHTML = '✕';
    delBtn.title = 'Delete chat';
    delBtn.onclick = (e) => {
      e.stopPropagation();
      deleteSession(session.id);
    };

    item.appendChild(titleSpan);
    item.appendChild(delBtn);
    listEl.appendChild(item);
  });
}

function startNewChat() {
  const newId = 'session_' + Date.now();
  state.currentSessionId = newId;
  state.conversations[newId] = {
    id: newId,
    title: 'New Conversation',
    timestamp: Date.now(),
    messages: []
  };
  saveConversations();
  renderHistoryList();
  renderMessages([]);
  showWelcomeScreen(true);

  // Close sidebar on mobile
  const sidebar = document.getElementById('sidebar');
  if (sidebar) sidebar.classList.remove('open');
}

function selectSession(sessionId) {
  if (!state.conversations[sessionId]) return;
  state.currentSessionId = sessionId;
  renderHistoryList();
  const msgs = state.conversations[sessionId].messages || [];
  renderMessages(msgs);
  showWelcomeScreen(msgs.length === 0);

  // Close sidebar on mobile
  const sidebar = document.getElementById('sidebar');
  if (sidebar) sidebar.classList.remove('open');
}

function deleteSession(sessionId) {
  delete state.conversations[sessionId];
  saveConversations();
  if (state.currentSessionId === sessionId) {
    const keys = Object.keys(state.conversations);
    if (keys.length > 0) {
      selectSession(keys[keys.length - 1]);
    } else {
      startNewChat();
    }
  } else {
    renderHistoryList();
  }
}

function clearAllHistory() {
  if (confirm("Are you sure you want to clear all chat history?")) {
    state.conversations = {};
    saveConversations();
    startNewChat();
    showToast("Chat history cleared.");
  }
}

function saveConversations() {
  try {
    localStorage.setItem('gramsathi_conversations', JSON.stringify(state.conversations));
  } catch (err) {
    console.warn("Storage quota exceeded or error:", err);
  }
}

function showWelcomeScreen(visible) {
  const welcome = document.getElementById('welcomeScreen');
  const stream = document.getElementById('messageStream');
  if (welcome) welcome.style.display = visible ? 'flex' : 'none';
  if (stream) stream.style.display = visible ? 'none' : 'flex';
}

// ============================================================================
// Messaging & Rendering
// ============================================================================

function renderMessages(messages) {
  const stream = document.getElementById('messageStream');
  if (!stream) return;
  stream.innerHTML = '';

  messages.forEach(msg => {
    appendMessageElement(msg, false);
  });

  scrollToBottom();
}

function appendMessageElement(msg, autoScroll = true) {
  const stream = document.getElementById('messageStream');
  if (!stream) return;

  const isUser = msg.role === 'user';
  const msgEl = document.createElement('div');
  msgEl.className = `chat-message ${isUser ? 'msg-user' : 'msg-bot'}`;

  // Avatar
  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.innerText = isUser ? '👤' : '🌾';

  // Bubble Wrapper
  const wrapper = document.createElement('div');
  wrapper.className = 'msg-bubble-wrapper';

  // Header meta for bot
  if (!isUser && msg.data) {
    const metaBar = document.createElement('div');
    metaBar.className = 'msg-header-meta';

    if (msg.data.winner_id) {
      const winnerTag = document.createElement('span');
      winnerTag.className = 'winner-tag';
      winnerTag.innerHTML = `🏆 ${escapeHtml(msg.data.winner_id)}`;
      metaBar.appendChild(winnerTag);
    }

    if (msg.data.similarities_count) {
      const simTag = document.createElement('span');
      simTag.className = 'stat-tag';
      simTag.innerText = `🔍 ${msg.data.similarities_count} Matches`;
      metaBar.appendChild(simTag);
    }

    if (msg.data.total_latency_seconds) {
      const timeTag = document.createElement('span');
      timeTag.className = 'stat-tag';
      timeTag.innerText = `⏱️ ${msg.data.total_latency_seconds}s`;
      metaBar.appendChild(timeTag);
    }

    wrapper.appendChild(metaBar);
  }

  // Bubble content
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble markdown-body';
  bubble.innerHTML = isUser ? escapeHtml(msg.content) : marked.parse(msg.content || '');
  wrapper.appendChild(bubble);

  // Top Matched Schemes preview pills for bot
  if (!isUser && msg.data && msg.data.top_ranked_schemes && msg.data.top_ranked_schemes.length > 0) {
    const schemesBox = document.createElement('div');
    schemesBox.className = 'matched-schemes-preview';
    schemesBox.innerHTML = '<div class="preview-title">Key Matching Schemes Detected:</div>';

    const chips = document.createElement('div');
    chips.className = 'preview-chips';

    msg.data.top_ranked_schemes.slice(0, 4).forEach(s => {
      const chip = document.createElement('div');
      chip.className = 'scheme-chip';
      chip.innerHTML = `
        <span class="scheme-chip-level">[${escapeHtml(s.level)}]</span>
        <span>${escapeHtml(s.schemeName)}</span>
      `;
      chip.title = `Ministry: ${s.nodalMinistryName} | Rank Score: ${s.composite_rank_score.toFixed(3)}`;
      chip.onclick = () => openSchemeDetail(s.index);
      chips.appendChild(chip);
    });

    schemesBox.appendChild(chips);
    wrapper.appendChild(schemesBox);
  }

  // Action Bar for Bot message
  if (!isUser && msg.data) {
    const actionsBar = document.createElement('div');
    actionsBar.className = 'msg-actions-bar';

    // 1. Text to Speech
    const ttsBtn = document.createElement('button');
    ttsBtn.className = 'action-chip-btn';
    ttsBtn.innerHTML = '<span>🔊</span> Listen';
    ttsBtn.onclick = () => toggleSpeechSynthesis(msg.content, ttsBtn);
    actionsBar.appendChild(ttsBtn);

    // 2. Copy Button
    const copyBtn = document.createElement('button');
    copyBtn.className = 'action-chip-btn';
    copyBtn.innerHTML = '<span>📋</span> Copy';
    copyBtn.onclick = () => {
      navigator.clipboard.writeText(msg.content);
      showToast('Copied welfare advice to clipboard!');
    };
    actionsBar.appendChild(copyBtn);

    // 3. AI Judge Scorecard Button
    if (msg.data.judge_evaluation) {
      const judgeBtn = document.createElement('button');
      judgeBtn.className = 'action-chip-btn btn-judge';
      judgeBtn.innerHTML = '<span>⚖️</span> AI Judge Scorecard';
      judgeBtn.onclick = () => openJudgeModal(msg.data);
      actionsBar.appendChild(judgeBtn);
    }

    // 4. Compare 5 Personas Button
    if (msg.data.candidate_responses && Object.keys(msg.data.candidate_responses).length > 1) {
      const candBtn = document.createElement('button');
      candBtn.className = 'action-chip-btn';
      candBtn.innerHTML = '<span>👥</span> 5 Personas';
      candBtn.onclick = () => openCandidatesModal(msg.data);
      actionsBar.appendChild(candBtn);
    }

    // 5. Retrieved Similarities
    if (msg.data.top_ranked_schemes && msg.data.top_ranked_schemes.length > 0) {
      const simBtn = document.createElement('button');
      simBtn.className = 'action-chip-btn';
      simBtn.innerHTML = `<span>🔍</span> Ranked Schemes (${msg.data.top_ranked_schemes.length})`;
      simBtn.onclick = () => openSchemesModal(msg.data);
      actionsBar.appendChild(simBtn);
    }

    // 6. Print / PDF Button
    const printBtn = document.createElement('button');
    printBtn.className = 'action-chip-btn';
    printBtn.innerHTML = '<span>🖨️</span> Print';
    printBtn.onclick = () => printMessage(msg.content, msg.data.winner_id);
    actionsBar.appendChild(printBtn);

    wrapper.appendChild(actionsBar);
  }

  msgEl.appendChild(avatar);
  msgEl.appendChild(wrapper);
  stream.appendChild(msgEl);

  if (autoScroll) scrollToBottom();
}

function scrollToBottom() {
  const container = document.getElementById('chatContainer');
  if (container) {
    setTimeout(() => {
      container.scrollTop = container.scrollHeight;
    }, 50);
  }
}

// ============================================================================
// Input & Query Execution
// ============================================================================

function handleInputKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResizeTextarea(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
}

function sendPreset(text) {
  const input = document.getElementById('chatInput');
  if (input) {
    input.value = text;
    autoResizeTextarea(input);
    sendMessage();
  }
}

async function sendMessage() {
  if (state.isGenerating) return;

  const inputEl = document.getElementById('chatInput');
  const query = inputEl.value.trim();
  if (!query) return;

  // Clear input
  inputEl.value = '';
  autoResizeTextarea(inputEl);

  showWelcomeScreen(false);

  // Append User message
  const userMsg = { role: 'user', content: query, timestamp: Date.now() };
  appendMessageElement(userMsg, true);

  // Add to session
  const session = state.conversations[state.currentSessionId];
  if (session) {
    session.messages.push(userMsg);
    if (session.messages.length === 1) {
      session.title = query.length > 38 ? query.substring(0, 38) + '...' : query;
    }
    saveConversations();
    renderHistoryList();
  }

  // Start Pipeline Tracking Animation
  startPipelineTracker(state.activeMode);

  try {
    state.isGenerating = true;
    document.getElementById('sendBtn').disabled = true;

    const payload = {
      message: query,
      mode: state.activeMode,
      state_filter: state.activeState,
      top_k: 5
    };

    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || `Server responded with status ${res.status}`);
    }

    const data = await res.json();
    state.lastQueryResult = data;

    // Append Bot message
    const botMsg = {
      role: 'assistant',
      content: data.final_best_response || "No response generated.",
      data: data,
      timestamp: Date.now()
    };

    appendMessageElement(botMsg, true);

    if (session) {
      session.messages.push(botMsg);
      saveConversations();
    }

  } catch (err) {
    console.error("Chat error:", err);
    const errorMsg = {
      role: 'assistant',
      content: `⚠️ **Error Processing Query:** ${err.message}\n\nPlease check your connection or try again.`,
      timestamp: Date.now()
    };
    appendMessageElement(errorMsg, true);
  } finally {
    stopPipelineTracker();
    state.isGenerating = false;
    document.getElementById('sendBtn').disabled = false;
    scrollToBottom();
  }
}

// ============================================================================
// Pipeline Progress Tracker Simulation
// ============================================================================

function startPipelineTracker(mode) {
  const card = document.getElementById('pipelineCard');
  if (!card) return;
  card.style.display = 'block';

  const title = document.getElementById('pipelineCurrentTitle');
  const sub = document.getElementById('pipelineCurrentSub');
  const elapsed = document.getElementById('pipelineElapsed');
  const fill = document.getElementById('pipelineProgressFill');

  const s1 = document.getElementById('pipeStep1');
  const s2 = document.getElementById('pipeStep2');
  const s3 = document.getElementById('pipeStep3');
  const s4 = document.getElementById('pipeStep4');

  [s1, s2, s3, s4].forEach(s => { s.className = 'pipe-step'; });
  s1.className = 'pipe-step active';
  fill.style.width = '20%';

  title.innerText = 'Scanning 5,061 Government Schemes...';
  sub.innerText = 'Dense semantic cosine matching + lexical BM25 + state filters';

  const startTime = Date.now();
  state.pipelineTimer = setInterval(() => {
    const sec = ((Date.now() - startTime) / 1000).toFixed(1);
    if (elapsed) elapsed.innerText = sec + 's';
  }, 100);

  if (mode === 'express') {
    setTimeout(() => {
      s1.className = 'pipe-step done';
      s2.className = 'pipe-step active';
      fill.style.width = '60%';
      title.innerText = 'Synthesizing Grounded Welfare Guidance...';
      sub.innerText = 'Plain Language Citizen Mitr persona crafting steps';
    }, 1000);
    setTimeout(() => {
      s2.className = 'pipe-step done';
      s4.className = 'pipe-step active';
      fill.style.width = '90%';
      title.innerText = 'Finalizing Verified Response...';
      sub.innerText = 'Formatting official eligibility and application links';
    }, 2200);
  } else {
    setTimeout(() => {
      s1.className = 'pipe-step done';
      s2.className = 'pipe-step active';
      fill.style.width = '45%';
      title.innerText = 'Generating 5 Specialized LLM Personas...';
      sub.innerText = 'Advisor, Roadmap, Eligibility Auditor, Financial Maximizer, Citizen Mitr';
    }, 1200);

    setTimeout(() => {
      s2.className = 'pipe-step done';
      s3.className = 'pipe-step active';
      fill.style.width = '75%';
      title.innerText = 'Executing AI-as-a-Judge Evaluation...';
      sub.innerText = 'Scoring all 5 candidates on Groundedness, Relevance, Actionability & Clarity';
    }, 3800);

    setTimeout(() => {
      s3.className = 'pipe-step done';
      s4.className = 'pipe-step active';
      fill.style.width = '92%';
      title.innerText = 'Crowning Winning Response...';
      sub.innerText = 'Compiling official rubric scorecard and verified citizen advice';
    }, 6500);
  }

  scrollToBottom();
}

function stopPipelineTracker() {
  const card = document.getElementById('pipelineCard');
  if (card) card.style.display = 'none';

  if (state.pipelineTimer) {
    clearInterval(state.pipelineTimer);
    state.pipelineTimer = null;
  }
}

// ============================================================================
// Voice Capabilities: Speech-to-Text & Text-to-Speech
// ============================================================================

function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    console.log("Web Speech Recognition API not supported in this browser.");
    return;
  }

  state.recognition = new SpeechRecognition();
  state.recognition.continuous = false;
  state.recognition.interimResults = true;
  state.recognition.lang = 'en-IN'; // Indian English / Hindi mix

  state.recognition.onstart = () => {
    state.isRecording = true;
    const btn = document.getElementById('micBtn');
    if (btn) btn.classList.add('listening');
    showToast("Listening... Speak your welfare question.");
  };

  state.recognition.onresult = (event) => {
    let transcript = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    const input = document.getElementById('chatInput');
    if (input) {
      input.value = transcript;
      autoResizeTextarea(input);
    }
  };

  state.recognition.onerror = (event) => {
    console.warn("Speech recognition error:", event.error);
    stopRecording();
    showToast(`Microphone: ${event.error}`);
  };

  state.recognition.onend = () => {
    stopRecording();
  };
}

function toggleSpeechRecognition() {
  if (!state.recognition) {
    showToast("Voice input is not supported in this browser.");
    return;
  }

  if (state.isRecording) {
    state.recognition.stop();
    stopRecording();
  } else {
    try {
      state.recognition.start();
    } catch (e) {
      console.warn("Recognition start error:", e);
    }
  }
}

function stopRecording() {
  state.isRecording = false;
  const btn = document.getElementById('micBtn');
  if (btn) btn.classList.remove('listening');
}

function toggleSpeechSynthesis(text, btn) {
  if (!state.speechSynth) {
    showToast("Text-to-speech is not supported on this device.");
    return;
  }

  if (state.speechSynth.speaking && state.activeSpeechBtn === btn) {
    state.speechSynth.cancel();
    btn.classList.remove('speaking');
    btn.innerHTML = '<span>🔊</span> Listen';
    state.activeSpeechBtn = null;
    return;
  }

  state.speechSynth.cancel();
  if (state.activeSpeechBtn) {
    state.activeSpeechBtn.classList.remove('speaking');
    state.activeSpeechBtn.innerHTML = '<span>🔊</span> Listen';
  }

  // Clean markdown for spoken text
  const cleanText = text
    .replace(/[#*`_\[\]()]/g, '')
    .replace(/>/g, '')
    .replace(/Sl\. No\..*?Amount/gi, '')
    .substring(0, 1000); // Read first section

  const utterance = new SpeechSynthesisUtterance(cleanText);
  utterance.lang = 'en-IN';
  utterance.rate = 1.0;

  utterance.onstart = () => {
    btn.classList.add('speaking');
    btn.innerHTML = '<span>⏹️</span> Stop';
    state.activeSpeechBtn = btn;
  };

  utterance.onend = () => {
    btn.classList.remove('speaking');
    btn.innerHTML = '<span>🔊</span> Listen';
    state.activeSpeechBtn = null;
  };

  utterance.onerror = () => {
    btn.classList.remove('speaking');
    btn.innerHTML = '<span>🔊</span> Listen';
    state.activeSpeechBtn = null;
  };

  state.speechSynth.speak(utterance);
}

// ============================================================================
// Modals Management
// ============================================================================

function openModal(id) {
  const modal = document.getElementById(id);
  if (modal) modal.classList.add('active');
}

function closeModal(id) {
  const modal = document.getElementById(id);
  if (modal) modal.classList.remove('active');
}

// Modal 1: AI Judge Scorecard
function openJudgeModal(data) {
  const evalData = data.judge_evaluation || {};
  const winnerId = data.winner_id || evalData.winner_id || 'Candidate 1';

  document.getElementById('modalWinnerLabel').innerText = `🏆 Crowned Winner: ${winnerId}`;
  document.getElementById('modalVerdictText').innerText = evalData.judge_verdict || "Evaluated by AI Judge across 5 criteria.";

  const sBody = document.getElementById('judgeScoreTableBody');
  sBody.innerHTML = '';

  const candidatesEval = evalData.candidates_evaluation || [];
  candidatesEval.forEach(ce => {
    const isW = ce.candidate_id === winnerId;
    const scores = ce.scores || {};
    const tr = document.createElement('tr');
    if (isW) tr.className = 'winner-row';

    tr.innerHTML = `
      <td><b>${escapeHtml(ce.candidate_id)}</b></td>
      <td>${scores.groundedness || '-'}</td>
      <td>${scores.relevance || '-'}</td>
      <td>${scores.actionability || '-'}</td>
      <td>${scores.completeness || '-'}</td>
      <td>${scores.clarity_tone || '-'}</td>
      <td><span class="score-badge ${isW ? 'score-high' : 'score-mid'}">${ce.total_score || '-'}</span></td>
      <td>${isW ? '<span style="color:#10b981;font-weight:700;">🏆 Winner</span>' : '<span style="color:#94a3b8;">Runner-up</span>'}</td>
    `;
    sBody.appendChild(tr);
  });

  // Strengths and Weaknesses cards
  const cardsGrid = document.getElementById('judgeCardsGrid');
  cardsGrid.innerHTML = '';
  candidatesEval.forEach(ce => {
    const card = document.createElement('div');
    card.className = 'eval-card';
    card.innerHTML = `
      <div class="eval-card-header">
        <span>${escapeHtml(ce.candidate_id)}</span>
        <span class="score-badge score-mid">${ce.total_score || 0}/50</span>
      </div>
      <div class="eval-card-strengths"><b>✓ Strengths:</b> ${escapeHtml(ce.strengths || 'Grounded output')}</div>
      <div class="eval-card-weaknesses"><b>⚠ Weaknesses:</b> ${escapeHtml(ce.weaknesses || 'Minor points for improvement')}</div>
    `;
    cardsGrid.appendChild(card);
  });

  openModal('judgeModal');
}

// Modal 2: 5 Candidates Viewer
function openCandidatesModal(data) {
  const candidateResponses = data.candidate_responses || {};
  const winnerId = data.winner_id;
  const nav = document.getElementById('candidateTabsNav');
  const content = document.getElementById('candidateTabContent');
  nav.innerHTML = '';

  const keys = Object.keys(candidateResponses);
  if (keys.length === 0) return;

  keys.forEach((cid, idx) => {
    const btn = document.createElement('button');
    btn.className = 'cand-tab-btn' + (idx === 0 ? ' active' : '');
    btn.innerHTML = `${cid === winnerId ? '🏆 ' : ''}${escapeHtml(cid)}`;
    btn.onclick = () => {
      document.querySelectorAll('.cand-tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderCandidateContent(candidateResponses[cid]);
    };
    nav.appendChild(btn);
  });

  renderCandidateContent(candidateResponses[keys[0]]);
  openModal('candidatesModal');
}

function renderCandidateContent(candInfo) {
  const content = document.getElementById('candidateTabContent');
  if (!content) return;
  if (!candInfo) {
    content.innerHTML = '<p>No content available.</p>';
    return;
  }
  content.innerHTML = marked.parse(candInfo.content || '');
}

// Modal 3: Top Retrieved Similarities
function openSchemesModal(data) {
  const schemes = data.top_ranked_schemes || [];
  const tbody = document.getElementById('retrievedSchemesTableBody');
  tbody.innerHTML = '';

  schemes.forEach(s => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><b>#${s.rank}</b></td>
      <td><b style="color:#38bdf8;font-family:var(--font-mono);">${s.composite_rank_score.toFixed(4)}</b></td>
      <td><b>${escapeHtml(s.schemeName)}</b></td>
      <td><span style="color:#10b981;font-weight:600;">${escapeHtml(s.level)}</span></td>
      <td>${escapeHtml((s.beneficiaryState || []).join(', ').substring(0, 30))}</td>
      <td style="font-family:var(--font-mono);">${s.dense_similarity.toFixed(4)}</td>
      <td style="font-family:var(--font-mono);">${s.lexical_score.toFixed(4)}</td>
      <td style="font-family:var(--font-mono);">${s.state_multiplier.toFixed(2)}x</td>
      <td>
        <button class="action-chip-btn" onclick="openSchemeDetail(${s.index})">View Full Text</button>
      </td>
    `;
    tbody.appendChild(tr);
  });

  openModal('schemesModal');
}

// Modal 4: 5,061 Schemes Explorer
function openExplorerModal() {
  state.explorerOffset = 0;
  loadExplorerSchemes(0);
  openModal('explorerModal');
}

function debounceExplorerSearch() {
  clearTimeout(state.debounceTimer);
  state.debounceTimer = setTimeout(() => {
    loadExplorerSchemes(0);
  }, 300);
}

async function loadExplorerSchemes(offset = 0) {
  state.explorerOffset = offset;
  const searchInput = document.getElementById('explorerSearchInput');
  const stateSelect = document.getElementById('explorerStateSelect');
  const catSelect = document.getElementById('explorerCategorySelect');
  const countEl = document.getElementById('explorerResultsCount');
  const grid = document.getElementById('explorerCardsGrid');
  const pagin = document.getElementById('explorerPagination');

  const q = searchInput ? searchInput.value.trim() : '';
  const st = stateSelect ? stateSelect.value : 'All';
  const cat = catSelect ? catSelect.value : 'All';

  if (countEl) countEl.innerText = 'Searching 5,061 schemes...';
  if (grid) grid.innerHTML = '<div style="color:var(--text-muted);padding:2rem 0;">Loading matching schemes...</div>';

  try {
    const params = new URLSearchParams({
      limit: state.explorerLimit,
      offset: offset
    });
    if (q) params.append('q', q);
    if (st && st !== 'All') params.append('state', st);
    if (cat && cat !== 'All') params.append('category', cat);

    const res = await fetch(`/api/schemes?${params.toString()}`);
    if (!res.ok) throw new Error("Failed to load schemes.");

    const data = await res.json();

    if (countEl) {
      countEl.innerText = `Showing ${data.offset + 1}-${Math.min(data.offset + data.limit, data.total)} of ${data.total.toLocaleString()} Schemes`;
    }

    if (grid) {
      grid.innerHTML = '';
      if (data.schemes.length === 0) {
        grid.innerHTML = '<div style="color:var(--text-muted);padding:2rem;">No schemes matched your search criteria.</div>';
        if (pagin) pagin.innerHTML = '';
        return;
      }

      data.schemes.forEach(s => {
        const card = document.createElement('div');
        card.className = 'explorer-card';
        card.innerHTML = `
          <div>
            <div class="explorer-card-meta">
              <span style="color:#10b981;font-weight:700;">[${escapeHtml(s.level)}]</span>
              <span>${escapeHtml(s.nodalMinistryName || 'Ministry of India')}</span>
            </div>
            <div class="explorer-card-title">${escapeHtml(s.schemeName)}</div>
            <div class="explorer-card-desc">${escapeHtml(s.briefDescription || 'No summary available.')}</div>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.5rem;">
            <span style="font-size:0.75rem;color:var(--text-muted);">${escapeHtml((s.beneficiaryState || []).join(', ').substring(0, 24) || 'All India')}</span>
            <button class="action-chip-btn" onclick="openSchemeDetail(${s.index})">Read Details →</button>
          </div>
        `;
        grid.appendChild(card);
      });
    }

    // Pagination
    if (pagin) {
      pagin.innerHTML = '';
      const totalPages = Math.ceil(data.total / state.explorerLimit);
      const currentPage = Math.floor(offset / state.explorerLimit) + 1;

      if (currentPage > 1) {
        const prevBtn = document.createElement('button');
        prevBtn.className = 'btn btn-secondary';
        prevBtn.innerText = '← Previous';
        prevBtn.onclick = () => loadExplorerSchemes(offset - state.explorerLimit);
        pagin.appendChild(prevBtn);
      }

      const pageIndicator = document.createElement('span');
      pageIndicator.style.cssText = 'font-size:0.85rem;color:var(--text-secondary);padding:0 0.5rem;';
      pageIndicator.innerText = `Page ${currentPage} of ${totalPages || 1}`;
      pagin.appendChild(pageIndicator);

      if (currentPage < totalPages) {
        const nextBtn = document.createElement('button');
        nextBtn.className = 'btn btn-secondary';
        nextBtn.innerText = 'Next →';
        nextBtn.onclick = () => loadExplorerSchemes(offset + state.explorerLimit);
        pagin.appendChild(nextBtn);
      }
    }

  } catch (err) {
    console.error("Schemes explorer error:", err);
    if (countEl) countEl.innerText = "Error loading schemes.";
  }
}

// Modal 5: Detailed Scheme Reader
async function openSchemeDetail(index) {
  const modal = document.getElementById('schemeDetailModal');
  const title = document.getElementById('detailSchemeTitle');
  const sub = document.getElementById('detailSchemeSubtitle');
  const body = document.getElementById('detailSchemeBody');

  if (title) title.innerText = 'Loading scheme...';
  if (body) body.innerHTML = '<p>Fetching complete scheme guidelines...</p>';
  openModal('schemeDetailModal');

  try {
    const res = await fetch(`/api/scheme/${index}`);
    if (!res.ok) throw new Error("Scheme not found.");
    const data = await res.json();
    const meta = data.metadata || {};

    if (title) title.innerText = meta.schemeName || 'Government Scheme';
    if (sub) {
      sub.innerText = `Ministry: ${meta.nodalMinistryName || 'Central/State'} | Level: ${meta.level || 'Central'} | States: ${(meta.beneficiaryState || []).join(', ') || 'All India'}`;
    }

    if (body) {
      body.innerHTML = marked.parse(data.page_content || 'No detailed content available.');
    }
  } catch (err) {
    if (body) body.innerHTML = `<p style="color:var(--danger)">Error: ${err.message}</p>`;
  }
}

// ============================================================================
// Printing / Saving Guide
// ============================================================================

function printMessage(content, winnerId) {
  const win = window.open('', '_blank');
  win.document.write(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>GramSathi Government Welfare Guide</title>
      <style>
        body { font-family: sans-serif; line-height: 1.6; max-width: 800px; margin: 40px auto; color: #1e293b; }
        h1, h2, h3 { color: #047857; }
        .header { border-bottom: 2px solid #047857; padding-bottom: 12px; margin-bottom: 24px; }
        .badge { background: #d1fae5; color: #065f46; padding: 4px 10px; border-radius: 20px; font-weight: bold; font-size: 13px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #cbd5e1; padding: 8px 12px; text-align: left; }
        th { background: #f8fafc; }
        a { color: #0284c7; }
      </style>
      <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    </head>
    <body>
      <div class="header">
        <h1>🌾 GramSathi Citizen Welfare Advisory</h1>
        <div>Verified Government Welfare Guidance • Evaluated by AI Judge</div>
        <div style="margin-top: 8px;"><span class="badge">Winning Persona: ${escapeHtml(winnerId || 'Verified Advice')}</span></div>
      </div>
      <div id="content"></div>
      <script>
        document.getElementById('content').innerHTML = marked.parse(${JSON.stringify(content)});
        setTimeout(() => { window.print(); }, 500);
      </script>
    </body>
    </html>
  `);
  win.document.close();
}

// ============================================================================
// Utilities
// ============================================================================

function showToast(message) {
  const toast = document.getElementById('toast');
  if (!toast) return;
  toast.innerText = message;
  toast.classList.add('show');
  setTimeout(() => {
    toast.classList.remove('show');
  }, 3000);
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
