import os
import time
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import SchemeRAGPipeline

app = FastAPI(title="GramSathi RAG & AI Judge", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[SchemeRAGPipeline] = None

@app.on_event("startup")
def startup_event():
    global pipeline
    print("Initializing GramSathi SchemeRAGPipeline in FastAPI...")
    pipeline = SchemeRAGPipeline()
    print("GramSathi Pipeline loaded and ready for queries!")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    min_similarity: float = 0.20

@app.get("/api/health")
def health_check():
    return {"status": "ok", "indexed_schemes": len(pipeline.retriever.documents) if pipeline else 0}

@app.post("/api/query")
def run_query(req: QueryRequest):
    if not pipeline:
        return JSONResponse(status_code=503, content={"error": "Pipeline is initializing. Please wait."})
    if not req.query.strip():
        return JSONResponse(status_code=400, content={"error": "Query cannot be empty."})
        
    result = pipeline.run(
        query=req.query,
        top_k_context=req.top_k,
        min_similarity_threshold=req.min_similarity
    )
    return result

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GramSathi | Multi-LLM RAG with AI Judge</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(18, 24, 38, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent: #3b82f6;
      --accent-glow: rgba(59, 130, 246, 0.25);
      --success: #10b981;
      --success-glow: rgba(16, 185, 129, 0.25);
      --warning: #f59e0b;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --radius: 14px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: radial-gradient(circle at top right, #172554, #090d16 55%);
      color: var(--text);
      font-family: 'Outfit', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 1.25rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--card-border);
      backdrop-filter: blur(12px);
      background: rgba(9, 13, 22, 0.8);
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .logo-badge {
      background: linear-gradient(135deg, #2563eb, #10b981);
      width: 40px;
      height: 40px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.3rem;
      box-shadow: 0 4px 14px var(--accent-glow);
    }
    .brand h1 { font-size: 1.4rem; font-weight: 700; letter-spacing: -0.02em; }
    .brand span { font-size: 0.8rem; color: #10b981; background: rgba(16,185,129,0.15); padding: 2px 8px; border-radius: 12px; margin-left: 6px; font-weight: 600; }
    .stats-badge {
      display: flex;
      gap: 1.25rem;
      font-size: 0.85rem;
      color: var(--text-muted);
    }
    .stats-badge b { color: var(--text); }
    
    main {
      flex: 1;
      max-width: 1280px;
      width: 100%;
      margin: 0 auto;
      padding: 2rem;
      display: flex;
      flex-direction: column;
      gap: 2rem;
    }
    
    /* Search Box */
    .search-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 1.75rem;
      box-shadow: 0 10px 30px rgba(0,0,0,0.4);
      backdrop-filter: blur(16px);
    }
    .search-input-wrapper {
      display: flex;
      gap: 0.75rem;
      margin-bottom: 1rem;
    }
    textarea {
      flex: 1;
      background: rgba(10, 15, 26, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 10px;
      color: var(--text);
      font-family: inherit;
      font-size: 1.05rem;
      padding: 1rem 1.25rem;
      resize: none;
      height: 70px;
      transition: all 0.2s;
    }
    textarea:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }
    .btn-submit {
      background: linear-gradient(135deg, #3b82f6, #2563eb);
      color: white;
      border: none;
      border-radius: 10px;
      padding: 0 2rem;
      font-family: inherit;
      font-size: 1.05rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      transition: transform 0.15s, box-shadow 0.15s;
    }
    .btn-submit:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px var(--accent-glow);
    }
    .btn-submit:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }
    
    .presets {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      align-items: center;
    }
    .presets span { font-size: 0.82rem; color: var(--text-muted); }
    .preset-pill {
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.08);
      color: #cbd5e1;
      font-size: 0.8rem;
      padding: 5px 12px;
      border-radius: 20px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .preset-pill:hover {
      background: rgba(59, 130, 246, 0.2);
      border-color: var(--accent);
      color: white;
    }
    
    /* Pipeline Step Tracker */
    .pipeline-tracker {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1rem;
      margin-top: 0.5rem;
    }
    .step-box {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 0.85rem 1rem;
      display: flex;
      align-items: center;
      gap: 0.75rem;
      transition: border-color 0.3s;
    }
    .step-box.active {
      border-color: var(--accent);
      box-shadow: 0 0 15px var(--accent-glow);
      background: rgba(59, 130, 246, 0.1);
    }
    .step-box.done {
      border-color: var(--success);
      background: rgba(16, 185, 129, 0.08);
    }
    .step-num {
      width: 28px;
      height: 28px;
      border-radius: 50%;
      background: rgba(255,255,255,0.1);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 0.85rem;
    }
    .step-box.done .step-num { background: var(--success); color: #000; }
    .step-title { font-size: 0.85rem; font-weight: 600; }
    .step-desc { font-size: 0.75rem; color: var(--text-muted); }
    
    /* Results Sections */
    .tabs-nav {
      display: flex;
      gap: 0.5rem;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 0.5rem;
    }
    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-family: inherit;
      font-size: 0.95rem;
      font-weight: 600;
      padding: 0.6rem 1.25rem;
      border-radius: 8px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      transition: all 0.2s;
    }
    .tab-btn:hover { color: var(--text); background: rgba(255,255,255,0.04); }
    .tab-btn.active {
      color: white;
      background: rgba(59, 130, 246, 0.2);
      border: 1px solid var(--accent);
    }
    
    .tab-pane { display: none; }
    .tab-pane.active { display: block; animation: fadeIn 0.3s ease; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    
    /* Best Response View */
    .best-response-card {
      background: var(--card-bg);
      border: 1px solid rgba(16, 185, 129, 0.3);
      border-radius: var(--radius);
      padding: 2.5rem;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5), 0 0 20px var(--success-glow);
    }
    .winner-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid var(--success);
      color: #34d399;
      font-size: 0.85rem;
      font-weight: 700;
      padding: 6px 14px;
      border-radius: 30px;
      margin-bottom: 1.5rem;
    }
    .markdown-content {
      line-height: 1.7;
      color: #e2e8f0;
    }
    .markdown-content h1 { font-size: 1.8rem; margin: 1.2rem 0 0.8rem; color: #fff; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 0.4rem; }
    .markdown-content h2 { font-size: 1.4rem; margin: 1.2rem 0 0.6rem; color: #60a5fa; }
    .markdown-content h3 { font-size: 1.15rem; margin: 1rem 0 0.4rem; color: #34d399; }
    .markdown-content p { margin-bottom: 0.85rem; }
    .markdown-content ul, .markdown-content ol { margin: 0.6rem 0 1rem 1.5rem; }
    .markdown-content li { margin-bottom: 0.35rem; }
    .markdown-content strong { color: #f8fafc; }
    .markdown-content a { color: #38bdf8; text-decoration: underline; }
    .markdown-content blockquote {
      border-left: 4px solid var(--accent);
      background: rgba(59, 130, 246, 0.08);
      padding: 0.75rem 1.25rem;
      border-radius: 4px;
      margin: 1rem 0;
    }
    
    /* Judge Scorecard Table */
    .judge-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 2rem;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 1.5rem 0;
    }
    th, td {
      padding: 0.85rem 1rem;
      text-align: left;
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    th {
      background: rgba(255,255,255,0.03);
      color: var(--text-muted);
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    tr.winner-row {
      background: rgba(16, 185, 129, 0.1);
      font-weight: 600;
    }
    tr.winner-row td:first-child {
      border-left: 3px solid var(--success);
    }
    .score-badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 0.9rem;
    }
    .score-high { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .score-mid { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    
    /* 5 Candidates Grid */
    .candidates-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 1.5rem;
    }
    .cand-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      position: relative;
    }
    .cand-card.is-winner {
      border-color: var(--success);
      box-shadow: 0 0 15px var(--success-glow);
    }
    .cand-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 1rem;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      padding-bottom: 0.75rem;
    }
    .cand-title { font-size: 1.1rem; font-weight: 700; color: #fff; }
    .cand-desc { font-size: 0.8rem; color: var(--text-muted); }
    .cand-body {
      font-size: 0.9rem;
      color: #cbd5e1;
      max-height: 400px;
      overflow-y: auto;
      padding-right: 0.5rem;
    }
    
    /* Retrieval Table */
    .retrieval-table {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 1.5rem;
    }
    
    /* Loader Spinner */
    .loader {
      display: none;
      text-align: center;
      padding: 3rem;
    }
    .spinner {
      border: 4px solid rgba(255,255,255,0.1);
      width: 48px;
      height: 48px;
      border-radius: 50%;
      border-left-color: var(--accent);
      animation: spin 1s linear infinite;
      margin: 0 auto 1rem;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="logo-badge">🌾</div>
      <div>
        <h1>GramSathi <span>RAG + AI Judge</span></h1>
      </div>
    </div>
    <div class="stats-badge">
      <div>Schemes Indexed: <b>5,061</b></div>
      <div>Embedding: <b>text-embedding-3-small (1536d)</b></div>
      <div>Multi-LLM: <b>5 Parallel Personas</b></div>
      <div>Judge: <b>Impartial AI Evaluator</b></div>
    </div>
  </header>

  <main>
    <div class="search-card">
      <div class="search-input-wrapper">
        <textarea id="queryInput" placeholder="Describe the citizen's profile and welfare need (e.g. 'I am a female farmer from Maharashtra with 2 acres looking for drip irrigation subsidy and crop insurance')..."></textarea>
        <button class="btn-submit" id="searchBtn" onclick="submitQuery()">
          <span>Run RAG Pipeline</span> 🚀
        </button>
      </div>
      <div class="presets">
        <span>Try Examples:</span>
        <button class="preset-pill" onclick="setQuery('I am a female farmer from Maharashtra with 2 acres looking for drip irrigation subsidy and crop insurance')">🌾 Maharashtra Female Farmer (Drip Subsidy)</button>
        <button class="preset-pill" onclick="setQuery('I am a girl student in Karnataka belonging to SC community looking for engineering scholarship and hostel allowance')">🎓 Karnataka SC Student (Scholarship)</button>
        <button class="preset-pill" onclick="setQuery('I am an unemployed youth in Bihar wanting to start a dairy farm and small grocery store loan')">🐄 Bihar Youth (Dairy / Mudra Loan)</button>
        <button class="preset-pill" onclick="setQuery('Senior citizen widow in Uttar Pradesh seeking old age pension and healthcare benefits')">👵 UP Senior Citizen (Pension)</button>
      </div>
      
      <div class="pipeline-tracker" id="pipelineTracker">
        <div class="step-box" id="step1">
          <div class="step-num">1</div>
          <div>
            <div class="step-title">Similarity Search</div>
            <div class="step-desc">Scan all 5,061 schemes</div>
          </div>
        </div>
        <div class="step-box" id="step2">
          <div class="step-num">2</div>
          <div>
            <div class="step-title">Hybrid Ranking</div>
            <div class="step-desc">Dense + Lexical + State match</div>
          </div>
        </div>
        <div class="step-box" id="step3">
          <div class="step-num">3</div>
          <div>
            <div class="step-title">5-LLM Generation</div>
            <div class="step-desc">5 Concurrent Personas</div>
          </div>
        </div>
        <div class="step-box" id="step4">
          <div class="step-num">4</div>
          <div>
            <div class="step-title">AI-as-a-Judge</div>
            <div class="step-desc">5-Criteria Rubric Scoring</div>
          </div>
        </div>
      </div>
    </div>

    <div class="loader" id="loader">
      <div class="spinner"></div>
      <h3 id="loaderText">Retrieving each and every similarity across 5,061 schemes...</h3>
      <p style="color: var(--text-muted); font-size: 0.9rem; margin-top: 0.5rem;">Generating 5 parallel candidate responses and running AI evaluation...</p>
    </div>

    <div id="resultsContainer" style="display: none;">
      <div class="tabs-nav">
        <button class="tab-btn active" onclick="switchTab('bestResponseTab')">🏆 Final Best Response</button>
        <button class="tab-btn" onclick="switchTab('judgeTab')">⚖️ AI Judge Scorecard</button>
        <button class="tab-btn" onclick="switchTab('candidatesTab')">👥 5 Candidate Responses</button>
        <button class="tab-btn" onclick="switchTab('retrievalTab')">🔍 Retrieved Similarities (<span id="simCount">0</span>)</button>
      </div>

      <!-- Tab 1: Final Best Response -->
      <div class="tab-pane active" id="bestResponseTab">
        <div class="best-response-card">
          <div class="winner-badge" id="winnerBadge">🏆 CROWNED WINNER: Candidate 2</div>
          <div class="markdown-content" id="bestResponseContent"></div>
        </div>
      </div>

      <!-- Tab 2: AI Judge Scorecard -->
      <div class="tab-pane" id="judgeTab">
        <div class="judge-card">
          <h2 style="margin-bottom: 0.5rem; color: #f59e0b;">⚖️ AI Judge Evaluation Rubric & Verdict</h2>
          <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1rem;">
            The Judge evaluates all 5 candidate responses against ground-truth retrieved schemes across Groundedness, Relevance, Actionability, Completeness, and Clarity (Scale 1-10).
          </p>
          <div id="judgeVerdictBox" style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem; color: #fde68a; line-height: 1.6;"></div>
          
          <table id="scoresTable">
            <thead>
              <tr>
                <th>Candidate</th>
                <th>Groundedness</th>
                <th>Relevance</th>
                <th>Actionability</th>
                <th>Completeness</th>
                <th>Clarity</th>
                <th>Total (/50)</th>
                <th>Decision</th>
              </tr>
            </thead>
            <tbody id="scoresTableBody"></tbody>
          </table>
        </div>
      </div>

      <!-- Tab 3: 5 Candidate Responses -->
      <div class="tab-pane" id="candidatesTab">
        <div class="candidates-grid" id="candidatesGrid"></div>
      </div>

      <!-- Tab 4: Retrieved Similarities -->
      <div class="tab-pane" id="retrievalTab">
        <div class="retrieval-table">
          <h2 style="margin-bottom: 0.5rem;">🔍 All Ranked Similar Schemes</h2>
          <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1rem;">
            Exhaustive similarity retrieval scanning all 5,061 schemes using OpenAI 1536-dim dense cosine similarity + lexical token matching + state compatibility multiplier.
          </p>
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Composite Score</th>
                <th>Scheme Name</th>
                <th>Level</th>
                <th>Applicable States</th>
                <th>Dense Sim</th>
                <th>Lexical Score</th>
                <th>State Match</th>
              </tr>
            </thead>
            <tbody id="retrievalTableBody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </main>

  <script>
    let currentData = null;

    function setQuery(text) {
      document.getElementById('queryInput').value = text;
      submitQuery();
    }

    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      
      const targetBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
      if (targetBtn) targetBtn.classList.add('active');
      const targetPane = document.getElementById(tabId);
      if (targetPane) targetPane.classList.add('active');
    }

    async function submitQuery() {
      const q = document.getElementById('queryInput').value.trim();
      if (!q) return;

      const btn = document.getElementById('searchBtn');
      const loader = document.getElementById('loader');
      const resultsContainer = document.getElementById('resultsContainer');
      const tracker = document.getElementById('pipelineTracker');

      btn.disabled = true;
      loader.style.display = 'block';
      resultsContainer.style.display = 'none';

      const step1 = document.getElementById('step1');
      const step2 = document.getElementById('step2');
      const step3 = document.getElementById('step3');
      const step4 = document.getElementById('step4');

      step1.className = 'step-box active';
      step2.className = 'step-box';
      step3.className = 'step-box';
      step4.className = 'step-box';

      try {
        setTimeout(() => { step1.className = 'step-box done'; step2.className = 'step-box active'; }, 800);
        setTimeout(() => { step2.className = 'step-box done'; step3.className = 'step-box active'; }, 1600);
        setTimeout(() => { step3.className = 'step-box done'; step4.className = 'step-box active'; }, 3200);

        const resp = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: q, top_k: 5 })
        });
        const data = await resp.json();
        currentData = data;

        step4.className = 'step-box done';
        renderResults(data);
      } catch (err) {
        alert("Pipeline error: " + err.message);
      } finally {
        btn.disabled = false;
        loader.style.display = 'none';
        resultsContainer.style.display = 'block';
      }
    }

    function renderResults(data) {
      const winnerId = data.winner_id;
      const winnerBadge = document.getElementById('winnerBadge');
      winnerBadge.innerText = "🏆 CROWNED WINNER: " + winnerId;
      
      // Best Response
      const contentEl = document.getElementById('bestResponseContent');
      contentEl.innerHTML = marked.parse(data.final_best_response || "No response");

      // Sim count
      document.getElementById('simCount').innerText = data.similarities_count;

      // Judge Tab
      const judgeData = data.judge_evaluation || {};
      document.getElementById('judgeVerdictBox').innerHTML = "<b>Judge Verdict & Selection Rationale:</b><br>" + (judgeData.judge_verdict || "Evaluated by AI Judge.");
      
      const sBody = document.getElementById('scoresTableBody');
      sBody.innerHTML = '';
      (judgeData.candidates_evaluation || []).forEach(ce => {
        const isW = (ce.candidate_id === winnerId);
        const scores = ce.scores || {};
        const tr = document.createElement('tr');
        if (isW) tr.className = 'winner-row';
        tr.innerHTML = `
          <td><b>${ce.candidate_id}</b></td>
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

      // 5 Candidates Grid
      const candGrid = document.getElementById('candidatesGrid');
      candGrid.innerHTML = '';
      for (const [cid, cinfo] of Object.entries(data.candidate_responses || {})) {
        const isW = (cid === winnerId);
        const card = document.createElement('div');
        card.className = 'cand-card' + (isW ? ' is-winner' : '');
        card.innerHTML = `
          <div class="cand-header">
            <div>
              <div class="cand-title">${cid}</div>
              <div class="cand-desc">${cinfo.title || ''}</div>
            </div>
            ${isW ? '<span class="winner-badge" style="margin:0;padding:3px 8px;font-size:0.75rem;">WINNER</span>' : ''}
          </div>
          <div class="cand-body markdown-content">${marked.parse(cinfo.content || '')}</div>
        `;
        candGrid.appendChild(card);
      }

      // Retrieval Table
      const retBody = document.getElementById('retrievalTableBody');
      retBody.innerHTML = '';
      (data.top_ranked_schemes || []).forEach(s => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><b>#${s.rank}</b></td>
          <td><b style="color:#38bdf8;">${s.composite_rank_score.toFixed(4)}</b></td>
          <td><b>${s.schemeName}</b></td>
          <td><span style="color:#10b981;">${s.level}</span></td>
          <td>${(s.beneficiaryState || []).join(', ').substring(0, 30)}</td>
          <td>${s.dense_similarity.toFixed(4)}</td>
          <td>${s.lexical_score.toFixed(4)}</td>
          <td>${s.state_multiplier.toFixed(2)}x</td>
        `;
        retBody.appendChild(tr);
      });

      switchTab('bestResponseTab');
    }
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
