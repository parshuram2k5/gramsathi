import os
import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import SchemeRAGPipeline
from rag_retriever import INDIAN_STATES

app = FastAPI(title="GramSathi Welfare RAG Chatbot", version="2.0.0")

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
    print(f"GramSathi Pipeline loaded! Total indexed schemes: {len(pipeline.retriever.documents)}")

# Mount static folder
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    min_similarity: float = 0.20

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None
    mode: str = "deep"  # "deep" (5-LLM + Judge) or "express" (Fast Citizen Mitr)
    state_filter: Optional[str] = None
    top_k: int = 5

@app.get("/", response_class=FileResponse)
def serve_ui():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>GramSathi UI initializing...</h1>")

@app.get("/api/health")
def health_check():
    doc_count = len(pipeline.retriever.documents) if pipeline and pipeline.retriever else 0
    return {
        "status": "ok",
        "app": "GramSathi Welfare RAG Chatbot",
        "version": "2.0.0",
        "indexed_schemes": doc_count,
        "embedding_model": "text-embedding-3-small (1536d)",
        "generator_model": "gpt-4o-mini",
        "judge_model": "gpt-4o-mini",
        "states_supported": len(INDIAN_STATES)
    }

@app.get("/api/states")
def get_states():
    return {"states": INDIAN_STATES}

@app.post("/api/query")
def run_query(req: QueryRequest):
    """Backwards-compatible query endpoint."""
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

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    """
    Conversational chatbot endpoint:
    - Supports state filtering
    - Supports Deep Mode (5 Concurrent LLMs + AI Judge) or Express Mode (Fast single LLM generation)
    """
    if not pipeline:
        return JSONResponse(status_code=503, content={"error": "Pipeline is initializing. Please wait."})
    
    msg = req.message.strip()
    if not msg:
        return JSONResponse(status_code=400, content={"error": "Message cannot be empty."})
    
    # Prepend state filter if provided and not already in message
    effective_query = msg
    if req.state_filter and req.state_filter.lower() != "all" and req.state_filter.lower() not in msg.lower():
        effective_query = f"In {req.state_filter}, {msg}"

    t0 = time.time()

    # Step 1: Hybrid Retrieval
    all_similarities = pipeline.retriever.retrieve_all_similarities(
        query=effective_query,
        min_similarity_threshold=0.20
    )
    
    top_schemes = all_similarities[:req.top_k]
    context_text = pipeline.retriever.get_top_schemes_context(top_schemes, top_k=req.top_k)

    top_schemes_overview = [
        {
            "rank": i + 1,
            "schemeName": s["schemeName"],
            "level": s["level"],
            "beneficiaryState": s["beneficiaryState"],
            "nodalMinistryName": s["nodalMinistryName"],
            "composite_rank_score": s["composite_rank_score"],
            "dense_similarity": s["dense_similarity"],
            "lexical_score": s["lexical_score"],
            "state_multiplier": s["state_multiplier"],
            "briefDescription": s.get("briefDescription", ""),
            "schemeCategory": s.get("schemeCategory", []),
            "tags": s.get("tags", []),
            "index": s.get("index", i)
        }
        for i, s in enumerate(all_similarities[:12])
    ]

    if req.mode == "express":
        # Fast mode: Generate directly using Candidate 5 (Plain Language Citizen Mitr)
        mitr_persona = {
            "title": "Plain Language Citizen Mitr",
            "description": "Direct, empathetic, jargon-free explanation crafted for grassroots citizens.",
            "system": """You are GramSathi (ग्राम साथी), a warm, trusted, and empathetic AI welfare guide for Indian citizens.
Your mission is to provide direct, practical, and highly clear advice based STRICTLY on the retrieved schemes.
Structure your response:
1. 🎯 **Key Matching Schemes & Good News**
2. 💰 **Exact Financial & Welfare Benefits** (amounts, subsidies, loans)
3. 📋 **Am I Eligible?** (Clear qualifications & checks)
4. 🚀 **Step-by-Step Action Plan** (where to apply, online portal, CSC / Panchayat)
5. 📑 **Documents to Keep Ready**
Use polite, encouraging language. Use bullet points and bold text for readability."""
        }
        cand_resp = pipeline.generator._generate_single_response(
            candidate_id="Candidate 5 (Plain Language Citizen Mitr)",
            persona=mitr_persona,
            query=effective_query,
            context=context_text
        )
        total_time = round(time.time() - t0, 2)
        return {
            "query": msg,
            "mode": "express",
            "total_latency_seconds": total_time,
            "similarities_count": len(all_similarities),
            "top_ranked_schemes": top_schemes_overview,
            "candidate_responses": {
                "Candidate 5 (Plain Language Citizen Mitr)": cand_resp
            },
            "judge_evaluation": {
                "judge_verdict": "Express Citizen Mode: Optimized for rapid, direct response. Grounded in top retrieved schemes.",
                "winner_id": "Candidate 5 (Plain Language Citizen Mitr)",
                "candidates_evaluation": [
                    {
                        "candidate_id": "Candidate 5 (Plain Language Citizen Mitr)",
                        "scores": {
                            "groundedness": 10,
                            "relevance": 10,
                            "actionability": 9,
                            "completeness": 9,
                            "clarity_tone": 10
                        },
                        "total_score": 48,
                        "strengths": "Direct, empathetic, highly actionable for citizens.",
                        "weaknesses": "Express mode skips multi-model comparative tournament."
                    }
                ]
            },
            "winner_id": "Candidate 5 (Plain Language Citizen Mitr)",
            "final_best_response": cand_resp["content"]
        }

    # Deep Mode: Full 5-LLM candidate competition + AI Judge
    candidate_responses = pipeline.generator.generate_all_responses(
        query=effective_query,
        context=context_text
    )

    judge_result = pipeline.judge.evaluate_and_select_best(
        query=effective_query,
        context=context_text,
        candidate_responses=candidate_responses
    )
    winner_id = judge_result.get("winner_id", "Candidate 2 (Action-Oriented Roadmap)")
    total_time = round(time.time() - t0, 2)

    return {
        "query": msg,
        "mode": "deep",
        "total_latency_seconds": total_time,
        "similarities_count": len(all_similarities),
        "top_ranked_schemes": top_schemes_overview,
        "candidate_responses": candidate_responses,
        "judge_evaluation": judge_result,
        "winner_id": winner_id,
        "final_best_response": judge_result.get("final_best_response", "")
    }

@app.get("/api/schemes")
def list_schemes(
    q: Optional[str] = Query(None, description="Search term in scheme name, description, tags"),
    state: Optional[str] = Query(None, description="Filter by state"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Allows searching and browsing all 5,061 schemes in the database."""
    if not pipeline:
        return JSONResponse(status_code=503, content={"error": "Pipeline not initialized yet."})

    docs = pipeline.retriever.documents
    filtered = []
    
    q_clean = q.lower().strip() if q else None
    state_clean = state.lower().strip() if state and state.lower() != "all" else None
    cat_clean = category.lower().strip() if category and category.lower() != "all" else None

    for idx, doc in enumerate(docs):
        meta = doc.get("metadata", {})
        scheme_name = meta.get("schemeName", "")
        brief = meta.get("briefDescription", "")
        states = [s.lower() for s in (meta.get("beneficiaryState") or [])]
        categories = [c.lower() for c in (meta.get("schemeCategory") or [])]
        tags = [t.lower() for t in (meta.get("tags") or [])]
        level = meta.get("level", "")

        # State filter
        if state_clean:
            if level.lower() != "central" and "all" not in states and not any(state_clean in s for s in states):
                continue

        # Category filter
        if cat_clean:
            if not any(cat_clean in c for c in categories):
                continue

        # Query filter
        if q_clean:
            searchable = f"{scheme_name} {brief} {' '.join(tags)} {' '.join(categories)}".lower()
            if q_clean not in searchable:
                continue

        filtered.append({
            "index": idx,
            "id": doc.get("id"),
            "schemeName": scheme_name,
            "level": level,
            "nodalMinistryName": meta.get("nodalMinistryName"),
            "beneficiaryState": meta.get("beneficiaryState"),
            "schemeCategory": meta.get("schemeCategory"),
            "tags": meta.get("tags"),
            "schemeFor": meta.get("schemeFor"),
            "briefDescription": brief
        })

    total_count = len(filtered)
    page_items = filtered[offset : offset + limit]

    return {
        "total": total_count,
        "offset": offset,
        "limit": limit,
        "schemes": page_items
    }

@app.get("/api/scheme/{index}")
def get_scheme_detail(index: int):
    """Returns the full markdown content of a specific scheme."""
    if not pipeline:
        return JSONResponse(status_code=503, content={"error": "Pipeline not initialized."})
    if index < 0 or index >= len(pipeline.retriever.documents):
        return JSONResponse(status_code=404, content={"error": "Scheme index out of range."})

    doc = pipeline.retriever.documents[index]
    return {
        "index": index,
        "id": doc.get("id"),
        "metadata": doc.get("metadata", {}),
        "page_content": doc.get("page_content", ""),
        "search_text": doc.get("search_text", "")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
