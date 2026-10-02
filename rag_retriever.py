import os
import json
import re
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CACHE_DIR = "rag_cache"
EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "scheme_embeddings.npy")
DOCS_FILE = os.path.join(CACHE_DIR, "schemes_store.json")

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal", "Delhi", "Jammu and Kashmir", "Ladakh",
    "Puducherry", "Chandigarh", "Andaman and Nicobar", "Dadra and Nagar Haveli",
    "Daman and Diu", "Lakshadweep"
]

class SchemeRAGRetriever:
    def __init__(self, embeddings_file: str = EMBEDDINGS_FILE, docs_file: str = DOCS_FILE):
        if not os.path.exists(embeddings_file) or not os.path.exists(docs_file):
            raise FileNotFoundError(
                f"Index files missing! Run `python index_dataset.py` first. Looking for: {embeddings_file} and {docs_file}"
            )
            
        print(f"Loading scheme embeddings matrix from {embeddings_file}...")
        self.embeddings = np.load(embeddings_file)
        
        print(f"Loading schemes documents store from {docs_file}...")
        with open(docs_file, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
            
        assert len(self.embeddings) == len(self.documents), (
            f"Embeddings count ({len(self.embeddings)}) does not match documents count ({len(self.documents)})!"
        )
        print(f"Retriever initialized successfully with {len(self.documents)} schemes.")
        self.client = OpenAI()

    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extract detected states, demographics, and key tokens from user query."""
        q_lower = query.lower()
        
        detected_states = []
        for state in INDIAN_STATES:
            if re.search(r'\b' + re.escape(state.lower()) + r'\b', q_lower):
                detected_states.append(state)
                
        # Common aliases
        if "up" in q_lower.split() or "uttar pradesh" in q_lower:
            if "Uttar Pradesh" not in detected_states:
                detected_states.append("Uttar Pradesh")
        if "mp" in q_lower.split() or "madhya pradesh" in q_lower:
            if "Madhya Pradesh" not in detected_states:
                detected_states.append("Madhya Pradesh")
        if "wb" in q_lower.split() or "bengal" in q_lower:
            if "West Bengal" not in detected_states:
                detected_states.append("West Bengal")
        if "tn" in q_lower.split():
            if "Tamil Nadu" not in detected_states:
                detected_states.append("Tamil Nadu")
                
        # Clean tokens for keyword matching
        words = re.findall(r'\b\w{3,}\b', q_lower)
        stopwords = {"the", "and", "for", "with", "from", "that", "this", "are", "can", "how", "what", "which", "want", "need", "get", "give", "help", "scheme", "schemes", "government", "govt", "india", "indian", "yojana"}
        keywords = [w for w in words if w not in stopwords]
        
        return {
            "detected_states": detected_states,
            "keywords": keywords,
            "query_lower": q_lower
        }

    def _compute_lexical_similarity(self, doc: Dict[str, Any], query_features: Dict[str, Any]) -> float:
        """Computes lexical / keyword matching score."""
        keywords = query_features["keywords"]
        if not keywords:
            return 0.0
            
        meta = doc.get("metadata", {})
        scheme_name = meta.get("schemeName", "").lower()
        tags = [t.lower() for t in meta.get("tags", [])]
        categories = [c.lower() for c in meta.get("schemeCategory", [])]
        brief = meta.get("briefDescription", "").lower()
        search_text = doc.get("search_text", "").lower()
        
        matched_score = 0.0
        
        for kw in keywords:
            # High weight for scheme name match
            if kw in scheme_name:
                matched_score += 1.5
            elif any(kw in t for t in tags):
                matched_score += 1.0
            elif any(kw in c for c in categories):
                matched_score += 0.8
            elif kw in brief:
                matched_score += 0.5
            elif kw in search_text:
                matched_score += 0.3
                
        # Exact match or acronym match bonus
        q_clean = query_features["query_lower"].replace("-", " ")
        name_clean = scheme_name.replace("-", " ")
        if name_clean in q_clean or q_clean in name_clean:
            matched_score += 3.0
            
        # Normalize roughly to [0, 1]
        max_possible = max(len(keywords) * 1.5, 1.0)
        return min(matched_score / max_possible, 1.0)

    def _compute_state_compatibility(self, doc: Dict[str, Any], detected_states: List[str]) -> float:
        """Evaluates whether the scheme matches the citizen's state."""
        if not detected_states:
            return 1.0  # No state specified; neutral
            
        meta = doc.get("metadata", {})
        level = meta.get("level", "")
        beneficiary_states = [s.strip().lower() for s in meta.get("beneficiaryState", [])]
        
        # Central schemes are universally applicable
        if level.lower() == "central" or "all" in beneficiary_states or not beneficiary_states:
            return 1.0
            
        # Check if any detected state is in beneficiary states
        for st in detected_states:
            st_lower = st.lower()
            if any(st_lower in bs or bs in st_lower for bs in beneficiary_states):
                return 1.25  # Boost state-specific matching schemes
                
        # Explicit mismatch (scheme is for another specific state)
        return 0.15

    def retrieve_all_similarities(
        self,
        query: str,
        dense_weight: float = 0.65,
        lexical_weight: float = 0.35,
        min_similarity_threshold: float = 0.20,
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves each and every possible similarity and ranks them using hybrid scoring.
        """
        # 1. Embed query
        query_resp = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        query_vec = np.array(query_resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
            
        # 2. Dense semantic similarity across all 5,061 schemes in matrix
        dense_similarities = np.dot(self.embeddings, query_vec)
        
        # 3. Query features extraction
        query_features = self._extract_query_features(query)
        detected_states = query_features["detected_states"]
        
        # 4. Hybrid ranking & score computation
        ranked_results = []
        for idx, doc in enumerate(self.documents):
            dense_sim = float(dense_similarities[idx])
            lex_score = self._compute_lexical_similarity(doc, query_features)
            state_multiplier = self._compute_state_compatibility(doc, detected_states)
            
            # Base hybrid score
            base_score = (dense_weight * max(0.0, dense_sim)) + (lexical_weight * lex_score)
            
            # Final composite score with state compatibility
            composite_score = base_score * state_multiplier
            
            # Include every possible match above minimal threshold
            if composite_score >= min_similarity_threshold or dense_sim >= 0.40 or lex_score >= 0.60:
                meta = doc.get("metadata", {})
                ranked_results.append({
                    "id": doc.get("id"),
                    "index": idx,
                    "schemeName": meta.get("schemeName"),
                    "nodalMinistryName": meta.get("nodalMinistryName"),
                    "level": meta.get("level"),
                    "beneficiaryState": meta.get("beneficiaryState"),
                    "schemeCategory": meta.get("schemeCategory"),
                    "tags": meta.get("tags"),
                    "schemeFor": meta.get("schemeFor"),
                    "briefDescription": meta.get("briefDescription"),
                    "dense_similarity": round(dense_sim, 4),
                    "lexical_score": round(lex_score, 4),
                    "state_multiplier": round(state_multiplier, 2),
                    "composite_rank_score": round(composite_score, 4),
                    "page_content": doc.get("page_content")
                })
                
        # 5. Sort by composite_rank_score descending
        ranked_results.sort(key=lambda x: x["composite_rank_score"], reverse=True)
        
        if top_n is not None:
            return ranked_results[:top_n]
        return ranked_results

    def get_top_schemes_context(self, ranked_schemes: List[Dict[str, Any]], top_k: int = 5) -> str:
        """Formats the top-K schemes into a clear context prompt for the LLM."""
        selected = ranked_schemes[:top_k]
        context_blocks = []
        
        for rank, scheme in enumerate(selected, 1):
            block = [
                f"==================================================",
                f"RANK #{rank}: {scheme['schemeName']} (Rank Score: {scheme['composite_rank_score']:.4f})",
                f"Ministry: {scheme.get('nodalMinistryName')} | Level: {scheme.get('level')}",
                f"Beneficiary States: {', '.join(scheme.get('beneficiaryState') or [])}",
                f"Categories: {', '.join(scheme.get('schemeCategory') or [])}",
                f"Target Beneficiaries: {scheme.get('schemeFor')}",
                f"--------------------------------------------------",
                scheme["page_content"],
                f"==================================================\n"
            ]
            context_blocks.append("\n".join(block))
            
        return "\n\n".join(context_blocks)

if __name__ == "__main__":
    retriever = SchemeRAGRetriever()
    test_q = "I am a female farmer from Maharashtra with 2 acres of land looking for drip irrigation subsidy and crop insurance"
    print(f"\n--- Testing Query: '{test_q}' ---")
    results = retriever.retrieve_all_similarities(test_q, top_n=10)
    print(f"Total similarity matches found: {len(results)}")
    print(f"Top 5 Ranked Schemes:")
    for i, r in enumerate(results[:5], 1):
        print(f"#{i} [{r['composite_rank_score']:.4f}] {r['schemeName']} | Level: {r['level']} | States: {r['beneficiaryState']}")
