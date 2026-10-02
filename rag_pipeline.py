import time
from typing import Dict, Any, List, Optional
from rag_retriever import SchemeRAGRetriever
from generator import SchemeResponseGenerator
from judge import SchemeResponseJudge

class SchemeRAGPipeline:
    def __init__(
        self,
        retriever: Optional[SchemeRAGRetriever] = None,
        generator_model: str = "gpt-4o-mini",
        judge_model: str = "gpt-4o-mini"
    ):
        print("Initializing SchemeRAGPipeline...")
        self.retriever = retriever or SchemeRAGRetriever()
        self.generator = SchemeResponseGenerator(model=generator_model)
        self.judge = SchemeResponseJudge(model=judge_model)
        print("SchemeRAGPipeline ready.")

    def run(
        self,
        query: str,
        top_k_context: int = 5,
        min_similarity_threshold: float = 0.20
    ) -> Dict[str, Any]:
        """
        Executes the full end-to-end RAG pipeline:
        1. Retrieve each and every possible similarity and rank them
        2. Format top schemes as rich context
        3. Generate 5 distinct candidate LLM responses
        4. Execute AI Judge evaluation across 5 dimensions
        5. Return the winning best response and full audit trail
        """
        t0 = time.time()
        
        # Step 1: Retrieve and rank similarities
        print(f"\n[1/4] Retrieving each and every similarity for query: '{query}'...")
        all_similarities = self.retriever.retrieve_all_similarities(
            query=query,
            min_similarity_threshold=min_similarity_threshold
        )
        print(f"      Found {len(all_similarities)} similarity matches across 5,061 schemes.")
        
        # Step 2: Format context for top candidates
        top_schemes = all_similarities[:top_k_context]
        context_text = self.retriever.get_top_schemes_context(top_schemes, top_k=top_k_context)
        print(f"[2/4] Formatted top {len(top_schemes)} schemes into grounding context.")
        
        # Step 3: Generate 5 LLM responses in parallel
        print("[3/4] Concurrently generating 5 specialized candidate LLM responses...")
        candidate_responses = self.generator.generate_all_responses(
            query=query,
            context=context_text
        )
        print("      Generated all 5 responses successfully.")
        
        # Step 4: AI as the Judge
        print("[4/4] Executing AI Judge to evaluate candidates and select best response...")
        judge_result = self.judge.evaluate_and_select_best(
            query=query,
            context=context_text,
            candidate_responses=candidate_responses
        )
        winner_id = judge_result.get("winner_id", "Candidate 1")
        print(f"      AI Judge crowned: {winner_id}!")
        
        total_time = round(time.time() - t0, 2)
        
        return {
            "query": query,
            "total_latency_seconds": total_time,
            "similarities_count": len(all_similarities),
            "top_ranked_schemes": [
                {
                    "rank": i + 1,
                    "schemeName": s["schemeName"],
                    "level": s["level"],
                    "beneficiaryState": s["beneficiaryState"],
                    "nodalMinistryName": s["nodalMinistryName"],
                    "composite_rank_score": s["composite_rank_score"],
                    "dense_similarity": s["dense_similarity"],
                    "lexical_score": s["lexical_score"],
                    "state_multiplier": s["state_multiplier"]
                }
                for i, s in enumerate(all_similarities[:15])  # Top 15 overview
            ],
            "candidate_responses": candidate_responses,
            "judge_evaluation": judge_result,
            "winner_id": winner_id,
            "final_best_response": judge_result.get("final_best_response", "")
        }

if __name__ == "__main__":
    pipeline = SchemeRAGPipeline()
    sample_query = "I am a girl student in Karnataka belonging to SC community looking for engineering scholarship and hostel allowance"
    result = pipeline.run(sample_query, top_k_context=4)
    print("\n" + "=" * 60)
    print(f"WINNER: {result['winner_id']}")
    print("=" * 60)
    print(result["final_best_response"])
