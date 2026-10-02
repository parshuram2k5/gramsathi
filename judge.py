import os
import json
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

JUDGE_SYSTEM_PROMPT = """You are an elite, impartial AI Judge and Public Policy Evaluator specializing in Indian government welfare schemes.
Your mission is to evaluate 5 candidate AI responses generated for a citizen's query based on ground-truth retrieved scheme data.

You must evaluate each candidate across 5 strict criteria (score 1 to 10 for each, 10 being perfect):
1. **Factual Groundedness & Accuracy (1-10)**: Is every fact, monetary amount, criteria, and link strictly supported by the retrieved scheme data? Zero tolerance for hallucinations.
2. **Relevance & Intent Alignment (1-10)**: Does the response directly solve the citizen's specific state, demographic, and stated needs?
3. **Actionability & Procedural Clarity (1-10)**: Does it guide the citizen on exact next steps, portal URLs, documents needed, and submission channels?
4. **Comprehensiveness & Exclusions (1-10)**: Does it mention important eligibility caveats, limits, or exclusion criteria to prevent rejected applications?
5. **Readability & Formatting (1-10)**: Structure, headings, bullet points, and clarity.

You must output a valid JSON object strictly matching this schema:
{
  "candidates_evaluation": [
    {
      "candidate_id": "Candidate 1 (Comprehensive Advisor)",
      "scores": {
        "groundedness": 9,
        "relevance": 9,
        "actionability": 8,
        "completeness": 9,
        "clarity_tone": 9
      },
      "total_score": 44,
      "strengths": "Clear summary, accurate financial breakdown.",
      "weaknesses": "Could highlight offline office visits more."
    },
    ... (for all 5 candidates)
  ],
  "rankings": [
    {"rank": 1, "candidate_id": "...", "total_score": 46},
    {"rank": 2, "candidate_id": "...", "total_score": 43},
    {"rank": 3, "candidate_id": "...", "total_score": 40},
    {"rank": 4, "candidate_id": "...", "total_score": 38},
    {"rank": 5, "candidate_id": "...", "total_score": 35}
  ],
  "winner_id": "Candidate 1 (Comprehensive Advisor)",
  "judge_verdict": "Detailed explanation of why this candidate won and why it is the most valuable and safe for the citizen.",
  "final_best_response": "The complete, polished best response ready for the citizen, formatted in clean Markdown."
}
"""

class SchemeResponseJudge:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI()

    def evaluate_and_select_best(
        self,
        query: str,
        context: str,
        candidate_responses: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates the 5 candidate responses and determines the winning best response.
        """
        t0 = time.time()
        
        candidates_formatted = []
        for cid, cand in candidate_responses.items():
            candidates_formatted.append(f"""
============================================================
### {cid}: {cand.get('title', '')}
------------------------------------------------------------
{cand.get('content', '')}
============================================================
""")
        
        candidates_text = "\n".join(candidates_formatted)
        
        prompt = f"""### CITIZEN QUERY:
{query}

### GROUND TRUTH RETRIEVED SCHEME CONTEXT:
{context}

### THE 5 CANDIDATE RESPONSES TO EVALUATE:
{candidates_text}

Analyze and score all 5 responses thoroughly according to the 5 evaluation criteria. Return your evaluation strictly in the requested JSON format."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            evaluation_raw = response.choices[0].message.content
            evaluation_data = json.loads(evaluation_raw)
            evaluation_data["latency_seconds"] = round(time.time() - t0, 2)
            
            # Ensure final_best_response is present
            if not evaluation_data.get("final_best_response"):
                winner = evaluation_data.get("winner_id")
                if winner and winner in candidate_responses:
                    evaluation_data["final_best_response"] = candidate_responses[winner]["content"]
                    
            return evaluation_data
            
        except Exception as e:
            # Fallback if parsing or API fails
            first_key = list(candidate_responses.keys())[0] if candidate_responses else "Candidate 1"
            return {
                "error": str(e),
                "winner_id": first_key,
                "judge_verdict": f"Automated evaluation encountered an issue: {e}. Defaulting to {first_key}.",
                "final_best_response": candidate_responses[first_key]["content"] if candidate_responses else "No response generated.",
                "latency_seconds": round(time.time() - t0, 2)
            }

if __name__ == "__main__":
    judge = SchemeResponseJudge()
    print("SchemeResponseJudge initialized.")
