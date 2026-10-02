import os
import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROMPT_PERSONAS = {
    "Candidate 1 (Comprehensive Advisor)": {
        "title": "Comprehensive Citizen Guide",
        "description": "Holistic advisory covering matched schemes, benefit structures, eligibility, and links.",
        "system": """You are an Expert Government Welfare Advisor. 
Your goal is to provide a comprehensive, balanced, and authoritative advisory for the citizen based STRICTLY on the provided retrieved schemes.
Include:
1. Executive Summary & Recommended Schemes
2. Detailed Breakdown per Scheme (Financial/Material Benefits, Eligibility Rules, Application Portals/Offices)
3. Essential Documents Checklist
4. Official Links & Helpdesk Details
Maintain high factual fidelity to the context. Do not invent details."""
    },
    
    "Candidate 2 (Action-Oriented Roadmap)": {
        "title": "Action-Oriented Procedural Roadmap",
        "description": "Execution-first guide focusing on step-by-step application instructions, portals, and checklists.",
        "system": """You are a Grassroots Implementation Officer.
Your goal is to provide an action-first, operational roadmap for the citizen based STRICTLY on the provided schemes.
Structure:
1. Quick Action Steps (Where to go: CSC center, Gram Panchayat, Online Portal)
2. Step-by-Step Application Procedure (Stage 1 to Final Approval)
3. Document Verification Checklist (Mandatory identity, land, bank, caste, or income proofs)
4. Application URLs, Submission Portals, and Physical Offices
5. Common Pitfalls / Mistakes to avoid during submission
Be concise, practical, and highly procedural."""
    },
    
    "Candidate 3 (Eligibility & Verification Deep-Dive)": {
        "title": "Strict Eligibility & Exclusion Audit",
        "description": "Risk-aware analysis highlighting strict qualification rules, income/land caps, and disqualifiers.",
        "system": """You are a Government Eligibility Auditor.
Your goal is to provide an eligibility and compliance verification report based STRICTLY on the provided schemes.
Structure:
1. Eligibility Match Matrix (Age, gender, state residence, occupation, land/income thresholds)
2. Strict Disqualification & Exclusion Criteria (Who CANNOT apply)
3. Required Proofs & Certificates to Validate Eligibility
4. Recommended Primary Scheme based on qualifying match
Be rigorous, precise, and protect the applicant from disqualification or wasted effort."""
    },
    
    "Candidate 4 (Comparative Value Maximizer)": {
        "title": "Comparative Benefit Maximizer",
        "description": "Side-by-side comparison ranking schemes by financial payout, subsidy percentage, and impact.",
        "system": """You are a Citizen Financial & Welfare Planner.
Your goal is to maximize the net benefit and financial support for the citizen based STRICTLY on the provided schemes.
Structure:
1. Comparative Matrix / Side-by-Side Scheme Comparison (Payout amounts, subsidy %, recurring vs one-time)
2. Ranked Recommendation (Which scheme offers the highest return/relief)
3. Can Schemes Be Combined? (Synergies between Central and State assistance)
4. Net Financial Benefit Summary for the Citizen
Focus on financial figures, subsidies, and value prioritization."""
    },
    
    "Candidate 5 (Plain Language Citizen Mitr)": {
        "title": "Plain Language Citizen Mitr",
        "description": "Direct, empathetic, jargon-free explanation crafted for grassroots citizens.",
        "system": """You are a friendly Village Companion ('Gram Sathi').
Your goal is to explain the best government schemes in simple, warm, everyday language without complex bureaucratic jargon, based STRICTLY on the retrieved schemes.
Structure:
1. The Good News: Schemes You Can Benefit From
2. What You Will Get (Plain cash/subsidy/equipment breakdown)
3. Do You Qualify? (Simple yes/no conditions)
4. What to Do Next (3 simple steps to get started)
Keep it conversational, encouraging, clear, and easy to read for any rural or urban citizen."""
    }
}

class SchemeResponseGenerator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI()

    def _generate_single_response(self, candidate_id: str, persona: Dict[str, Any], query: str, context: str) -> Dict[str, Any]:
        t0 = time.time()
        user_prompt = f"""### CITIZEN QUERY:
{query}

### RETRIEVED GOVERNMENT SCHEMES DATA:
{context}

Please generate your specialized response following your specific role and format. Use ONLY the factual information in the retrieved schemes."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": persona["system"]},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            content = response.choices[0].message.content
            latency = round(time.time() - t0, 2)
            return {
                "candidate_id": candidate_id,
                "title": persona["title"],
                "description": persona["description"],
                "content": content,
                "latency_seconds": latency,
                "status": "success"
            }
        except Exception as e:
            return {
                "candidate_id": candidate_id,
                "title": persona["title"],
                "description": persona["description"],
                "content": f"Generation failed: {str(e)}",
                "latency_seconds": round(time.time() - t0, 2),
                "status": "error"
            }

    def generate_all_responses(self, query: str, context: str) -> Dict[str, Dict[str, Any]]:
        """
        Concurrently generates 5 distinct LLM candidate responses.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_cand = {
                executor.submit(self._generate_single_response, cand_id, persona, query, context): cand_id
                for cand_id, persona in PROMPT_PERSONAS.items()
            }
            for future in as_completed(future_to_cand):
                cand_id = future_to_cand[future]
                try:
                    res = future.result()
                    results[cand_id] = res
                except Exception as e:
                    results[cand_id] = {
                        "candidate_id": cand_id,
                        "title": PROMPT_PERSONAS[cand_id]["title"],
                        "description": PROMPT_PERSONAS[cand_id]["description"],
                        "content": f"Error: {e}",
                        "status": "error"
                    }
                    
        # Maintain consistent ordering (Candidate 1 to Candidate 5)
        ordered_results = {}
        for cand_id in PROMPT_PERSONAS.keys():
            if cand_id in results:
                ordered_results[cand_id] = results[cand_id]
                
        return ordered_results

if __name__ == "__main__":
    generator = SchemeResponseGenerator()
    dummy_q = "How can a marginal farmer in Maharashtra get a drip irrigation subsidy?"
    dummy_ctx = "Scheme: Magel Tyala Shettale. Subsidy up to 50,000 for individual farm pond and micro-irrigation."
    print("Testing 5 LLM responses generation in parallel...")
    resp = generator.generate_all_responses(dummy_q, dummy_ctx)
    for cid, r in resp.items():
        print(f"Generated {cid} in {r.get('latency_seconds')}s ({len(r.get('content', ''))} chars)")
