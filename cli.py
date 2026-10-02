import sys
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rag_pipeline import SchemeRAGPipeline

console = Console()

def run_cli():
    parser = argparse.ArgumentParser(description="GramSathi Multi-LLM RAG with AI Judge")
    parser.add_argument("--query", "-q", type=str, help="Citizen query")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of top schemes for context (default: 5)")
    parser.add_argument("--show-similarities", "-s", action="store_true", help="Display top ranked similarity table")
    parser.add_argument("--show-all-candidates", "-c", action="store_true", help="Display all 5 generated candidate responses")
    parser.add_argument("--show-judge-scores", "-j", action="store_true", help="Display the AI Judge's detailed scorecard")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive chat loop")
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]🌾 GramSathi Intelligent Welfare RAG Pipeline[/bold cyan]\n"
        "[dim]Exhaustive Similarity Retrieval → Multi-Factor Ranking → 5 LLM Persona Generation → AI Judge Evaluation[/dim]",
        border_style="cyan"
    ))
    
    with console.status("[bold green]Loading vector matrix and initializing RAG pipeline..."):
        pipeline = SchemeRAGPipeline()
        
    def process_query(user_query: str):
        console.rule(f"[bold yellow]Processing Query[/bold yellow]: {user_query}")
        with console.status("[bold yellow]Executing multi-retrieval, 5-LLM generation, and AI Judge evaluation..."):
            result = pipeline.run(user_query, top_k_context=args.top_k)
            
        console.print(f"[green]✓ Completed in {result['total_latency_seconds']}s | Found {result['similarities_count']} similarity matches[/green]\n")
        
        # 1. Similarity Table
        if args.show_similarities or not args.query:
            table = Table(title="Top Ranked Similar Schemes (Ground Truth Retrieval)", border_style="blue")
            table.add_column("Rank", justify="center", style="cyan", no_wrap=True)
            table.add_column("Composite Score", justify="right", style="magenta")
            table.add_column("Scheme Name", style="bold white")
            table.add_column("Level", style="green")
            table.add_column("States", style="yellow")
            table.add_column("Dense / Lexical / State Multiplier", style="dim")
            
            for s in result["top_ranked_schemes"][:8]:
                table.add_row(
                    str(s["rank"]),
                    f"{s['composite_rank_score']:.4f}",
                    s["schemeName"],
                    s["level"],
                    ", ".join(s["beneficiaryState"] or [])[:25],
                    f"D:{s['dense_similarity']:.2f} | L:{s['lexical_score']:.2f} | M:{s['state_multiplier']:.1f}x"
                )
            console.print(table)
            console.print()
            
        # 2. Show All Candidates if requested
        if args.show_all_candidates:
            console.rule("[bold purple]The 5 Candidate LLM Responses[/bold purple]")
            for cid, cdata in result["candidate_responses"].items():
                is_winner = (cid == result["winner_id"])
                border_color = "green" if is_winner else "dim"
                title_badge = f"{cid} - {cdata.get('title')} {'🏆 WINNER' if is_winner else ''}"
                console.print(Panel(
                    Markdown(cdata.get("content", "")),
                    title=f"[bold]{title_badge}[/bold]",
                    border_style=border_color,
                    subtitle=f"[dim]Generated in {cdata.get('latency_seconds')}s[/dim]"
                ))
                console.print()
                
        # 3. Judge Evaluation Scorecard
        judge_data = result["judge_evaluation"]
        if args.show_judge_scores or not args.query:
            score_table = Table(title="⚖️ AI Judge Evaluation Scorecard (1-10 Scale)", border_style="gold1")
            score_table.add_column("Candidate", style="bold cyan")
            score_table.add_column("Groundedness", justify="center")
            score_table.add_column("Relevance", justify="center")
            score_table.add_column("Actionability", justify="center")
            score_table.add_column("Completeness", justify="center")
            score_table.add_column("Clarity", justify="center")
            score_table.add_column("Total Score (/50)", justify="center", style="bold yellow")
            score_table.add_column("Result", justify="center")
            
            for ce in judge_data.get("candidates_evaluation", []):
                cid = ce.get("candidate_id", "")
                scores = ce.get("scores", {})
                is_winner = (cid == result["winner_id"])
                score_table.add_row(
                    cid,
                    str(scores.get("groundedness", "-")),
                    str(scores.get("relevance", "-")),
                    str(scores.get("actionability", "-")),
                    str(scores.get("completeness", "-")),
                    str(scores.get("clarity_tone", "-")),
                    str(ce.get("total_score", "-")),
                    "[bold green]🏆 Winner[/bold green]" if is_winner else "[dim]Runner-up[/dim]"
                )
            console.print(score_table)
            console.print(Panel(
                f"[bold yellow]Judge Verdict:[/bold yellow]\n{judge_data.get('judge_verdict', '')}",
                title="[bold gold1]⚖️ Official Verdict Rationale[/bold gold1]",
                border_style="gold1"
            ))
            console.print()
            
        # 4. Final Best Response
        console.rule("[bold green]🏆 FINAL BEST RESPONSE (Selected by AI Judge)[/bold green]")
        console.print(Panel(
            Markdown(result["final_best_response"]),
            title=f"[bold green]Verified Best Response ({result['winner_id']})[/bold green]",
            border_style="green",
            padding=(1, 2)
        ))

    if args.interactive:
        console.print("[bold yellow]Interactive Mode Activated. Type 'exit' or 'quit' to end.[/bold yellow]\n")
        while True:
            try:
                user_q = console.input("[bold cyan]Enter Citizen Query > [/bold cyan]").strip()
                if not user_q:
                    continue
                if user_q.lower() in ["exit", "quit", "q"]:
                    console.print("[dim]Goodbye![/dim]")
                    break
                process_query(user_q)
            except (KeyboardInterrupt, EOFError):
                break
    elif args.query:
        process_query(args.query)
    else:
        # Default run with sample query
        default_q = "I am a female farmer from Maharashtra with 2 acres looking for drip irrigation subsidy and crop insurance"
        process_query(default_q)

if __name__ == "__main__":
    run_cli()
