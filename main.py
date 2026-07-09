"""
Financial Deep Research Agent — Main Entry Point

Interactive CLI that guides users through:
  1. Query submission
  2. Research plan review & approval
  3. Deep research execution (with live progress)
  4. Comprehensive report generation & saving

Usage:
    python main.py
    python main.py --query "Analyze the Indian IT services sector"
    python main.py --query "..." --no-approval  # Skip plan approval
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.text import Text

from agents.it_agent import ITSectorAgent
from agents.pharma_agent import PharmaSectorAgent
from config import config
from core.router import QueryRouter

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,  # Suppress verbose logs in CLI; set to DEBUG for dev
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


# ── Progress Callback ─────────────────────────────────────────────────────────

class ProgressTracker:
    """Tracks research progress for display."""

    def __init__(self) -> None:
        self.steps: list[str] = []
        self._progress: Progress | None = None
        self._task_id = None

    def callback(self, step_num: int, total: int, message: str) -> None:
        self.steps.append(f"Step {step_num}: {message}")
        console.print(
            f"  [dim cyan]▶ Step {step_num}/{total}:[/dim cyan] {message}"
        )


# ── Main Application ──────────────────────────────────────────────────────────

def build_router() -> QueryRouter:
    """Initialize and configure the query router with all sector agents."""
    router = QueryRouter()
    router.register_agent("IT", ITSectorAgent())
    router.register_agent("Pharma", PharmaSectorAgent())
    return router


def print_welcome() -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold blue]Financial Deep Research Agent[/bold blue]\n"
            "[dim]Powered by multi-step iterative research with LLM synthesis[/dim]\n\n"
            "[green]Supported Sectors:[/green] IT Technology | Pharmaceutical & Healthcare\n"
            "[dim]Type 'exit' or 'quit' to stop[/dim]",
            border_style="blue",
        )
    )
    console.print()


def run_research_workflow(
    router: QueryRouter,
    user_query: str,
    require_approval: bool = True,
) -> None:
    """Execute the full research workflow for a given query."""

    console.print(Rule("[bold]Step 1: Query Analysis & Research Planning[/bold]"))
    console.print()

    # Route the query
    with console.status("[cyan]Analyzing query and routing to sector agent...[/cyan]"):
        sector, agent, message = router.route(user_query)

    if agent is None:
        console.print(Panel(message, title="[yellow]Query Routing[/yellow]", border_style="yellow"))
        return

    console.print(f"  [green]✓[/green] Routed to: [bold]{sector} Sector Agent[/bold]")
    console.print()

    # Generate research plan
    with console.status("[cyan]Generating research plan...[/cyan]"):
        analysis, plan, formatted_plan = agent.get_plan(user_query)

    console.print(formatted_plan)

    # User approval step
    if require_approval:
        console.print(Rule("[bold]Step 2: Plan Approval[/bold]"))
        console.print()

        approved = Confirm.ask(
            "  [bold]Proceed with this research plan?[/bold]",
            default=True,
        )

        if not approved:
            modify = Confirm.ask(
                "  Would you like to modify the query before proceeding?",
                default=False,
            )
            if modify:
                user_query = Prompt.ask("  Enter modified query")
                # Re-run with modified query
                run_research_workflow(router, user_query, require_approval)
                return
            else:
                console.print("[yellow]Research cancelled.[/yellow]")
                return

    # Execute deep research
    console.print()
    console.print(Rule("[bold]Step 3: Deep Research Execution[/bold]"))
    console.print()
    console.print(
        f"  [dim]Running {config.MIN_RESEARCH_STEPS}–{config.MAX_RESEARCH_STEPS} "
        f"research steps. This may take a few minutes...[/dim]"
    )
    console.print()

    tracker = ProgressTracker()
    start_time = time.time()

    report, file_path = agent.run(
        user_query=user_query,
        approved_plan=plan,
        progress_callback=tracker.callback,
        save_report=True,
    )

    elapsed = time.time() - start_time
    console.print()
    console.print(
        f"  [green]✓[/green] Research complete in [bold]{elapsed:.1f}s[/bold] "
        f"({len(tracker.steps)} steps executed)"
    )

    # Display report
    console.print()
    console.print(Rule("[bold]Step 4: Research Report[/bold]"))
    console.print()

    # Show report as markdown
    try:
        console.print(Markdown(report))
    except Exception:
        console.print(report)

    # Show save location
    if file_path:
        console.print()
        console.print(
            Panel(
                f"[green]Report saved to:[/green] [bold]{file_path}[/bold]",
                border_style="green",
            )
        )


def interactive_mode(router: QueryRouter, require_approval: bool = True) -> None:
    """Run the agent in interactive loop mode."""
    print_welcome()

    while True:
        console.print()
        try:
            user_query = Prompt.ask("[bold blue]Enter your research query[/bold blue]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if user_query.lower() in ("exit", "quit", "q"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        if not user_query.strip():
            console.print("[dim]Please enter a query.[/dim]")
            continue

        try:
            run_research_workflow(router, user_query, require_approval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Research interrupted.[/yellow]")
        except Exception as exc:
            console.print(f"[red]Error during research: {exc}[/red]")
            logger.exception("Research workflow error")

        console.print()
        another = Confirm.ask("Research another topic?", default=True)
        if not another:
            console.print("[yellow]Goodbye![/yellow]")
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Financial Deep Research Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --query "Analyze the Indian IT services sector outlook"
  python main.py --query "Compare Sun Pharma and Cipla financials" --no-approval
  python main.py --query "Biosimilar market trends in India" --debug
        """,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Research query (if not provided, runs in interactive mode)",
    )
    parser.add_argument(
        "--no-approval",
        action="store_true",
        help="Skip the plan approval step and execute immediately",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate configuration
    try:
        config.validate()
    except EnvironmentError as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        console.print(
            "[dim]Copy .env.example to .env and fill in your API keys.[/dim]"
        )
        sys.exit(1)

    # Build router
    router = build_router()
    require_approval = not args.no_approval

    if args.query:
        # Single query mode
        print_welcome()
        try:
            run_research_workflow(router, args.query, require_approval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Research interrupted.[/yellow]")
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
            if args.debug:
                raise
            sys.exit(1)
    else:
        # Interactive mode
        interactive_mode(router, require_approval)


if __name__ == "__main__":
    main()
