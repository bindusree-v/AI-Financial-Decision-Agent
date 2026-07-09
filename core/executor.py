"""
Research Executor — the deep research loop engine.

This is the heart of the system. It runs iterative, adaptive research steps:
  1. Execute a search/data action
  2. Extract insights from results
  3. Dynamically decide the next research step based on findings
  4. Repeat until MIN_RESEARCH_STEPS is reached and research is comprehensive

The executor adapts dynamically:
  - Discovers a trend → digs deeper into it
  - Finds a company → fetches its financial data
  - Identifies a regulation → researches its impact
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import config
from core.llm import get_llm
from tools.finance_api import get_finance_tool
from tools.rag import get_rag_tool
from tools.web_search import get_search_tool

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

STEP_ANALYSIS_SYSTEM = """You are a Financial Research Agent. After each research step, extract insights and decide the next action.

Respond in this EXACT format (no extra text):
INSIGHTS:
<3-5 bullet points of key findings>

NEXT_ACTION: web_search | financial_data | rag_documents | COMPLETE
NEXT_QUERY: <specific next query>
NEXT_REASON: <one line reason>"""

COMPLETION_CHECK_SYSTEM = """You are evaluating whether a financial research task is complete.

Given the research conducted so far, determine if the research is comprehensive enough.

Consider:
- Has the main question been addressed?
- Are key companies/players covered?
- Are financial metrics included?
- Are trends and risks analyzed?
- Is there a forward-looking perspective?

Respond with:
COMPLETE: yes | no
REASON: <brief explanation>
MISSING: <what is still needed if not complete>"""


@dataclass
class ResearchStep:
    """Represents a single step in the research loop."""
    step_number: int
    action: str          # web_search | financial_data | rag_documents
    query: str
    purpose: str
    raw_results: str = ""
    insights: str = ""
    next_action: str = ""
    next_query: str = ""
    next_reason: str = ""


@dataclass
class ResearchState:
    """Accumulates all research findings across the loop."""
    user_query: str
    sector: str
    analysis: dict[str, Any]
    plan: dict[str, Any]
    steps: list[ResearchStep] = field(default_factory=list)
    all_insights: list[str] = field(default_factory=list)
    financial_data: dict[str, Any] = field(default_factory=dict)
    rag_context: str = ""
    is_complete: bool = False

    def get_research_summary(self) -> str:
        """Compile all insights into a single text block for report generation."""
        parts = [f"Research Query: {self.user_query}\n"]
        for step in self.steps:
            parts.append(
                f"\n--- Step {step.step_number}: {step.action} ---\n"
                f"Query: {step.query}\n"
                f"Insights:\n{step.insights}"
            )
        if self.financial_data:
            parts.append("\n--- Financial Data ---")
            for ticker, data in self.financial_data.items():
                parts.append(data)
        if self.rag_context:
            parts.append(f"\n--- Document Intelligence ---\n{self.rag_context}")
        return "\n".join(parts)


class ResearchExecutor:
    """
    Executes the deep research loop for a given query and plan.
    """

    def __init__(self, sector: str = "general") -> None:
        self.sector = sector
        self.llm = get_llm(temperature=0.1)
        self.search_tool = get_search_tool()
        self.finance_tool = get_finance_tool()
        self.rag_tool = get_rag_tool(sector)
        self.min_steps = config.MIN_RESEARCH_STEPS
        self.max_steps = config.MAX_RESEARCH_STEPS

    def execute(
        self,
        user_query: str,
        analysis: dict[str, Any],
        plan: dict[str, Any],
        progress_callback=None,
    ) -> ResearchState:
        """
        Run the full deep research loop.

        Args:
            user_query: Original user query.
            analysis: Output from ResearchPlanner.analyze_query().
            plan: Output from ResearchPlanner.create_research_plan().
            progress_callback: Optional callable(step_num, total, message) for UI updates.

        Returns:
            ResearchState with all findings.
        """
        state = ResearchState(
            user_query=user_query,
            sector=self.sector,
            analysis=analysis,
            plan=plan,
        )

        # Store search domains for targeted web search
        self._search_domains = analysis.get("search_domains", None)
        # Store focus areas for sector-aware step analysis
        self._focus_areas = analysis.get("focus_areas", [])

        # Seed the loop with the planned steps
        planned_steps = plan.get("research_steps", [])
        step_queue = [
            {
                "action": s.get("action", "web_search"),
                "query": s.get("query", user_query),
                "purpose": s.get("purpose", ""),
            }
            for s in planned_steps
        ]

        # Ensure we have at least min_steps queued
        if len(step_queue) < self.min_steps:
            step_queue.extend(self._generate_fallback_steps(user_query, analysis))

        step_num = 0
        while step_num < self.max_steps:
            # Get next step from queue or stop
            if step_queue:
                next_step_info = step_queue.pop(0)
            else:
                # Check if we've done enough
                if step_num >= self.min_steps and self._is_research_complete(state):
                    state.is_complete = True
                    break
                # Generate a follow-up step dynamically
                next_step_info = self._generate_next_step(state)
                if not next_step_info:
                    break

            step_num += 1
            action = next_step_info.get("action", "web_search")
            query = next_step_info.get("query", user_query)
            purpose = next_step_info.get("purpose", "")

            if progress_callback:
                progress_callback(step_num, self.max_steps, f"Step {step_num}: {query[:60]}...")

            logger.info("Step %d [%s]: %s", step_num, action, query)

            # Execute the step
            research_step = ResearchStep(
                step_number=step_num,
                action=action,
                query=query,
                purpose=purpose,
            )

            raw_results = self._execute_action(action, query, analysis)
            research_step.raw_results = raw_results

            # Analyze results and decide next step
            insights, next_action, next_query, next_reason = self._analyze_step_results(
                query, raw_results, state
            )
            research_step.insights = insights
            research_step.next_action = next_action
            research_step.next_query = next_query
            research_step.next_reason = next_reason

            state.steps.append(research_step)
            state.all_insights.append(insights)

            # If LLM suggests a follow-up and we haven't hit max, queue it
            if (
                next_action
                and next_action.upper() != "COMPLETE"
                and next_query
                and step_num < self.max_steps - 1
            ):
                step_queue.insert(
                    0,
                    {
                        "action": next_action,
                        "query": next_query,
                        "purpose": next_reason,
                    },
                )

            # After min steps, check completion
            if step_num >= self.min_steps and not step_queue:
                if self._is_research_complete(state):
                    state.is_complete = True
                    break

        # Fetch financial data for identified tickers
        tickers = analysis.get("entities", {}).get("tickers", [])
        if tickers:
            self._fetch_financial_data(tickers, state)

        # Retrieve relevant documents from RAG
        rag_results = self.rag_tool.retrieve(user_query)
        if rag_results:
            state.rag_context = self.rag_tool.format_retrieved_as_text(rag_results)

        logger.info(
            "Research complete: %d steps executed for query '%s'.",
            len(state.steps),
            user_query,
        )
        return state

    # ── Private: Action Execution ─────────────────────────────────────────────

    def _execute_action(
        self, action: str, query: str, analysis: dict[str, Any]
    ) -> str:
        """Dispatch to the appropriate tool."""
        if action == "financial_data":
            return self._execute_financial_data(query, analysis)
        elif action == "rag_documents":
            return self._execute_rag(query)
        else:
            return self._execute_web_search(query)

    def _execute_web_search(self, query: str) -> str:
        # Use sector-specific domains if available from analysis
        domains = getattr(self, '_search_domains', None)
        results = self.search_tool.search(query, search_depth="basic", max_results=3,
                                          include_domains=domains if domains else None)
        return self.search_tool.format_results_as_text(results)

    def _execute_financial_data(self, query: str, analysis: dict[str, Any]) -> str:
        tickers = analysis.get("entities", {}).get("tickers", [])
        if not tickers:
            # Fall back to web search for financial data
            return self._execute_web_search(f"{query} financial results revenue profit")

        parts = []
        for ticker in tickers[:2]:  # Limit to 2 tickers per step
            info = self.finance_tool.get_stock_info(ticker)
            metrics = self.finance_tool.get_financial_metrics(ticker)
            price = self.finance_tool.get_price_history(ticker)

            parts.append(
                f"=== {info.get('company_name', ticker)} ({ticker}) ===\n"
                f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}\n"
                f"Market Cap: {self.finance_tool._to_billions(info.get('market_cap'))}B\n"
                f"{self.finance_tool.format_metrics_as_text(metrics)}\n"
                f"Price (1Y): {price.get('latest_price', 'N/A')} | "
                f"Return: {price.get('price_return_pct', 'N/A')}%"
            )
        return "\n\n".join(parts) if parts else "No financial data available."

    def _execute_rag(self, query: str) -> str:
        results = self.rag_tool.retrieve(query)
        return self.rag_tool.format_retrieved_as_text(results)

    # ── Private: LLM Step Analysis ────────────────────────────────────────────

    def _analyze_step_results(
        self, query: str, raw_results: str, state: ResearchState
    ) -> tuple[str, str, str, str]:
        """
        Use LLM to extract insights and decide the next research step.
        Returns: (insights, next_action, next_query, next_reason)
        """
        prior_context = "\n".join(state.all_insights[-2:]) if state.all_insights else "None yet"

        focus_hint = ""
        if hasattr(self, '_focus_areas') and self._focus_areas:
            focus_hint = f"\nSector Focus Areas: {', '.join(self._focus_areas[:4])}"

        prompt = (
            f"Sector: {self.sector}{focus_hint}\n"
            f"Original Research Query: {state.user_query}\n\n"
            f"Current Step Query: {query}\n\n"
            f"Search Results:\n{raw_results[:1500]}\n\n"
            f"Prior Context:\n{prior_context[:800]}\n\n"
            f"Steps done: {len(state.steps)} / min: {self.min_steps} / max: {self.max_steps}"
        )

        messages = [
            SystemMessage(content=STEP_ANALYSIS_SYSTEM),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            return self._parse_step_response(response.content)
        except Exception as exc:
            logger.error("Step analysis LLM call failed: %s", exc)
            return (
                f"Findings from search: {raw_results[:500]}",
                "web_search",
                f"{state.user_query} additional analysis",
                "Continue research",
            )

    def _parse_step_response(
        self, text: str
    ) -> tuple[str, str, str, str]:
        """Parse the structured LLM step response."""
        insights = ""
        next_action = "web_search"
        next_query = ""
        next_reason = ""

        lines = text.strip().split("\n")
        current_section = None

        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("INSIGHTS:"):
                current_section = "insights"
                rest = line_stripped[len("INSIGHTS:"):].strip()
                if rest:
                    insights += rest + "\n"
            elif line_stripped.startswith("NEW_QUESTIONS:"):
                current_section = "questions"
            elif line_stripped.startswith("NEXT_ACTION:"):
                current_section = None
                val = line_stripped[len("NEXT_ACTION:"):].strip()
                next_action = val.lower().replace(" ", "_") if val else "web_search"
            elif line_stripped.startswith("NEXT_QUERY:"):
                current_section = None
                next_query = line_stripped[len("NEXT_QUERY:"):].strip()
            elif line_stripped.startswith("NEXT_REASON:"):
                current_section = None
                next_reason = line_stripped[len("NEXT_REASON:"):].strip()
            elif current_section == "insights" and line_stripped:
                insights += line_stripped + "\n"

        return insights.strip(), next_action, next_query, next_reason

    def _is_research_complete(self, state: ResearchState) -> bool:
        """Skip LLM call — complete when min steps done and queue empty."""
        return len(state.steps) >= self.min_steps

    def _generate_next_step(
        self, state: ResearchState
    ) -> dict[str, Any] | None:
        """Dynamically generate the next research step when the queue is empty."""
        if not state.all_insights:
            return None
        last_insights = state.all_insights[-1]
        return {
            "action": "web_search",
            "query": f"{state.user_query} {last_insights[:100]} analysis",
            "purpose": "Follow-up on latest findings",
        }

    def _fetch_financial_data(
        self, tickers: list[str], state: ResearchState
    ) -> None:
        """Fetch and store financial data for all identified tickers."""
        for ticker in tickers[:3]:  # Cap at 3 tickers
            try:
                info = self.finance_tool.get_stock_info(ticker)
                price = self.finance_tool.get_price_history(ticker, period="6mo")
                formatted = (
                    f"=== {info.get('company_name', ticker)} ({ticker}) ===\n"
                    f"Market Cap: {self.finance_tool._to_billions(info.get('market_cap'))}B | "
                    f"P/E: {info.get('pe_ratio', 'N/A')}\n"
                    f"Price: {price.get('latest_price', 'N/A')} | "
                    f"6M Return: {price.get('price_return_pct', 'N/A')}% | "
                    f"52W High: {price.get('52w_high', 'N/A')} | "
                    f"52W Low: {price.get('52w_low', 'N/A')}"
                )
                state.financial_data[ticker] = formatted
            except Exception as exc:
                logger.warning("Financial data fetch failed for %s: %s", ticker, exc)

    @staticmethod
    def _generate_fallback_steps(
        query: str, analysis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate generic fallback steps if the plan has too few."""
        sector = analysis.get("sector", "")
        entities = analysis.get("entities", {})
        companies = entities.get("companies", [])
        company_str = " ".join(companies[:2]) if companies else ""

        return [
            {"action": "web_search", "query": f"{query} market overview 2024 2025",
             "purpose": "Broad market landscape"},
            {"action": "web_search", "query": f"{sector} sector trends {company_str}",
             "purpose": "Sector-specific trends"},
            {"action": "financial_data", "query": "financial metrics comparison",
             "purpose": "Quantitative financial analysis"},
            {"action": "web_search", "query": f"{query} regulatory environment",
             "purpose": "Regulatory landscape"},
            {"action": "web_search", "query": f"{query} investment risks challenges",
             "purpose": "Risk assessment"},
            {"action": "web_search", "query": f"{query} future outlook growth projections",
             "purpose": "Forward-looking analysis"},
        ]
