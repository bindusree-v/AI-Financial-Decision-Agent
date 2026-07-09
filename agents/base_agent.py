"""
Base Sector Agent — shared logic for all sector-specific agents.

Each sector agent inherits from this and provides:
  - sector name
  - sector-specific search domains
  - sector-specific ticker mappings
  - sector-specific research focus areas
"""
from __future__ import annotations

import logging
from typing import Any

from core.analyzer import FinancialAnalyzer
from core.executor import ResearchExecutor, ResearchState
from core.planner import ResearchPlanner
from output.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class BaseSectorAgent:
    """
    Base class for all sector-specific research agents.

    Subclasses must define:
        SECTOR_NAME: str
        SECTOR_KEYWORDS: list[str]
        KNOWN_TICKERS: dict[str, str]  # company_name -> ticker
        SEARCH_DOMAINS: list[str]      # preferred domains for web search
    """

    SECTOR_NAME: str = "General"
    SECTOR_KEYWORDS: list[str] = []
    KNOWN_TICKERS: dict[str, str] = {}
    SEARCH_DOMAINS: list[str] = []

    def __init__(self) -> None:
        self.planner = ResearchPlanner()
        self.executor = ResearchExecutor(sector=self.SECTOR_NAME)
        self.analyzer = FinancialAnalyzer()
        self.report_generator = ReportGenerator()
        logger.info("%s Agent initialized.", self.SECTOR_NAME)

    def run(
        self,
        user_query: str,
        approved_plan: dict[str, Any] | None = None,
        progress_callback=None,
        save_report: bool = True,
    ) -> tuple[str, str]:
        """
        Execute the full research workflow.

        Args:
            user_query: The user's research question.
            approved_plan: Pre-approved plan (skip planning step if provided).
            progress_callback: Optional callable(step, total, message) for progress updates.
            save_report: Whether to save the report to disk.

        Returns:
            Tuple of (report_text, file_path_or_empty_string).
        """
        # Step 1: Analyze query
        logger.info("[%s] Analyzing query: %s", self.SECTOR_NAME, user_query)
        analysis = self.planner.analyze_query(user_query)

        # Enrich analysis with sector-specific knowledge
        analysis = self._enrich_analysis(analysis, user_query)

        # Step 2: Create research plan (if not pre-approved)
        if approved_plan is None:
            plan = self.planner.create_research_plan(user_query, analysis)
        else:
            plan = approved_plan

        # Step 3: Execute deep research
        logger.info("[%s] Starting research execution.", self.SECTOR_NAME)
        state: ResearchState = self.executor.execute(
            user_query=user_query,
            analysis=analysis,
            plan=plan,
            progress_callback=progress_callback,
        )

        # Step 4: Generate report
        logger.info("[%s] Generating report.", self.SECTOR_NAME)
        report = self.report_generator.generate(state)

        # Step 5: Save report
        file_path = ""
        if save_report:
            file_path = self.report_generator.save_report(report, state)
            logger.info("[%s] Report saved to: %s", self.SECTOR_NAME, file_path)

        return report, file_path

    def get_plan(self, user_query: str) -> tuple[dict[str, Any], dict[str, Any], str]:
        """
        Generate and return the research plan without executing it.
        Used for the user approval step.

        Returns:
            Tuple of (analysis, plan, formatted_plan_text).
        """
        analysis = self.planner.analyze_query(user_query)
        analysis = self._enrich_analysis(analysis, user_query)
        plan = self.planner.create_research_plan(user_query, analysis)
        formatted = self.planner.format_plan_for_display(plan, analysis)
        return analysis, plan, formatted

    def _enrich_analysis(
        self, analysis: dict[str, Any], user_query: str
    ) -> dict[str, Any]:
        """
        Enrich LLM analysis with sector-specific knowledge.
        - Injects sector name
        - Adds known tickers for mentioned companies (with partial matching)
        - Injects research focus areas for better planning
        - Injects search domains for targeted web search
        """
        # Override sector if not detected
        if analysis.get("sector") == "Unknown":
            analysis["sector"] = self.SECTOR_NAME

        # Enrich tickers from known mappings (partial + exact matching)
        entities = analysis.setdefault("entities", {})
        companies = entities.get("companies", [])
        tickers = entities.get("tickers", [])

        query_upper = user_query.upper()
        for company, ticker in self.KNOWN_TICKERS.items():
            if ticker is None:
                continue  # skip private companies with no ticker
            # Match full name or any word of the company name
            company_upper = company.upper()
            words = [w for w in company_upper.split() if len(w) > 2]
            if company_upper in query_upper or any(w in query_upper for w in words):
                if ticker not in tickers:
                    tickers.append(ticker)
                if company not in companies:
                    companies.append(company)

        entities["companies"] = companies
        entities["tickers"] = tickers

        # Inject sector-specific focus areas for the planner
        if hasattr(self, "RESEARCH_FOCUS_AREAS") and self.RESEARCH_FOCUS_AREAS:
            analysis["focus_areas"] = self.RESEARCH_FOCUS_AREAS

        # Inject search domains for targeted web search
        if hasattr(self, "SEARCH_DOMAINS") and self.SEARCH_DOMAINS:
            analysis["search_domains"] = self.SEARCH_DOMAINS

        return analysis
