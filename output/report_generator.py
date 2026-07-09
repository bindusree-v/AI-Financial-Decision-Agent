"""
Report Generator — synthesizes all research findings into a comprehensive report.

Adapts report structure based on query intent:
  - sector_analysis → Market Overview, Key Players, Trends, Risks, Outlook
  - company_analysis → Company Profile, Financials, Competitive Position, Outlook
  - comparison → Comparison Criteria, Individual Analysis, Comparative Tables
  - trend_analysis → Trend Overview, Drivers, Impact, Future Projections
  - regulation_analysis → Regulatory Overview, Impact Assessment, Compliance
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import config
from core.analyzer import FinancialAnalyzer
from core.executor import ResearchState
from core.llm import get_llm

logger = logging.getLogger(__name__)

# ── Report Prompts ────────────────────────────────────────────────────────────

REPORT_SYSTEM = """You are a Senior Financial Research Analyst writing a comprehensive research report.

Your task is to synthesize all research findings into a well-structured, professional report.

Instructions:
- Be detailed and analytical — provide insights, not just data summaries
- Maintain a professional, objective tone
- Use clear markdown headings and sections
- Include specific data points and metrics where available
- Highlight key insights and actionable takeaways
- Synthesize across all research steps — don't just list findings sequentially
- Ensure logical flow between sections

IMPORTANT:
- Do NOT fabricate data or statistics
- Clearly attribute findings to sources when possible
- If data is unavailable, state it clearly rather than guessing
- Financial figures should be clearly labeled with units"""

SECTOR_ANALYSIS_STRUCTURE = """
Report Structure for Sector Analysis:
1. Executive Summary (3-5 key takeaways)
2. Market Overview (size, growth, key dynamics)
3. Key Players & Market Share
4. Financial Performance Analysis
5. Technology & Innovation Trends
6. Regulatory Environment
7. Competitive Landscape
8. Investment Opportunities
9. Risks & Challenges
10. Future Outlook & Projections
11. Conclusion
"""

COMPANY_ANALYSIS_STRUCTURE = """
Report Structure for Company Analysis:
1. Executive Summary
2. Company Overview & Business Model
3. Financial Performance (Revenue, Margins, Cash Flow)
4. Balance Sheet & Financial Health
5. Competitive Positioning
6. Strategic Initiatives & Growth Drivers
7. Management & Governance
8. Risks & Challenges
9. Valuation & Investment Thesis
10. Future Outlook
11. Conclusion
"""

COMPARISON_STRUCTURE = """
Report Structure for Comparative Analysis:
1. Executive Summary
2. Comparison Methodology & Criteria
3. Individual Company Profiles
4. Financial Metrics Comparison (with table)
5. Operational Comparison
6. Strategic Positioning
7. Strengths & Weaknesses Matrix
8. Investment Ranking & Rationale
9. Risks
10. Conclusion & Recommendations
"""

TREND_ANALYSIS_STRUCTURE = """
Report Structure for Trend Analysis:
1. Executive Summary
2. Trend Overview & Context
3. Key Drivers & Catalysts
4. Market Impact Assessment
5. Company-Level Implications
6. Geographic & Segment Analysis
7. Regulatory & Policy Factors
8. Investment Implications
9. Risks to the Trend
10. Future Projections
11. Conclusion
"""

INTENT_TO_STRUCTURE = {
    "sector_analysis": SECTOR_ANALYSIS_STRUCTURE,
    "company_analysis": COMPANY_ANALYSIS_STRUCTURE,
    "comparison": COMPARISON_STRUCTURE,
    "trend_analysis": TREND_ANALYSIS_STRUCTURE,
    "regulation_analysis": TREND_ANALYSIS_STRUCTURE,  # reuse trend structure
}


class ReportGenerator:
    """
    Generates comprehensive research reports from ResearchState.
    """

    def __init__(self) -> None:
        self.llm = get_llm(temperature=0.2)
        self.analyzer = FinancialAnalyzer()
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    def generate(self, state: ResearchState) -> str:
        """
        Generate a full research report from the research state.

        Returns:
            Complete report as a markdown string.
        """
        intent = state.analysis.get("intent", "sector_analysis")
        structure = INTENT_TO_STRUCTURE.get(intent, SECTOR_ANALYSIS_STRUCTURE)

        # Build the research data package for the LLM
        research_package = self._build_research_package(state)

        # Generate the main report body
        report_body = self._generate_report_body(
            state.user_query, research_package, structure, state.analysis
        )

        # Add financial analysis section if we have structured data
        financial_section = ""
        if state.financial_data:
            companies = state.analysis.get("entities", {}).get("companies", [])
            if len(state.financial_data) >= 2:
                financial_section = self.analyzer.compare_companies(
                    state.financial_data, companies, state.user_query
                )
            else:
                financial_section = self.analyzer.analyze_financial_data(
                    state.financial_data, state.user_query
                )

        # Add document intelligence section if RAG context exists
        doc_section = ""
        if state.rag_context:
            doc_section = self.analyzer.analyze_documents(
                state.rag_context, state.user_query
            )

        # Assemble the final report
        report = self._assemble_report(
            state, report_body, financial_section, doc_section
        )

        return report

    def save_report(self, report: str, state: ResearchState) -> str:
        """
        Save the report to a file and return the file path.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sector = state.sector.lower()
        intent = state.analysis.get("intent", "research")

        # Create a safe filename from the query
        safe_query = "".join(
            c if c.isalnum() or c in (" ", "-", "_") else ""
            for c in state.user_query[:50]
        ).strip().replace(" ", "_")

        filename = f"{sector}_{intent}_{safe_query}_{timestamp}.md"
        filepath = Path(config.OUTPUT_DIR) / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info("Report saved to: %s", filepath)
        return str(filepath)

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_research_package(self, state: ResearchState) -> str:
        """Compile all research findings into a structured text package."""
        parts = []

        # Step-by-step insights
        parts.append("=== RESEARCH FINDINGS (Step-by-Step) ===\n")
        for step in state.steps:
            parts.append(
                f"Step {step.step_number} [{step.action.upper()}]\n"
                f"Query: {step.query}\n"
                f"Key Insights:\n{step.insights}\n"
            )

        # Financial data
        if state.financial_data:
            parts.append("\n=== STRUCTURED FINANCIAL DATA ===\n")
            for ticker, data in state.financial_data.items():
                parts.append(data)

        # RAG context
        if state.rag_context:
            parts.append("\n=== DOCUMENT INTELLIGENCE (Annual Reports / Filings) ===\n")
            parts.append(state.rag_context[:2000])

        return "\n".join(parts)

    def _generate_report_body(
        self,
        user_query: str,
        research_package: str,
        structure: str,
        analysis: dict[str, Any],
    ) -> str:
        """Use LLM to generate the main report body."""
        sector = analysis.get("sector", "Financial")
        entities = analysis.get("entities", {})
        companies = entities.get("companies", [])
        company_context = f"Key Companies: {', '.join(companies)}" if companies else ""

        prompt = (
            f"Research Query: {user_query}\n"
            f"Sector: {sector}\n"
            f"{company_context}\n\n"
            f"Report Structure to Follow:\n{structure}\n\n"
            f"All Research Data:\n{research_package[:8000]}"
        )

        messages = [
            SystemMessage(content=REPORT_SYSTEM),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as exc:
            logger.error("Report generation LLM call failed: %s", exc)
            return f"Report generation failed: {exc}\n\nRaw Research Data:\n{research_package}"

    def _assemble_report(
        self,
        state: ResearchState,
        report_body: str,
        financial_section: str,
        doc_section: str,
    ) -> str:
        """Assemble the final report with metadata header and appendices."""
        timestamp = datetime.now().strftime("%B %d, %Y %H:%M")
        sector = state.sector
        steps_count = len(state.steps)

        header = (
            f"# Financial Deep Research Report\n\n"
            f"**Query:** {state.user_query}\n"
            f"**Sector:** {sector}\n"
            f"**Research Steps Executed:** {steps_count}\n"
            f"**Generated:** {timestamp}\n\n"
            f"---\n\n"
        )

        sections = [header, report_body]

        if financial_section:
            sections.append(
                "\n\n---\n\n## Detailed Financial Analysis\n\n" + financial_section
            )

        if doc_section:
            sections.append(
                "\n\n---\n\n## Document Intelligence (Annual Reports & Filings)\n\n"
                + doc_section
            )

        # Research methodology appendix
        appendix = self._build_appendix(state)
        sections.append(appendix)

        return "\n".join(sections)

    def _build_appendix(self, state: ResearchState) -> str:
        """Build a research methodology appendix."""
        lines = [
            "\n\n---\n\n## Appendix: Research Methodology\n",
            f"**Total Research Steps:** {len(state.steps)}\n",
            "**Research Actions Performed:**\n",
        ]
        for step in state.steps:
            lines.append(
                f"- Step {step.step_number} [{step.action}]: {step.query}"
            )

        tickers = list(state.financial_data.keys())
        if tickers:
            lines.append(f"\n**Financial Data Retrieved For:** {', '.join(tickers)}")

        if state.rag_context:
            lines.append("\n**Document Intelligence:** Annual reports and filings analyzed via RAG")

        lines.append(
            "\n\n*This report was generated by the Financial Deep Research Agent. "
            "All financial data is sourced from public market data. "
            "This is not investment advice.*"
        )
        return "\n".join(lines)
