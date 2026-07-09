"""
Financial Analyzer — processes structured financial data and RAG documents.

Provides:
- Document intelligence (RAG context analysis)
- Structured financial data analysis
- Cross-company comparison
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.llm import get_llm

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

DOCUMENT_ANALYST_SYSTEM = """You are a Financial Document Analyst.
You are given extracted text from financial documents such as:
- Annual reports
- Earnings calls
- Investor presentations

Your task:
- Extract key financial insights
- Identify:
  - Revenue trends
  - Profitability
  - Strategic initiatives
  - Risks
- Summarize in structured format with clear headings
- Be specific with numbers when available
- Do NOT fabricate data"""

FINANCIAL_DATA_ANALYST_SYSTEM = """You are a Financial Analyst.
Using the provided structured data, perform financial analysis.

Instructions:
- Analyze: Revenue growth, Profit margins, EBITDA, Financial stability
- Compare if multiple companies are present
- Identify strengths and weaknesses
- Highlight trends and anomalies
- Provide investment-relevant insights

IMPORTANT:
- Do NOT guess or fabricate numbers
- Use ONLY the provided data
- Clearly state when data is unavailable"""

SECTOR_COMPARISON_SYSTEM = """You are a Senior Financial Analyst specializing in sector comparisons.
Given financial metrics for multiple companies, provide:
1. A comparative analysis table (in markdown)
2. Key differentiators between companies
3. Relative strengths and weaknesses
4. Which companies appear financially stronger and why
5. Sector-level insights from the comparison

Be analytical, not just descriptive. Provide actionable insights."""


class FinancialAnalyzer:
    """
    Analyzes financial documents and structured data using LLM reasoning.
    """

    def __init__(self) -> None:
        self.llm = get_llm(temperature=0.1)

    def analyze_documents(
        self, retrieved_documents: str, user_query: str
    ) -> str:
        """
        Analyze RAG-retrieved financial documents for relevant insights.

        Args:
            retrieved_documents: Text from RAG retrieval.
            user_query: Original research question.

        Returns:
            Structured analysis as text.
        """
        if not retrieved_documents or retrieved_documents.startswith("No relevant"):
            return "No relevant documents found in the knowledge base for this query."

        messages = [
            SystemMessage(content=DOCUMENT_ANALYST_SYSTEM),
            HumanMessage(
                content=(
                    f"Context (Financial Documents):\n{retrieved_documents}\n\n"
                    f"Question: {user_query}"
                )
            ),
        ]
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as exc:
            logger.error("Document analysis failed: %s", exc)
            return f"Document analysis unavailable: {exc}"

    def analyze_financial_data(
        self, financial_data: dict[str, str], user_query: str
    ) -> str:
        """
        Analyze structured financial metrics for identified companies.

        Args:
            financial_data: Dict of {ticker: formatted_metrics_text}.
            user_query: Original research question.

        Returns:
            Financial analysis as text.
        """
        if not financial_data:
            return "No structured financial data available for analysis."

        data_text = "\n\n".join(financial_data.values())

        messages = [
            SystemMessage(content=FINANCIAL_DATA_ANALYST_SYSTEM),
            HumanMessage(
                content=(
                    f"Financial Data:\n{data_text}\n\n"
                    f"Research Question: {user_query}"
                )
            ),
        ]
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as exc:
            logger.error("Financial data analysis failed: %s", exc)
            return f"Financial analysis unavailable: {exc}"

    def compare_companies(
        self,
        financial_data: dict[str, str],
        companies: list[str],
        user_query: str,
    ) -> str:
        """
        Generate a comparative analysis across multiple companies.

        Args:
            financial_data: Dict of {ticker: formatted_metrics_text}.
            companies: List of company names for context.
            user_query: Original research question.

        Returns:
            Comparative analysis as markdown text.
        """
        if not financial_data or len(financial_data) < 2:
            return "Insufficient data for comparison (need at least 2 companies)."

        data_text = "\n\n".join(financial_data.values())
        company_list = ", ".join(companies) if companies else "the identified companies"

        messages = [
            SystemMessage(content=SECTOR_COMPARISON_SYSTEM),
            HumanMessage(
                content=(
                    f"Companies to Compare: {company_list}\n\n"
                    f"Financial Data:\n{data_text}\n\n"
                    f"Research Context: {user_query}"
                )
            ),
        ]
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as exc:
            logger.error("Company comparison failed: %s", exc)
            return f"Comparison analysis unavailable: {exc}"

    def extract_key_metrics_summary(
        self, financial_data: dict[str, str]
    ) -> dict[str, Any]:
        """
        Extract a simplified metrics summary for report embedding.
        Returns a dict of {ticker: {metric: value}} for easy table rendering.
        """
        # This is a lightweight pass — the full analysis is done by analyze_financial_data
        summary: dict[str, Any] = {}
        for ticker, data_text in financial_data.items():
            summary[ticker] = {"raw": data_text}
        return summary
