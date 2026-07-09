"""
Query Router — intelligently routes financial queries to the correct sector agent.

Routing logic:
  - Keyword matching (fast path)
  - LLM-based classification (fallback)
  - Cross-sector detection
  - Non-financial query rejection
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from config import config
from core.llm import get_llm

if TYPE_CHECKING:
    from agents.base_agent import BaseSectorAgent

logger = logging.getLogger(__name__)

ROUTER_SYSTEM = """You are a Financial Query Router.

Your job:
- Identify which sector the query belongs to
- Determine if the query is financial in nature

Supported sectors:
- IT: software, IT services, cloud, AI, semiconductors, tech companies
- Pharma: drugs, vaccines, biotech, R&D, clinical trials, pharmaceutical companies
- Finance: banking, NBFCs, lending, fintech, digital payments, UPI, insurance, capital markets
- Ecommerce: online retail, marketplaces, quick commerce, D2C, GMV, logistics
- Automotive: automobiles, EVs, electric vehicles, two-wheelers, commercial vehicles
- Healthcare: hospitals, diagnostics, health services, health insurance, telemedicine

Rules:
- Route to the best matching sector from the list above
- If clearly non-financial (recipes, sports, entertainment, personal advice) → return "NonFinancial"
- If financial but truly outside all supported sectors → return "Unknown"

Return ONLY valid JSON:
{
  "sector": "IT" | "Pharma" | "Finance" | "Ecommerce" | "Automotive" | "Healthcare" | "Unknown" | "NonFinancial",
  "confidence": "high" | "medium" | "low",
  "reason": "brief explanation"
}"""

# Non-financial rejection message
NON_FINANCIAL_MESSAGE = (
    "I'm a specialized financial research agent. "
    "I can't help with that query, but I'd be happy to research topics like:\n"
    "  • IT / Technology sector (TCS, Infosys, cloud, AI)\n"
    "  • Pharma sector (drug pipelines, biosimilars, FDA approvals)\n"
    "  • Finance (banking, NBFCs, fintech, UPI, digital payments)\n"
    "  • E-Commerce (Zomato, Nykaa, quick commerce, GMV trends)\n"
    "  • Automotive (EVs, auto sales, OEMs, Tata Motors, Maruti)\n"
    "  • Healthcare (hospitals, diagnostics, health insurance)\n\n"
    "Please ask a finance-related question within these domains."
)

UNKNOWN_SECTOR_MESSAGE = (
    "Your query appears to be financial in nature, but it falls outside my supported sectors.\n"
    "I cover: IT, Pharma, Finance (Banking & Fintech), E-Commerce, Automotive, and Healthcare.\n\n"
    "Could you rephrase your query to relate to one of these sectors?"
)


class QueryRouter:
    """
    Routes user queries to the appropriate sector agent.
    """

    def __init__(self) -> None:
        self.llm = get_llm(temperature=0.0)
        self._agents: dict[str, "BaseSectorAgent"] = {}

    def register_agent(self, sector: str, agent: "BaseSectorAgent") -> None:
        """Register a sector agent."""
        self._agents[sector.upper()] = agent
        logger.info("Registered agent for sector: %s", sector)

    def route(self, user_query: str) -> tuple[str, "BaseSectorAgent | None", str]:
        """
        Route a query to the appropriate agent.

        Returns:
            Tuple of (sector, agent_or_None, message).
            - If routed successfully: (sector, agent, "")
            - If non-financial: ("NonFinancial", None, rejection_message)
            - If unknown sector: ("Unknown", None, clarification_message)
        """
        # Fast path: keyword matching
        sector = self._keyword_route(user_query)

        # Fallback: LLM routing
        if sector is None:
            sector = self._llm_route(user_query)

        if sector == "NonFinancial":
            return "NonFinancial", None, NON_FINANCIAL_MESSAGE

        if sector == "Unknown" or sector not in self._agents:
            return "Unknown", None, UNKNOWN_SECTOR_MESSAGE

        agent = self._agents[sector]
        logger.info("Query routed to %s agent.", sector)
        return sector, agent, ""

    def _keyword_route(self, query: str) -> str | None:
        """Fast keyword-based routing."""
        query_lower = query.lower()

        # Check for clearly non-financial queries
        non_financial_keywords = [
            "recipe", "cook", "food", "sport", "football", "cricket",
            "movie", "film", "music", "song", "weather", "travel",
            "hotel", "restaurant", "fashion", "celebrity",
        ]
        if any(kw in query_lower for kw in non_financial_keywords):
            return "NonFinancial"

        # Check registered agents' keywords
        for sector, agent in self._agents.items():
            keywords = [kw.lower() for kw in agent.SECTOR_KEYWORDS]
            if any(kw in query_lower for kw in keywords):
                return sector

        return None  # Needs LLM routing

    def _llm_route(self, query: str) -> str:
        """LLM-based routing for ambiguous queries."""
        messages = [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Query: {query}"),
        ]
        try:
            response = self.llm.invoke(messages)
            result = self._parse_json(response.content)
            sector = result.get("sector", "Unknown")
            logger.info(
                "LLM routed '%s' to '%s' (confidence: %s)",
                query[:50],
                sector,
                result.get("confidence", "?"),
            )
            return sector
        except Exception as exc:
            logger.error("LLM routing failed: %s", exc)
            return "Unknown"

    @staticmethod
    def _parse_json(text: str) -> dict:
        cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {"sector": "Unknown", "confidence": "low"}
