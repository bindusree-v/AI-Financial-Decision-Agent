"""
Research Planner — Step 1 of the deep research workflow.

Responsibilities:
1. Analyze the user query (sector, intent, entities, depth)
2. Generate a detailed multi-step research plan
3. Present the plan to the user for approval
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.llm import get_llm

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

QUERY_ANALYSIS_SYSTEM = """You are a Financial Query Analyzer.
Your job is to analyze the user's query and extract structured information for a financial deep research system.

----------------------
🎯 TASKS
----------------------

1. Identify the most relevant sector from the list below:

Sectors:
- IT (Information Technology, software, AI, cloud, SaaS)
- Pharma (Pharmaceuticals, biotech, drugs, vaccines, R&D)
- Banking (commercial banks, NBFCs, lending, interest rates)
- Fintech (digital payments, neobanks, blockchain finance)
- Energy (oil, gas, renewable energy, power sector)
- Automotive (automobiles, EVs, manufacturing vehicles)
- FMCG (consumer goods, retail products, household brands)
- Telecom (telecommunications, 5G, mobile networks)
- Infrastructure (construction, real estate, logistics)
- Metals & Mining (steel, aluminum, mining companies)
- E-commerce (online retail, marketplaces)
- Healthcare (hospitals, healthcare services, diagnostics)

If the query does not clearly belong to any sector, return: "sector": "Unknown"

----------------------
2. Identify the intent of the query:
Choose one:
- sector_analysis
- company_analysis
- comparison
- trend_analysis
- regulation_analysis

----------------------
3. Determine research depth:
- low  → simple factual query
- medium → moderate explanation or analysis
- high → requires deep research, multi-step reasoning, or trends

----------------------
4. Extract key entities:
- Company names (e.g., Infosys, TCS)
- Country/Region (e.g., India, US)
- Topics (e.g., AI adoption, EV growth, inflation)
- Identify relevant stock tickers if companies are mentioned (use standard exchange tickers)

----------------------
📤 OUTPUT FORMAT (STRICT JSON ONLY)

{
  "sector": "",
  "intent": "",
  "depth": "",
  "entities": {
    "companies": [],
    "tickers": [],
    "country": "",
    "topics": []
  },
  "is_financial": true,
  "decline_reason": null
}"""

RESEARCH_PLAN_SYSTEM = """You are a Financial Research Planner.
Create a focused, step-by-step research plan.

Instructions:
- Generate EXACTLY {min_steps} to {max_steps} research steps — no more, no less
- Each step must use one of: web_search, financial_data, rag_documents
- Cover: latest news, financial metrics, trends, risks, outlook
- Be specific — each query should be targeted, not generic

Return ONLY valid JSON in this format:
{{
  "research_title": "...",
  "estimated_steps": {min_steps},
  "research_areas": [{{"area": "...", "description": "..."}}],
  "research_steps": [
    {{
      "step": 1,
      "action": "web_search | financial_data | rag_documents",
      "query": "...",
      "purpose": "..."
    }}
  ],
  "expected_output_structure": ["Executive Summary", "Market Overview", "Financial Analysis", "Risks", "Outlook"]
}}"""


class ResearchPlanner:
    """
    Analyzes queries and generates structured research plans.
    """

    def __init__(self) -> None:
        self.llm = get_llm(temperature=0.0)

    def analyze_query(self, user_query: str) -> dict[str, Any]:
        """
        Parse the user query into structured metadata.

        Returns:
            Dict with sector, intent, depth, entities, is_financial, decline_reason.
        """
        messages = [
            SystemMessage(content=QUERY_ANALYSIS_SYSTEM),
            HumanMessage(content=f"User Query: {user_query}"),
        ]
        response = self.llm.invoke(messages)
        return self._parse_json(response.content, fallback={
            "sector": "Unknown",
            "intent": "sector_analysis",
            "depth": "medium",
            "entities": {},
            "is_financial": True,
            "decline_reason": None,
        })

    def create_research_plan(
        self, user_query: str, analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate a research plan sized to match MIN/MAX_RESEARCH_STEPS.
        Uses sector-specific focus areas when available.
        """
        from config import config as _cfg
        min_s = _cfg.MIN_RESEARCH_STEPS
        max_s = _cfg.MAX_RESEARCH_STEPS

        # Build the system prompt with exact step counts
        system_prompt = RESEARCH_PLAN_SYSTEM.format(
            min_steps=min_s, max_steps=max_s
        )

        focus_areas = analysis.get("focus_areas", [])
        focus_block = ""
        if focus_areas:
            focus_block = "\n\nSector-Specific Focus Areas to cover:\n" + \
                "\n".join(f"- {f}" for f in focus_areas)

        context = (
            f"User Query: {user_query}\n\n"
            f"Sector: {analysis.get('sector','Unknown')}\n"
            f"Intent: {analysis.get('intent','sector_analysis')}\n"
            f"Companies: {', '.join(analysis.get('entities',{}).get('companies',[]))}\n"
            f"Tickers: {', '.join(analysis.get('entities',{}).get('tickers',[]))}"
            f"{focus_block}"
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context),
        ]
        response = self.llm.invoke(messages)
        plan = self._parse_json(response.content, fallback={
            "research_title": user_query,
            "estimated_steps": min_s,
            "research_areas": [],
            "research_steps": self._default_steps(user_query, analysis),
            "expected_output_structure": [
                "Executive Summary", "Market Overview", "Key Players",
                "Financial Analysis", "Trends", "Risks", "Outlook",
            ],
        })
        return plan

    def format_plan_for_display(
        self, plan: dict[str, Any], analysis: dict[str, Any]
    ) -> str:
        """
        Render the research plan as a human-readable string for user approval.
        """
        lines = [
            "=" * 70,
            f"  RESEARCH PLAN: {plan.get('research_title', 'Financial Research')}",
            "=" * 70,
            "",
            f"  Sector:  {analysis.get('sector', 'Unknown')}",
            f"  Intent:  {analysis.get('intent', 'analysis')}",
            f"  Depth:   {analysis.get('depth', 'medium')}",
            f"  Est. Steps: {plan.get('estimated_steps', '?')}",
            "",
        ]

        # Research areas
        areas = plan.get("research_areas", [])
        if areas:
            lines.append("  RESEARCH AREAS:")
            for a in areas:
                lines.append(f"    • {a.get('area', '')}: {a.get('description', '')}")
            lines.append("")

        # Step-by-step plan
        steps = plan.get("research_steps", [])
        if steps:
            lines.append("  RESEARCH STEPS:")
            for s in steps:
                lines.append(
                    f"    Step {s.get('step', '?')} [{s.get('action', 'web_search')}]"
                )
                lines.append(f"      Query:   {s.get('query', '')}")
                lines.append(f"      Purpose: {s.get('purpose', '')}")
            lines.append("")

        # Expected output
        output_sections = plan.get("expected_output_structure", [])
        if output_sections:
            lines.append("  EXPECTED REPORT SECTIONS:")
            for sec in output_sections:
                lines.append(f"    • {sec}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str, fallback: dict) -> dict:
        """Extract and parse JSON from LLM response."""
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        logger.warning("Could not parse JSON from LLM response; using fallback.")
        return fallback

    @staticmethod
    def _default_steps(query: str, analysis: dict | None = None) -> list[dict]:
        sector = (analysis or {}).get("sector", "")
        focus = (analysis or {}).get("focus_areas", [])
        focus_query = focus[0] if focus else "financial performance"
        return [
            {"step": 1, "action": "web_search",
             "query": f"{query} {sector} overview 2025",
             "purpose": "Get broad market landscape"},
            {"step": 2, "action": "web_search",
             "query": f"{query} key companies latest results",
             "purpose": "Identify major players and recent performance"},
            {"step": 3, "action": "financial_data",
             "query": "financial metrics",
             "purpose": "Quantitative financial analysis"},
            {"step": 4, "action": "web_search",
             "query": f"{query} {focus_query}",
             "purpose": "Sector-specific deep dive"},
            {"step": 5, "action": "web_search",
             "query": f"{query} risks challenges outlook",
             "purpose": "Risk assessment and forward view"},
        ]
