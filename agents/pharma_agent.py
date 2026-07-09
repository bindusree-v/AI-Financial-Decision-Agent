"""
Pharma Sector Research Agent

Specializes in:
- Indian and global pharmaceutical companies
- Drug discovery and R&D pipelines
- Biosimilars and generics
- Clinical trials and regulatory approvals (FDA, EMA, CDSCO)
- Biotech and life sciences
- Healthcare and medical devices
"""
from __future__ import annotations

from agents.base_agent import BaseSectorAgent


class PharmaSectorAgent(BaseSectorAgent):
    """
    Deep research agent for the Pharmaceutical & Healthcare sector.

    Covers: pharma companies, biotech, drug pipelines, regulatory landscape,
    generics, biosimilars, R&D spending, and clinical trials.
    """

    SECTOR_NAME = "Pharma"

    SECTOR_KEYWORDS = [
        "pharma", "pharmaceutical", "drug", "medicine", "biotech",
        "biotechnology", "vaccine", "clinical trial", "FDA", "EMA", "CDSCO",
        "generic", "biosimilar", "API", "active pharmaceutical ingredient",
        "R&D", "research and development", "oncology", "diabetes",
        "Sun Pharma", "Dr Reddy", "Cipla", "Lupin", "Aurobindo",
        "Divi's", "Biocon", "Pfizer", "Novartis", "Roche",
        "healthcare", "life sciences", "medical device",
    ]

    # Major pharma companies with their primary exchange tickers
    KNOWN_TICKERS: dict[str, str] = {
        # Indian Pharma (NSE)
        "Sun Pharma": "SUNPHARMA.NS",
        "Sun Pharmaceutical": "SUNPHARMA.NS",
        "Dr Reddy's": "DRREDDY.NS",
        "Dr. Reddy's": "DRREDDY.NS",
        "Cipla": "CIPLA.NS",
        "Lupin": "LUPIN.NS",
        "Aurobindo Pharma": "AUROPHARMA.NS",
        "Aurobindo": "AUROPHARMA.NS",
        "Divi's Laboratories": "DIVISLAB.NS",
        "Divi's": "DIVISLAB.NS",
        "Biocon": "BIOCON.NS",
        "Torrent Pharma": "TORNTPHARM.NS",
        "Alkem Laboratories": "ALKEM.NS",
        "Ipca Laboratories": "IPCALAB.NS",
        "Glenmark": "GLENMARK.NS",
        "Mankind Pharma": "MANKIND.NS",
        "Zydus Lifesciences": "ZYDUSLIFE.NS",
        "Abbott India": "ABBOTINDIA.NS",
        # Global Pharma
        "Pfizer": "PFE",
        "Novartis": "NVS",
        "Roche": "RHHBY",
        "AstraZeneca": "AZN",
        "Johnson & Johnson": "JNJ",
        "Abbott": "ABT",
        "Merck": "MRK",
        "Bristol-Myers Squibb": "BMY",
        "Eli Lilly": "LLY",
    }

    SEARCH_DOMAINS = [
        "economictimes.indiatimes.com",
        "pharmabiz.com",
        "fiercepharma.com",
        "fda.gov",
        "clinicaltrials.gov",
        "reuters.com",
        "businessstandard.com",
        "livemint.com",
    ]

    # Sector-specific research focus areas
    RESEARCH_FOCUS_AREAS = [
        "Drug pipeline and FDA/EMA approvals",
        "R&D spending as % of revenue",
        "Biosimilar and generic market opportunities",
        "US generics pricing pressure",
        "API manufacturing and supply chain",
        "Regulatory compliance and warning letters",
        "Emerging therapy areas: oncology, rare diseases",
        "Patent cliffs and exclusivity expirations",
        "India domestic market vs export revenue split",
    ]
