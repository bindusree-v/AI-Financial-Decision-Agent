"""Healthcare Services Sector Research Agent"""
from __future__ import annotations
from agents.base_agent import BaseSectorAgent

class HealthcareAgent(BaseSectorAgent):
    SECTOR_NAME = "Healthcare"
    SECTOR_KEYWORDS = [
        "healthcare", "hospital", "diagnostics", "health services", "medical",
        "pathology", "radiology", "health insurance", "telemedicine",
        "Apollo Hospitals", "Fortis", "Max Healthcare", "Narayana Health",
        "Dr Lal PathLabs", "Metropolis", "SRL Diagnostics", "Thyrocare",
        "Star Health", "Niva Bupa", "Medanta", "HCG", "Aster DM",
        "ARPOB", "occupancy rate", "bed capacity", "health tech",
        "ayushman bharat", "health insurance penetration",
    ]
    KNOWN_TICKERS: dict[str, str] = {
        "Apollo Hospitals": "APOLLOHOSP.NS",
        "Fortis Healthcare": "FORTIS.NS",
        "Max Healthcare": "MAXHEALTH.NS",
        "Narayana Hrudayalaya": "NH.NS",
        "Dr Lal PathLabs": "LALPATHLAB.NS",
        "Metropolis Healthcare": "METROPOLIS.NS",
        "Thyrocare": "THYROCARE.NS",
        "Star Health Insurance": "STARHEALTH.NS",
        "Aster DM Healthcare": "ASTERDM.NS",
        "HCG": "HCG.NS",
        "Medanta": "MEDANTA.NS",
    }
    SEARCH_DOMAINS = [
        "economictimes.indiatimes.com", "livemint.com", "healthcareittoday.com",
        "businessstandard.com", "reuters.com", "mohfw.gov.in",
    ]
    RESEARCH_FOCUS_AREAS = [
        "ARPOB (Average Revenue Per Occupied Bed) trends",
        "Bed capacity expansion and new hospital openings",
        "Diagnostic test volumes and realization",
        "Health insurance penetration and claims ratio",
        "Telemedicine and digital health adoption",
        "Medical tourism revenue",
        "Government health scheme (Ayushman Bharat) impact",
        "Margin improvement through operational efficiency",
    ]
