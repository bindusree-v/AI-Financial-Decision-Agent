"""Automotive Sector Research Agent"""
from __future__ import annotations
from agents.base_agent import BaseSectorAgent

class AutomotiveAgent(BaseSectorAgent):
    SECTOR_NAME = "Automotive"
    SECTOR_KEYWORDS = [
        "automobile", "automotive", "car", "vehicle", "EV", "electric vehicle",
        "two wheeler", "commercial vehicle", "truck", "bus", "tractor",
        "Maruti", "Tata Motors", "Mahindra", "Hyundai", "Hero MotoCorp",
        "Bajaj Auto", "TVS Motor", "Eicher Motors", "Ashok Leyland",
        "auto sector", "auto sales", "OEM", "auto ancillary", "EV adoption",
        "battery", "charging infrastructure", "PLI scheme automobile",
    ]
    KNOWN_TICKERS: dict[str, str] = {
        "Maruti Suzuki": "MARUTI.NS",
        "Tata Motors": "TATAMOTORS.NS",
        "Mahindra & Mahindra": "M&M.NS",
        "Mahindra": "M&M.NS",
        "Hero MotoCorp": "HEROMOTOCO.NS",
        "Bajaj Auto": "BAJAJ-AUTO.NS",
        "TVS Motor": "TVSMOTOR.NS",
        "Eicher Motors": "EICHERMOT.NS",
        "Ashok Leyland": "ASHOKLEY.NS",
        "Hyundai India": "HYUNDAI.NS",
        "Toyota": "TM",
        "Tesla": "TSLA",
        "Ford": "F",
        "General Motors": "GM",
    }
    SEARCH_DOMAINS = [
        "economictimes.indiatimes.com", "autocarindia.com", "livemint.com",
        "businessstandard.com", "reuters.com", "siam.in",
    ]
    RESEARCH_FOCUS_AREAS = [
        "Monthly auto sales volumes (SIAM data)",
        "EV adoption rate and model launches",
        "Raw material cost (steel, aluminium) impact",
        "Export performance and global demand",
        "PLI scheme benefits and capex plans",
        "Two-wheeler rural demand trends",
        "Commercial vehicle freight cycle",
        "Battery technology and supply chain",
    ]
