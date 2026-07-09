"""E-Commerce Sector Research Agent"""
from __future__ import annotations
from agents.base_agent import BaseSectorAgent

class EcommerceAgent(BaseSectorAgent):
    SECTOR_NAME = "Ecommerce"
    SECTOR_KEYWORDS = [
        "ecommerce", "e-commerce", "online retail", "marketplace", "online shopping",
        "quick commerce", "q-commerce", "D2C", "direct to consumer",
        "Amazon India", "Flipkart", "Meesho", "Nykaa", "Zomato", "Swiggy",
        "Blinkit", "Zepto", "BigBasket", "JioMart", "Myntra",
        "GMV", "gross merchandise value", "logistics", "last mile delivery",
        "dark store", "hyperlocal", "social commerce", "ONDC",
    ]
    KNOWN_TICKERS: dict[str, str] = {
        "Nykaa": "NYKAA.NS",
        "FSN E-Commerce": "NYKAA.NS",
        "Zomato": "ZOMATO.NS",
        "Delhivery": "DELHIVERY.NS",
        "Indiamart": "INDIAMART.NS",
        "Info Edge": "NAUKRI.NS",
        "Amazon": "AMZN",
        "Alibaba": "BABA",
        "Shopify": "SHOP",
    }
    SEARCH_DOMAINS = [
        "inc42.com", "entrackr.com", "economictimes.indiatimes.com",
        "livemint.com", "techcrunch.com", "reuters.com",
    ]
    RESEARCH_FOCUS_AREAS = [
        "GMV growth and take rate trends",
        "Quick commerce market share battle",
        "Unit economics and path to profitability",
        "Customer acquisition cost and retention",
        "Logistics network and delivery speed",
        "ONDC impact on marketplace dynamics",
        "D2C brand growth on platforms",
        "Festive season sales performance",
    ]
