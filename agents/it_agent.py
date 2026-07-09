"""
IT Sector Research Agent

Specializes in:
- Indian and global IT services companies
- Cloud computing, AI/ML, digital transformation
- Software products and SaaS
- IT consulting and outsourcing
- Semiconductor and hardware companies
"""
from __future__ import annotations

from agents.base_agent import BaseSectorAgent


class ITSectorAgent(BaseSectorAgent):
    """
    Deep research agent for the Information Technology sector.

    Covers: IT services, cloud, AI/ML, software, semiconductors,
    digital transformation, cybersecurity, and related sub-sectors.
    """

    SECTOR_NAME = "IT"

    SECTOR_KEYWORDS = [
        "IT", "information technology", "software", "cloud", "SaaS",
        "AI", "artificial intelligence", "machine learning", "digital transformation",
        "cybersecurity", "semiconductor", "chip", "data center",
        "Infosys", "TCS", "Wipro", "HCL", "Tech Mahindra",
        "Accenture", "IBM", "Microsoft", "Google", "Amazon",
        "IT services", "outsourcing", "BPO", "ERP", "SAP",
    ]

    # Major IT companies with their primary exchange tickers
    KNOWN_TICKERS: dict[str, str] = {
        # Indian IT (NSE tickers via yfinance use .NS suffix)
        "TCS": "TCS.NS",
        "Tata Consultancy Services": "TCS.NS",
        "Infosys": "INFY",
        "Wipro": "WIT",
        "HCL Technologies": "HCLTECH.NS",
        "HCL": "HCLTECH.NS",
        "Tech Mahindra": "TECHM.NS",
        "Mphasis": "MPHASIS.NS",
        "LTIMindtree": "LTIM.NS",
        "Persistent Systems": "PERSISTENT.NS",
        "Coforge": "COFORGE.NS",
        # Global IT
        "Accenture": "ACN",
        "IBM": "IBM",
        "Cognizant": "CTSH",
        "Capgemini": "CAP.PA",
        "Microsoft": "MSFT",
        "Salesforce": "CRM",
        "Oracle": "ORCL",
    }

    SEARCH_DOMAINS = [
        "economictimes.indiatimes.com",
        "livemint.com",
        "businessstandard.com",
        "nasscom.in",
        "gartner.com",
        "idc.com",
        "techcrunch.com",
        "reuters.com",
    ]

    # Sector-specific research focus areas injected into plans
    RESEARCH_FOCUS_AREAS = [
        "Revenue growth and deal wins",
        "AI and automation adoption",
        "Headcount and attrition trends",
        "BFSI, healthcare, retail vertical performance",
        "US and European market demand",
        "Cloud migration and digital transformation projects",
        "Margin pressure and cost optimization",
        "Emerging technologies: GenAI, edge computing",
    ]
