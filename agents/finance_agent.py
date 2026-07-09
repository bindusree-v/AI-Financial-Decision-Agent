"""
Finance Sector Research Agent

Covers Banking, NBFCs, Fintech, digital payments, lending, and capital markets.
"""
from __future__ import annotations
from agents.base_agent import BaseSectorAgent


class FinanceAgent(BaseSectorAgent):
    SECTOR_NAME = "Finance"

    SECTOR_KEYWORDS = [
        # Banking
        "bank", "banking", "NBFC", "non-banking financial", "lending", "loan",
        "interest rate", "NPA", "credit", "deposit", "RBI", "monetary policy",
        "HDFC", "ICICI", "SBI", "Kotak", "Axis Bank", "IndusInd", "Bajaj Finance",
        "microfinance", "retail banking", "corporate banking", "net interest margin",
        "CASA ratio", "credit growth", "bad loans", "provisioning",
        # Fintech
        "fintech", "digital payment", "neobank", "blockchain", "crypto",
        "UPI", "payment gateway", "digital wallet", "insurtech", "wealthtech",
        "Paytm", "PhonePe", "Razorpay", "BharatPe", "Zerodha", "Groww",
        "BNPL", "buy now pay later", "open banking", "DeFi",
        "digital lending", "peer to peer lending", "P2P", "robo advisor",
        # Capital markets
        "stock market", "equity", "mutual fund", "IPO", "SEBI", "NSE", "BSE",
        "portfolio", "investment", "wealth management", "insurance",
    ]

    KNOWN_TICKERS: dict[str, str] = {
        # Banks
        "HDFC Bank": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "SBI": "SBIN.NS",
        "State Bank of India": "SBIN.NS",
        "Kotak Mahindra Bank": "KOTAKBANK.NS",
        "Axis Bank": "AXISBANK.NS",
        "IndusInd Bank": "INDUSINDBK.NS",
        "Yes Bank": "YESBANK.NS",
        "Bank of Baroda": "BANKBARODA.NS",
        "Punjab National Bank": "PNB.NS",
        "Federal Bank": "FEDERALBNK.NS",
        "IDFC First Bank": "IDFCFIRSTB.NS",
        # NBFCs
        "Bajaj Finance": "BAJFINANCE.NS",
        "Bajaj Finserv": "BAJAJFINSV.NS",
        "Shriram Finance": "SHRIRAMFIN.NS",
        "Muthoot Finance": "MUTHOOTFIN.NS",
        "Cholamandalam": "CHOLAFIN.NS",
        "LIC Housing Finance": "LICHSGFIN.NS",
        # Fintech (listed)
        "Paytm": "PAYTM.NS",
        "One97 Communications": "PAYTM.NS",
        "PB Fintech": "POLICYBZR.NS",
        "PolicyBazaar": "POLICYBZR.NS",
        "Angel One": "ANGELONE.NS",
        # Global
        "PayPal": "PYPL",
        "Visa": "V",
        "Mastercard": "MA",
    }

    SEARCH_DOMAINS = [
        "rbi.org.in",
        "economictimes.indiatimes.com",
        "livemint.com",
        "businessstandard.com",
        "moneycontrol.com",
        "inc42.com",
        "reuters.com",
    ]

    RESEARCH_FOCUS_AREAS = [
        "Net interest margin (NIM) and credit growth trends",
        "NPA levels, provisioning, and asset quality",
        "CASA ratio and deposit mobilisation",
        "RBI monetary policy and rate cycle impact",
        "UPI transaction volumes and digital payment market share",
        "Digital lending growth and credit risk",
        "Fintech funding, valuations, and regulatory landscape",
        "BNPL and buy-now-pay-later market dynamics",
        "Capital adequacy and tier-1 ratios",
        "Insurance penetration and wealth management growth",
    ]
