"""Sector-specific research agents."""

from agents.it_agent import ITSectorAgent
from agents.pharma_agent import PharmaSectorAgent
from agents.finance_agent import FinanceAgent
from agents.ecommerce_agent import EcommerceAgent
from agents.automotive_agent import AutomotiveAgent
from agents.healthcare_agent import HealthcareAgent

__all__ = [
    "ITSectorAgent", "PharmaSectorAgent", "FinanceAgent",
    "EcommerceAgent", "AutomotiveAgent", "HealthcareAgent",
]
