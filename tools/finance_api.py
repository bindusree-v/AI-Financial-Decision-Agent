"""
Financial data tool.
Uses yfinance for real-time/historical stock data and Alpha Vantage for
fundamentals. All calculations are done programmatically (no LLM math).
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from config import config

logger = logging.getLogger(__name__)


class FinanceAPITool:
    """
    Fetches and calculates financial metrics programmatically.
    Supports stock prices, key ratios, income statement, balance sheet,
    and cash flow data.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def get_stock_info(self, ticker: str) -> dict[str, Any]:
        """Return key info dict for a ticker symbol."""
        try:
            t = yf.Ticker(ticker)
            info = t.info
            return self._extract_key_info(info, ticker)
        except Exception as exc:
            logger.error("get_stock_info failed for %s: %s", ticker, exc)
            return {"error": str(exc), "ticker": ticker}

    def get_financial_metrics(self, ticker: str) -> dict[str, Any]:
        """
        Return computed financial metrics from income statement,
        balance sheet, and cash flow — all calculated programmatically.
        """
        try:
            t = yf.Ticker(ticker)
            income_stmt = t.financials          # annual income statement
            balance_sheet = t.balance_sheet
            cash_flow = t.cashflow

            metrics: dict[str, Any] = {"ticker": ticker}

            # ── Revenue & Growth ──────────────────────────────────────────
            if income_stmt is not None and not income_stmt.empty:
                if "Total Revenue" in income_stmt.index:
                    rev = income_stmt.loc["Total Revenue"]
                    metrics["revenue_latest"] = self._safe_float(rev.iloc[0])
                    metrics["revenue_prior"] = self._safe_float(rev.iloc[1]) if len(rev) > 1 else None
                    metrics["revenue_growth_yoy"] = self._pct_change(
                        metrics["revenue_latest"], metrics["revenue_prior"]
                    )

                if "Net Income" in income_stmt.index:
                    ni = income_stmt.loc["Net Income"]
                    metrics["net_income_latest"] = self._safe_float(ni.iloc[0])
                    metrics["net_income_prior"] = self._safe_float(ni.iloc[1]) if len(ni) > 1 else None
                    metrics["net_income_growth_yoy"] = self._pct_change(
                        metrics["net_income_latest"], metrics["net_income_prior"]
                    )

                # Net profit margin
                if metrics.get("revenue_latest") and metrics.get("net_income_latest"):
                    metrics["net_profit_margin_pct"] = round(
                        (metrics["net_income_latest"] / metrics["revenue_latest"]) * 100, 2
                    )

                # EBITDA
                ebitda = self._get_ebitda(income_stmt)
                if ebitda is not None:
                    metrics["ebitda"] = ebitda
                    if metrics.get("revenue_latest"):
                        metrics["ebitda_margin_pct"] = round(
                            (ebitda / metrics["revenue_latest"]) * 100, 2
                        )

            # ── Balance Sheet ─────────────────────────────────────────────
            if balance_sheet is not None and not balance_sheet.empty:
                if "Total Assets" in balance_sheet.index:
                    metrics["total_assets"] = self._safe_float(
                        balance_sheet.loc["Total Assets"].iloc[0]
                    )
                if "Total Debt" in balance_sheet.index:
                    metrics["total_debt"] = self._safe_float(
                        balance_sheet.loc["Total Debt"].iloc[0]
                    )
                if "Stockholders Equity" in balance_sheet.index:
                    equity = self._safe_float(
                        balance_sheet.loc["Stockholders Equity"].iloc[0]
                    )
                    metrics["stockholders_equity"] = equity
                    if metrics.get("total_debt") and equity and equity != 0:
                        metrics["debt_to_equity"] = round(
                            metrics["total_debt"] / equity, 2
                        )

            # ── Cash Flow ─────────────────────────────────────────────────
            if cash_flow is not None and not cash_flow.empty:
                if "Free Cash Flow" in cash_flow.index:
                    metrics["free_cash_flow"] = self._safe_float(
                        cash_flow.loc["Free Cash Flow"].iloc[0]
                    )
                elif "Operating Cash Flow" in cash_flow.index:
                    metrics["operating_cash_flow"] = self._safe_float(
                        cash_flow.loc["Operating Cash Flow"].iloc[0]
                    )

            return metrics

        except Exception as exc:
            logger.error("get_financial_metrics failed for %s: %s", ticker, exc)
            return {"error": str(exc), "ticker": ticker}

    def get_price_history(
        self, ticker: str, period: str = "1y"
    ) -> dict[str, Any]:
        """Return price history summary for a ticker."""
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period)
            if hist.empty:
                return {"error": "No price data", "ticker": ticker}

            latest_price = float(hist["Close"].iloc[-1])
            start_price = float(hist["Close"].iloc[0])
            high_52w = float(hist["High"].max())
            low_52w = float(hist["Low"].min())
            price_return = self._pct_change(latest_price, start_price)

            return {
                "ticker": ticker,
                "latest_price": round(latest_price, 2),
                "period_start_price": round(start_price, 2),
                "price_return_pct": price_return,
                "52w_high": round(high_52w, 2),
                "52w_low": round(low_52w, 2),
                "avg_volume": int(hist["Volume"].mean()),
            }
        except Exception as exc:
            logger.error("get_price_history failed for %s: %s", ticker, exc)
            return {"error": str(exc), "ticker": ticker}

    def compare_companies(self, tickers: list[str]) -> pd.DataFrame:
        """
        Return a DataFrame comparing key metrics across multiple tickers.
        Useful for sector comparison tables.
        """
        rows = []
        for ticker in tickers:
            info = self.get_stock_info(ticker)
            metrics = self.get_financial_metrics(ticker)
            row = {
                "Ticker": ticker,
                "Company": info.get("company_name", ticker),
                "Market Cap (B)": self._to_billions(info.get("market_cap")),
                "Revenue (B)": self._to_billions(metrics.get("revenue_latest")),
                "Revenue Growth YoY (%)": metrics.get("revenue_growth_yoy"),
                "Net Margin (%)": metrics.get("net_profit_margin_pct"),
                "EBITDA Margin (%)": metrics.get("ebitda_margin_pct"),
                "D/E Ratio": metrics.get("debt_to_equity"),
                "P/E Ratio": info.get("pe_ratio"),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def format_metrics_as_text(self, metrics: dict[str, Any]) -> str:
        """Convert metrics dict to a readable text block."""
        if "error" in metrics:
            return f"Data unavailable for {metrics.get('ticker', 'unknown')}: {metrics['error']}"

        lines = [f"Financial Metrics — {metrics.get('ticker', '')}"]
        field_labels = {
            "revenue_latest": "Revenue (Latest)",
            "revenue_growth_yoy": "Revenue Growth YoY (%)",
            "net_income_latest": "Net Income (Latest)",
            "net_income_growth_yoy": "Net Income Growth YoY (%)",
            "net_profit_margin_pct": "Net Profit Margin (%)",
            "ebitda": "EBITDA",
            "ebitda_margin_pct": "EBITDA Margin (%)",
            "total_assets": "Total Assets",
            "total_debt": "Total Debt",
            "stockholders_equity": "Stockholders Equity",
            "debt_to_equity": "Debt-to-Equity Ratio",
            "free_cash_flow": "Free Cash Flow",
            "operating_cash_flow": "Operating Cash Flow",
        }
        for key, label in field_labels.items():
            val = metrics.get(key)
            if val is not None:
                if key in ("revenue_latest", "net_income_latest", "ebitda",
                           "total_assets", "total_debt", "stockholders_equity",
                           "free_cash_flow", "operating_cash_flow"):
                    lines.append(f"  {label}: {self._format_large_number(val)}")
                else:
                    lines.append(f"  {label}: {val}")
        return "\n".join(lines)

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _extract_key_info(self, info: dict, ticker: str) -> dict[str, Any]:
        return {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "description": info.get("longBusinessSummary", "")[:500],
        }

    def _get_ebitda(self, income_stmt: pd.DataFrame) -> float | None:
        """Compute EBITDA = Operating Income + D&A if available."""
        try:
            ebitda = None
            if "EBITDA" in income_stmt.index:
                ebitda = self._safe_float(income_stmt.loc["EBITDA"].iloc[0])
            elif "Operating Income" in income_stmt.index:
                op_income = self._safe_float(income_stmt.loc["Operating Income"].iloc[0])
                da = 0.0
                if "Depreciation And Amortization" in income_stmt.index:
                    da = self._safe_float(
                        income_stmt.loc["Depreciation And Amortization"].iloc[0]
                    ) or 0.0
                if op_income is not None:
                    ebitda = op_income + da
            return ebitda
        except Exception:
            return None

    @staticmethod
    def _safe_float(val: Any) -> float | None:
        try:
            f = float(val)
            return None if pd.isna(f) else f
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _pct_change(current: float | None, prior: float | None) -> float | None:
        if current is None or prior is None or prior == 0:
            return None
        return round(((current - prior) / abs(prior)) * 100, 2)

    @staticmethod
    def _to_billions(val: float | None) -> float | None:
        if val is None:
            return None
        return round(val / 1e9, 2)

    @staticmethod
    def _format_large_number(val: float) -> str:
        if abs(val) >= 1e9:
            return f"${val / 1e9:.2f}B"
        if abs(val) >= 1e6:
            return f"${val / 1e6:.2f}M"
        return f"${val:,.0f}"


# Module-level singleton
_finance_tool: FinanceAPITool | None = None


def get_finance_tool() -> FinanceAPITool:
    global _finance_tool
    if _finance_tool is None:
        _finance_tool = FinanceAPITool()
    return _finance_tool
