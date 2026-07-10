"""
FastAPI Backend for Financial Deep Research Agent
...
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

# Disable chromadb telemetry to suppress noisy startup errors
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY", "false")

# ── Activity log file ─────────────────────────────────────────────────────────
ACTIVITY_FILE = Path(__file__).parent / "activity_log.json"

def _load_activity() -> list[dict]:
    try:
        return json.loads(ACTIVITY_FILE.read_text())
    except Exception:
        return []

def _save_activity(log: list[dict]) -> None:
    ACTIVITY_FILE.write_text(json.dumps(log, indent=2))

def _log_activity(email: str, action: str, sector: str = "", query: str = "") -> None:
    log = _load_activity()
    log.append({
        "email": email,
        "action": action,
        "sector": sector,
        "query": query[:120] if query else "",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    # Keep last 1000 entries
    _save_activity(log[-1000:])

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from agents.it_agent import ITSectorAgent
from agents.pharma_agent import PharmaSectorAgent
from agents.finance_agent import FinanceAgent
from agents.ecommerce_agent import EcommerceAgent
from agents.automotive_agent import AutomotiveAgent
from agents.healthcare_agent import HealthcareAgent
from config import config
from core.router import QueryRouter

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm heavy models on startup so first request is fast."""
    import asyncio
    loop = asyncio.get_running_loop()
    try:
        logger.warning("Pre-warming sentence-transformer embedding model...")
        from chromadb.utils import embedding_functions
        from config import config as _config
        def _warm():
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=_config.EMBEDDING_MODEL
            )
        await loop.run_in_executor(None, _warm)
        logger.warning("Embedding model ready.")
    except Exception as e:
        logger.warning("Model pre-warm failed (non-fatal): %s", e)
    yield

app = FastAPI(
    title="Financial Deep Research Agent API",
    version="1.0.0",
    description="Multi-step iterative financial research powered by LLM synthesis",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── User store (users.json) ───────────────────────────────────────────────────
USERS_FILE = Path(__file__).parent / "users.json"

def _load_users() -> list[dict]:
    try:
        return json.loads(USERS_FILE.read_text())
    except Exception:
        return []

def _save_users(users: list[dict]) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2))

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _find_user(email: str) -> dict | None:
    return next((u for u in _load_users() if u["email"].lower() == email.lower()), None)

# ── In-memory session store: token -> email ───────────────────────────────────
_sessions: dict[str, str] = {}

# ── Admin ─────────────────────────────────────────────────────────────────────
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Admin@2026")
_admin_sessions: set[str] = set()

# ── Validation ────────────────────────────────────────────────────────────────
def _validate_email(email: str) -> bool:
    """Must be a valid Gmail address ending with @gmail.com."""
    return bool(re.match(r'^[a-zA-Z0-9][a-zA-Z0-9._%+\-]*@gmail\.com$', email))

def _validate_password(password: str) -> bool:
    """Starts with capital letter, min 6 chars, contains a special character."""
    if not password or len(password) < 6:
        return False
    if not password[0].isupper():
        return False
    return bool(re.search(r'[^a-zA-Z0-9]', password))

def _get_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return request.cookies.get("fintel_token")

def require_auth(request: Request) -> str:
    token = _get_token(request)
    if not token or token not in _sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return _sessions[token]

# ── Router (initialised once) ─────────────────────────────────────────────────
_router: QueryRouter | None = None

def get_router() -> QueryRouter:
    global _router
    if _router is None:
        _router = QueryRouter()
        _router.register_agent("IT", ITSectorAgent())
        _router.register_agent("Pharma", PharmaSectorAgent())
        _router.register_agent("Finance", FinanceAgent())
        _router.register_agent("Ecommerce", EcommerceAgent())
        _router.register_agent("Automotive", AutomotiveAgent())
        _router.register_agent("Healthcare", HealthcareAgent())
    return _router

# ── Request models ────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

class AdminLoginRequest(BaseModel):
    password: str

class PlanRequest(BaseModel):
    query: str

class ResearchRequest(BaseModel):
    query: str
    plan: dict[str, Any] | None = None
    skip_approval: bool = False

# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/api/signup")
async def signup(req: SignupRequest):
    name = req.name.strip()
    email = req.email.strip()
    password = req.password

    if not name:
        raise HTTPException(status_code=400, detail="Full name is required.")

    if not _validate_email(email):
        raise HTTPException(
            status_code=400,
            detail="Only Gmail addresses are allowed (e.g. yourname@gmail.com)"
        )
    if not _validate_password(password):
        raise HTTPException(
            status_code=400,
            detail="Password must start with a capital letter and contain a special character (e.g. Yash@123)"
        )

    if _find_user(email):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    users = _load_users()
    users.append({
        "name": name,
        "email": email,
        "password_hash": _hash_password(password),
        "joined": datetime.now().strftime("%b %d, %Y"),
    })
    _save_users(users)

    # Auto-login after signup
    token = secrets.token_urlsafe(32)
    _sessions[token] = email
    response = JSONResponse({"token": token, "email": email, "name": name})
    response.set_cookie(key="fintel_token", value=token, httponly=True, samesite="lax", max_age=86400)
    return response


@app.post("/api/login")
async def login(req: LoginRequest):
    email = req.email.strip()
    password = req.password

    if not _validate_email(email):
        raise HTTPException(
            status_code=400,
            detail="Only Gmail addresses are allowed (e.g. yourname@gmail.com)"
        )
    if not _validate_password(password):
        raise HTTPException(
            status_code=400,
            detail="Password must start with a capital letter and contain a special character (e.g. Yash@123)"
        )

    user = _find_user(email)
    if not user or user["password_hash"] != _hash_password(password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = secrets.token_urlsafe(32)
    _sessions[token] = email
    response = JSONResponse({"token": token, "email": email, "name": user.get("name", "")})
    response.set_cookie(key="fintel_token", value=token, httponly=True, samesite="lax", max_age=86400)
    return response


@app.post("/api/logout")
async def logout(request: Request):
    token = _get_token(request)
    if token and token in _sessions:
        del _sessions[token]
    response = JSONResponse({"status": "logged out"})
    response.delete_cookie("fintel_token")
    return response


# ── Profile endpoints ─────────────────────────────────────────────────────────

@app.get("/api/profile")
async def get_profile(email: str = Depends(require_auth)):
    user = _find_user(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Count reports for this user
    reports_dir = Path(config.OUTPUT_DIR)
    report_count = len(list(reports_dir.glob("*.md"))) if reports_dir.exists() else 0
    return {
        "name": user.get("name", ""),
        "email": user.get("email", ""),
        "joined": user.get("joined", "N/A"),
        "report_count": report_count,
    }


@app.post("/api/change-password")
async def change_password(req: ChangePasswordRequest, email: str = Depends(require_auth)):
    user = _find_user(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user["password_hash"] != _hash_password(req.current_password):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")
    if not _validate_password(req.new_password):
        raise HTTPException(
            status_code=400,
            detail="New password must start with a capital letter and contain a special character."
        )
    users = _load_users()
    for u in users:
        if u["email"].lower() == email.lower():
            u["password_hash"] = _hash_password(req.new_password)
            break
    _save_users(users)
    return {"status": "ok", "message": "Password changed successfully."}


# ── Report history endpoints ──────────────────────────────────────────────────

@app.get("/api/reports")
async def list_reports(email: str = Depends(require_auth)):
    reports_dir = Path(config.OUTPUT_DIR)
    if not reports_dir.exists():
        return {"reports": []}

    reports = []
    for f in sorted(reports_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True):
        # Parse filename: sector_intent_query_YYYYMMDD_HHMMSS.md
        parts = f.stem.split("_")
        sector = parts[0].upper() if parts else "Unknown"
        # Extract date from last two parts
        try:
            date_str = parts[-2] + parts[-1]
            dt = datetime.strptime(date_str, "%Y%m%d%H%M%S")
            date_display = dt.strftime("%b %d, %Y %H:%M")
        except Exception:
            date_display = "Unknown date"
        # Build a readable title from middle parts
        title_parts = parts[2:-2] if len(parts) > 4 else parts[1:-2]
        title = " ".join(title_parts).replace("_", " ").title() or f.stem

        # Read first few lines to get the query
        query = title
        try:
            lines = f.read_text(encoding="utf-8").split("\n")
            for line in lines:
                if line.startswith("**Query:**"):
                    query = line.replace("**Query:**", "").strip()
                    break
        except Exception:
            pass

        reports.append({
            "filename": f.name,
            "sector": sector,
            "title": query,
            "date": date_display,
            "size_kb": round(f.stat().st_size / 1024, 1),
        })

    return {"reports": reports}


@app.get("/api/reports/{filename}")
async def get_report(filename: str, email: str = Depends(require_auth)):
    safe_name = Path(filename).name
    report_path = Path(config.OUTPUT_DIR) / safe_name
    if not report_path.exists() or not safe_name.endswith(".md"):
        raise HTTPException(status_code=404, detail="Report not found")
    content = report_path.read_text(encoding="utf-8")
    return {"filename": safe_name, "content": content}


# ── Stock price endpoint ──────────────────────────────────────────────────────

@app.get("/api/stock/{ticker}")
async def get_stock(ticker: str, email: str = Depends(require_auth)):
    import yfinance as yf
    import math
    def safe(v):
        """Convert NaN/Inf floats to None for JSON safety."""
        try:
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else round(f, 2)
        except (TypeError, ValueError):
            return None
    try:
        def fetch():
            t = yf.Ticker(ticker.upper())
            info = t.info
            hist = t.history(period="5d")
            price = safe(hist["Close"].iloc[-1]) if not hist.empty else None
            prev  = safe(hist["Close"].iloc[-2]) if len(hist) > 1 else price
            change     = round(price - prev, 2) if price is not None and prev is not None else 0
            change_pct = round((change / prev) * 100, 2) if prev else 0
            spark_hist = t.history(period="1mo")
            sparkline  = [safe(v) for v in spark_hist["Close"].tolist() if safe(v) is not None] if not spark_hist.empty else []
            return {
                "ticker":     ticker.upper(),
                "name":       info.get("longName", ticker.upper()),
                "price":      price,
                "change":     change,
                "change_pct": change_pct,
                "market_cap": safe(info.get("marketCap")),
                "pe_ratio":   safe(info.get("trailingPE")),
                "week_high":  safe(info.get("fiftyTwoWeekHigh")),
                "week_low":   safe(info.get("fiftyTwoWeekLow")),
                "volume":     safe(info.get("volume")),
                "currency":   info.get("currency", "USD"),
                "sparkline":  sparkline,
            }
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, fetch)
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch stock data: {str(exc)}")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ── Research endpoints ────────────────────────────────────────────────────────

@app.post("/api/plan")
async def get_plan(req: PlanRequest, email: str = Depends(require_auth)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    router = get_router()
    loop = asyncio.get_running_loop()

    try:
        sector, agent, message = await loop.run_in_executor(None, router.route, req.query)
    except Exception as exc:
        logger.exception("Routing error")
        raise HTTPException(status_code=500, detail=str(exc))

    if agent is None:
        return {"routed": False, "sector": sector, "message": message, "plan": None, "analysis": None}

    try:
        analysis, plan, formatted_plan = await loop.run_in_executor(None, agent.get_plan, req.query)
    except Exception as exc:
        logger.exception("Planning error")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "routed": True, "sector": sector, "message": "",
        "analysis": analysis, "plan": plan, "formatted_plan": formatted_plan,
    }


@app.post("/api/research")
async def run_research(req: ResearchRequest, email: str = Depends(require_auth)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    async def event_stream() -> AsyncGenerator[str, None]:
        def sse(event: str, data: Any) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        router = get_router()
        loop = asyncio.get_running_loop()

        try:
            sector, agent, message = await loop.run_in_executor(None, router.route, req.query)
        except Exception as exc:
            yield sse("error", {"message": str(exc)}); yield sse("done", {}); return

        yield sse("routing", {"sector": sector, "routed": agent is not None, "message": message})
        if agent is None:
            yield sse("done", {}); return

        approved_plan = req.plan
        if approved_plan is None:
            try:
                analysis, approved_plan, formatted_plan = await loop.run_in_executor(None, agent.get_plan, req.query)
                yield sse("planning", {"analysis": analysis, "plan": approved_plan, "formatted_plan": formatted_plan})
            except Exception as exc:
                yield sse("error", {"message": str(exc)}); yield sse("done", {}); return

        progress_events: asyncio.Queue = asyncio.Queue()

        def progress_callback(step_num: int, total: int, message: str) -> None:
            loop.call_soon_threadsafe(
                progress_events.put_nowait,
                {"step": step_num, "total": total, "message": message},
            )

        research_future = loop.run_in_executor(
            None,
            lambda: agent.run(
                user_query=req.query,
                approved_plan=approved_plan,
                progress_callback=progress_callback,
                save_report=True,
            ),
        )

        while not research_future.done():
            try:
                event = await asyncio.wait_for(progress_events.get(), timeout=0.25)
                yield sse("progress", event)
            except asyncio.TimeoutError:
                pass

        while not progress_events.empty():
            yield sse("progress", progress_events.get_nowait())

        try:
            report, file_path = await research_future
        except Exception as exc:
            logger.exception("Research execution error")
            yield sse("error", {"message": str(exc)}); yield sse("done", {}); return

        yield sse("report", {"report": report, "file_path": file_path})
        yield sse("done", {})
        # Log activity
        _log_activity(email, "research", sector=sector, query=req.query)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Activity log endpoint ─────────────────────────────────────────────────────

@app.get("/api/activity")
async def get_activity(email: str = Depends(require_auth)):
    """Return activity log for the current user."""
    log = _load_activity()
    user_log = [e for e in log if e.get("email") == email]
    return {"activity": list(reversed(user_log[-50:]))}


@app.get("/api/dashboard")
async def get_dashboard(email: str = Depends(require_auth)):
    """Return analytics data for the sector dashboard."""
    from collections import Counter
    import re as _re

    log = _load_activity()
    research_log = [e for e in log if e.get("action") == "research"]

    # Sector breakdown
    sector_counts = Counter(e.get("sector", "Unknown").upper() for e in research_log)
    sectors = [{"sector": s, "count": c} for s, c in sector_counts.most_common()]

    # Activity over time (last 14 days by date)
    date_counts: dict[str, int] = {}
    for e in research_log:
        ts = e.get("timestamp", "")
        date = ts[:10] if ts else "unknown"
        date_counts[date] = date_counts.get(date, 0) + 1
    timeline = [{"date": d, "count": c} for d, c in sorted(date_counts.items())[-14:]]

    # Top companies (extract from queries)
    company_pattern = _re.compile(
        r'\b(TCS|Infosys|Wipro|HCL|HDFC|ICICI|SBI|Reliance|Tata|Zomato|Nykaa|'
        r'Airtel|Jio|ONGC|Adani|Sun Pharma|Cipla|Maruti|Bajaj|ITC|HUL|'
        r'Apollo|Fortis|JSW|Hindalco|Vedanta|Paytm|L&T)\b', _re.IGNORECASE
    )
    company_counts: dict[str, int] = {}
    for e in research_log:
        q = e.get("query", "")
        for m in company_pattern.findall(q):
            key = m.upper()
            company_counts[key] = company_counts.get(key, 0) + 1
    top_companies = [{"company": c, "count": n}
                     for c, n in sorted(company_counts.items(), key=lambda x: -x[1])[:10]]

    # Word cloud — extract meaningful words from queries
    stop_words = {
        "analyze","analysis","and","the","of","in","for","a","an","to","with",
        "vs","compare","indian","india","sector","market","growth","trends",
        "outlook","strategy","performance","research","study","impact","latest",
        "2024","2025","2026","how","what","why","is","are","was","were"
    }
    word_counts: dict[str, int] = {}
    for e in research_log:
        words = _re.findall(r'\b[a-zA-Z]{4,}\b', e.get("query", "").lower())
        for w in words:
            if w not in stop_words:
                word_counts[w] = word_counts.get(w, 0) + 1
    word_cloud = [{"word": w, "count": c}
                  for w, c in sorted(word_counts.items(), key=lambda x: -x[1])[:40]]

    # Summary stats
    total_queries = len(research_log)
    unique_sectors = len(sector_counts)
    reports_dir = Path(config.OUTPUT_DIR)
    total_reports = len(list(reports_dir.glob("*.md"))) if reports_dir.exists() else 0

    return {
        "total_queries": total_queries,
        "unique_sectors": unique_sectors,
        "total_reports": total_reports,
        "sectors": sectors,
        "timeline": timeline,
        "top_companies": top_companies,
        "word_cloud": word_cloud,
    }


# ── Admin endpoints ───────────────────────────────────────────────────────────

def require_admin(request: Request) -> None:
    token = _get_token(request)
    if not token or token not in _admin_sessions:
        raise HTTPException(status_code=401, detail="Admin access required")

@app.post("/api/admin/login")
async def admin_login(req: AdminLoginRequest):
    if req.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid admin password")
    token = secrets.token_urlsafe(32)
    _admin_sessions.add(token)
    response = JSONResponse({"token": token})
    response.set_cookie(key="fds_admin_token", value=token, httponly=True, samesite="lax", max_age=3600)
    return response

@app.post("/api/admin/logout")
async def admin_logout(request: Request):
    token = _get_token(request)
    _admin_sessions.discard(token or "")
    response = JSONResponse({"status": "logged out"})
    response.delete_cookie("fds_admin_token")
    return response

@app.get("/api/admin/stats")
async def admin_stats(request: Request, _=Depends(require_admin)):
    users = _load_users()
    reports_dir = Path(config.OUTPUT_DIR)
    report_files = list(reports_dir.glob("*.md")) if reports_dir.exists() else []
    log = _load_activity()

    # Most active users
    from collections import Counter
    user_counts = Counter(e["email"] for e in log if e.get("action") == "research")
    most_active = [{"email": e, "count": c} for e, c in user_counts.most_common(5)]

    # Sector breakdown
    sector_counts = Counter(e["sector"] for e in log if e.get("sector"))
    sectors = [{"sector": s, "count": c} for s, c in sector_counts.most_common()]

    # Recent activity
    recent = list(reversed(log[-20:]))

    # Reports by sector from filenames
    report_sectors = Counter()
    for f in report_files:
        parts = f.stem.split("_")
        if parts:
            report_sectors[parts[0].upper()] += 1

    return {
        "total_users": len(users),
        "total_reports": len(report_files),
        "total_queries": len([e for e in log if e.get("action") == "research"]),
        "most_active": most_active,
        "sector_breakdown": sectors,
        "report_sectors": [{"sector": s, "count": c} for s, c in report_sectors.most_common()],
        "recent_activity": recent,
        "users": [{"name": u.get("name",""), "email": u.get("email",""), "joined": u.get("joined","N/A")} for u in users],
    }


# ── Serve frontend ─────────────────────────────────────────────────────────────
UI_DIR = Path(__file__).parent / "ui"

@app.get("/admin")
async def serve_admin():
    page = UI_DIR / "admin.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Admin page not found")
    return FileResponse(str(page))

@app.get("/login")
async def serve_login():
    page = UI_DIR / "login.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Login page not found")
    return FileResponse(str(page))

@app.get("/history")
async def serve_history():
    page = UI_DIR / "history.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="History page not found")
    return FileResponse(str(page))

@app.get("/activity")
async def serve_activity():
    page = UI_DIR / "activity.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Activity page not found")
    return FileResponse(str(page))

@app.get("/dashboard")
async def serve_dashboard():
    page = UI_DIR / "dashboard.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Dashboard page not found")
    return FileResponse(str(page))

@app.get("/profile")
async def serve_profile():
    page = UI_DIR / "profile.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Profile page not found")
    return FileResponse(str(page))

@app.get("/")
async def serve_ui():
    index = UI_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(str(index))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    try:
        config.validate()
    except EnvironmentError as e:
        print(f"[ERROR] Configuration: {e}")
        print("Copy .env.example to .env and fill in your API keys.")
        sys.exit(1)

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False, log_level="warning")
