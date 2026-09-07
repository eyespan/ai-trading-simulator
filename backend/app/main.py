from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router

load_dotenv()

app = FastAPI(
    title="AI Trading Simulator",
    description=(
        "Paper-trading simulator combining technical-indicator signals, an "
        "AI decision-review layer, and an independent risk engine. "
        "Educational/portfolio project — not investment advice."
    ),
    version="0.1.0",
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

app.include_router(api_router, prefix="/api")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")


@app.get("/")
def serve_dashboard():
    return FileResponse(FRONTEND_DIR / "templates" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}
