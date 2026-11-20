import collections
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .engine import Engine

STATE: dict = {}
LAT = collections.deque(maxlen=500)
COUNT = {"requests": 0, "errors": 0}
INDEX = Path(__file__).resolve().parent.parent / "static" / "index.html"


def api_stats():
    lat = sorted(LAT)
    p95 = lat[int(0.95 * (len(lat) - 1))] if lat else 0.0
    return dict(COUNT, p95_s=round(p95, 4))


@asynccontextmanager
async def lifespan(app):
    STATE["e"] = Engine(int(os.getenv("SEED", "7")))  # ~5s: simulate, fit, forecast, investigate
    yield


app = FastAPI(title="AirWatch AI", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def meter(request, call_next):
    t0 = time.perf_counter()
    try:
        resp = await call_next(request)
    except Exception:
        COUNT["errors"] += 1
        raise
    COUNT["requests"] += 1
    COUNT["errors"] += resp.status_code >= 500
    LAT.append(time.perf_counter() - t0)
    return resp


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=300)
    llm: bool = False


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX)


@app.get("/api/summary")
def summary():
    return STATE["e"].summary()


@app.get("/api/stations")
def stations():
    return STATE["e"].stations()


@app.get("/api/stations/{sid}/series")
def series(sid: int, hours: int = 48):
    s = STATE["e"].series(sid, hours)
    if s is None:
        raise HTTPException(404, f"unknown station {sid}")
    return s


@app.get("/api/incidents")
def incidents():
    return STATE["e"].incidents


@app.get("/api/forecast")
def forecast():
    e = STATE["e"]
    return dict(warnings=e.warnings, stations=e.forecast, metrics=e.metrics)


@app.get("/api/infra")
def infra():
    return STATE["e"].infra(api_stats())


@app.post("/api/ask")
def ask(body: Ask):
    return STATE["e"].ask(body.question, body.llm)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return STATE["e"].metrics_text(api_stats())
