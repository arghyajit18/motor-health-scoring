"""
Motor health scoring service.

Wraps the batch pipeline (features.py + scoring.py) as an HTTP API:
- POST /score  score one waveform reading, returns health 0-100,
  status band, suspected cause and recommended action
- GET  /fleet  latest fleet ranking from the batch run
- GET  /health liveness plus baseline info

Run locally: uvicorn api:app --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features import extract_feature_row
from scoring import fit_baseline, score_features

BASELINE_CSV = "data/motor_features.csv"


class Metadata(BaseModel):
    motor_id: str = "unknown"
    load_pct: float = 70.0
    temp_c: float = 45.0
    f0_hz: float = 50.0
    fr_hz: float = 24.0
    slip: float = 0.04


class ScoreRequest(BaseModel):
    phase_a: list[float] = Field(min_length=256)
    phase_b: list[float] = Field(min_length=256)
    phase_c: list[float] = Field(min_length=256)
    fs: float = Field(default=2000.0, gt=100, le=100000)
    vibration: list[float] | None = None
    metadata: Metadata = Metadata()


class ScoreResponse(BaseModel):
    health: float
    status: str
    suspected_cause: str
    action: str
    degradation: float
    iforest_anomaly: bool


BASELINE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    df = pd.read_csv(BASELINE_CSV)
    BASELINE.update(fit_baseline(df))
    yield


app = FastAPI(title="Motor Health Scoring", lifespan=lifespan)


def get_baseline():
    if not BASELINE:
        df = pd.read_csv(BASELINE_CSV)
        BASELINE.update(fit_baseline(df))
    return BASELINE


@app.get("/health")
def health():
    base = get_baseline()
    return {"status": "ok", "features": base["cols"],
            "baseline_median_d": round(float(base["d50"]), 3)}


@app.get("/fleet")
def fleet():
    df = pd.read_csv("outputs/fleet_ranking.csv")
    return {"ranking": df.to_dict(orient="records")}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    n = len(req.phase_a)
    if not (len(req.phase_b) == len(req.phase_c) == n):
        raise HTTPException(status_code=422, detail="phase arrays must share a length")
    if req.vibration is not None and len(req.vibration) != n:
        raise HTTPException(status_code=422, detail="vibration must match phase length")
    wave = np.stack([np.asarray(req.phase_a, dtype=float),
                     np.asarray(req.phase_b, dtype=float),
                     np.asarray(req.phase_c, dtype=float)], axis=1)
    vib = np.asarray(req.vibration, dtype=float) if req.vibration is not None else None
    feat = extract_feature_row(wave, vib, req.metadata.model_dump(), fs=req.fs)
    return score_features(feat, get_baseline())
