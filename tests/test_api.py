"""API contract tests: POST /score must match the batch pipeline."""
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _body(idx=0):
    bundle = np.load("data/waves_clean.npz")
    meta = pd.read_csv("data/clean_meta.csv")
    row = meta.iloc[idx]
    vibs = bundle["vibs"] if "vibs" in bundle else None
    payload = {
        "phase_a": bundle["waves"][idx][:, 0].tolist(),
        "phase_b": bundle["waves"][idx][:, 1].tolist(),
        "phase_c": bundle["waves"][idx][:, 2].tolist(),
        "fs": 2000.0,
        "metadata": {
            "motor_id": str(row["motor_id"]),
            "load_pct": float(row["load_pct"]),
            "temp_c": float(row["temp_c"]),
            "f0_hz": float(row.get("f0_hz", 50) or 50),
            "fr_hz": float(row.get("fr_hz", 24) or 24),
            "slip": float(row.get("slip", 0.04) or 0.04),
        },
    }
    if vibs is not None:
        payload["vibration"] = vibs[idx].tolist()
    return payload


def test_health_endpoint():
    assert client.get("/health").status_code == 200


def test_score_matches_batch():
    trend = pd.read_csv("outputs/health_trend.csv")
    for idx in (0, len(trend) // 2, len(trend) - 1):
        got = client.post("/score", json=_body(idx)).json()
        exp = trend.iloc[idx]
        assert abs(got["health"] - exp["health"]) < 0.05
        assert got["status"] == exp["status"]
        assert got["suspected_cause"] == exp["suspected_cause"]


def test_score_rejects_short_waveform():
    bad = {"phase_a": [1.0] * 10, "phase_b": [1.0] * 10, "phase_c": [1.0] * 10}
    assert client.post("/score", json=bad).status_code == 422


def test_fleet_endpoint():
    ranking = client.get("/fleet").json()["ranking"]
    assert len(ranking) == pd.read_csv("outputs/fleet_ranking.csv").shape[0]
