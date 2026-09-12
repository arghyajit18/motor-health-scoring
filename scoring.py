"""
Step 4: score every reading 0-100 and turn scores into fleet decisions.

Method:
- Fit a StandardScaler + IsolationForest on early healthy readings only.
  The forest acts as an independent cross-check on the rule score.
- Rule score: z-scores of each fault indicator against the same healthy
  baseline, combined into a degradation index calibrated so the baseline
  median scores 100 and its 95th percentile scores 80.
- Status bands: Healthy >= 80, Watch 60-80, Alert 40-60, Critical < 40.
- Suspected cause is the largest weighted contributor; each cause maps to
  one shop-floor action.
- Detection quality is measured against the seeded truth (severity > 0.35
  counts as truly faulty): precision/recall plus mean early-warning lead
  time in days.

Importable: api.py calls fit_baseline() once at startup and
score_features() per request. Running this file directly executes the
batch pipeline over data/motor_features.csv.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ALL_FEATS = ["rms_mean", "crest", "thd", "sideband_ratio",
               "ecc_ratio", "bearing_band", "unbalance_pct",
               "vib_rms", "vib_crest", "vib_high_ratio"]
WEIGHTS = {"sideband_ratio": 1.0, "bearing_band": 1.0, "ecc_ratio": 1.0,
           "unbalance_pct": 0.8, "thd": 0.5, "crest": 0.5,
           "vib_high_ratio": 1.2, "vib_rms": 0.6, "vib_crest": 0.5}

RAMP_DAYS = 12

CAUSE_ACTION = {
    "rotor": ("Broken rotor bar signatures",
              "Schedule rotor inspection; plan rewind at next shutdown"),
    "bearing": ("Bearing defect frequencies",
                "Replace DE/NDE bearings at next planned stop"),
    "eccentricity": ("Air-gap eccentricity sidebands",
                     "Check alignment, air gap and mounting; re-measure in 7 days"),
    "supply": ("Phase unbalance / overload",
               "Check supply balance, terminal tightness and winding insulation"),
}

CAUSE_MAP = {"sideband_ratio": "rotor", "bearing_band": "bearing",
             "ecc_ratio": "eccentricity", "unbalance_pct": "supply",
             "thd": "supply", "crest": "supply", "rms_mean": "supply",
             "vib_high_ratio": "bearing", "vib_rms": "bearing", "vib_crest": "bearing"}


def status_of(health):
    if health > 80:
        return "Healthy"
    if health > 60:
        return "Watch"
    if health > 40:
        return "Alert"
    return "Critical"


def fit_baseline(df, feat_cols=None):
    cols = [c for c in (feat_cols or ALL_FEATS) if c in df.columns]
    base = df[(df["true_fault"] == "healthy") & (df["day"] < 10)]
    if len(base) == 0:
        base = df[df["true_fault"] == "healthy"]
    mu = base[cols].mean()
    sigma = base[cols].std().replace(0, 1e-9)
    scaler = StandardScaler().fit(base[cols])
    forest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    forest.fit(scaler.transform(base[cols]))
    d = ((base[cols] - mu) / sigma).clip(lower=0).mul(
        pd.Series({c: WEIGHTS.get(c, 0.5) for c in cols})).sum(axis=1)
    d50 = float(d.median())
    scale = max((float(d.quantile(0.95)) - d50) / 0.2231, 0.5)
    return {"cols": cols, "mu": mu, "sigma": sigma, "d50": d50,
            "scale": scale, "scaler": scaler, "forest": forest}


def score_features(feat, base):
    cols = base["cols"]
    x = pd.Series({c: float(feat.get(c, 0.0)) for c in cols})
    z = ((x - base["mu"]) / base["sigma"]).clip(lower=0)
    contrib = z.mul(pd.Series({c: WEIGHTS.get(c, 0.5) for c in cols}))
    degradation = float(contrib.sum())
    health = float(np.clip(
        100 * np.exp(-max(0.0, degradation - base["d50"]) / base["scale"]), 0, 100))
    cause = str(contrib.idxmax())
    cause = CAUSE_MAP.get(cause, "supply")
    anomaly = bool(base["forest"].predict(
        base["scaler"].transform(pd.DataFrame([x])))[0] == -1)
    return {
        "health": round(health, 1),
        "status": status_of(health),
        "suspected_cause": CAUSE_ACTION[cause][0],
        "action": CAUSE_ACTION[cause][1],
        "degradation": round(degradation, 3),
        "iforest_anomaly": anomaly,
    }


def main():
    df = pd.read_csv("data/motor_features.csv", parse_dates=["date"])
    base = fit_baseline(df)
    cols = base["cols"]

    scored = [score_features(row.to_dict(), base) for _, row in df.iterrows()]
    s = pd.DataFrame(scored)
    df = pd.concat([df.reset_index(drop=True), s], axis=1)

    df.to_csv("outputs/health_trend.csv", index=False)

    latest = df.sort_values("day").groupby("motor_id").tail(1)
    ranking = latest[["motor_id", "motor_type", "day", "health", "status",
                      "suspected_cause", "action", "iforest_anomaly"]].sort_values("health")
    ranking.to_csv("outputs/fleet_ranking.csv", index=False)

    alerts = df[df["status"].isin(["Alert", "Critical"])].sort_values(["date", "motor_id"])
    alerts[["date", "motor_id", "motor_type", "health", "status",
            "suspected_cause", "action"]].to_csv("outputs/alerts.csv", index=False)

    true_bad = df["true_severity"] > 0.35
    pred_bad = df["status"].isin(["Alert", "Critical"])
    tp = int((true_bad & pred_bad).sum())
    precision = round(tp / max(int(pred_bad.sum()), 1), 3)
    recall = round(tp / max(int(true_bad.sum()), 1), 3)

    leads = []
    for mid, g in df[df["true_fault"] != "healthy"].groupby("motor_id"):
        first_day = int(g["day"].min())
        onset_days = g.loc[g["true_severity"] > 0, "day"]
        if len(onset_days) == 0:
            continue
        onset = int(onset_days.min())
        if onset <= first_day:
            continue
        hit = g[pred_bad.loc[g.index]]
        if len(hit):
            leads.append(max(0, onset + RAMP_DAYS - int(hit["day"].min())))
    mean_lead = round(float(np.mean(leads)), 1) if leads else None

    metrics = {
        "readings": len(df),
        "motors": int(df["motor_id"].nunique()),
        "precision_alert_critical": precision,
        "recall_alert_critical": recall,
        "mean_early_warning_lead_days": mean_lead,
        "motors_by_status": df.sort_values("day").groupby("motor_id").tail(1)["status"].value_counts().to_dict(),
        "baseline": "healthy motors, first 10 days",
    }
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(base["scaler"], "outputs/scaler.joblib")
    joblib.dump(base["forest"], "outputs/iforest.joblib")

    print(f"Scored {len(df)} readings across {metrics['motors']} motors")
    print(f"Precision={precision} Recall={recall} Mean lead={mean_lead} days")
    print("\nFleet ranking (latest):")
    print(ranking[["motor_id", "health", "status", "suspected_cause"]].to_string(index=False))


if __name__ == "__main__":
    main()
