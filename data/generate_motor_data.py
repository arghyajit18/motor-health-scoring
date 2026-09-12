"""
Generate a synthetic 3-phase motor current dataset with seeded electrical and
mechanical faults. Each motor gets one 1-second waveform per day (2 kHz), with
fault signatures that grow over time after an onset day.

Fault physics encoded here:
- Broken rotor bar: sidebands at 50 +/- 2*s*50 Hz (slip s = 0.04 -> 46/54 Hz)
- Bearing defect: tones near BPFO (~74 Hz) and BPFI (~119 Hz), amplitude
  modulated at rotor speed (24 Hz)
- Eccentricity: sidebands at 50 +/- rotor frequency (26/74 Hz)
- Supply unbalance / overload: magnitude shift on one phase plus heat rise

Swap this file's output for real drive/clamp-meter recordings with the same
layout (one waveform per motor per day) to run the rest of the pipeline
unchanged.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

FS = 2000
DUR = 1.0
N = int(FS * DUR)
T = np.arange(N) / FS

F0 = 50.0
SLIP = 0.04
FR = 24.0
BPFO = 73.7
BPFI = 118.8

BASE_LOAD = {"Pump": 72.0, "Fan": 64.0, "Conveyor": 78.0, "Compressor": 82.0}

MOTORS = [
    ("M-01", "Pump", "healthy", None),
    ("M-02", "Fan", "healthy", None),
    ("M-03", "Conveyor", "healthy", None),
    ("M-04", "Compressor", "healthy", None),
    ("M-05", "Pump", "bearing", 18),
    ("M-06", "Fan", "bearing", 12),
    ("M-07", "Conveyor", "bearing", 25),
    ("M-08", "Pump", "rotor", 15),
    ("M-09", "Fan", "rotor", 22),
    ("M-10", "Conveyor", "eccentricity", 10),
    ("M-11", "Compressor", "eccentricity", 28),
    ("M-12", "Pump", "unbalance", 20),
]

DAYS = 45
RAMP_DAYS = 12


def severity(onset, day):
    if onset is None or day < onset:
        return 0.0
    return float(min(1.0, (day - onset) / RAMP_DAYS))


def waveform(load_pct, fault, sev):
    amp = 4.0 + 6.0 * load_pct / 100.0
    phases = np.zeros((N, 3))
    for p in range(3):
        shift = p * 2 * np.pi / 3
        sig = amp * np.sin(2 * np.pi * F0 * T + shift)
        sig += 0.020 * amp * np.sin(2 * np.pi * 150 * T + shift)
        sig += 0.008 * amp * np.sin(2 * np.pi * 250 * T + shift)
        if fault == "rotor":
            sig += sev * 0.06 * amp * (
                np.sin(2 * np.pi * 46 * T + shift)
                + np.sin(2 * np.pi * 54 * T + shift)
            )
        elif fault == "bearing":
            mod = 1.0 + 0.3 * np.sin(2 * np.pi * FR * T)
            sig += sev * 0.03 * amp * mod * (
                np.sin(2 * np.pi * BPFO * T + shift)
                + np.sin(2 * np.pi * BPFI * T + shift)
            )
        elif fault == "eccentricity":
            sig += sev * 0.05 * amp * (
                np.sin(2 * np.pi * (F0 - FR) * T + shift)
                + np.sin(2 * np.pi * (F0 + FR) * T + shift)
            )
        if fault == "unbalance" and p == 0:
            sig *= 1.0 + 0.10 * sev
        sig *= 1.0 + 0.005 * rng.standard_normal(N)
        sig += 0.004 * amp * rng.standard_normal(N)
        phases[:, p] = sig
    return phases.astype(np.float32)


waves = []
meta = []
dates = pd.date_range(start="2024-01-01", periods=DAYS, freq="D")

for motor_id, motor_type, fault, onset in MOTORS:
    for day in range(DAYS):
        sev = severity(onset, day)
        load = BASE_LOAD[motor_type] + 4 * np.sin(2 * np.pi * day / 7) + rng.normal(0, 1.5)
        load = float(np.clip(load, 40, 100))
        temp = 52 + 0.35 * load + 12 * sev + rng.normal(0, 0.8)
        w = waveform(load, fault, sev)
        waves.append(w)
        meta.append({
            "motor_id": motor_id,
            "motor_type": motor_type,
            "day": day,
            "date": dates[day].date().isoformat(),
            "load_pct": round(load, 1),
            "temp_c": round(float(temp), 1),
            "true_fault": fault,
            "true_severity": round(sev, 3),
        })

waves = np.stack(waves)

missing_idx = rng.choice(len(waves), size=int(0.01 * len(waves)), replace=False)
waves[missing_idx] = np.nan

np.savez_compressed("data/waveforms.npz", waves=waves)
pd.DataFrame(meta).to_csv("data/readings_meta.csv", index=False)

print(f"Generated {len(meta)} readings ({len(MOTORS)} motors x {DAYS} days) -> data/waveforms.npz")
print(f"Fault mix: {[m[2] for m in MOTORS]}")
print(f"Injected {len(missing_idx)} missing readings for the cleaning step")
