"""
Real-data loader: Paderborn University bearing dataset (KAT Data Center).

Reads downloaded bearing .rar/.mat files at one operating point
(N15_M07_F10: 1500 rpm, 0.7 Nm, 1000 N), takes the two measured current
phases, reconstructs the third (isolated 3-wire system: i1+i2+i3 = 0),
resamples 64 kHz -> 2 kHz, and slices each 4 s record into four 1 s
windows. Output layout matches generate_motor_data.py exactly, so the
rest of the pipeline (cleaning -> FFT features -> scoring) runs unchanged.

Per-window supply frequency and rotor speed are estimated from the data
and stored in the metadata; features.py builds its sideband windows
from those instead of fixed 50 Hz assumptions.

Usage: place the bearing .rar files in data/paderborn_rars/ (or .mat
files in data/paderborn_mats/), then run this script.
"""
import glob
import os
import re
import subprocess
import numpy as np
import pandas as pd
from scipy import signal as spsig
import scipy.io as sio

RAR_DIR = "data/paderborn_rars"
MAT_DIR = "data/paderborn_mats"
SEVEN_ZIP = r"C:\Program Files\7-Zip\7z.exe"

BEARINGS = {
    "K001": ("healthy", 0.0),
    "K002": ("healthy", 0.0),
    "KI04": ("bearing", 1.0),
    "KI16": ("bearing", 1.0),
}
CONDITION = "N15_M07_F10"
FS_TARGET = 2000


def extract_rars():
    os.makedirs(MAT_DIR, exist_ok=True)
    for code in BEARINGS:
        rar = os.path.join(RAR_DIR, f"{code}.rar")
        if not os.path.exists(rar):
            print(f"Missing {rar}, skipping {code}")
            continue
        cmd = [SEVEN_ZIP, "e", rar, f"-o{MAT_DIR}",
               f"{code}\\{CONDITION}_{code}_*.mat", "-y"]
        subprocess.run(cmd, capture_output=True, check=False)
    print("Mats on disk:", len(glob.glob(os.path.join(MAT_DIR, "*.mat"))))


def entry_vec(e):
    return np.asarray(e["Data"]).ravel().astype(np.float64)


def load_channels(path):
    var = os.path.splitext(os.path.basename(path))[0]
    s = sio.loadmat(path)[var][0, 0]
    chans = {}
    for ch in s["Y"][0]:
        chans[str(ch["Name"][0])] = entry_vec(ch)
    n = len(chans["phase_current_1"])
    t = None
    for x in s["X"][0]:
        v = entry_vec(x)
        if len(v) == n:
            t = v
            break
    if t is None:
        raise ValueError(f"No matching time base in {path}")
    fs = float((len(t) - 1) / (t[-1] - t[0]))
    return chans, fs


def estimate_f0(x, fs):
    n = len(x)
    spec = np.abs(np.fft.rfft(x - x.mean()))
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    sel = (freqs >= 10) & (freqs <= 500)
    return float(freqs[sel][np.argmax(spec[sel])])


waves, vibs, meta = [], [], []
base_date = pd.Timestamp("2024-01-01")

extract_rars()
files = sorted(glob.glob(os.path.join(MAT_DIR, "*.mat")))
files = [p for p in files if re.match(r"N\d+_M\d+_F\d+_(K[A-Z0-9]+)_.*\.mat",
                                      os.path.basename(p))]
if not files:
    raise SystemExit("No .mat files found. Put bearing .rar files in data/paderborn_rars/ first.")

day_counter = {}
for path in files:
    m = re.match(r"N\d+_M\d+_F\d+_(K[A-Z0-9]+)_.*\.mat", os.path.basename(path))
    if not m or m.group(1) not in BEARINGS:
        continue
    code = m.group(1)
    fault, sev = BEARINGS[code]
    chans, fs = load_channels(path)
    i1, i2 = chans["phase_current_1"], chans["phase_current_2"]
    i3 = -(i1 + i2)
    vib = chans["vibration_1"]
    rpm = float(chans["speed"].mean())
    fr = rpm / 60.0
    temp = float(chans["temp_2_bearing_module"].mean())
    n_total = min(len(i1), len(vib))
    n_sec = int(n_total // fs)
    n_use = int(n_sec * fs)
    stacked = np.stack([i1[:n_use], i2[:n_use], i3[:n_use]], axis=1)
    n2k = n_sec * FS_TARGET
    down = np.stack([spsig.resample(stacked[:, p], n2k) for p in range(3)], axis=1)
    vib_use = vib[:n_use]
    vib2k = spsig.resample(vib_use, n2k)
    f0 = estimate_f0(down[:FS_TARGET, 0], FS_TARGET)
    slip = max((1500.0 - rpm) / 1500.0, 0.002)
    for w in range(n_sec):
        seg = down[w * FS_TARGET:(w + 1) * FS_TARGET]
        vseg = vib2k[w * FS_TARGET:(w + 1) * FS_TARGET]
        d = day_counter.get(code, 0)
        day_counter[code] = d + 1
        waves.append(seg.astype(np.float32))
        vibs.append(vseg.astype(np.float32))
        meta.append({
            "motor_id": code, "motor_type": "TestRig", "day": d,
            "date": (base_date + pd.Timedelta(days=d)).date().isoformat(),
            "load_pct": 70.0, "temp_c": round(temp, 1),
            "true_fault": fault, "true_severity": sev,
            "f0_hz": round(f0, 2), "fr_hz": round(fr, 2),
            "slip": round(float(slip), 4),
        })

waves = np.stack(waves)
vibs = np.stack(vibs)
np.savez_compressed("data/waveforms.npz", waves=waves, vibs=vibs)
pd.DataFrame(meta).to_csv("data/readings_meta.csv", index=False)
print(f"Converted {len(meta)} windows from {len(files)} files "
      f"-> data/waveforms.npz, data/readings_meta.csv")
print(pd.DataFrame(meta).groupby(["motor_id", "true_fault"]).size().to_string())
