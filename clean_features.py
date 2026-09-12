"""
Step 2: validate raw waveforms, drop missing or flat readings, and report
what was removed. Keeps a clean waveform file plus matching metadata so the
feature step below works on verified data only.
"""
import numpy as np
import pandas as pd

bundle = np.load("data/waveforms.npz")
waves = bundle["waves"].astype(np.float64)
vibs = bundle["vibs"].astype(np.float64) if "vibs" in bundle else None
meta = pd.read_csv("data/readings_meta.csv")

n_total = len(meta)
bad = np.isnan(waves).any(axis=(1, 2))
if vibs is not None:
    bad = bad | np.isnan(vibs).any(axis=1)
flat = (~bad) & (waves.std(axis=(1, 2)) < 1e-6)
drop = bad | flat

print(f"Total readings: {n_total}")
print(f"Missing (NaN) readings dropped: {int(bad.sum())}")
print(f"Flat-line readings dropped: {int(flat.sum())}")

waves_clean = waves[~drop]
meta_clean = meta.loc[~drop].reset_index(drop=True)

cap = float(np.nanpercentile(np.abs(waves_clean), 99.9))
n_clipped = int((np.abs(waves_clean) > cap).sum())
waves_clean = np.clip(waves_clean, -cap, cap).astype(np.float32)

out = {"waves": waves_clean}
if vibs is not None:
    vibs_clean = vibs[~drop]
    vcap = float(np.nanpercentile(np.abs(vibs_clean), 99.9))
    vibs_clean = np.clip(vibs_clean, -vcap, vcap).astype(np.float32)
    out["vibs"] = vibs_clean
    print(f"Clipped vibration extremes at +/-{vcap:.3f}")

np.savez_compressed("data/waves_clean.npz", **out)
meta_clean.to_csv("data/clean_meta.csv", index=False)

print(f"Clipped {n_clipped} extreme current samples at +/-{cap:.2f} A")
print(f"Clean readings: {len(meta_clean)} -> data/waves_clean.npz, data/clean_meta.csv")
