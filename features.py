"""
Step 3: turn each 1-second 3-phase waveform into motor current signature
features with an FFT (1 Hz bins fall out naturally from the 1 s window).

Per reading:
- rms_a/b/c, rms_mean, crest factor and THD of phase A
- sideband ratio: rotor-bar bands over fundamental
- eccentricity ratio and bearing band energy, located from the
  per-reading supply frequency and rotor speed in the metadata
- voltage/current unbalance proxy: phase RMS spread in percent
Plus operating point (load, temperature) carried from metadata.
"""
import numpy as np
import pandas as pd

FS = 2000


def mag_at(spec, freqs, f):
    return float(spec[int(np.argmin(np.abs(freqs - f)))])


def band_peak(spec, freqs, lo, hi):
    sel = (freqs >= lo) & (freqs <= hi)
    return float(spec[sel].max())


def extract_feature_row(wave, vib, meta, fs=FS):
    n = wave.shape[0]
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    rms = [float(np.sqrt(np.mean(wave[:, p] ** 2))) for p in range(3)]
    rms_mean = float(np.mean(rms))
    peak = float(np.abs(wave[:, 0]).max())
    crest = peak / rms[0] if rms[0] > 0 else 0.0

    spec = np.abs(np.fft.rfft(wave[:, 0])) / (n / 2)
    f0 = float(meta.get("f0_hz", 50.0) or 50.0)
    fr = float(meta.get("fr_hz", 24.0) or 24.0)
    slip = float(meta.get("slip", 0.04) or 0.04)
    fund = mag_at(spec, freqs, f0) + 1e-9
    thd = float(np.sqrt(sum(mag_at(spec, freqs, k * f0) ** 2 for k in (2, 3, 4, 5))) / fund)
    sb = 2 * slip * f0
    sideband = (mag_at(spec, freqs, f0 - sb) + mag_at(spec, freqs, f0 + sb)) / fund
    ecc = (mag_at(spec, freqs, f0 - fr) + mag_at(spec, freqs, f0 + fr)) / fund
    bearing = (band_peak(spec, freqs, 3.07 * fr - 2.5, 3.07 * fr + 2.5)
               + band_peak(spec, freqs, 4.95 * fr - 2.5, 4.95 * fr + 2.5)) / fund
    unbalance = (max(rms) - min(rms)) / (rms_mean + 1e-9) * 100.0

    row = {
        **dict(meta),
        "rms_a": round(rms[0], 3), "rms_b": round(rms[1], 3), "rms_c": round(rms[2], 3),
        "rms_mean": round(rms_mean, 3), "crest": round(crest, 3),
        "thd": round(thd, 4), "sideband_ratio": round(float(sideband), 4),
        "ecc_ratio": round(float(ecc), 4), "bearing_band": round(float(bearing), 4),
        "unbalance_pct": round(float(unbalance), 2),
    }
    if vib is not None:
        v_rms = float(np.sqrt(np.mean(vib ** 2)))
        v_spec = np.abs(np.fft.rfft(vib)) / (len(vib) / 2)
        v_freqs = np.fft.rfftfreq(len(vib), d=1 / fs)
        hi = ((v_freqs >= 350) & (v_freqs <= 950))
        row["vib_rms"] = round(v_rms, 4)
        row["vib_crest"] = round(float(np.abs(vib).max() / (v_rms + 1e-12)), 3)
        row["vib_high_ratio"] = round(float((v_spec[hi] ** 2).sum() / ((v_spec ** 2).sum() + 1e-12)), 4)
    return row


def main():
    bundle = np.load("data/waves_clean.npz")
    waves = bundle["waves"].astype(np.float64)
    vibs = bundle["vibs"].astype(np.float64) if "vibs" in bundle else None
    meta = pd.read_csv("data/clean_meta.csv")

    rows = []
    for idx, (w, (_, m)) in enumerate(zip(waves, meta.iterrows())):
        rows.append(extract_feature_row(w, vibs[idx] if vibs is not None else None,
                                        m.to_dict()))

    feat = pd.DataFrame(rows)
    feat.to_csv("data/motor_features.csv", index=False)
    print(f"Extracted {len(feat)} feature rows -> data/motor_features.csv")
    print(feat[["sideband_ratio", "bearing_band", "unbalance_pct"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
