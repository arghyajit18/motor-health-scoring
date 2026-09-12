"""
Step 5: exploratory plots for the report and dashboard.
Saves health trends, example current spectra (healthy vs faulty),
score distribution and indicator heatmap to outputs/.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv("outputs/health_trend.csv", parse_dates=["date"])
feat = pd.read_csv("data/motor_features.csv", parse_dates=["date"])
out = "outputs"

fig, ax = plt.subplots(figsize=(12, 5))
for mid, g in df.groupby("motor_id"):
    g = g.sort_values("day")
    ax.plot(g["day"], g["health"], linewidth=1.2, label=mid)
ax.axhline(80, color="green", linestyle="--", linewidth=1, label="Healthy line (80)")
ax.axhline(60, color="orange", linestyle="--", linewidth=1, label="Watch line (60)")
ax.axhline(40, color="red", linestyle="--", linewidth=1, label="Alert line (40)")
ax.set_title("Motor Health Score Over Time (12-motor fleet)")
ax.set_xlabel("Day")
ax.set_ylabel("Health (0-100)")
ax.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.savefig(f"{out}/01_health_trend.png", dpi=120)
plt.close()

bundle = np.load("data/waves_clean.npz")
waves = bundle["waves"].astype(np.float64)
meta = pd.read_csv("data/clean_meta.csv")
FS = 2000
freqs = np.fft.rfftfreq(waves.shape[1], d=1 / FS)


def spectrum(motor_id, day):
    i = meta[(meta["motor_id"] == motor_id) & (meta["day"] == day)].index
    if len(i) == 0:
        return None, None
    spec = np.abs(np.fft.rfft(waves[i[0]][:, 0])) / (waves.shape[1] / 2)
    return freqs, spec


fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
healthy_ids = feat[feat["true_fault"] == "healthy"]["motor_id"].unique()
faulty_ids = [m for m in feat["motor_id"].unique() if m not in set(healthy_ids)]
examples = [(healthy_ids[0], "Healthy"), (faulty_ids[0], "Faulty"),
            (faulty_ids[1] if len(faulty_ids) > 1 else faulty_ids[0], "Faulty")]
for ax, (mid, kind) in zip(axes, examples):
    day = int(feat[feat["motor_id"] == mid]["day"].max())
    f, spec = spectrum(mid, day)
    if f is None:
        continue
    sel = f <= 150
    ax.plot(f[sel], spec[sel], linewidth=0.9, color="#2563eb")
    ax.set_title(f"{kind} ({mid}, day {day})")
    ax.set_ylabel("Current (A)")
axes[-1].set_xlabel("Frequency (Hz)")
plt.tight_layout()
plt.savefig(f"{out}/02_example_spectra.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(8, 4))
order = ["Healthy", "Watch", "Alert", "Critical"]
counts = df.sort_values("day").groupby("motor_id").tail(1)["status"].value_counts()
ax.bar([s for s in order if s in counts.index],
       [counts[s] for s in order if s in counts.index], color="#16a34a")
ax.set_title("Fleet Status Today (latest reading per motor)")
ax.set_ylabel("Motors")
plt.tight_layout()
plt.savefig(f"{out}/03_fleet_status.png", dpi=120)
plt.close()

pivot = feat.pivot_table(index="motor_id",
                         values=["sideband_ratio", "bearing_band", "ecc_ratio", "unbalance_pct"],
                         aggfunc="max")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
ax.set_title("Peak Fault-Indicator Values per Motor (45 days)")
plt.tight_layout()
plt.savefig(f"{out}/04_indicator_heatmap.png", dpi=120)
plt.close()

print("Saved 4 EDA plots to outputs/")
