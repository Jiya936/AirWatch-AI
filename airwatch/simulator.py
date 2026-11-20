"""Synthetic 50-station network with labelled faults.

Why a simulator: the point of the project is telling a pollution event from a broken sensor,
and real feeds don't come with ground truth for that. Here every injected problem is
labelled, so the detector can be scored. Swap `make_network` for a loader (CPCB / OpenAQ)
that fills the same `Network` arrays to run on real data - nothing downstream changes.

The last hour of every array is "now". Hourly cadence, 21 days.
"""
from dataclasses import dataclass

import numpy as np

# Station indexes (id = 101 + index) that receive a fault in the final 48h.
SPIKE, STUCK, OFFLINE, IMPLAUSIBLE, SPARSE = 3, 11, 20, 33, 40


@dataclass
class Network:
    ids: list
    lat: np.ndarray
    lon: np.ndarray
    pm25: np.ndarray   # [n, T], NaN = no report
    pm10: np.ndarray
    no2: np.ndarray
    temp: np.ndarray
    hum: np.ndarray
    nbrs: np.ndarray   # [n, k] nearest-station indexes
    truth: dict        # station id -> injected label

    @property
    def T(self) -> int:
        return self.pm25.shape[1]


def make_network(n=50, days=21, seed=7, k=5) -> Network:
    rng = np.random.default_rng(seed)
    T = days * 24
    hod = np.arange(T) % 24
    lat = 28.40 + 0.40 * rng.random(n)
    lon = 77.00 + 0.40 * rng.random(n)
    d = np.hypot(lat[:, None] - lat[None], (lon[:, None] - lon[None]) * 0.88)  # cos(28.6deg)
    np.fill_diagonal(d, np.inf)
    nbrs = np.argsort(d, axis=1)[:, :k]

    # City-wide level: slow multi-day episodes (AR(1)) x daily double peak. All stations
    # share it, which is what makes neighbour comparison informative.
    ar = np.zeros(T)
    for t in range(1, T):
        ar[t] = 0.97 * ar[t - 1] + 0.05 * rng.standard_normal()
    diurnal = 1 + 0.25 * np.sin(2 * np.pi * (hod - 6) / 24) + 0.15 * np.sin(4 * np.pi * (hod - 3) / 24)
    f = np.clip(1 + 0.8 * (lat - 28.6) + 0.08 * rng.standard_normal(n), 0.6, None)

    noise = lambda s: rng.standard_normal((n, T)) * s  # noqa: E731
    pm25 = 70 * np.exp(ar) * diurnal * f[:, None] * np.exp(noise(0.06))
    pm10 = pm25 * 1.7 * np.exp(noise(0.05)) + 10          # PM10 contains PM2.5, so it is always larger
    no2 = 25 + 0.25 * pm25 + noise(3)
    wave = np.sin(2 * np.pi * (hod - 9) / 24)
    temp = 24 + 8 * wave + noise(0.5)
    hum = np.clip(60 - 20 * wave + noise(2), 20, 100)
    arrays = (pm25, pm10, no2, temp, hum)
    ids = [101 + i for i in range(n)]
    truth = {}

    def blank(i, cols):
        for a in arrays:
            a[i, cols] = np.nan

    # 1. Sensor spike: neighbours stay normal, earlier glitches, irregular reporting.
    pm25[SPIKE, -3:] = 320 + 5 * rng.standard_normal(3)
    pm10[SPIKE, -3:] = 380 + 5 * rng.standard_normal(3)
    pm25[SPIKE, [-20, -14]] *= 2.5
    blank(SPIKE, np.arange(T - 30, T - 3)[np.arange(T - 30, T - 3) % 3 == 0])
    truth[ids[SPIKE]] = "sensor_fault"
    # 2. Stuck: last good value repeated for 12h.
    pm25[STUCK, -12:] = pm25[STUCK, -13]
    pm10[STUCK, -12:] = pm10[STUCK, -13]
    truth[ids[STUCK]] = "sensor_fault"
    # 3. Offline for 9h.
    blank(OFFLINE, slice(T - 9, T))
    truth[ids[OFFLINE]] = "data_pipeline"
    # 4. Physically impossible: PM2.5 above PM10.
    pm25[IMPLAUSIBLE, -6:] = pm10[IMPLAUSIBLE, -6:] * 1.6
    truth[ids[IMPLAUSIBLE]] = "sensor_fault"
    # 5. Reports only every 3rd hour for 24h.
    t24 = np.arange(T - 24, T)
    blank(SPARSE, t24[t24 % 3 != 0])
    truth[ids[SPARSE]] = "data_pipeline"
    # 6. Real event: a 6-station cluster far from the spike station gains up to +130 ug/m3 over 6h.
    #    Additive, not a multiplier: a plume adds pollution, and x2.6 of an evening low is invisible.
    faulty = {SPIKE, STUCK, OFFLINE, IMPLAUSIBLE, SPARSE}
    dist = np.where(np.isinf(d[SPIKE]), -1, d[SPIKE])
    dist[list(faulty)] = -1  # the plume centre must not be an already-faulty station
    far = dist.argmax()
    cluster = [far] + [j for j in nbrs[far] if j not in faulty]
    cluster = cluster[:6]
    ramp = np.clip((np.arange(T) - (T - 6)) / 5, 0, 1)
    for j in cluster:
        pm25[j] += 130 * ramp
        pm10[j] += 200 * ramp
        no2[j] += 20 * ramp
        truth[ids[j]] = "genuine_event"

    # ~1% random dropouts in older data (recent window left clean so labels stay exact).
    miss = rng.random((n, T)) < 0.01
    miss[:, T - 12:] = False
    for a in arrays:
        a[miss] = np.nan
    return Network(ids, lat, lon, pm25, pm10, no2, temp, hum, nbrs, truth)
