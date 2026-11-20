"""PM2.5 forecast at +6h and +24h: one gradient-boosted model per horizon, shared by all stations.

Chosen over XGBoost/LSTM on purpose: HistGradientBoosting handles NaN (missing reports)
natively, trains in seconds, and keeps the Render image small. Every model is scored
against two naive baselines, because a forecaster that can't beat 'same as now' is noise.
"""
import warnings
from numpy.lib.stride_tricks import sliding_window_view as swv

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .anomaly import neighbour_median
from .aqi import aqi_pm25, category, level

HORIZONS = (6, 24)
# Small, heavily regularised trees: with ~16 days of history a bigger model fits noise and
# lost to 'same as now' at +24h in every config we tried.
PARAMS = dict(max_iter=60, learning_rate=0.05, max_depth=3, min_samples_leaf=300,
              l2_regularization=5.0, random_state=0)
LAGS = (0, 1, 2, 3, 6, 12, 24)


def _roll_mean(x, w):
    pad = np.full((x.shape[0], w - 1), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(swv(np.hstack([pad, x]), w, axis=1), axis=2)


def _X(pm, nb, temp, hum, t):
    """Returns (features, base). Features are divided by the trailing 24h mean and the model
    predicts target/base, i.e. a correction to 'same as now'. Trees can't extrapolate to a new
    pollution level, and our first two versions lost to 'same as now' at +24h for that reason."""
    n = pm.shape[0]
    r6, scale = _roll_mean(pm, 6), _roll_mean(pm, 24)[:, t]
    hr = np.broadcast_to(2 * np.pi * (t % 24) / 24, (n, len(t)))
    cols = [pm[:, t - L] / scale for L in LAGS] + [r6[:, t] / scale, nb[:, t] / scale, np.log(scale),
                                                   temp[:, t], hum[:, t], np.sin(hr), np.cos(hr)]
    base = np.where(np.isfinite(pm[:, t]), pm[:, t], scale)  # 'same as now', the thing to beat
    return np.stack(cols, axis=-1).reshape(-1, len(cols)), base.reshape(-1)  # station-major rows


def fit(net):
    """Train < T-96h, validate T-96h..T-48h (clean), never touch the final 48h of incidents."""
    pm, T = net.pm25, net.T
    nb = neighbour_median(net, pm)
    models, metrics = {}, {}
    for h in HORIZONS:
        t = np.arange(24, T - h)
        tr, va = t[t + h < T - 96], t[(t + h >= T - 96) & (t + h < T - 48)]
        Xtr, btr = _X(pm, nb, net.temp, net.hum, tr)
        ytr = pm[:, tr + h].reshape(-1) / btr
        m = np.isfinite(ytr)
        model = HistGradientBoostingRegressor(**PARAMS)
        model.fit(Xtr[m], ytr[m])
        (Xva, bva), yva = _X(pm, nb, net.temp, net.hum, va), pm[:, va + h].reshape(-1)
        seasonal = pm[:, va + h - 24].reshape(-1)
        m = np.isfinite(yva) & np.isfinite(Xva[:, 0]) & np.isfinite(seasonal)
        mae = lambda p: round(float(np.abs(p - yva[m]).mean()), 2)  # noqa: E731
        metrics[h] = dict(mae=mae(model.predict(Xva[m]) * bva[m]), persistence_mae=mae(bva[m]),
                          seasonal_naive_mae=mae(seasonal[m]))
        metrics[h]['skilled'] = metrics[h]['mae'] < metrics[h]['persistence_mae']
        models[h] = model
    return models, metrics


def predict(net, models, quarantine, metrics):
    """Quarantined stations (faulty/offline) get their history replaced by the neighbour
    median, so one broken sensor can't poison its own forecast."""
    pm = net.pm25.copy()
    nb = neighbour_median(net, pm)
    for i in quarantine:
        pm[i] = nb[i]
    X, base = _X(pm, nb, net.temp, net.hum, np.array([net.T - 1]))
    pred = {h: np.clip(m.predict(X) * base, 0, None) for h, m in models.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        now = np.nanmean(pm[:, -24:], axis=1)
    rows, warns = [], []
    for i, sid in enumerate(net.ids):
        a_now = aqi_pm25(float(now[i]))
        row = dict(id=sid, pm25_24h_mean=round(float(now[i]), 1), aqi_now=round(a_now),
                   quarantined=i in quarantine)
        for h in HORIZONS:
            a = aqi_pm25(float(pred[h][i]))
            row[f"pm25_{h}h"] = round(float(pred[h][i]), 1)
            row[f"aqi_{h}h"] = round(a)
            # Warn on a worse band than now, or anything Poor+. Compares instant PM2.5 mapped
            # through NAQI breakpoints with a 24h mean: indicative, not an official AQI.
            if metrics[h]["skilled"] and (a > 200 or level(a) > level(a_now)):
                warns.append(dict(id=sid, horizon_h=h, pm25=row[f"pm25_{h}h"], aqi=round(a),
                                  category=category(a), quarantined=i in quarantine))
        rows.append(row)
    warns.sort(key=lambda w: -w["aqi"])
    return rows, warns
