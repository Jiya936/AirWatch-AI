"""Stage 1 of fault finding: cheap rules + Isolation Forest. Flags *suspicious* readings only.

Deciding whether a flagged reading is a broken sensor or a real event is stage 2 (rca.py).
Keeping them apart matters: a genuine plume looks anomalous to any single-station model.
"""
import warnings
from numpy.lib.stride_tricks import sliding_window_view as swv

import numpy as np
from sklearn.ensemble import IsolationForest

W = 6  # "recent" window in hours


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def neighbour_median(net, x):
    return _quiet(np.nanmedian, x[net.nbrs], axis=1)  # [n,k,T] -> [n,T]


def _roll_median(x, w=24):
    pad = np.full((x.shape[0], w - 1), np.nan)
    return _quiet(np.nanmedian, swv(np.hstack([pad, x]), w, axis=1), axis=2)


def features(net):
    """Four log-ratios, so 'twice the usual' looks the same at 40 and at 200 ug/m3:
    vs neighbours, PM2.5/PM10 (should be <1), vs own 24h median, hour-on-hour jump."""
    pm = net.pm25
    nb = neighbour_median(net, pm)
    jump = np.abs(np.diff(pm, axis=1, prepend=np.nan))
    with np.errstate(all="ignore"):
        return np.stack([np.log(pm / nb), np.log(pm / net.pm10),
                         np.log(pm / _roll_median(pm)), np.log1p(jump / nb)], axis=-1)


def fit(net, train_end):
    """Train on history before the incident window. contamination=0.5% because the
    training data is assumed mostly clean - this is a simulation-time assumption."""
    F = features(net)[:, 24:train_end].reshape(-1, 4)
    F = F[np.isfinite(F).all(1)]
    return IsolationForest(n_estimators=100, contamination=0.005, random_state=0).fit(F)


def detect(net, iso):
    F = features(net)
    ok = np.isfinite(F).all(-1)
    hit = np.zeros(ok.shape, bool)
    hit[ok] = iso.predict(F[ok]) == -1
    out = []
    for i in range(len(net.ids)):
        p25, p10 = net.pm25[i, -W:], net.pm10[i, -W:]
        v = p25[np.isfinite(p25)]
        rules = []
        if ((v < 0) | (v > 1000)).any():
            rules.append("out_of_range")
        if (p25 > 1.05 * p10).sum() >= 3:  # NaN compares False, so gaps don't count
            rules.append("pm25_exceeds_pm10")
        if len(v) >= 5 and np.ptp(v) < 1e-6:
            rules.append("stuck_value")
        recent, prior = int(hit[i, -W:].sum()), int(hit[i, -48:-W].sum())
        # Surge check: Isolation Forest only sees odd *combinations*, and a slow area-wide rise
        # isn't one. A 1.6x jump (last-3h mean; the 6h mean hides a ramp behind the evening dip) over the station's own 42h median is flagged for RCA to rule on.
        base = _quiet(np.nanmedian, net.pm25[i, -48:-W])
        surge = bool(len(v) and base > 0 and v[-3:].mean() / base >= 1.6)
        out.append(dict(rules=rules, iso_recent=recent, iso_prior=prior, surge=surge,
                        flagged=bool(rules or recent >= 2 or surge)))
    return out
