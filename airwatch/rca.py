"""Stage 2: root-cause analysis for every flagged or unhealthy station.

Question asked of each one: do the neighbours agree? A real event moves a whole area;
a broken sensor moves alone. Rules fire first because they are unambiguous (PM2.5 above
PM10 cannot happen physically). Neighbour comparison decides the rest.

Confidence numbers are heuristic scores (more independent evidence -> higher), not
calibrated probabilities.
"""
import warnings

import numpy as np

from .anomaly import W

ACTIONS = {
    "sensor_fault": "Quarantine from city AQI and forecasts; send a technician to compare against a reference monitor.",
    "genuine_event": "Treat as real: advisory for the affected area, look for local sources (burning, construction, traffic), watch the forecast.",
    "data_pipeline": "Check power, connectivity and the data logger. Readings themselves are not suspect yet.",
    "watch": "No clear cause. Re-check in 3 hours.",
}
PRIORITY = {"sensor_fault": 0, "genuine_event": 1, "data_pipeline": 2, "watch": 3}


def _rec_base(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(x[..., -3:], axis=-1), np.nanmedian(x[..., -48:-W], axis=-1)


def investigate(net, det, health):
    out = []
    for i, sid in enumerate(net.ids):
        d, h = det[i], health[i]
        if not d["flagged"] and h["status"] == "ok":
            continue
        rec, base = _rec_base(net.pm25[i])
        nrec, nbase = _rec_base(net.pm25[net.nbrs[i]])
        ok = np.isfinite(nrec) & np.isfinite(nbase) & (nbase > 0)
        rise = rec / base if np.isfinite(rec) and base > 0 else np.nan
        up = int((nrec[ok] / nbase[ok] > 1.4).sum())
        frac = up / ok.sum() if ok.any() else 0.0
        ev = []
        if np.isfinite(rec) and ok.any():
            ev.append(f"PM2.5 {rec:.0f} ug/m3 over last 3h vs neighbour median {np.median(nrec[ok]):.0f}")
            ev.append(f"{up} of {int(ok.sum())} neighbours up >40% on their own baseline")

        if d["rules"]:
            verdict, conf = "sensor_fault", 0.9
            ev.insert(0, "physical/logic rule broken: " + ", ".join(d["rules"]))
        elif h["status"] == "offline":
            verdict, conf = "data_pipeline", 0.8
            ev.insert(0, f"no report for {h['hours_since_report']}h")
        elif rise > 1.5:
            if frac >= 0.4:
                verdict, conf = "genuine_event", 0.5 + 0.5 * frac
                ev.insert(0, f"station rose {rise:.1f}x and the surrounding stations moved with it")
            else:
                verdict, conf = "sensor_fault", 0.6
                ev.insert(0, f"station rose {rise:.1f}x but neighbours did not")
        elif h["status"] == "degraded":
            verdict, conf = "data_pipeline", 0.6
        else:
            verdict, conf = "watch", 0.3
        if h["status"] != "ok" and verdict != "data_pipeline":
            ev.append(f"reporting irregular: {h['reports_24h']}/24 hourly reports in the last day")
            conf += 0.05
        elif h["status"] == "degraded":
            ev.append(f"only {h['reports_24h']}/24 hourly reports in the last day")
        if d["iso_prior"] > 0 and verdict == "sensor_fault":
            ev.append(f"{d['iso_prior']} other unusual readings in the 42h before")
            conf += 0.05
        out.append(dict(id=sid, index=i, verdict=verdict, confidence=round(min(conf, 0.99), 2),
                        evidence=ev, action=ACTIONS[verdict], status=h["status"]))
    out.sort(key=lambda r: (PRIORITY[r["verdict"]], -r["confidence"]))
    return out
