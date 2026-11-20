"""Wires the stages together once at startup; the API only reads from this object."""
import warnings

import numpy as np

from . import anomaly, forecast, health, rca
from .aqi import aqi_pm25, category
from .assistant import answer
from .simulator import make_network


def _clean(arr):
    return [None if not np.isfinite(v) else round(float(v), 1) for v in arr]


class Engine:
    def __init__(self, seed=7):
        net = self.net = make_network(seed=seed)
        self.iso = anomaly.fit(net, net.T - 72)
        self.det = anomaly.detect(net, self.iso)
        self.health = health.station_health(net)
        self.incidents = rca.investigate(net, self.det, self.health)
        self.verdict = {r["id"]: r for r in self.incidents}
        # Stations whose numbers we do not trust. Offline ones are stale, faulty ones are wrong.
        self.quarantine = {r["index"] for r in self.incidents
                           if r["verdict"] == "sensor_fault" or r["status"] == "offline"}
        models, self.metrics = forecast.fit(net)
        self.forecast, self.warnings = forecast.predict(net, models, self.quarantine, self.metrics)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            good = [i for i in range(len(net.ids)) if i not in self.quarantine]
            self.city_now = float(np.nanmedian(np.nanmean(net.pm25[good, -24:], axis=1)))
            self.city_prev = float(np.nanmedian(np.nanmean(net.pm25[good, -48:-24], axis=1)))

    # --- views -------------------------------------------------------------------------
    def summary(self):
        a = aqi_pm25(self.city_now)
        counts = {s: sum(h["status"] == s for h in self.health) for s in ("ok", "degraded", "offline")}
        verdicts = {}
        for r in self.incidents:
            verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
        return dict(city_aqi=round(a), category=category(a),
                    pm25_change_vs_prior_24h_pct=round(100 * (self.city_now / self.city_prev - 1), 1),
                    stations=len(self.net.ids), station_status=counts, incident_verdicts=verdicts,
                    quarantined=len(self.quarantine), forecast_mae=self.metrics,
                    note="Simulated network; AQI is PM2.5-only NAQI.")

    def stations(self):
        net, out = self.net, []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean24 = np.nanmean(net.pm25[:, -24:], axis=1)
        for i, sid in enumerate(net.ids):
            last = net.pm25[i][np.isfinite(net.pm25[i])]
            a = aqi_pm25(float(mean24[i]))
            out.append(dict(id=sid, lat=round(float(net.lat[i]), 4), lon=round(float(net.lon[i]), 4),
                            status=self.health[i]["status"],
                            pm25_latest=round(float(last[-1]), 1) if len(last) else None,
                            aqi_24h=None if np.isnan(a) else round(a), category=category(a),
                            verdict=self.verdict.get(sid, {}).get("verdict")))
        return out

    def series(self, sid, hours=48):
        if sid not in self.net.ids:
            return None
        i, hours = self.net.ids.index(sid), max(6, min(hours, 336))
        nb = anomaly.neighbour_median(self.net, self.net.pm25)[i, -hours:]
        return dict(id=sid, hours_ago=list(range(hours - 1, -1, -1)), pm25=_clean(self.net.pm25[i, -hours:]),
                    pm10=_clean(self.net.pm10[i, -hours:]), neighbour_median_pm25=_clean(nb))

    def infra(self, api):
        return dict(stations=self.health, server=health.server_stats(api))

    def ask(self, q, llm=False):
        return answer(self, q, llm)

    def metrics_text(self, api):
        s, srv = self.summary(), health.server_stats(api)
        lines = [f'airwatch_stations{{status="{k}"}} {v}' for k, v in s["station_status"].items()]
        lines += [f'airwatch_incidents{{verdict="{k}"}} {v}' for k, v in s["incident_verdicts"].items()]
        lines += [f'airwatch_forecast_mae{{horizon="{h}"}} {m["mae"]}' for h, m in self.metrics.items()]
        lines += [f"airwatch_city_aqi {s['city_aqi']}", f"airwatch_process_peak_rss_mb {srv['peak_rss_mb']}",
                  f"airwatch_api_requests_total {api['requests']}", f"airwatch_api_errors_total {api['errors']}",
                  f"airwatch_api_latency_p95_seconds {api['p95_s']}"]
        return "\n".join(lines) + "\n"
