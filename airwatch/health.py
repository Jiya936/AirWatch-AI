"""AIOps layer: is the *monitoring system* healthy, independent of what the air looks like?"""
import os
import resource

import numpy as np


def station_health(net, offline_after=3, min_rate=0.8):
    ok = np.isfinite(net.pm25)
    rows = []
    for i, sid in enumerate(net.ids):
        seen = np.flatnonzero(ok[i])
        silent = int(net.T - 1 - seen[-1]) if len(seen) else 999
        got = int(ok[i, -24:].sum())
        status = "offline" if silent >= offline_after else "degraded" if got < 24 * min_rate else "ok"
        rows.append(dict(id=sid, status=status, hours_since_report=silent, reports_24h=got))
    return rows


def server_stats(api):
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB on Linux (peak RSS)
    return dict(peak_rss_mb=round(rss_mb, 1), load_1m=load, **api)
