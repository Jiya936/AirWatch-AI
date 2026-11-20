# AirWatch AI

A small FastAPI service that watches an air-quality monitoring network and answers one question: **when a station reports a spike, is it pollution or is the sensor broken?**

It runs on a simulated 50-station network (Delhi-area coordinates) where every fault is injected and labelled, so the detector can be scored rather than just demoed. It is not connected to live CPCB data, and the dashboard's clock does not advance.

## What it does

| Stage | Method | File |
|---|---|---|
| Flag suspicious readings | Three rules (impossible values, PM2.5 > PM10, flat-lined sensor), Isolation Forest on four log-ratios, and a 1.6× surge check | `anomaly.py` |
| Check the network itself | Time since last report, reports in the last 24h, process memory and API latency | `health.py` |
| Decide the cause | Compare the station to its 5 nearest neighbours: a real event moves an area, a broken sensor moves alone | `rca.py` |
| Forecast PM2.5 at +6h and +24h | Gradient-boosted trees predicting a correction to "same as now" | `forecast.py` |
| Answer operator questions | Keyword routing over live state; optional LLM rewrite | `assistant.py` |

The two-stage split is deliberate. A genuine plume looks anomalous to any single-station model, so stage 1 only says "look here" and stage 2 rules on it. Stations judged faulty or offline are quarantined: they are dropped from the city AQI and their history is replaced by the neighbour median before forecasting, so one bad sensor can't poison its own forecast.

Example from the running app (station 104, simulated): PM2.5 322 µg/m³ over the last 3 hours against a neighbour median of 41, with only 1 of 5 neighbours up. Its reporting has also been irregular (17 of 24 hourly reports). Verdict: `sensor_fault`, confidence 0.65. Action: quarantine it and compare against a reference monitor.

## Run it

```bash
pip install -r requirements-dev.txt
python -m pytest -q                      # 14 tests; the API test needs fastapi + httpx
python evaluate.py                       # scores the detector over 8 seeds
uvicorn airwatch.app:app --reload        # http://localhost:8000
```

Startup takes a few seconds (simulate, fit, forecast, investigate). All state lives in memory, so there is no database to run.

## API

| Endpoint | Returns |
|---|---|
| `GET /` | Dashboard: map, incidents with evidence, early warnings, assistant box |
| `GET /api/summary` | City AQI, station status counts, incident counts, forecast scores |
| `GET /api/stations`, `/api/stations/{id}/series?hours=48` | Per-station status and AQI; PM2.5/PM10 against the neighbour median |
| `GET /api/incidents` | Verdict, confidence, evidence and recommended action per station |
| `GET /api/forecast` | Warnings, per-station forecasts, validation scores |
| `GET /api/infra` | Station health plus server memory, load and API latency |
| `POST /api/ask` | `{"question": "...", "llm": false}` |
| `GET /metrics` | Prometheus text format, hand-written (no client library) |
| `GET /healthz` | Render health check |

## Results on the simulator

`python evaluate.py`, 8 random networks, 4–6 injected plume stations plus 5 faults each:

- **80 of 84** injected problems diagnosed correctly, **0 false alarms** on healthy stations.
- The 4 misses are stations at the edge of a plume. Their neighbours are mostly outside it, so they look isolated and get called `sensor_fault`. Real plumes have edges too; expect this failure on real data.
- **+6h forecast:** MAE 4–11 µg/m³ against 12–26 for "same as now", better on every seed.
- **+24h forecast: no reliable skill.** It beat the naive baseline on only 3 of 8 seeds and lost on the rest. The simulator's pollution level drifts like a random walk, so past the daily cycle there is little to learn. The app marks that horizon as unskilled and excludes it from warnings.

These numbers measure the code against *my* simulator. They say nothing about CPCB data, where faults are messier. The simulated faults are crisp by construction, so read 80/84 as "the logic works", not as an accuracy claim.

## How little it spends

- **Default: zero LLM tokens.** The assistant routes by keyword and answers from live state. The model, ML pipeline and rules do the work; an LLM is the least reliable component for arithmetic over sensor data.
- **Optional polish:** set `ANTHROPIC_API_KEY` and send `"llm": true`. The already-computed answer (capped at 900 characters) goes to `claude-haiku-5-5` with `max_tokens=160`. Identical facts are cached, and any API error falls back to the plain answer. The model never sees raw readings and is told to add no facts. Override with `LLM_MODEL`.
- No XGBoost, PyTorch or pandas in the image: numpy and scikit-learn only, to fit Render's free 512 MB tier and keep builds fast.

## Deploy to Render

1. Push this folder to a GitHub repository (below).
2. In Render choose **New → Blueprint**, select the repo. `render.yaml` sets the build, start command, Python 3.12.8 and the health check.
3. Optionally add `ANTHROPIC_API_KEY` in the service's environment settings.

Free instances sleep after inactivity, and the first request after a sleep waits for the app to start again.

## Push to GitHub

```bash
git remote add origin https://github.com/<you>/airwatch-ai.git
git push -u origin main
```

## What differs from the original brief, and why

| Brief | Here | Reason |
|---|---|---|
| PostgreSQL | In-memory | Simulated data regenerates deterministically; a database adds ops work with nothing to persist. |
| React | One static HTML file | No build step to break on deploy. |
| Prometheus + Grafana | `/metrics` endpoint only | Scrapeable by either; running both is outside a single Render service. |
| XGBoost / LSTM | `HistGradientBoostingRegressor` | Handles missing reports natively, trains in seconds, no extra dependency. An LSTM on 16 days of history would overfit. |
| DBSCAN, Random Forest failure prediction | Not built | The injected failures have no real precursors, so a failure-prediction model would only learn my simulator's rules. |
| RAG over documents | Routing over live state | There is no document corpus yet. Adding one means embeddings, which this version avoids. |
| Real CPCB / OpenAQ data | Not connected | Fill the `Network` arrays in `simulator.py` from a loader; nothing downstream changes. |

## Known limits

- AQI is PM2.5-only and mapped through NAQI breakpoints. Official NAQI uses 24h averages and the worst pollutant. Forecast bands compare an instantaneous value to a 24h mean, so treat them as indicative.
- Confidence values are heuristic scores that rise with independent evidence. They are not calibrated probabilities.
- Isolation Forest trains on history assumed to be mostly clean. On real data you would need to mask known-bad periods first.
- The surge check uses the station's own 42-hour median, so a slow event over several days won't trigger it.
- Hygroscopic growth can inflate PM2.5 on low-cost optical sensors at high humidity; humidity is not yet used in the verdict.
