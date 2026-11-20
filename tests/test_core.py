import pytest

from airwatch.aqi import aqi_pm25, category
from airwatch.assistant import route
from airwatch.engine import Engine


@pytest.fixture(scope="module")
def e():
    return Engine(7)


def test_aqi_breakpoints():
    assert [aqi_pm25(c) for c in (0, 30, 60, 90, 120, 250)] == [0, 50, 100, 200, 300, 400]
    assert category(aqi_pm25(75)) == "Moderate"


def test_faults_and_outages_are_diagnosed(e):
    v = {r["id"]: r["verdict"] for r in e.incidents}
    assert all(v[i] == "sensor_fault" for i in (104, 112, 134))
    assert all(v[i] == "data_pipeline" for i in (121, 141))


def test_real_event_is_not_called_a_fault(e):
    plume = [i for i, label in e.net.truth.items() if label == "genuine_event"]
    hits = sum(e.verdict.get(i, {}).get("verdict") == "genuine_event" for i in plume)
    assert hits >= len(plume) - 1  # plume-edge stations can look isolated; see README


def test_no_alarms_on_healthy_stations(e):
    assert all(r["id"] in e.net.truth for r in e.incidents if r["verdict"] != "watch")


def test_spike_station_evidence_cites_neighbours(e):
    assert any("neighbours" in line for line in e.verdict[104]["evidence"])


def test_quarantined_station_does_not_poison_its_forecast(e):
    row = next(r for r in e.forecast if r["id"] == 104)
    assert row["quarantined"] and row["pm25_6h"] < 150


def test_short_horizon_beats_persistence(e):
    m = e.metrics[6]
    assert m["mae"] < m["persistence_mae"] and m["skilled"]


@pytest.mark.parametrize("q,intent", [
    ("Why did AQI suddenly increase?", "why"),
    ("Which monitoring stations are offline?", "offline"),
    ("Which sensors are producing suspicious readings?", "suspicious"),
    ("What should the maintenance team investigate?", "maintenance"),
    ("hello", "help")])
def test_routing(q, intent):
    assert route(q) == intent


def test_assistant_uses_no_llm_by_default(e):
    r = e.ask("Which sensors are suspicious?")
    assert not r["llm_used"] and "104" in r["answer"]


def test_api():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from airwatch.app import app
    with TestClient(app) as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/api/summary").json()["stations"] == 50
        assert c.get("/api/stations/999/series").status_code == 404
        assert len(c.get("/api/stations/104/series?hours=24").json()["pm25"]) == 24
        assert "104" in c.post("/api/ask", json={"question": "suspicious sensors?"}).json()["answer"]
        assert c.post("/api/ask", json={"question": ""}).status_code == 422
        assert "airwatch_city_aqi" in c.get("/metrics").text
        assert "AirWatch" in c.get("/").text
