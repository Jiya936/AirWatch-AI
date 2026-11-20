"""Operations assistant, built to spend as few tokens as possible.

Default path uses zero LLM tokens: the question is routed by keyword to a function that reads
live engine state and returns a factual answer. If ANTHROPIC_API_KEY is set and the caller asks
for it (llm=true), the answer text - already capped at ~900 chars - is sent to a small model with
max_tokens=160 purely to make it read smoothly. Identical facts are cached, and any API failure
falls back to the plain answer. The model never sees raw data and can't add facts.
"""
import json
import os
import re
import urllib.request

INTENTS = [
    ("offline", r"offline|not (sending|reporting|transmitting)|stopped|\bdown\b|silent|missing data|outage"),
    ("maintenance", r"maintenance|inspect|investigate|fix|dispatch|priorit|to-?do|crew"),
    ("suspicious", r"suspicious|faulty|malfunction|broken|bad sensor|wrong|stuck|unreliable"),
    ("forecast", r"forecast|predict|tomorrow|next|warning|later|ahead"),
    ("why", r"why|increase|spike|rise|risen|jump|worse|deteriorat|spiked"),
]
_CACHE: dict = {}
HELP = ("I can answer: why AQI rose, which stations are offline, which sensors look faulty, "
        "what maintenance should check first, and what the forecast warns about.")


def _s(ids):
    return ", ".join(str(i) for i in ids) or "none"


def _facts(e, intent):
    inc = e.incidents
    by = lambda v: [r for r in inc if r["verdict"] == v]  # noqa: E731
    if intent == "offline":
        bad = [h for h in e.health if h["status"] != "ok"]
        if not bad:
            return "All stations are reporting normally."
        return " ".join(f"Station {h['id']}: {h['status']}, last report {h['hours_since_report']}h ago, "
                        f"{h['reports_24h']}/24 reports in the last day." for h in bad)
    if intent == "suspicious":
        bad = by("sensor_fault")
        if not bad:
            return "No sensor currently looks faulty."
        return " ".join(f"Station {r['id']} ({r['confidence']:.2f}): {'; '.join(r['evidence'][:3])}." for r in bad)
    if intent == "maintenance":
        if not inc:
            return "Nothing needs inspection right now."
        return " ".join(f"{k}. Station {r['id']} - {r['verdict'].replace('_', ' ')}: {r['action']}"
                        for k, r in enumerate([r for r in inc if r["verdict"] != "watch"][:5], 1))
    if intent == "forecast":
        w = e.warnings[:5]
        if not w:
            return "No station is forecast to move into a worse AQI band."
        s = " ".join(f"Station {x['id']} +{x['horizon_h']}h: PM2.5 {x['pm25']} ({x['category']})"
                     f"{' [data untrusted]' if x['quarantined'] else ''}." for x in w)
        m = e.metrics[6]
        return f"{s} Model MAE at +6h is {m['mae']} vs {m['persistence_mae']} for 'same as now'."
    if intent == "why":
        ch = 100 * (e.city_now / e.city_prev - 1)
        ev, fl = by("genuine_event"), by("sensor_fault")
        s = f"City PM2.5 is {ch:+.0f}% vs the previous 24h (faulty and offline stations excluded)."
        if ev:
            s += (f" Stations {_s(r['id'] for r in ev)} rose sharply and their neighbours rose too, "
                  "so this looks like a real local event.")
        if fl:
            s += (f" Stations {_s(r['id'] for r in fl)} show spikes or impossible values that "
                  "neighbours don't confirm - treat those as sensor problems, not pollution.")
        return s
    return HELP


def _llm(text):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    text = text[:900]
    if text in _CACHE:
        return _CACHE[text]
    body = json.dumps({
        "model": os.getenv("LLM_MODEL", "claude-haiku-5-5"), "max_tokens": 160,
        "system": "You brief an air-quality duty officer. Rewrite the facts as a short plain answer. "
                  "Use only the facts given, keep every number and station ID, say so if unsure.",
        "messages": [{"role": "user", "content": text}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", body, {
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            _CACHE[text] = json.load(r)["content"][0]["text"]
        return _CACHE[text]
    except Exception:  # network, auth, quota: fall back silently, answer is still correct
        return None


def route(q):
    q = q.lower()
    return next((name for name, pat in INTENTS if re.search(pat, q)), "help")


def answer(e, q, use_llm=False):
    intent = route(q)
    facts = _facts(e, intent)
    polished = _llm(facts) if use_llm and intent != "help" else None
    return dict(intent=intent, answer=polished or facts, llm_used=bool(polished))
