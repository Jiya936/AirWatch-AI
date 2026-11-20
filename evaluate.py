"""Score the detector against the simulator's injected labels across seeds: python evaluate.py"""
from airwatch.engine import Engine

tot = ok = fps = 0
print("seed  correct   false_alarms  +6h MAE (naive)   +24h MAE (naive)")
for seed in range(1, 9):
    e = Engine(seed)
    got = {r["id"]: r["verdict"] for r in e.incidents}
    good = sum(got.get(i) == v for i, v in e.net.truth.items())
    fp = sum(1 for i, v in got.items() if i not in e.net.truth and v != "watch")
    m = e.metrics
    print(f"{seed:>4}  {good:>2}/{len(e.net.truth):<5} {fp:>8}       {m[6]['mae']:>5} ({m[6]['persistence_mae']})"
          f"      {m[24]['mae']:>5} ({m[24]['persistence_mae']})")
    tot, ok, fps = tot + len(e.net.truth), ok + good, fps + fp
print(f"total {ok}/{tot} correct, {fps} false alarms")
