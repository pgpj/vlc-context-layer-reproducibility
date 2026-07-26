from pathlib import Path
import json
import pandas as pd

root = Path(__file__).resolve().parent
required = [
    root / "config" / "fixed_protocol.json",
    root / "results" / "per_seed_all_conditions.csv",
    root / "results" / "aggregate_all_conditions.csv",
    root / "results" / "snr_sweep_aggregate.csv",
    root / "results" / "proposed_overhead_summary.json",
]
missing = [str(p.relative_to(root)) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"Missing required files: {missing}")

with open(root / "config" / "fixed_protocol.json", encoding="utf-8") as f:
    config = json.load(f)
per_seed = pd.read_csv(root / "results" / "per_seed_all_conditions.csv")
aggregate = pd.read_csv(root / "results" / "aggregate_all_conditions.csv")
print("Archive verification passed")
print(f"Configured seeds: {config.get('seeds', 'see configuration')}")
print(f"Per-seed rows: {len(per_seed)}")
print(f"Aggregate rows: {len(aggregate)}")
