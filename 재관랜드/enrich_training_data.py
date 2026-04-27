from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


CACHE_DIR = Path(".cache")
TRAINING_DATA_PATH = CACHE_DIR / "trade_training_data.jsonl"


def infer_market_session(recorded_at: datetime) -> str:
    if recorded_at.hour < 9:
        return "pre_open"
    if recorded_at.hour < 12:
        return "morning"
    if recorded_at.hour < 15 or (recorded_at.hour == 15 and recorded_at.minute <= 20):
        return "afternoon"
    return "after_close"


def main() -> int:
    if not TRAINING_DATA_PATH.exists():
        print(f"missing={TRAINING_DATA_PATH}")
        return 1

    rows: list[str] = []
    enriched = 0
    total = 0
    for raw_line in TRAINING_DATA_PATH.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        total += 1
        if row.get("sample_id") in (None, ""):
            row["sample_id"] = f"{row.get('data_source', 'real')}-{total:07d}"
            enriched += 1
        recorded_at_raw = str(row.get("recorded_at", "") or "")
        recorded_at: datetime | None = None
        if recorded_at_raw:
            try:
                recorded_at = datetime.fromisoformat(recorded_at_raw)
            except ValueError:
                recorded_at = None

        if recorded_at is not None:
            if "time_bucket" not in row:
                row["time_bucket"] = recorded_at.strftime("%H:%M")
                enriched += 1
            if "market_session" not in row:
                row["market_session"] = infer_market_session(recorded_at)
                enriched += 1
            if "weekday" not in row:
                row["weekday"] = recorded_at.weekday()
                enriched += 1
            if "minute_of_day" not in row:
                row["minute_of_day"] = recorded_at.hour * 60 + recorded_at.minute
                enriched += 1
        if "data_source" not in row or row.get("data_source") in (None, ""):
            row["data_source"] = "real"
            enriched += 1
        if row.get("side") == "buy" and row.get("entry_price") in (None, "", 0):
            row["entry_price"] = row.get("price")
            enriched += 1
        if row.get("stop_loss_percent") in (None, ""):
            row["stop_loss_percent"] = 5.0
            enriched += 1
        if row.get("take_profit_percent") in (None, ""):
            row["take_profit_percent"] = 3.0
            enriched += 1
        defaults = {
            "day_open": row.get("price"),
            "day_high": row.get("price"),
            "day_low": row.get("price"),
            "previous_diff": 0,
            "traded_value": 0,
            "market_cap": 0,
            "per": -1.0,
            "pbr": -1.0,
            "eps": -1.0,
            "bps": -1.0,
            "day_range": 0,
            "intraday_range_pct": 0.0,
            "open_gap_pct": 0.0,
            "price_vs_entry_pct": 0.0,
            "drawdown_from_high_pct": 0.0,
            "rebound_from_low_pct": 0.0,
            "price_position_in_range": 0.5,
            "turnover_value_per_share": 0.0,
            "position_notional": int(row.get("price", 0) or 0) * int(row.get("qty", 0) or 0),
            "tracked_invested": 0,
            "remaining_budget": 0,
            "budget_utilization": 0.0,
            "tracked_position_count": 0,
            "holding_minutes": 0.0,
            "model_score": 0.0,
            "candidate_score": 0.0,
        }
        for key, value in defaults.items():
            if row.get(key) in (None, ""):
                row[key] = value
                enriched += 1
        rows.append(json.dumps(row, ensure_ascii=False))

    TRAINING_DATA_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"rows={total}")
    print(f"enriched_fields={enriched}")
    print(f"training_file={TRAINING_DATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
