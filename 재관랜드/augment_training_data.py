from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from kis_autotrade import KST, TRAINING_DATA_PATH, normalize_training_row


def iter_rows(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = normalize_training_row(json.loads(line))
            except json.JSONDecodeError:
                continue
            if row is not None:
                yield row


def load_real_base_rows(path: Path) -> list[dict]:
    rows = [
        row
        for row in iter_rows(path)
        if row.get("side") == "buy"
        and isinstance(row.get("label_success"), (int, float))
        and row.get("data_source", "real") == "real"
    ]
    if not rows:
        rows = [
            row
            for row in iter_rows(path)
            if row.get("side") == "buy" and isinstance(row.get("label_success"), (int, float))
        ]
    return rows


def current_row_count(path: Path) -> int:
    return sum(1 for _ in iter_rows(path))


def ensure_sample_ids(path: Path) -> int:
    rows: list[str] = []
    updated = 0
    for idx, row in enumerate(iter_rows(path), start=1):
        if not row.get("sample_id"):
            row["sample_id"] = f"{row.get('data_source', 'real')}-{idx:07d}"
            updated += 1
        rows.append(json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return updated


def generate_augmented_row(base: dict, synthetic_index: int) -> dict:
    base_time = datetime.fromisoformat(str(base["recorded_at"]).replace("Z", "+00:00"))
    shifted_time = base_time + timedelta(minutes=synthetic_index + 1)

    price = max(100, int(float(base["price"]) * random.uniform(0.93, 1.07)))
    change_rate = round(float(base["change_rate"]) + random.uniform(-2.2, 2.2), 2)
    volume = max(1000, int(float(base["volume"]) * random.uniform(0.55, 1.7)))

    day_open = max(100, int(price * random.uniform(0.96, 1.04)))
    day_low = max(100, min(price, day_open) - random.randint(0, max(1, int(price * 0.035))))
    day_high = max(price, day_open) + random.randint(0, max(1, int(price * 0.035)))
    day_range = day_high - day_low

    label_success = float(base.get("label_success", 0.0))
    if random.random() < 0.18:
        label_success = 1.0 - label_success

    tracked_invested = random.randint(3_000, 48_000)
    remaining_budget = random.randint(200, 25_000)
    total_budget = max(1, tracked_invested + remaining_budget)
    qty = max(1, int(base.get("qty", 1)))

    row = {
        "sample_id": f"synthetic-{synthetic_index:07d}",
        "recorded_at": shifted_time.astimezone(KST).isoformat(),
        "side": "buy",
        "symbol": str(base["symbol"]),
        "name": str(base.get("name", "")).replace('"', "").strip(),
        "price": price,
        "change_rate": change_rate,
        "volume": volume,
        "qty": qty,
        "reason": str(base.get("reason", "auto_pick")),
        "label_success": label_success,
        "pnl_percent": round(random.uniform(-7.0, 7.0), 4),
        "time_bucket": shifted_time.strftime("%H:%M"),
        "market_session": (
            "pre_open"
            if shifted_time.hour < 9
            else "morning"
            if shifted_time.hour < 12
            else "afternoon"
            if shifted_time.hour < 15 or (shifted_time.hour == 15 and shifted_time.minute <= 20)
            else "after_close"
        ),
        "weekday": shifted_time.weekday(),
        "minute_of_day": shifted_time.hour * 60 + shifted_time.minute,
        "data_source": "synthetic",
        "model_score": round(random.uniform(0.1, 0.92), 6),
        "candidate_score": round(random.uniform(0.2, 12.0), 6),
        "entry_price": price,
        "tracked_invested": tracked_invested,
        "remaining_budget": remaining_budget,
        "budget_utilization": round(tracked_invested / total_budget, 6),
        "tracked_position_count": random.randint(1, 12),
        "holding_minutes": round(random.uniform(1.0, 720.0), 4),
        "stop_loss_percent": 5.0,
        "take_profit_percent": 3.0,
        "day_open": day_open,
        "day_high": day_high,
        "day_low": day_low,
        "previous_diff": int(round(price * change_rate / 100.0)),
        "traded_value": volume * price,
        "market_cap": random.randint(30_000_000_000, 3_000_000_000_000),
        "per": round(random.uniform(-10.0, 60.0), 4),
        "pbr": round(random.uniform(0.05, 12.0), 4),
        "eps": round(random.uniform(-1000.0, 8000.0), 4),
        "bps": round(random.uniform(50.0, 40000.0), 4),
        "day_range": day_range,
        "intraday_range_pct": round(((day_high - day_low) / day_low * 100.0) if day_low > 0 else 0.0, 6),
        "open_gap_pct": round(((price - day_open) / day_open * 100.0) if day_open > 0 else 0.0, 6),
        "price_vs_entry_pct": round(random.uniform(-7.0, 7.0), 6),
        "drawdown_from_high_pct": round(((price - day_high) / day_high * 100.0) if day_high > 0 else 0.0, 6),
        "rebound_from_low_pct": round(((price - day_low) / day_low * 100.0) if day_low > 0 else 0.0, 6),
        "price_position_in_range": round(((price - day_low) / day_range) if day_range > 0 else 0.5, 6),
        "turnover_value_per_share": round((volume * price / volume) if volume > 0 else 0.0, 6),
        "position_notional": price * qty,
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand the single training file to a larger unique dataset.")
    parser.add_argument("--count", type=int, default=0, help="How many new synthetic rows to append")
    parser.add_argument("--target-total", type=int, default=0, help="Expand until this total row count is reached")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=5000)
    args = parser.parse_args()

    random.seed(args.seed)
    updated = ensure_sample_ids(TRAINING_DATA_PATH)
    base_rows = load_real_base_rows(TRAINING_DATA_PATH)
    if not base_rows:
        print("no_base_rows")
        return 1

    current_total = current_row_count(TRAINING_DATA_PATH)
    if args.target_total > 0:
        to_append = max(0, args.target_total - current_total)
    else:
        to_append = max(0, args.count)
    if to_append <= 0:
        print(
            json.dumps(
                {
                    "updated_sample_ids": updated,
                    "current_total": current_total,
                    "target_total": args.target_total,
                    "appended_synthetic_rows": 0,
                    "training_file": str(TRAINING_DATA_PATH),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    synthetic_start_index = current_total + 1
    appended = 0
    buffer: list[str] = []
    with TRAINING_DATA_PATH.open("a", encoding="utf-8") as handle:
        for offset in range(to_append):
            synthetic_index = synthetic_start_index + offset
            row = generate_augmented_row(base_rows[offset % len(base_rows)], synthetic_index)
            normalized = normalize_training_row(row)
            if normalized is None:
                continue
            buffer.append(json.dumps(normalized, ensure_ascii=False))
            appended += 1
            if len(buffer) >= args.chunk_size:
                handle.write("\n".join(buffer) + "\n")
                buffer.clear()
        if buffer:
            handle.write("\n".join(buffer) + "\n")

    print(
        json.dumps(
            {
                "updated_sample_ids": updated,
                "base_rows": len(base_rows),
                "previous_total": current_total,
                "target_total": args.target_total or None,
                "appended_synthetic_rows": appended,
                "new_total": current_total + appended,
                "training_file": str(TRAINING_DATA_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
