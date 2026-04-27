from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from kis_autotrade import TRAINING_CLEANED_DATA_PATH, TRAINING_SYNTHETIC_DATA_PATH, read_training_rows


def generate_row(base: dict, index: int) -> dict:
    price = max(100, int(float(base["price"]) * random.uniform(0.97, 1.03)))
    change_rate = round(float(base["change_rate"]) + random.uniform(-1.2, 1.2), 2)
    volume = max(1000, int(float(base["volume"]) * random.uniform(0.75, 1.35)))
    qty = max(1, int(base.get("qty", 1)))

    row = dict(base)
    row["recorded_at"] = f"{base['recorded_at']}#aug{index}"
    row["price"] = price
    row["change_rate"] = change_rate
    row["volume"] = volume
    row["qty"] = qty
    row["is_synthetic"] = True
    row["data_source"] = "synthetic"
    row["base_recorded_at"] = base.get("recorded_at")
    row["base_symbol"] = base.get("symbol")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic training rows for model warm-up.")
    parser.add_argument("--target-count", type=int, default=6000, help="Target synthetic row count")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    base_rows = read_training_rows(TRAINING_CLEANED_DATA_PATH if TRAINING_CLEANED_DATA_PATH.exists() else Path(".cache/trade_training_data.jsonl"))
    if not base_rows:
        print("no_base_rows")
        return 1

    synthetic_rows: list[dict] = []
    for idx in range(args.target_count):
        base = base_rows[idx % len(base_rows)]
        synthetic_rows.append(generate_row(base, idx))

    TRAINING_SYNTHETIC_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_SYNTHETIC_DATA_PATH.open("w", encoding="utf-8") as handle:
        for row in synthetic_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"base_rows": len(base_rows), "synthetic_rows": len(synthetic_rows), "output": str(TRAINING_SYNTHETIC_DATA_PATH)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
