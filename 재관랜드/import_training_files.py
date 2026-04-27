from __future__ import annotations

import argparse
import json
from pathlib import Path

from kis_autotrade import TRAINING_DATA_PATH, normalize_training_row


DEFAULT_INPUTS = [
    Path.home() / "Downloads" / "trade_training_data_100.jsonl",
    Path.home() / "Downloads" / "trade_training_data_add_400.jsonl",
    Path.home() / "Downloads" / "trade_training_data_add_500.jsonl",
    Path.home() / "Downloads" / "trade_training_data_add_5000_diverse.jsonl",
]


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            normalized = normalize_training_row(row)
            if normalized is not None:
                yield normalized


def existing_keys() -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    if not TRAINING_DATA_PATH.exists():
        return keys
    for row in iter_rows(TRAINING_DATA_PATH):
        keys.add(
            (
                str(row.get("recorded_at", "")),
                str(row.get("symbol", "")),
                str(row.get("side", "")),
                str(row.get("reason", "")),
            )
        )
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description="Import external JSONL training files into the main training dataset.")
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--source", default="imported")
    args = parser.parse_args()

    keys = existing_keys()
    imported = 0
    skipped = 0
    missing = 0

    with TRAINING_DATA_PATH.open("a", encoding="utf-8") as output:
        for path in args.paths:
            if not path.exists():
                missing += 1
                continue
            for row in iter_rows(path):
                key = (
                    str(row.get("recorded_at", "")),
                    str(row.get("symbol", "")),
                    str(row.get("side", "")),
                    str(row.get("reason", "")),
                )
                if key in keys:
                    skipped += 1
                    continue
                keys.add(key)
                row["data_source"] = args.source
                row["sample_id"] = f"{args.source}-{len(keys):07d}"
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                imported += 1

    print(
        json.dumps(
            {
                "training_file": str(TRAINING_DATA_PATH),
                "imported": imported,
                "skipped_duplicates": skipped,
                "missing_files": missing,
                "source": args.source,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
