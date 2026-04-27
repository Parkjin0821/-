from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / ".cache" / "trade_training_data.jsonl"


def extract_text(line: str, key: str) -> str | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', line)
    return match.group(1) if match else None


def extract_number(line: str, key: str) -> int | float | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+(?:\.\d+)?)', line)
    if not match:
        return None
    raw = match.group(1)
    return float(raw) if "." in raw else int(raw)


def extract_name(line: str) -> str:
    match = re.search(r'"name"\s*:\s*(.*?),\s*"price"\s*:', line)
    if not match:
        return ""
    raw = match.group(1).strip()
    if raw.startswith('"'):
        raw = raw[1:]
    if raw.endswith('"'):
        raw = raw[:-1]
    return raw.replace('"', "").replace("\x00", " ").replace("\r", " ").replace("\n", " ").strip()


def salvage_line(line: str) -> dict | None:
    try:
        row = json.loads(line)
        if isinstance(row.get("name"), str):
            row["name"] = row["name"].replace('"', "").replace("\x00", " ").strip()
        return row
    except json.JSONDecodeError:
        pass

    row = {
        "recorded_at": extract_text(line, "recorded_at"),
        "side": extract_text(line, "side"),
        "symbol": extract_text(line, "symbol"),
        "name": extract_name(line),
        "price": extract_number(line, "price"),
        "change_rate": extract_number(line, "change_rate"),
        "volume": extract_number(line, "volume"),
        "qty": extract_number(line, "qty"),
        "reason": extract_text(line, "reason"),
        "pnl_percent": extract_number(line, "pnl_percent"),
        "label_success": extract_number(line, "label_success"),
    }

    required_keys = ("recorded_at", "side", "symbol", "price", "change_rate", "volume", "qty", "reason")
    if any(row.get(key) is None for key in required_keys):
        return None

    return {key: value for key, value in row.items() if value is not None}


def main() -> int:
    if not DATA_PATH.exists():
        print(f"not_found={DATA_PATH}")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = DATA_PATH.with_name(f"{DATA_PATH.stem}.backup_{timestamp}.jsonl")
    lines = DATA_PATH.read_text(encoding="utf-8").splitlines()

    cleaned_rows: list[dict] = []
    invalid_count = 0
    salvaged_count = 0

    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
            if isinstance(row.get("name"), str):
                row["name"] = row["name"].replace('"', "").replace("\x00", " ").strip()
            cleaned_rows.append(row)
            continue
        except json.JSONDecodeError:
            recovered = salvage_line(raw)
            if recovered is None:
                invalid_count += 1
                continue
            salvaged_count += 1
            cleaned_rows.append(recovered)

    backup_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    with DATA_PATH.open("w", encoding="utf-8") as handle:
        for row in cleaned_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "source": str(DATA_PATH),
                "backup": str(backup_path),
                "original_lines": len(lines),
                "clean_rows": len(cleaned_rows),
                "salvaged_rows": salvaged_count,
                "dropped_rows": invalid_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
