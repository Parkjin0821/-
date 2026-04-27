from __future__ import annotations

import json
from pathlib import Path

from kis_autotrade import TRAINING_CLEANED_DATA_PATH, TRAINING_DATA_PATH, normalize_training_row


def main() -> int:
    if not TRAINING_DATA_PATH.exists():
        print(f"not_found={TRAINING_DATA_PATH}")
        return 1

    rows = []
    seen: set[tuple[str, str, str, str]] = set()
    dropped = 0
    for line in TRAINING_DATA_PATH.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            dropped += 1
            continue
        normalized = normalize_training_row(parsed)
        if normalized is None:
            dropped += 1
            continue
        normalized["data_source"] = "real"
        dedupe_key = (
            str(normalized.get("recorded_at", "")),
            str(normalized.get("symbol", "")),
            str(normalized.get("side", "")),
            str(normalized.get("reason", "")),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        rows.append(normalized)

    TRAINING_CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_CLEANED_DATA_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"cleaned_rows": len(rows), "dropped_rows": dropped, "output": str(TRAINING_CLEANED_DATA_PATH)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
