from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


CACHE_DIR = Path(".cache")
TRAINING_DATA_PATH = CACHE_DIR / "trade_training_data.jsonl"
MODEL_STATE_PATH = CACHE_DIR / "trade_model_state.json"
REPORT_PATH = CACHE_DIR / "training_quality_report.json"


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    if not TRAINING_DATA_PATH.exists():
        print(f"missing={TRAINING_DATA_PATH}")
        return 1

    total = 0
    bad = 0
    duplicate_ids = 0
    ids: set[str] = set()
    sources: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    sessions: Counter[str] = Counter()
    numeric_values: dict[str, list[float]] = defaultdict(list)

    numeric_fields = (
        "price",
        "change_rate",
        "volume",
        "pnl_percent",
        "candidate_score",
        "model_score",
        "budget_utilization",
        "intraday_range_pct",
        "price_position_in_range",
    )

    with TRAINING_DATA_PATH.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue

            total += 1
            sample_id = str(row.get("sample_id", ""))
            if sample_id:
                if sample_id in ids:
                    duplicate_ids += 1
                ids.add(sample_id)

            sources[str(row.get("data_source", "unknown"))] += 1
            labels[str(row.get("label_success", "missing"))] += 1
            reasons[str(row.get("reason", "unknown"))] += 1
            sessions[str(row.get("market_session", "unknown"))] += 1

            for field in numeric_fields:
                if field in row and row.get(field) not in (None, ""):
                    numeric_values[field].append(safe_float(row.get(field)))

    model_state = {}
    if MODEL_STATE_PATH.exists():
        try:
            model_state = json.loads(MODEL_STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            model_state = {}

    numeric_summary = {
        field: {
            "count": len(values),
            "avg": mean(values) if values else 0.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }
        for field, values in numeric_values.items()
    }

    report = {
        "generated_at": datetime.now().isoformat(),
        "training_file": str(TRAINING_DATA_PATH),
        "rows": total,
        "bad_rows": bad,
        "unique_sample_ids": len(ids),
        "duplicate_sample_ids": duplicate_ids,
        "sources": dict(sources),
        "labels": dict(labels),
        "reasons_top20": dict(reasons.most_common(20)),
        "sessions": dict(sessions),
        "numeric_summary": numeric_summary,
        "model_state": model_state,
        "notes": [
            "real 데이터와 synthetic 데이터는 반드시 분리해서 해석해야 합니다.",
            "배포 설명에서는 수익률 과장이 아니라 학습 파이프라인과 검증 가능성을 강조하세요.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"rows={total}")
    print(f"bad_rows={bad}")
    print(f"duplicate_sample_ids={duplicate_ids}")
    print(f"sources={dict(sources)}")
    print(f"labels={dict(labels)}")
    print(f"report={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
