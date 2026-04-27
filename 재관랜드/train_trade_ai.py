from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from kis_autotrade import (
    KST,
    MODEL_STATE_PATH,
    TRAINING_DATA_PATH,
    ModelWeights,
    extract_features,
    normalize_training_row,
    save_model_weights,
    sigmoid,
    write_training_meta,
)
from datetime import datetime


def iter_training_rows(path: Path):
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
            if row is None:
                continue
            yield row


def choose_epochs(labeled_count: int) -> int:
    if labeled_count >= 1_000_000:
        return 4
    if labeled_count >= 500_000:
        return 5
    if labeled_count >= 200_000:
        return 8
    if labeled_count >= 100_000:
        return 12
    return 20


def train_streaming_model(path: Path, *, learning_rate: float = 0.08) -> tuple[ModelWeights, Counter, Counter, int]:
    sources: Counter = Counter()
    labels: Counter = Counter()
    labeled_count = 0
    for row in iter_training_rows(path):
        sources[row.get("data_source", "unknown")] += 1
        label = row.get("label_success")
        if row.get("side") == "buy" and isinstance(label, (int, float)):
            labels[float(label)] += 1
            labeled_count += 1

    if labeled_count < 8:
        weights = ModelWeights(sample_count=labeled_count)
        return weights, sources, labels, labeled_count

    epochs = choose_epochs(labeled_count)
    weights = ModelWeights(sample_count=labeled_count)

    for _ in range(epochs):
        grad_intercept = 0.0
        grad_price = 0.0
        grad_change = 0.0
        grad_volume = 0.0
        grad_mid = 0.0

        for row in iter_training_rows(path):
            label = row.get("label_success")
            if row.get("side") != "buy" or not isinstance(label, (int, float)):
                continue

            features = extract_features(
                int(row.get("price", 0) or 0),
                float(row.get("change_rate", 0.0) or 0.0),
                int(row.get("volume", 0) or 0),
            )
            prediction = sigmoid(
                weights.intercept
                + weights.price_weight * features["price_norm"]
                + weights.change_weight * features["change_norm"]
                + weights.volume_weight * features["volume_norm"]
                + weights.bias_to_mid_momentum * features["mid_momentum"]
            )
            error = prediction - float(label)
            grad_intercept += error
            grad_price += error * features["price_norm"]
            grad_change += error * features["change_norm"]
            grad_volume += error * features["volume_norm"]
            grad_mid += error * features["mid_momentum"]

        scale = 1.0 / labeled_count
        weights = replace(
            weights,
            intercept=weights.intercept - learning_rate * grad_intercept * scale,
            price_weight=weights.price_weight - learning_rate * grad_price * scale,
            change_weight=weights.change_weight - learning_rate * grad_change * scale,
            volume_weight=weights.volume_weight - learning_rate * grad_volume * scale,
            bias_to_mid_momentum=weights.bias_to_mid_momentum - learning_rate * grad_mid * scale,
        )

    weights = replace(weights, sample_count=labeled_count)
    return weights, sources, labels, labeled_count


def main() -> int:
    weights, sources, labels, labeled_count = train_streaming_model(TRAINING_DATA_PATH)
    save_model_weights(weights)
    write_training_meta(
        {
            "last_retrained_at": datetime.now(KST).isoformat(),
            "last_sample_count": labeled_count,
            "training_mode": "streaming_offline",
        }
    )
    print(f"training_file={TRAINING_DATA_PATH}")
    print(f"model_path={MODEL_STATE_PATH}")
    print(f"training_rows={sum(sources.values())}")
    print(f"sample_count={labeled_count}")
    print(f"sources={dict(sources)}")
    print(f"labels={dict(labels)}")
    print(
        "weights="
        f"intercept={weights.intercept:.4f}, "
        f"price={weights.price_weight:.4f}, "
        f"change={weights.change_weight:.4f}, "
        f"volume={weights.volume_weight:.4f}, "
        f"mid_momentum={weights.bias_to_mid_momentum:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
