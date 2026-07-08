import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    average_latency_ms: float | None
    sample_count: int
    class_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate labeled emergency hand-gesture results from a CSV file."
    )
    parser.add_argument(
        "csv_path",
        help="Path to a CSV file with at least true_label and predicted_label columns.",
    )
    parser.add_argument(
        "--label-column",
        default="true_label",
        help="Column name containing the ground-truth label.",
    )
    parser.add_argument(
        "--prediction-column",
        default="predicted_label",
        help="Column name containing the predicted label.",
    )
    parser.add_argument(
        "--latency-column",
        default="latency_ms",
        help="Optional column name containing latency in milliseconds.",
    )
    parser.add_argument(
        "--positive-labels",
        default="EMERGENCY,SOS,HELP,ALERT",
        help="Comma-separated list of labels treated as emergency positives.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def compute_metrics(
    rows: list[dict[str, str]],
    label_column: str,
    prediction_column: str,
    latency_column: str,
    positive_labels: set[str],
) -> Metrics:
    if not rows:
        raise ValueError("No rows found in the labeled evaluation file.")

    y_true: list[str] = []
    y_pred: list[str] = []
    latencies: list[float] = []

    for row in rows:
        true_label = (row.get(label_column) or "").strip()
        predicted_label = (row.get(prediction_column) or "").strip()
        if not true_label or not predicted_label:
            continue
        y_true.append(true_label)
        y_pred.append(predicted_label)
        latency_value = safe_float(row.get(latency_column))
        if latency_value is not None:
            latencies.append(latency_value)

    if not y_true:
        raise ValueError(
            f"No valid rows found. Ensure '{label_column}' and '{prediction_column}' are populated."
        )

    total = len(y_true)
    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    accuracy = correct / total

    positive_truth = [label in positive_labels for label in y_true]
    positive_pred = [label in positive_labels for label in y_pred]

    tp = sum(1 for truth, pred in zip(positive_truth, positive_pred) if truth and pred)
    fp = sum(1 for truth, pred in zip(positive_truth, positive_pred) if not truth and pred)
    fn = sum(1 for truth, pred in zip(positive_truth, positive_pred) if truth and not pred)
    tn = sum(1 for truth, pred in zip(positive_truth, positive_pred) if not truth and not pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

    average_latency_ms = mean(latencies) if latencies else None
    class_count = len(Counter(y_true))

    return Metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=false_positive_rate,
        average_latency_ms=average_latency_ms,
        sample_count=total,
        class_count=class_count,
    )


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)
    rows = load_rows(csv_path)
    positive_labels = {label.strip() for label in args.positive_labels.split(",") if label.strip()}

    metrics = compute_metrics(
        rows=rows,
        label_column=args.label_column,
        prediction_column=args.prediction_column,
        latency_column=args.latency_column,
        positive_labels=positive_labels,
    )

    print(f"Samples: {metrics.sample_count}")
    print(f"Gesture classes: {metrics.class_count}")
    print(f"Accuracy: {metrics.accuracy * 100:.2f}%")
    print(f"Precision: {metrics.precision * 100:.2f}%")
    print(f"Recall: {metrics.recall * 100:.2f}%")
    print(f"F1: {metrics.f1 * 100:.2f}%")
    print(f"False positive rate: {metrics.false_positive_rate * 100:.2f}%")
    if metrics.average_latency_ms is not None:
        print(f"Average latency: {metrics.average_latency_ms:.2f} ms")
    else:
        print("Average latency: not provided")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())