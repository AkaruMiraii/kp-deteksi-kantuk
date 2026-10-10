import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .smart_yawn import TemporalFeatures

POSITIVES = {"yawn_like_candidate_wide", "repeated_wide_opening_candidate"}
NEGATIVES = {"small_or_closed", "small_opening", "small_or_moderate_opening"}


def label_at(frame, intervals, margin=0):
    for interval in intervals:
        start, end = int(interval["start_frame"]), int(interval["end_frame"])
        if start + margin <= frame <= end - margin:
            if interval["label"] in POSITIVES:
                return 1
            if interval["label"] in NEGATIVES:
                return 0
    return None


def metrics(target, prediction):
    p, r, f, _ = precision_recall_fscore_support(
        target, prediction, average="binary", zero_division=0
    )
    return dict(
        precision=float(p),
        recall=float(r),
        f1=float(f),
        confusion_matrix=confusion_matrix(target, prediction, labels=[0, 1]).tolist(),
        support=dict(negative=int(np.sum(target == 0)), positive=int(np.sum(target == 1))),
        zero_division_policy="zero for unavailable class metric; inspect support",
    )


def fit(x, y):
    if len(set(y)) != 2:
        raise ValueError("Training requires both classes.")
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    )
    model.fit(x, y)
    return model


def build_dataset(session, annotations):
    with annotations.open(encoding="utf-8", newline="") as source:
        intervals = [row for row in csv.DictReader(source) if row["channel"] == "mouth"]
    summary = json.loads((session / "session.json").read_text(encoding="utf-8"))
    origin = None
    trackers = {}
    last_sample = {}
    dataset = []
    counts = Counter()
    segments = [("A", 0, 38), ("B", 43, 74), ("C", 79, 120), ("D", 125, 159)]
    with (session / "metrics.csv").open(encoding="utf-8", newline="") as source:
        for index, row in enumerate(csv.DictReader(source)):
            if int(row["frame_id"]) != index:
                raise ValueError("Noncontiguous source frame IDs.")
            ts = int(row["timestamp_ms"])
            origin = ts if origin is None else origin
            seconds = (ts - origin) / 1000
            group = next((g for g, a, b in segments if a <= seconds < b), None)
            if group is None:
                counts["transition_excluded"] += 1
                continue
            if group not in trackers:
                trackers[group] = TemporalFeatures()
            values = [float(row[k]) if row[k] else None for k in ("ear_r", "ear_l", "mar")]
            if row["reason"]:
                values = [None, None, None]
            features = trackers[group].update(ts, *values)
            target = label_at(
                index, [r for r in intervals if r["subject_segment"] == group], margin=15
            )
            if target is None:
                counts["unknown_occluded_or_boundary_excluded"] += 1
                continue
            if features is None:
                counts["labeled_invalid_excluded"] += 1
                continue
            if ts - last_sample.get(group, -1000) < 100:
                continue
            last_sample[group] = ts
            dataset.append(
                dict(frame_id=index, timestamp_ms=ts, group=group, target=target, features=features)
            )
    if len(dataset) < 20:
        raise ValueError("Insufficient labeled data.")
    return dataset, dict(counts), summary


def train(session, annotations, output):
    dataset, exclusions, summary = build_dataset(session, annotations)
    x = np.array([r["features"] for r in dataset])
    y = np.array([r["target"] for r in dataset])
    groups = np.array([r["group"] for r in dataset])
    predictions = np.zeros(len(y), dtype=int)
    fold_reports = []
    for group in sorted(set(groups)):
        holdout = groups == group
        model = fit(x[~holdout], y[~holdout])
        scores = model.predict_proba(x[holdout])[:, 1]
        predictions[holdout] = scores >= 0.5
        base = x[holdout, 0] > 0.5
        fold_reports.append(
            dict(
                subject_segment=group,
                model=metrics(y[holdout], predictions[holdout]),
                mar_baseline=metrics(y[holdout], base),
                train_groups=sorted(set(groups[~holdout])),
            )
        )
        for row, score in zip([r for r in dataset if r["group"] == group], scores):
            row["oof_score"] = float(score)
    fitted = fit(x, y)
    scaler, classifier = fitted.steps[0][1], fitted.steps[1][1]
    artifact = dict(
        schema_version=1,
        feature_names=TemporalFeatures.names,
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
        coefficients=classifier.coef_[0].tolist(),
        intercept=float(classifier.intercept_[0]),
        score_threshold=0.5,
        persistence_seconds=0.4,
        target="draft_visual_yawn_candidate",
        calibrated_probability=False,
        label_status="AI visual draft, not final ground truth",
        source_video_sha256=summary["raw_video"]["video_sha256"],
        source_metrics_sha256=hashlib.sha256((session / "metrics.csv").read_bytes()).hexdigest(),
        annotations_sha256=hashlib.sha256(annotations.read_bytes()).hexdigest(),
        landmark_model_sha256=summary["model_sha256"],
        sklearn_version=sklearn.__version__,
        training_rows=len(y),
        threshold_selection="fixed before fit, no test tuning",
        causal_window_seconds=1.5,
    )
    report = dict(
        method="leave-one-foreground-subject-segment-out; not identity verification",
        folds=fold_reports,
        pooled_oof=metrics(y, predictions),
        pooled_baseline=metrics(y, x[:, 0] > 0.5),
        exclusions=exclusions,
        sample_count=len(y),
        subject_class_counts={
            g: dict(Counter(str(int(v)) for v in y[groups == g])) for g in sorted(set(groups))
        },
        limitation="Approximate AI labels of simulated yawn-like movement, one recording, temporal correlation. D may lack negatives. Scores uncalibrated; no claim of actual yawns/drowsiness.",
        evaluation_unit="sampled labeled valid frames, before live persistence; unknown/occluded not negatives",
        deployment="final model refit on all labeled samples; this model has no unseen external test",
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "smart_yawn.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    (output / "training-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "oof-predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["frame_id", "timestamp_ms", "group", "target", "oof_score"]
        )
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in dataset)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Train experimental EAR/MAR yawn candidate classifier."
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    train(args.session, args.annotations, args.output)


if __name__ == "__main__":
    main()
