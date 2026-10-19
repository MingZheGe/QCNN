"""Train matched CNN and four-qubit hybrid models without modifying the parent project."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    # Keep this directory first so local data.py/model.py cannot be shadowed
    # by files with the same names in the parent project.
    sys.path.insert(1, str(PARENT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader

from config import BATCH_SIZE, FEATURES, LEARNING_RATE, SEQ_LEN, TRAIN_FRACTION
from data_pipeline import engineer_features, load_panel
from data import HybridDataset
from model import CNNBaseline, HybridClassifier


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def validation_loss(model, dataset, loss_fn, device):
    model.eval(); total = seen = 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False):
            batch = move(batch, device); loss = loss_fn(model(batch), batch["label"])
            total += loss.item() * len(batch["label"]); seen += len(batch["label"])
    return total / max(seen, 1)


def evaluate(model, dataset, device):
    model.eval(); labels = []; preds = []; rows = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False):
            cpu_batch = batch; batch = move(batch, device)
            probability = torch.softmax(model(batch), 1).cpu(); prediction = probability.argmax(1)
            labels.extend(cpu_batch["label"].tolist()); preds.extend(prediction.tolist())
            for i in range(len(prediction)):
                rows.append({"timestamp": cpu_batch["timestamp"][i], "symbol": cpu_batch["symbol"][i],
                             "target_return": float(cpu_batch["target_return"][i]), "actual": int(cpu_batch["label"][i]),
                             "prob_down": float(probability[i, 0]), "prob_up": float(probability[i, 1]),
                             "predicted": int(prediction[i])})
    return np.asarray(labels), np.asarray(preds), rows


RAW_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def build_window_lookup(data: pd.DataFrame, dataset: HybridDataset) -> dict:
    """Retain the exact raw and preprocessed rows used by each test window."""
    lookup = {}
    needed = {(str(sample[2]), pd.Timestamp(sample[3])) for sample in dataset.samples}
    columns = ["timestamp", *RAW_COLUMNS, *FEATURES]
    for symbol, group in data.groupby("symbol"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for end in range(SEQ_LEN - 1, len(group)):
            key = (str(symbol), pd.Timestamp(group.loc[end, "timestamp"]))
            if key in needed:
                lookup[key] = group.loc[end - SEQ_LEN + 1:end, columns].copy()
    return lookup


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    counts = np.bincount(labels, minlength=2)
    predicted_counts = np.bincount(predictions, minlength=2)
    accuracy = float(accuracy_score(labels, predictions))
    return {
        "samples_test": int(len(labels)),
        "test_class_counts": counts.tolist(),
        "predicted_class_counts": predicted_counts.tolist(),
        "majority_class_baseline_accuracy": float(counts.max() / max(len(labels), 1)),
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "down_recall": float(report["0"]["recall"]),
        "up_recall": float(report["1"]["recall"]),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "classification_report": report,
    }


def write_per_stock_outputs(rows: list[dict], window_lookup: dict, seed_dir: Path) -> dict:
    """Write separate predictions, metrics, and full-window audit CSVs per stock."""
    frame = pd.DataFrame(rows)
    metrics_by_symbol = {}
    summary_rows = []
    for symbol, symbol_frame in frame.groupby("symbol", sort=True):
        symbol = str(symbol)
        symbol_frame = symbol_frame.reset_index(drop=True)
        symbol_dir = seed_dir / f"symbol_{symbol}"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        symbol_frame.to_csv(symbol_dir / "predictions.csv", index=False, encoding="utf-8-sig")

        labels = symbol_frame["actual"].to_numpy(dtype=int)
        predictions = symbol_frame["predicted"].to_numpy(dtype=int)
        symbol_metrics = classification_metrics(labels, predictions)
        metrics_by_symbol[symbol] = symbol_metrics
        (symbol_dir / "metrics.json").write_text(
            json.dumps(symbol_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary_rows.append({
            "symbol": symbol,
            **{key: value for key, value in symbol_metrics.items() if isinstance(value, (int, float))},
        })

        audit_rows = []
        for row in symbol_frame.to_dict("records"):
            window = window_lookup[(symbol, pd.Timestamp(row["timestamp"]))]
            audit = {
                "symbol": symbol,
                "window_start": str(window.iloc[0]["timestamp"]),
                "window_end": str(window.iloc[-1]["timestamp"]),
                "target_return": row["target_return"],
                "actual_label": row["actual"],
                "prob_down": row["prob_down"],
                "prob_up": row["prob_up"],
                "predicted_label": row["predicted"],
                "correct": int(row["actual"] == row["predicted"]),
            }
            for position, (_, window_row) in enumerate(window.iterrows(), start=1):
                prefix = f"window_{position:02d}"
                audit[f"{prefix}_timestamp"] = str(window_row["timestamp"])
                for column in RAW_COLUMNS:
                    audit[f"{prefix}_{column}_raw"] = float(window_row[column])
                for feature in FEATURES:
                    audit[f"{prefix}_{feature}_model_input"] = float(window_row[feature])
            audit_rows.append(audit)
        pd.DataFrame(audit_rows).to_csv(
            symbol_dir / "test_audit.csv", index=False, encoding="utf-8-sig"
        )

    pd.DataFrame(summary_rows).to_csv(
        seed_dir / "per_stock_metrics.csv", index=False, encoding="utf-8-sig"
    )
    return metrics_by_symbol


def train_seed(model_type, args, seed, train, validation, test, weights, output, window_lookup):
    set_seed(seed); started = time.perf_counter(); device = torch.device("cpu")
    noise = ({"p1": args.p1, "p2": args.p2, "readout": args.readout_error,
              "shots": args.shots} if args.noise else None)
    model = (CNNBaseline(args.input_mode, args.gaf_mode) if model_type == "cnn"
             else HybridClassifier(args.input_mode, args.gaf_mode, args.differentiator, noise=noise)).to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_loss = float("inf"); best_state = None; best_epoch = 0; stale = 0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = seen = 0
        for batch in loader:
            batch = move(batch, device); optimizer.zero_grad()
            loss = loss_fn(model(batch), batch["label"]); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += loss.item() * len(batch["label"]); seen += len(batch["label"])
        train_loss = total / seen; val_loss = validation_loss(model, validation, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"[{model_type} seed={seed}] epoch={epoch} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if val_loss < best_loss - args.min_delta:
            best_loss = val_loss; best_epoch = epoch; stale = 0; best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(best_state)
    labels, predictions, rows = evaluate(model, test, device)
    test_metrics = classification_metrics(labels, predictions)
    metrics = {
        "model_type": model_type, "input_mode": args.input_mode, "gaf_mode": args.gaf_mode,
        "seed": seed, "horizon": args.current_horizon, "stride": args.stride,
        "qubits": 4 if model_type == "qcnn" else 0, "quantum_layers": 2 if model_type == "qcnn" else 0,
        "quantum_parameters": 8 if model_type == "qcnn" else 0,
        "differentiator": args.differentiator if model_type == "qcnn" else None,
        "noise_evaluation": noise if model_type == "qcnn" else None,
        "training_quantum_mode": "ideal_unitary_adjoint" if model_type == "qcnn" else None,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "samples_train": len(train), "samples_validation": len(validation), "samples_test": len(test),
        "class_weights": weights.tolist(), "best_epoch": best_epoch, "best_validation_loss": best_loss,
        **test_metrics,
        "accuracy_gain_over_majority_baseline": (
            test_metrics["accuracy"] - test_metrics["majority_class_baseline_accuracy"]
        ),
        "duration_seconds": time.perf_counter() - started,
    }
    seed_dir = output / model_type / f"seed_{seed}"; seed_dir.mkdir(parents=True, exist_ok=True)
    metrics["per_stock"] = write_per_stock_outputs(rows, window_lookup, seed_dir)
    (seed_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (seed_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(seed_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    torch.save({
        "model_type": model_type,
        "model_state_dict": model.state_dict(),
        "input_mode": args.input_mode,
        "gaf_mode": args.gaf_mode,
        "seed": seed,
        "horizon": args.current_horizon,
        "stride": args.stride,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "features": FEATURES,
        "sequence_length": SEQ_LEN,
        "qubits": 4 if model_type == "qcnn" else 0,
        "differentiator": args.differentiator if model_type == "qcnn" else None,
        "noise_evaluation": noise if model_type == "qcnn" else None,
    }, seed_dir / "model.pt")
    return metrics


def aggregate(results):
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc"]
    return {key: {"mean": float(np.mean([r[key] for r in results])),
                  "std": float(np.std([r[key] for r in results], ddof=1)) if len(results) > 1 else 0.0}
            for key in metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mode", choices=("conv1d", "conv2d", "fusion"), default="fusion")
    parser.add_argument("--gaf-mode", choices=("gasf", "gadf", "both"), default="both")
    parser.add_argument("--model", choices=("cnn", "qcnn", "both"), default="both")
    parser.add_argument("--horizon", choices=("1", "5", "both"), default="both")
    parser.add_argument("--stride", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--differentiator", choices=("adjoint",), default="adjoint")
    parser.add_argument("--noise", action="store_true", help="Use noisy finite-shot QCNN for validation/test only")
    parser.add_argument("--p1", type=float, default=0.001, help="Single-qubit depolarizing probability")
    parser.add_argument("--p2", type=float, default=0.01, help="Approximate two-qubit gate error probability")
    parser.add_argument("--readout-error", type=float, default=0.02, help="Readout bit-flip probability")
    parser.add_argument("--shots", type=int, default=1024, help="Shots per circuit during noisy validation/test")
    args = parser.parse_args()
    if not 0 <= args.p1 <= 0.75: parser.error("--p1 must be between 0 and 0.75")
    if not 0 <= args.p2 <= 0.75: parser.error("--p2 must be between 0 and 0.75")
    if not 0 <= args.readout_error <= 0.5: parser.error("--readout-error must be between 0 and 0.5")
    if args.shots < 1: parser.error("--shots must be positive")
    if args.input_mode == "conv1d" and args.gaf_mode != "both":
        print("Note: --gaf-mode is unused by conv1d.")
    data = engineer_features(load_panel())
    dates = np.array(sorted(data.timestamp.unique()))[::args.stride]
    outer_cut = max(2, int(len(dates) * TRAIN_FRACTION)); train_val_dates = dates[:outer_cut]
    inner_cut = max(1, int(len(train_val_dates) * 0.85))
    first_val = pd.Timestamp(train_val_dates[inner_cut]); first_test = pd.Timestamp(dates[outer_cut])
    test_dates = set(pd.to_datetime(dates[outer_cut:]))
    horizons = [1, 5] if args.horizon == "both" else [int(args.horizon)]
    model_types = ["cnn", "qcnn"] if args.model == "both" else [args.model]
    root = HERE / "results" / f"{args.input_mode}_{args.gaf_mode}_stride{args.stride}"
    for horizon in horizons:
        args.current_horizon = horizon
        train_dates = {pd.Timestamp(t) for t in train_val_dates[:inner_cut] if pd.Timestamp(t) + pd.Timedelta(minutes=horizon) < first_val}
        val_dates = {pd.Timestamp(t) for t in train_val_dates[inner_cut:] if pd.Timestamp(t) + pd.Timedelta(minutes=horizon) < first_test}
        train = HybridDataset(data, train_dates, horizon, args.gaf_mode, args.input_mode)
        validation = HybridDataset(data, val_dates, horizon, args.gaf_mode, args.input_mode)
        test = HybridDataset(data, test_dates, horizon, args.gaf_mode, args.input_mode)
        window_lookup = build_window_lookup(data, test)
        counts = np.bincount([sample[1] for sample in train.samples], minlength=2)
        weights = len(train) / (2 * counts.astype(float))
        output = root / f"horizon_{horizon}"; summaries = {}
        for model_type in model_types:
            runs = [train_seed(model_type, args, seed, train, validation, test, weights, output, window_lookup) for seed in args.seeds]
            summaries[model_type] = aggregate(runs)
        summary = {"input_mode": args.input_mode, "gaf_mode": args.gaf_mode, "horizon": horizon,
                   "stride": args.stride, "train_class_counts": counts.tolist(), "class_weights": weights.tolist(),
                   "models": summaries}
        if all(name in summaries for name in ("cnn", "qcnn")):
            summary["qcnn_minus_cnn"] = {key: summaries["qcnn"][key]["mean"] - summaries["cnn"][key]["mean"]
                                          for key in summaries["cnn"]}
        output.mkdir(parents=True, exist_ok=True)
        (output / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
