# Four-qubit hybrid CNN/QCNN for five stocks

This repository contains the standalone `hybrid_qcnn_small` experiment and the
five one-minute stock CSV files used by it.  It compares a parameter-matched CNN
baseline with a four-qubit variational quantum hidden layer.

## Included inputs

- Raw 1-D factor windows (`conv1d`)
- Feature-wise GASF images
- Feature-wise GADF images
- GASF and GADF as paired channels
- Fusion of the raw 1-D and any of the three image representations

The data files cover stock codes `600000`, `600030`, `600036`, `600519`, and
`601318`.  The training code uses chronological train/validation/test splits.

## Environment

Python 3.11 is recommended.  Create a fresh environment rather than committing
or copying a platform-specific virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

Run from the repository root:

```powershell
cd hybrid_qcnn_small
..\.venv\Scripts\python.exe train.py `
  --input-mode fusion `
  --gaf-mode both `
  --model both `
  --horizon 1 `
  --stride 1 `
  --epochs 20 `
  --seeds 42 43 44 45 46 `
  --patience 5
```

Training uses an ideal four-qubit PennyLane simulator with adjoint
differentiation.  The optional `--noise` flag applies finite-shot noisy
validation and test evaluation; it does not submit jobs to quantum hardware.

## Outputs

Outputs are written under `hybrid_qcnn_small/results/` and are ignored by Git.
For every model and seed, the runner saves the best-validation checkpoint,
pooled predictions, per-stock predictions and metrics, and full-window audit
CSVs containing timestamps, raw OHLCVA values, model-input features, predicted
probabilities, and actual targets.

