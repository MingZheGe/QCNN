from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SYMBOLS = ["600000", "600030", "600036", "600519", "601318"]
SEQ_LEN = 20
GAF_WINDOW = 60
GAF_FRAMES = 3
GAF_IMAGE_SIZE = 20
GAF_COLORMAP = "jet"
QUBITS = 8
QUANTUM_LAYERS = 3
QCONVX_CHANNELS = (4, 8)
# One of the paper's valid lookback-20 pairs (Table 2 / Appendix G).
QCONVX_KERNELS = (2, 3)
QCONVX_POOL = 2
NEUTRAL_THRESHOLD = 0.0005  # 5 basis points for one-minute returns; configurable.
TRAIN_FRACTION = 0.8
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
SEED = 42

FEATURES = [
    "ret1", "mom1", "mom2", "mom5", "mom10", "mom20",
    "vol5", "vol10", "vol20", "dv_mean5", "dv_mean10", "dv_mean20",
    "volume_mean5", "volume_mean10", "volume_mean20", "log_volume", "log_dv",
]
