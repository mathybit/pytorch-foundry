"""
=============
EXPERIMENT: example_regressor_dnn — exp01
=============

TASK TYPE:
    Single-output regression (continuous target)

MODEL ARCHITECTURE:
    Feedforward DNN with configurable hidden layers.
"""


# ============================================================================
# Experiment Identification
# ============================================================================
EXP_NAME = "exp01"
EXP_DESCRIPTION = "Baseline regression (single output, 30 features)"

# ============================================================================
# Task Configuration
# ============================================================================
TASK_TYPE = "regression"
OUTPUT_HEAD_NAME = "RegressionHead"

# Single-output regression — one continuous target
# For multi-output regression: e.g. TARGET_PARAMS = {"outputs": ["target_0", "target_1"]}
TARGET_PARAMS = {
    "outputs": ["target"],
}

# Feature columns — must match the columns in the episode CSV
FEATURE_COLUMNS = [f"feature_{i:02d}" for i in range(30)]

# Episode metadata
EPISODES_CSV = "datasets/episodes_reg_v1.csv"
SPLIT_DIR = "data/example_regressor_dnn/splits/exp01"

# Split configuration
TEST_SPLIT_RATIO = 0.10
VALID_SPLIT_RATIO = 0.10
TRAIN_SPLIT_RATIO = 1.0 - VALID_SPLIT_RATIO - TEST_SPLIT_RATIO

# Random seeds
RANDOM_SEED_PYTHON = 51241
RANDOM_SEED_NP = 142341
RANDOM_SEED_PT = 75192
SPLIT_RANDOM_SEED = 42

# ============================================================================
# Model Architecture Configuration
# ============================================================================
MODEL_CONFIG_PARAMS = {
    "d_input": 30,
    "d_hidden": [128, 64, 32],
    "d_output": len(TARGET_PARAMS["outputs"]),
    "dropout": 0.1,
    "activation": "relu",
}

# ============================================================================
# Training Configuration
# ============================================================================
N_EPOCHS = 50
BATCH_SIZE = 64
N_WORKERS = 4

# Learning rate
LR0 = 1e-3
LR_SCHEDULER_START_END = [1.0, 0.01]
LR_SCHEDULER_TOTAL_ITERS = 40

# Loss function
LOSS_FN = "MSELoss"

# Early stopping
PATIENCE = 10
EARLY_STOPPING_MODE = "min"
METRIC_WEIGHTS = {
    "mae": -1.0,
    "rmse": -1.0,
}

# Checkpointing
SAVE_AFTER_EVERY_EPOCH = True
RESUME_TRAINING = False
