"""
=============
EXPERIMENT: example_classifier_dnn — exp01
=============

TASK TYPE:
    Single-label multi-class classification

MODEL ARCHITECTURE:
    Feedforward DNN with configurable hidden layers.
"""


# ============================================================================
# Experiment Identification
# ============================================================================
EXP_NAME = "exp01"
EXP_DESCRIPTION = "Baseline multi-class classification (5 classes, 30 features)"
MODEL_NAME = "example_classifier_dnn"

# ============================================================================
# Task Configuration
# ============================================================================
TASK_TYPE = "classification"
OUTPUT_HEAD_NAME = "ClassificationHead"

# Class names — string labels for each class. Mapped to integer IDs (0, 1, 2, ...)
# in the episode CSV. N_CLASSES is derived automatically.
TARGET_PARAMS = {
    "classes": ["class_A", "class_B", "class_C", "class_D", "class_E"],
}
N_CLASSES = len(TARGET_PARAMS["classes"])

# Feature columns — must match the columns in the episode CSV
FEATURE_COLUMNS = [f"feature_{i:02d}" for i in range(30)]

# Episode metadata
EPISODES_CSV = f"data/{MODEL_NAME}/datasets/episodes_cls_v1.csv"
SPLIT_DIR = f"data/{MODEL_NAME}/splits/exp01"

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
    "d_output": N_CLASSES,
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
LOSS_FN = "CrossEntropyLoss"

# Per-class weights for CrossEntropyLoss (optional; set to None for uniform)
CLASS_WEIGHTS = None

# Early stopping
PATIENCE = 10
EARLY_STOPPING_MODE = "max"
METRIC_WEIGHTS = {
    "accuracy": 1.0,
}

# Checkpointing
SAVE_AFTER_EVERY_EPOCH = True
RESUME_TRAINING = False
