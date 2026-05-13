"""Config serialization utilities."""

import json
from pathlib import Path
from typing import Any, Dict


def save_model_config(config: Dict[str, Any], output_dir: Path) -> Path:
    """Save model config as JSON to output_dir.

    Args:
        config: Configuration dict.
        output_dir: Directory to save the config file.

    Returns:
        Path to the saved config file.
    """
    path = output_dir / "model_config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return path


def load_model_config(config_path: Path) -> Dict[str, Any]:
    """Load model config from JSON.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        Configuration dict.
    """
    with open(config_path) as f:
        return json.load(f)
