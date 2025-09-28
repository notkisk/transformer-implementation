"""
Configuration validation for Transformer implementation.
Validates configuration parameters against paper specifications.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config
from typing import Dict, Any, List
import warnings


class ConfigError(Exception):
    """Custom exception for configuration validation errors."""

    pass


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration parameters against paper specifications.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List of validation warnings
    """
    warnings_list = []

    # Validate model dimension constraints
    if config["d_model"] <= 0:
        raise ConfigError(f"d_model ({config['d_model']}) must be positive")

    if config["d_model"] % 8 != 0:
        warnings_list.append(
            f"d_model ({config['d_model']}) should be divisible by number of attention heads (recommended: 8)"
        )

    # Validate number of layers
    if config.get("num_layers", 6) != 6:
        warnings_list.append(
            f"paper uses 6 layers, but config specifies {config.get('num_layers', 6)}"
        )

    # Validate attention heads
    num_heads = config.get("num_heads", 8)
    if config["d_model"] % num_heads != 0:
        raise ConfigError(
            f"d_model ({config['d_model']}) must be divisible by number of heads ({num_heads})"
        )

    # Validate sequence length
    if config["seq_len"] > 512:
        warnings_list.append(
            f"positional encodings may not work well for sequences longer than 512 (current: {config['seq_len']})"
        )

    # Validate learning rate schedule parameters
    if config.get("warmup_steps", 4000) != 4000:
        warnings_list.append(
            f"paper uses 4000 warmup steps (current: {config.get('warmup_steps', 4000)})"
        )

    # Validate dropout rate
    if config.get("dropout", 0.1) != 0.1:
        warnings_list.append(
            f"paper uses 0.1 dropout rate (current: {config.get('dropout', 0.1)})"
        )

    # Validate feed-forward dimension
    d_ff = config.get("d_ff", 2048)
    if d_ff != 2048:
        warnings_list.append(f"paper uses d_ff=2048 (current: {d_ff})")

    return warnings_list


def validate_and_get_config() -> Dict[str, Any]:
    """
    Get configuration and validate it against paper specifications.

    Returns:
        Validated configuration dictionary
    """
    config = get_config()
    warnings_list = validate_config(config)

    for warning in warnings_list:
        warnings.warn(warning)

    print("PASS Configuration validation passed")
    return config


if __name__ == "__main__":
    # Example usage
    try:
        config = validate_and_get_config()
        print("Configuration is valid for paper implementation")
    except ConfigError as e:
        print(f"Configuration error: {e}")
