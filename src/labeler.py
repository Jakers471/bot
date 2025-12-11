"""
Label management module for ML model training.

Handles storing, loading, and managing user-created labels for market data.
Labels are stored as JSON files in the labels directory.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from .config import LABELS_DIR, LABEL_CLASSES


def _get_label_path(symbol: str, interval: str) -> Path:
    """Get the file path for a symbol/interval label file."""
    return LABELS_DIR / f"{symbol}_{interval}_labels.json"


def _create_empty_labels_structure(symbol: str, interval: str) -> dict:
    """Create an empty labels data structure."""
    return {
        "labels": [],
        "metadata": {
            "symbol": symbol,
            "interval": interval,
            "label_classes": {str(k): v for k, v in LABEL_CLASSES.items()}
        }
    }


def load_labels(symbol: str, interval: str) -> dict:
    """
    Load labels from JSON file for given symbol and interval.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '5m')

    Returns:
        Dictionary containing labels and metadata
    """
    label_path = _get_label_path(symbol, interval)

    if not label_path.exists():
        # Return empty structure if file doesn't exist
        return _create_empty_labels_structure(symbol, interval)

    with open(label_path, 'r') as f:
        return json.load(f)


def save_labels(symbol: str, interval: str, labels_data: dict) -> None:
    """
    Save labels to JSON file.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '5m')
        labels_data: Dictionary containing labels and metadata
    """
    label_path = _get_label_path(symbol, interval)

    # Ensure labels directory exists
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_path, 'w') as f:
        json.dump(labels_data, f, indent=2)


def add_label(symbol: str, interval: str, start_idx: int, end_idx: int, label: int) -> None:
    """
    Add a new label to the labels file.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '5m')
        start_idx: Starting index of the labeled region
        end_idx: Ending index of the labeled region
        label: Label class (0=ranging, 1=trending_up, 2=trending_down)
    """
    labels_data = load_labels(symbol, interval)

    # Validate label
    if label not in LABEL_CLASSES:
        raise ValueError(f"Invalid label {label}. Must be one of {list(LABEL_CLASSES.keys())}")

    # Validate indices
    if start_idx >= end_idx:
        raise ValueError(f"start_idx ({start_idx}) must be less than end_idx ({end_idx})")

    # Create new label entry
    new_label = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "label": label,
        "created_at": datetime.now().isoformat()
    }

    labels_data["labels"].append(new_label)
    save_labels(symbol, interval, labels_data)


def remove_label(symbol: str, interval: str, label_index: int) -> None:
    """
    Remove a label by its index in the labels list.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '5m')
        label_index: Index of the label to remove
    """
    labels_data = load_labels(symbol, interval)

    if label_index < 0 or label_index >= len(labels_data["labels"]):
        raise IndexError(f"Label index {label_index} out of range")

    labels_data["labels"].pop(label_index)
    save_labels(symbol, interval, labels_data)


def get_labeled_mask(symbol: str, interval: str, df_length: int) -> np.ndarray:
    """
    Create a mask array where each index contains its label or -1 if unlabeled.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '5m')
        df_length: Length of the dataframe to create mask for

    Returns:
        NumPy array of shape (df_length,) with labels or -1 for unlabeled
    """
    labels_data = load_labels(symbol, interval)

    # Initialize mask with -1 (unlabeled)
    mask = np.full(df_length, -1, dtype=np.int8)

    # Fill in labeled regions
    for label_entry in labels_data["labels"]:
        start_idx = label_entry["start_idx"]
        end_idx = label_entry["end_idx"]
        label = label_entry["label"]

        # Ensure indices are within bounds
        start_idx = max(0, min(start_idx, df_length - 1))
        end_idx = max(0, min(end_idx, df_length))

        # Fill the range with the label
        mask[start_idx:end_idx] = label

    return mask


def labels_to_dataframe(symbol: str, interval: str, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'label' column to the features dataframe.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '5m')
        features_df: DataFrame containing features

    Returns:
        DataFrame with 'label' column added (-1 for unlabeled)
    """
    df = features_df.copy()
    mask = get_labeled_mask(symbol, interval, len(df))
    df['label'] = mask
    return df


def export_training_data(symbol: str, interval: str, features_df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Export labeled data ready for sklearn training.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '5m')
        features_df: Optional DataFrame with features. If None, will attempt to load.

    Returns:
        Tuple of (X, y) where X is feature matrix and y is label array.
        Only returns rows that have labels (excludes unlabeled data).

    Raises:
        ValueError: If no labeled data is found or features_df is None
    """
    if features_df is None:
        raise ValueError("features_df must be provided. Load features first.")

    # Add labels to dataframe
    df_with_labels = labels_to_dataframe(symbol, interval, features_df)

    # Filter out unlabeled data
    labeled_df = df_with_labels[df_with_labels['label'] != -1].copy()

    if len(labeled_df) == 0:
        raise ValueError(f"No labeled data found for {symbol} {interval}")

    # Separate features and labels
    y = labeled_df['label'].values
    X = labeled_df.drop(columns=['label']).values

    return X, y


def get_label_statistics(symbol: str, interval: str) -> Dict:
    """
    Get statistics about the labels for a given symbol/interval.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '5m')

    Returns:
        Dictionary with label statistics
    """
    labels_data = load_labels(symbol, interval)

    stats = {
        "total_labels": len(labels_data["labels"]),
        "label_counts": {LABEL_CLASSES[k]: 0 for k in LABEL_CLASSES},
        "total_bars_labeled": 0,
        "labels": labels_data["labels"]
    }

    # Count labels by class
    for label_entry in labels_data["labels"]:
        label = label_entry["label"]
        label_name = LABEL_CLASSES[label]
        stats["label_counts"][label_name] += 1

        # Count total bars
        bars = label_entry["end_idx"] - label_entry["start_idx"]
        stats["total_bars_labeled"] += bars

    return stats
