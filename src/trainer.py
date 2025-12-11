"""
ML Model Training Module for Pattern Recognition

This module handles training classifiers to predict market regimes:
- ranging (0)
- trending_up (1)
- trending_down (2)

Uses XGBoost as primary model with RandomForest as backup.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import xgboost as xgb

try:
    from src.config import MODELS_DIR, TRAIN_TEST_SPLIT, RANDOM_STATE, FEATURES_DIR, LABELS_DIR
except ImportError:
    from .config import MODELS_DIR, TRAIN_TEST_SPLIT, RANDOM_STATE, FEATURES_DIR, LABELS_DIR


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, list]:
    """
    Prepare feature matrix from DataFrame by dropping non-feature columns.

    Args:
        df: DataFrame with features and metadata

    Returns:
        Tuple of (X array, feature_names list)
    """
    # Columns to drop (metadata, not features)
    drop_cols = ['open_time', 'label', 'timestamp', 'date', 'datetime',
                 'close_time', 'open', 'high', 'low', 'close', 'volume']

    # Only drop columns that actually exist
    cols_to_drop = [col for col in drop_cols if col in df.columns]

    X = df.drop(columns=cols_to_drop)
    feature_names = X.columns.tolist()

    return X.values, feature_names


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'xgboost'
) -> Any:
    """
    Train a classifier model.

    Args:
        X: Feature matrix
        y: Target labels (0=ranging, 1=trending_up, 2=trending_down)
        model_type: 'xgboost' or 'randomforest'

    Returns:
        Trained model
    """
    if model_type.lower() == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
    elif model_type.lower() == 'randomforest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'xgboost' or 'randomforest'")

    model.fit(X, y)
    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate model performance on test data.

    Args:
        model: Trained model
        X_test: Test feature matrix
        y_test: Test labels

    Returns:
        Dictionary with metrics:
        - accuracy: Overall accuracy
        - precision: Weighted precision across classes
        - recall: Weighted recall across classes
        - f1: Weighted F1 score
        - confusion_matrix: Confusion matrix
    """
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    return metrics


def save_model(model: Any, symbol: str, interval: str) -> Path:
    """
    Save trained model to disk.

    Args:
        model: Trained model
        symbol: Trading symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h')

    Returns:
        Path to saved model file
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{symbol}_{interval}_model.joblib"
    joblib.dump(model, model_path)

    return model_path


def load_model(symbol: str, interval: str) -> Any:
    """
    Load trained model from disk.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h')

    Returns:
        Loaded model

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    model_path = MODELS_DIR / f"{symbol}_{interval}_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Make predictions using trained model.

    Args:
        model: Trained model
        X: Feature matrix

    Returns:
        Array of predicted class labels
    """
    return model.predict(X)


def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Get probability predictions for each class.

    Args:
        model: Trained model
        X: Feature matrix

    Returns:
        Array of shape (n_samples, n_classes) with probabilities
        Columns: [ranging, trending_up, trending_down]
    """
    return model.predict_proba(X)


def get_feature_importance(
    model: Any,
    feature_names: list
) -> pd.DataFrame:
    """
    Get feature importances from trained model.

    Args:
        model: Trained model (must have feature_importances_ attribute)
        feature_names: List of feature names

    Returns:
        DataFrame with columns ['feature', 'importance'] sorted by importance
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")

    importances = model.feature_importances_

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    return df.sort_values('importance', ascending=False).reset_index(drop=True)


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'xgboost',
    cv: int = 5
) -> Dict[str, Any]:
    """
    Perform cross-validation on the model.

    Args:
        X: Feature matrix
        y: Target labels
        model_type: 'xgboost' or 'randomforest'
        cv: Number of cross-validation folds

    Returns:
        Dictionary with:
        - scores: Array of CV scores
        - mean_score: Mean CV score
        - std_score: Standard deviation of CV scores
    """
    # Create model instance
    if model_type.lower() == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
    elif model_type.lower() == 'randomforest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    return {
        'scores': scores.tolist(),
        'mean_score': scores.mean(),
        'std_score': scores.std()
    }


def train_pipeline(
    symbol: str,
    interval: str,
    model_type: str = 'xgboost',
    save: bool = True
) -> Dict[str, Any]:
    """
    Complete training pipeline: load data, train, evaluate, save.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h')
        model_type: 'xgboost' or 'randomforest'
        save: Whether to save the trained model

    Returns:
        Dictionary with:
        - model: Trained model
        - metrics: Evaluation metrics
        - feature_names: List of feature names
        - model_path: Path to saved model (if save=True)
    """
    # Load features
    features_path = FEATURES_DIR / f"{symbol}_{interval}_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    df = pd.read_csv(features_path)

    # Load labels if they exist in a separate file
    labels_path = LABELS_DIR / f"{symbol}_{interval}_labels.csv"
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        df['label'] = labels_df['label']

    # Filter to only labeled rows
    if 'label' not in df.columns:
        raise ValueError("No 'label' column found in features or labels file")

    df_labeled = df.dropna(subset=['label']).copy()

    if len(df_labeled) == 0:
        raise ValueError("No labeled data found")

    print(f"Training on {len(df_labeled)} labeled samples")
    print(f"Label distribution:\n{df_labeled['label'].value_counts()}")

    # Prepare features
    y = df_labeled['label'].values.astype(int)
    X, feature_names = prepare_features(df_labeled)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=(1 - TRAIN_TEST_SPLIT),
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train model
    print(f"\nTraining {model_type} model...")
    model = train_model(X_train, y_train, model_type=model_type)

    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {metrics['confusion_matrix']}")

    result = {
        'model': model,
        'metrics': metrics,
        'feature_names': feature_names
    }

    # Save model
    if save:
        model_path = save_model(model, symbol, interval)
        result['model_path'] = model_path
        print(f"\nModel saved to: {model_path}")

    return result


def main():
    """
    Demonstration of training pipeline with sample data.
    """
    print("=" * 60)
    print("ML Model Training Demo")
    print("=" * 60)

    # Create sample data if features don't exist
    symbol = "BTCUSDT"
    interval = "1h"

    features_path = FEATURES_DIR / f"{symbol}_{interval}_features.csv"

    if not features_path.exists():
        print("\nGenerating sample training data...")
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)

        # Create sample features
        np.random.seed(RANDOM_STATE)
        n_samples = 1000

        # Sample features (simulating MAR-based features)
        data = {
            'open_time': pd.date_range('2024-01-01', periods=n_samples, freq='1h'),
            'close': np.random.randn(n_samples).cumsum() + 50000,
            'ma_5': np.random.randn(n_samples).cumsum() + 50000,
            'ma_10': np.random.randn(n_samples).cumsum() + 50000,
            'ma_20': np.random.randn(n_samples).cumsum() + 50000,
            'spread': np.abs(np.random.randn(n_samples)) * 100,
            'compression': np.random.uniform(0, 1, n_samples),
            'slope_5': np.random.randn(n_samples) * 10,
            'slope_10': np.random.randn(n_samples) * 8,
            'slope_20': np.random.randn(n_samples) * 5,
            'position_in_range': np.random.uniform(0, 1, n_samples),
            'volatility': np.abs(np.random.randn(n_samples)) * 50,
        }

        # Generate labels based on simple rules
        labels = []
        for i in range(n_samples):
            if data['compression'][i] > 0.7:
                labels.append(0)  # ranging
            elif data['slope_20'][i] > 5:
                labels.append(1)  # trending_up
            elif data['slope_20'][i] < -5:
                labels.append(2)  # trending_down
            else:
                labels.append(0)  # ranging

        data['label'] = labels

        df = pd.DataFrame(data)
        df.to_csv(features_path, index=False)
        print(f"Sample data saved to: {features_path}")

    # Train both model types
    for model_type in ['xgboost', 'randomforest']:
        print(f"\n{'=' * 60}")
        print(f"Training {model_type.upper()} Model")
        print(f"{'=' * 60}")

        try:
            result = train_pipeline(
                symbol=symbol,
                interval=interval,
                model_type=model_type,
                save=True
            )

            # Show feature importances
            print("\nTop 10 Feature Importances:")
            importance_df = get_feature_importance(
                result['model'],
                result['feature_names']
            )
            print(importance_df.head(10).to_string(index=False))

            # Cross-validation
            print(f"\nPerforming 5-fold cross-validation...")
            features_df = pd.read_csv(features_path)
            features_df = features_df.dropna(subset=['label'])
            y = features_df['label'].values.astype(int)
            X, _ = prepare_features(features_df)

            cv_results = cross_validate(X, y, model_type=model_type, cv=5)
            print(f"CV Scores: {[f'{s:.4f}' for s in cv_results['scores']]}")
            print(f"Mean CV Score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")

        except Exception as e:
            print(f"Error training {model_type}: {e}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
