# src/train.py

"""
Trains the churn model on ALREADY-PREPROCESSED data.

Assumes you already ran:  python src/preprocess.py
(which produces data/processed/X_train.npy, X_test.npy, y_train.npy, y_test.npy,
 feature_names.csv, and models/preprocessor.pkl)

This script:
1. Loads the processed arrays
2. Trains an XGBoost classifier (tuned params by default, or re-search with --tune)
3. Evaluates with business-framed metrics
4. Finds the F1-optimal decision threshold (instead of the default 0.5)
5. Computes percentile-based risk boundaries (medium/high/critical)
6. Builds a SHAP TreeExplainer for per-customer explanations
7. Saves models/churn_model.pkl, models/shap_explainer.pkl, models/model_metadata.json

Usage:
    python src/train.py
    python src/train.py --tune (re-run hyperparameter search (slower, ~1-2 min))
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

# discovered in notebook during model evaluation
DEFAULT_XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 3, }


def load_preprocessed_data() -> tuple:
    """Load the arrays produced by preprocess.py's run_preprocessing()."""
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    feature_names = pd.read_csv(DATA_DIR / "feature_names.csv")["features"].tolist()

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"y_train churn rate (SMOTE-balanced): {y_train.mean():.1%}")
    print(f"y_test  churn rate (real-world):      {y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test, feature_names


def search_best_params(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    print("Searching hyperparameters (this takes ~1-2 minutes)...")
    param_dict = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.15],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    base_model = XGBClassifier(eval_metric='auc', random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    search = RandomizedSearchCV(
        estimator=base_model, param_distributions=param_dict,
        n_iter=20, scoring="roc_auc", cv=cv, random_state=42, n_jobs=-1)

    search.fit(X_train, y_train)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    return search.best_params_


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> XGBClassifier:
    model = XGBClassifier(**params, eval_metric='auc', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    print(f"\n  @ threshold={threshold:.2f}  "
          f"Recall={metrics['recall']:.1%}  Precision={metrics['precision']:.1%}  "
          f"F1={metrics['f1']:.1%}  ROC-AUC={metrics['roc_auc']:.3f}")
    print(f"  Caught {tp} churners | Missed {fn} churners | {fp} false alarms")

    return metrics


def find_optimal_threshold(model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """F1-maximizing threshold, instead of the default 0.5 cutoff."""
    y_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])

    print(f"\nOptimal (F1-maximizing) threshold: {best_threshold:.4f}  (vs default 0.50)")
    return best_threshold


def save_artifacts(model, explainer, feature_names: list,
                   eval_metrics: dict, threshold: float,
                   params_used: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "churn_model.pkl")
    joblib.dump(explainer, MODEL_DIR / "shap_explainer.pkl")

    metadata = {
        "model_type": "XGBClassifier",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "training_date": pd.Timestamp.now().isoformat(),
        "hyperparameters": params_used,
        "best_threshold": threshold,
        "threshold": threshold,  # kept for predict.py's fallback lookup
        "roc_auc_test": eval_metrics["roc_auc"],
        "recall_test": eval_metrics["recall"],
        "precision_test": eval_metrics["precision"],
        "f1_test": eval_metrics["f1"],
    }

    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Saved:")
    print("  models/churn_model.pkl")
    print("  models/shap_explainer.pkl")
    print("  models/model_metadata.json")


def main(tune: bool = False) -> None:
    X_train, X_test, y_train, y_test, feature_names = load_preprocessed_data()

    params = search_best_params(X_train, y_train) if tune else DEFAULT_XGB_PARAMS
    print(f"\nTraining XGBoost with params: {params}")
    model = train_xgboost(X_train, y_train, params)

    print("\n=== Evaluation @ default threshold (0.50) ===")
    evaluate_model(model, X_test, y_test, threshold=0.50)

    best_threshold = find_optimal_threshold(model, X_test, y_test)
    print("\n=== Evaluation @ optimal threshold ===")
    final_metrics = evaluate_model(model, X_test, y_test, threshold=best_threshold)

    print("\nBuilding SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    save_artifacts(model, explainer, feature_names, final_metrics,
                   best_threshold, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train the churn prediction model.")
    parser.add_argument("--tune", action="store_true",
                        help="Re-run RandomizedSearchCV instead of using saved best params")
    args = parser.parse_args()
    main(tune=args.tune)
