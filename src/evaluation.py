from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error

from .models import LABEL_ORDER


def evaluate_classification(y_true: pd.Series, y_pred_proba: pd.DataFrame) -> dict[str, float]:
    y_pred = y_pred_proba.idxmax(axis=1)
    y_onehot = pd.get_dummies(y_true).reindex(columns=LABEL_ORDER, fill_value=0).to_numpy()
    proba = y_pred_proba[LABEL_ORDER].clip(1e-9, 1.0).to_numpy()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, proba, labels=LABEL_ORDER)),
        "brier_score": float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1))),
    }


def evaluate_goals(y_true_home, y_pred_home, y_true_away, y_pred_away) -> dict[str, float]:
    return {
        "home_goals_mae": float(mean_absolute_error(y_true_home, y_pred_home)),
        "away_goals_mae": float(mean_absolute_error(y_true_away, y_pred_away)),
        "home_goals_rmse": float(mean_squared_error(y_true_home, y_pred_home) ** 0.5),
        "away_goals_rmse": float(mean_squared_error(y_true_away, y_pred_away) ** 0.5),
    }


def evaluate_scoreline_topk(actual_scores: list[str], predicted_scorelines: list[list[dict]], k: int = 3) -> float:
    hits = 0
    for actual, predicted in zip(actual_scores, predicted_scorelines):
        if actual in {row["score"] for row in predicted[:k]}:
            hits += 1
    return hits / max(len(actual_scores), 1)
