from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS


LABEL_ORDER = ["away_win", "draw", "home_win"]


def train_outcome_model(X_train: pd.DataFrame, y_train: pd.Series, sample_weight=None):
    try:
        from xgboost import XGBClassifier

        label_map = {label: idx for idx, label in enumerate(LABEL_ORDER)}
        y_numeric = y_train.map(label_map)
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=42,
        )
        model.fit(X_train[FEATURE_COLUMNS], y_numeric, sample_weight=sample_weight)
        model.label_order_ = LABEL_ORDER
        return model
    except Exception:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(X_train[FEATURE_COLUMNS], y_train, clf__sample_weight=sample_weight)
        return model


def train_goals_models(X_train: pd.DataFrame, y_home_goals: pd.Series, y_away_goals: pd.Series, sample_weight=None):
    home_model = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    away_model = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    fit_kwargs = {"reg__sample_weight": sample_weight} if sample_weight is not None else {}
    home_model.fit(X_train[FEATURE_COLUMNS], y_home_goals, **fit_kwargs)
    away_model.fit(X_train[FEATURE_COLUMNS], y_away_goals, **fit_kwargs)
    return home_model, away_model


def predict_proba_frame(model, X: pd.DataFrame) -> pd.DataFrame:
    raw = model.predict_proba(X[FEATURE_COLUMNS])
    classes = getattr(model, "classes_", getattr(model, "label_order_", LABEL_ORDER))
    if hasattr(classes, "tolist"):
        classes = classes.tolist()
    if classes and isinstance(classes[0], (int, np.integer)):
        classes = [LABEL_ORDER[int(c)] for c in classes]
    proba = pd.DataFrame(raw, columns=classes, index=X.index)
    for label in LABEL_ORDER:
        if label not in proba:
            proba[label] = 0.0
    return proba[LABEL_ORDER]


def predict_match(outcome_model, home_goals_model, away_goals_model, match_features: pd.DataFrame):
    proba = predict_proba_frame(outcome_model, match_features).iloc[0].to_dict()
    expected_home = float(home_goals_model.predict(match_features[FEATURE_COLUMNS])[0])
    expected_away = float(away_goals_model.predict(match_features[FEATURE_COLUMNS])[0])
    return proba, max(expected_home, 0.05), max(expected_away, 0.05)
