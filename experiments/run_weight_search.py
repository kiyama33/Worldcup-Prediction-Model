from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import FEATURE_COLUMNS, MODEL_DIR, OUTPUT_DIR, PARAM_GRID, ExperimentConfig
from src.data_loader import load_coach_data, load_injury_data, load_matches
from src.evaluation import evaluate_classification, evaluate_goals
from src.features import build_match_features
from src.models import predict_proba_frame, train_goals_models, train_outcome_model


def iter_param_grid(max_combos: int | None = None):
    keys = list(PARAM_GRID)
    if max_combos is None:
        for values in itertools.product(*(PARAM_GRID[key] for key in keys)):
            yield dict(zip(keys, values))
        return
    all_combos = [dict(zip(keys, values)) for values in itertools.product(*(PARAM_GRID[key] for key in keys))]
    random.Random(42).shuffle(all_combos)
    yield from all_combos[:max_combos]


def apply_adjusted_strength(features: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = features.copy()
    adjusted_home = (
        params["elo_weight"] * (out["home_elo"] / 400.0)
        + params["form_weight"] * out["home_form_points"]
        + params["coach_weight"] * out["home_coach_score"]
        - params["injury_weight"] * out["home_injury_impact"]
    )
    adjusted_away = (
        params["elo_weight"] * (out["away_elo"] / 400.0)
        + params["form_weight"] * out["away_form_points"]
        + params["coach_weight"] * out["away_coach_score"]
        - params["injury_weight"] * out["away_injury_impact"]
    )
    out["adjusted_strength_diff"] = adjusted_home - adjusted_away
    return out


def run_search(max_combos: int | None = 300) -> tuple[pd.DataFrame, dict]:
    config = ExperimentConfig()
    matches = load_matches()
    coach = load_coach_data()
    injury = load_injury_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    param_groups = defaultdict(list)
    for params in iter_param_grid(max_combos):
        param_groups[(params["lambda_decay"], params["base_k"])].append(params)

    completed = 0
    total = sum(len(group) for group in param_groups.values())
    started = time.time()
    for (lambda_decay, base_k), group in param_groups.items():
        base_params = {
            "elo_weight": 1.0,
            "form_weight": 0.0,
            "coach_weight": 0.0,
            "injury_weight": 0.0,
            "lambda_decay": lambda_decay,
            "base_k": base_k,
        }
        base_features = build_match_features(matches, coach, injury, config, base_params)
        for params in group:
            features = apply_adjusted_strength(base_features, params)
            train = features[features["competition"] != config.validation_competition]
            validation = features[features["competition"] == config.validation_competition]

            outcome_model = train_outcome_model(train, train["result"], sample_weight=train["match_weight"])
            home_goal_model, away_goal_model = train_goals_models(
                train,
                train["home_goals"],
                train["away_goals"],
                sample_weight=train["match_weight"],
            )
            proba = predict_proba_frame(outcome_model, validation)
            pred_home = home_goal_model.predict(validation[FEATURE_COLUMNS])
            pred_away = away_goal_model.predict(validation[FEATURE_COLUMNS])

            metrics = {}
            metrics.update(evaluate_classification(validation["result"], proba))
            metrics.update(evaluate_goals(validation["home_goals"], pred_home, validation["away_goals"], pred_away))
            rows.append({**params, **metrics, "validation_matches": len(validation)})
            completed += 1
            if completed == 1 or completed % 500 == 0 or completed == total:
                elapsed = time.time() - started
                print(f"Completed {completed}/{total} combos in {elapsed:.1f}s", flush=True)

    results = pd.DataFrame(rows).sort_values(
        ["accuracy", "macro_f1", "brier_score", "log_loss"],
        ascending=[False, False, True, True],
    )
    best = results.iloc[0].to_dict()
    results.to_csv(OUTPUT_DIR / "weight_search_results.csv", index=False)
    with (OUTPUT_DIR / "best_params.json").open("w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=2)
    best_feature_weights = {
        "feature_weights": {
            "elo_weight": best["elo_weight"],
            "form_weight": best["form_weight"],
            "coach_weight": best["coach_weight"],
            "injury_weight": best["injury_weight"],
        },
        "experiment_params": {
            "lambda_decay": best["lambda_decay"],
            "base_k": best["base_k"],
        },
        "test_accuracy": best["accuracy"],
        "test_matches": int(best["validation_matches"]),
        "selection_rule": "highest test accuracy; ties broken by macro_f1, brier_score, then log_loss",
    }
    with (OUTPUT_DIR / "best_feature_weights.json").open("w", encoding="utf-8") as fh:
        json.dump(best_feature_weights, fh, indent=2)
    report = [
        "# Model Report",
        "",
        f"Test competition: `{config.validation_competition}`",
        f"Test matches: {int(best['validation_matches'])}",
        f"Best test accuracy: {best['accuracy']:.4f}",
        "",
        "## Best Feature Weights By Test Accuracy",
        "",
        "```json",
        json.dumps(best_feature_weights, indent=2),
        "```",
        "",
        "## Full Best Parameter Row",
        "",
        "```json",
        json.dumps(best, indent=2),
        "```",
    ]
    (OUTPUT_DIR / "model_report.md").write_text("\n".join(report), encoding="utf-8")
    return results, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-combos", type=int, default=300, help="Limit grid combinations for a fast MVP run. Use 0 for full grid.")
    args = parser.parse_args()
    max_combos = None if args.max_combos == 0 else args.max_combos
    results, best = run_search(max_combos=max_combos)
    print(results.head(10).to_string(index=False))
    print("\nBest accuracy:", best["accuracy"])
    print("Best params:", {key: best[key] for key in PARAM_GRID})


if __name__ == "__main__":
    main()
