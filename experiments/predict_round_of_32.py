from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, FEATURE_COLUMNS, OUTPUT_DIR, ExperimentConfig
from src.data_loader import load_coach_data, load_injury_data, load_matches
from src.evaluation import evaluate_classification
from src.features import build_match_features
from src.models import predict_match, predict_proba_frame, train_goals_models, train_outcome_model
from src.poisson_score import get_top_scorelines, scoreline_distribution, scoreline_wdl_probabilities


R32_PATH = DATA_DIR / "raw" / "worldcup_2026_round_of_32.csv"
DISAGREEMENT_THRESHOLDS = [0.15, 0.20, 0.25, 0.30]
MODEL_WARNING = "classification model and goal model disagree."


def load_best_params() -> dict:
    raw = json.loads((OUTPUT_DIR / "best_feature_weights.json").read_text(encoding="utf-8"))
    return {
        **raw["feature_weights"],
        **raw["experiment_params"],
    }


def read_round_of_32() -> pd.DataFrame:
    fixtures = pd.read_csv(R32_PATH)
    fixtures["date"] = pd.to_datetime(fixtures["date"])
    return fixtures


def predicted_advancer(row: pd.Series) -> str:
    if row["p_home_win"] >= row["p_away_win"]:
        return row["home_team"]
    return row["away_team"]


def model_conflict(row: pd.Series) -> bool:
    return bool(
        (row["p_home_win"] > 0.60 and row["expected_home_goals"] < row["expected_away_goals"])
        or (row["p_away_win"] > 0.60 and row["expected_away_goals"] < row["expected_home_goals"])
    )


def max_wdl_probability_diff(row: pd.Series) -> float:
    return float(
        max(
            abs(row["p_home_win"] - row["poisson_home_win"]),
            abs(row["p_draw"] - row["poisson_draw"]),
            abs(row["p_away_win"] - row["poisson_away_win"]),
        )
    )


def warning_mask(rows: pd.DataFrame, threshold: float) -> pd.Series:
    return rows["model_conflict"] | (rows["max_wdl_probability_diff"] > threshold)


def select_disagreement_threshold(validation_rows: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    threshold_rows = []
    for threshold in DISAGREEMENT_THRESHOLDS:
        mask = warning_mask(validation_rows, threshold)
        warned = validation_rows[mask]
        warning_count = int(mask.sum())
        if warning_count:
            classification_error_rate = float((~warned["result_correct"]).mean())
            advancer_error_rate = float((~warned["advancer_correct"]).mean())
            warning_precision = float(((~warned["result_correct"]) | (~warned["advancer_correct"])).mean())
        else:
            classification_error_rate = 0.0
            advancer_error_rate = 0.0
            warning_precision = -1.0
        threshold_rows.append(
            {
                "threshold": threshold,
                "warning_count": warning_count,
                "classification_error_rate_in_warnings": classification_error_rate,
                "advancer_error_rate_in_warnings": advancer_error_rate,
                "warning_precision": warning_precision,
                "is_default": threshold == 0.20,
            }
        )
    summary = pd.DataFrame(threshold_rows)
    ranked = summary.sort_values(
        ["warning_precision", "warning_count", "is_default"],
        ascending=[False, True, False],
    )
    return float(ranked.iloc[0]["threshold"]), summary


def add_consistency_diagnostics(rows: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    out = rows.copy()
    out["model_conflict"] = out.apply(model_conflict, axis=1)
    out["max_wdl_probability_diff"] = out.apply(max_wdl_probability_diff, axis=1)
    if threshold is not None:
        mask = warning_mask(out, threshold)
        out["model_warning"] = ""
        out.loc[mask, "model_warning"] = MODEL_WARNING
    return out


def predict_goal_outputs(home_goals_model, away_goals_model, features: pd.DataFrame) -> pd.DataFrame:
    expected_home = home_goals_model.predict(features[FEATURE_COLUMNS])
    expected_away = away_goals_model.predict(features[FEATURE_COLUMNS])
    return pd.DataFrame(
        {
            "expected_home_goals": [max(float(value), 0.05) for value in expected_home],
            "expected_away_goals": [max(float(value), 0.05) for value in expected_away],
        },
        index=features.index,
    )


def poisson_outputs(expected_home: float, expected_away: float) -> tuple[list[dict], dict[str, float]]:
    distribution = scoreline_distribution(expected_home, expected_away)
    return get_top_scorelines(distribution, 5), scoreline_wdl_probabilities(distribution)


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_future_feature(
    base_known_matches: pd.DataFrame,
    fixture: pd.Series,
    coach: pd.DataFrame,
    injury: pd.DataFrame,
    config: ExperimentConfig,
    params: dict,
) -> pd.DataFrame:
    dummy = fixture.copy()
    dummy["home_goals"] = 0
    dummy["away_goals"] = 0
    combined = pd.concat([base_known_matches, pd.DataFrame([dummy])], ignore_index=True, sort=False)
    features = build_match_features(combined, coach, injury, config, params)
    return features.tail(1)


def build_validation_rows(
    validation: pd.DataFrame,
    completed: pd.DataFrame,
    outcome_model,
    home_goals_model,
    away_goals_model,
) -> tuple[pd.DataFrame, dict[str, float]]:
    validation_proba = predict_proba_frame(outcome_model, validation)
    validation_rows = validation[["date", "home_team", "away_team", "home_goals", "away_goals", "result"]].join(
        validation_proba.rename(columns={"home_win": "p_home_win", "draw": "p_draw", "away_win": "p_away_win"})
    )
    validation_rows = validation_rows.join(predict_goal_outputs(home_goals_model, away_goals_model, validation))
    poisson_rows = []
    for _, row in validation_rows.iterrows():
        top_scorelines, poisson_wdl = poisson_outputs(row["expected_home_goals"], row["expected_away_goals"])
        poisson_rows.append(
            {
                **poisson_wdl,
                "top_scorelines": json.dumps(top_scorelines, ensure_ascii=False),
            }
        )
    validation_rows = validation_rows.join(pd.DataFrame(poisson_rows, index=validation_rows.index))
    validation_rows["predicted_result"] = validation_proba.idxmax(axis=1)
    validation_rows["actual_advancer"] = completed["actual_advancer"].to_numpy()
    validation_rows["predicted_advancer"] = validation_rows.apply(predicted_advancer, axis=1)
    validation_rows["result_correct"] = validation_rows["predicted_result"] == validation_rows["result"]
    validation_rows["advancer_correct"] = validation_rows["predicted_advancer"] == validation_rows["actual_advancer"]
    validation_rows = add_consistency_diagnostics(validation_rows)
    validation_metrics = evaluate_classification(validation["result"], validation_proba)
    validation_metrics["advancer_accuracy"] = float(validation_rows["advancer_correct"].mean())
    return validation_rows, validation_metrics


def main() -> None:
    params = load_best_params()
    config = ExperimentConfig()
    coach = load_coach_data()
    injury = load_injury_data()
    base_matches = load_matches()
    r32 = read_round_of_32()
    completed = r32[r32["status"] == "completed"].copy()
    upcoming = r32[r32["status"] == "upcoming"].copy()

    train_features = build_match_features(base_matches, coach, injury, config, params)
    validation_matches = pd.concat(
        [base_matches, completed.drop(columns=["status", "actual_advancer", "match_no"], errors="ignore")],
        ignore_index=True,
        sort=False,
    )
    validation_features = build_match_features(validation_matches, coach, injury, config, params)
    validation = validation_features[validation_features["competition"] == "world_cup_2026_r32"].copy()

    outcome_model = train_outcome_model(train_features, train_features["result"], sample_weight=train_features["match_weight"])
    home_goals_model, away_goals_model = train_goals_models(
        train_features,
        train_features["home_goals"],
        train_features["away_goals"],
        sample_weight=train_features["match_weight"],
    )
    validation_rows, validation_metrics = build_validation_rows(
        validation,
        completed,
        outcome_model,
        home_goals_model,
        away_goals_model,
    )
    selected_threshold, threshold_summary = select_disagreement_threshold(validation_rows)
    validation_rows = add_consistency_diagnostics(validation_rows, selected_threshold)
    validation_metrics["selected_disagreement_threshold"] = selected_threshold

    final_known_matches = pd.concat(
        [base_matches, completed.drop(columns=["status", "actual_advancer", "match_no"], errors="ignore")],
        ignore_index=True,
        sort=False,
    )
    final_known_features = build_match_features(final_known_matches, coach, injury, config, params)
    final_outcome_model = train_outcome_model(
        final_known_features,
        final_known_features["result"],
        sample_weight=final_known_features["match_weight"],
    )
    final_home_goals_model, final_away_goals_model = train_goals_models(
        final_known_features,
        final_known_features["home_goals"],
        final_known_features["away_goals"],
        sample_weight=final_known_features["match_weight"],
    )

    prediction_rows = []
    for _, fixture in upcoming.sort_values(["date", "match_no"]).iterrows():
        feature_row = build_future_feature(final_known_matches, fixture, coach, injury, config, params)
        proba, expected_home, expected_away = predict_match(
            final_outcome_model,
            final_home_goals_model,
            final_away_goals_model,
            feature_row,
        )
        poisson_top, poisson_wdl = poisson_outputs(expected_home, expected_away)
        prediction_rows.append(
            {
                "date": fixture["date"].strftime("%Y-%m-%d"),
                "match_no": int(fixture["match_no"]),
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "p_home_win": proba["home_win"],
                "p_draw": proba["draw"],
                "p_away_win": proba["away_win"],
                "predicted_result": max(proba, key=proba.get),
                "predicted_advancer": fixture["home_team"] if proba["home_win"] >= proba["away_win"] else fixture["away_team"],
                "expected_home_goals": expected_home,
                "expected_away_goals": expected_away,
                "top_scorelines": json.dumps(poisson_top, ensure_ascii=False),
                **poisson_wdl,
            }
        )

    predictions = add_consistency_diagnostics(pd.DataFrame(prediction_rows), selected_threshold)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validation_path = OUTPUT_DIR / "round_of_32_validation.csv"
    predictions_path = OUTPUT_DIR / "round_of_32_predictions.csv"
    report_path = OUTPUT_DIR / "round_of_32_report.md"
    threshold_path = OUTPUT_DIR / "round_of_32_threshold_selection.csv"
    validation_rows.to_csv(validation_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    threshold_summary.to_csv(threshold_path, index=False)

    report = [
        "# Round of 32 Validation and Predictions",
        "",
        "## Validation Metrics",
        "",
        "```json",
        json.dumps(validation_metrics, indent=2),
        "```",
        "",
        "## Threshold Selection",
        "",
        markdown_table(threshold_summary),
        "",
        "## Test Predictions",
        "",
        markdown_table(
            predictions[
                [
                    "date",
                    "home_team",
                    "away_team",
                    "predicted_advancer",
                    "p_home_win",
                    "p_draw",
                    "p_away_win",
                    "expected_home_goals",
                    "expected_away_goals",
                    "poisson_home_win",
                    "poisson_draw",
                    "poisson_away_win",
                    "max_wdl_probability_diff",
                    "model_conflict",
                    "model_warning",
                ]
            ]
        ),
        "",
        "## Test Warnings",
        "",
        markdown_table(
            predictions[predictions["model_warning"] != ""][
                [
                    "date",
                    "home_team",
                    "away_team",
                    "predicted_advancer",
                    "p_home_win",
                    "p_draw",
                    "p_away_win",
                    "expected_home_goals",
                    "expected_away_goals",
                    "poisson_home_win",
                    "poisson_draw",
                    "poisson_away_win",
                    "max_wdl_probability_diff",
                    "model_conflict",
                    "model_warning",
                ]
            ]
        ),
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(json.dumps({"validation_metrics": validation_metrics}, indent=2))
    print(
        predictions[
            [
                "date",
                "home_team",
                "away_team",
                "predicted_advancer",
                "p_home_win",
                "p_draw",
                "p_away_win",
                "expected_home_goals",
                "expected_away_goals",
                "poisson_home_win",
                "poisson_draw",
                "poisson_away_win",
                "model_conflict",
                "model_warning",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote {validation_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {threshold_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
