from __future__ import annotations

from dataclasses import replace
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, FEATURE_COLUMNS, OUTPUT_DIR, ExperimentConfig
from src.data_loader import load_coach_data, load_injury_data, load_matches
from src.features import build_match_features
from src.models import predict_match, train_goals_models, train_outcome_model
from src.poisson_score import get_top_scorelines, scoreline_distribution, scoreline_wdl_probabilities


R32_PATH = DATA_DIR / "raw" / "worldcup_2026_round_of_32.csv"
MODEL_WARNING = "classification model and goal model disagree."
ALPHAS = [round(value / 10, 1) for value in range(0, 11)]
GROUP_WEIGHTS = [1.10, 1.50, 2.00, 2.50, 3.00, 4.00]
HISTORICAL_SCALES = [1.00, 0.75, 0.50, 0.35, 0.20]
DECAYS = [0.002, 0.005, 0.01]


def load_best_params() -> dict:
    raw = json.loads((OUTPUT_DIR / "best_feature_weights.json").read_text(encoding="utf-8"))
    return {**raw["feature_weights"], **raw["experiment_params"]}


def weighted_config(group_weight: float, historical_scale: float) -> ExperimentConfig:
    base = ExperimentConfig()
    weights = dict(base.competition_weight)
    for competition in list(weights):
        if competition == "world_cup_2026_group":
            weights[competition] = group_weight
        elif competition != "world_cup_2026_r32":
            weights[competition] = weights[competition] * historical_scale
    return replace(base, competition_weight=weights)


def result_label(home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def predicted_advancer(row: pd.Series, prefix: str = "final") -> str:
    if row[f"{prefix}_home_win"] >= row[f"{prefix}_away_win"]:
        return row["home_team"]
    return row["away_team"]


def model_conflict(row: pd.Series, prefix: str = "final") -> bool:
    return bool(
        (row[f"{prefix}_home_win"] > 0.60 and row["expected_home_goals"] < row["expected_away_goals"])
        or (row[f"{prefix}_away_win"] > 0.60 and row["expected_away_goals"] < row["expected_home_goals"])
    )


def max_wdl_probability_diff(row: pd.Series, prefix: str = "final") -> float:
    return float(
        max(
            abs(row[f"{prefix}_home_win"] - row["poisson_home_win"]),
            abs(row[f"{prefix}_draw"] - row["poisson_draw"]),
            abs(row[f"{prefix}_away_win"] - row["poisson_away_win"]),
        )
    )


def strategy_probabilities(row: pd.Series, alpha: float, conflict_gate: bool) -> dict[str, float]:
    classifier = {
        "home_win": row["classifier_home_win"],
        "draw": row["classifier_draw"],
        "away_win": row["classifier_away_win"],
    }
    poisson = {
        "home_win": row["poisson_home_win"],
        "draw": row["poisson_draw"],
        "away_win": row["poisson_away_win"],
    }
    raw_conflict = (
        (classifier["home_win"] > 0.60 and row["expected_home_goals"] < row["expected_away_goals"])
        or (classifier["away_win"] > 0.60 and row["expected_away_goals"] < row["expected_home_goals"])
    )
    if conflict_gate and raw_conflict:
        final = poisson
    else:
        final = {key: alpha * classifier[key] + (1 - alpha) * poisson[key] for key in classifier}
    total = sum(final.values()) or 1.0
    return {key: value / total for key, value in final.items()}


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


def round_float_columns(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    rounded = df.copy()
    float_columns = rounded.select_dtypes(include=["float", "float64"]).columns
    rounded[float_columns] = rounded[float_columns].round(digits)
    return rounded


def top_scorelines_json(expected_home_goals: float, expected_away_goals: float, top_n: int = 5) -> str:
    distribution = scoreline_distribution(expected_home_goals, expected_away_goals)
    rounded = [
        {"score": row["score"], "probability": round(float(row["probability"]), 3)}
        for row in get_top_scorelines(distribution, top_n)
    ]
    return json.dumps(rounded, ensure_ascii=False)


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        if path.exists():
            path.unlink()
        round_float_columns(df).to_csv(path, index=False, float_format="%.3f")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated{path.suffix}")
        if fallback.exists():
            fallback.unlink()
        round_float_columns(df).to_csv(fallback, index=False, float_format="%.3f")
        return fallback


def write_text(text: str, path: Path) -> Path:
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated{path.suffix}")
        fallback.write_text(text, encoding="utf-8")
        return fallback


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
    return build_match_features(combined, coach, injury, config, params).tail(1)


def base_predictions(test_matches: pd.DataFrame, known_matches: pd.DataFrame, config: ExperimentConfig, params: dict) -> pd.DataFrame:
    coach = load_coach_data()
    injury = load_injury_data()
    train_features = build_match_features(known_matches, coach, injury, config, params)
    outcome_model = train_outcome_model(train_features, train_features["result"], sample_weight=train_features["match_weight"])
    home_goals_model, away_goals_model = train_goals_models(
        train_features,
        train_features["home_goals"],
        train_features["away_goals"],
        sample_weight=train_features["match_weight"],
    )

    rows = []
    for _, fixture in test_matches.sort_values(["date", "match_no"]).iterrows():
        feature_row = build_future_feature(known_matches, fixture, coach, injury, config, params)
        classifier_proba, expected_home, expected_away = predict_match(
            outcome_model,
            home_goals_model,
            away_goals_model,
            feature_row,
        )
        distribution = scoreline_distribution(expected_home, expected_away)
        poisson_wdl = scoreline_wdl_probabilities(distribution)
        rows.append(
            {
                "date": fixture["date"].strftime("%Y-%m-%d"),
                "match_no": int(fixture["match_no"]),
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "home_goals": fixture["home_goals"],
                "away_goals": fixture["away_goals"],
                "actual_result": result_label(fixture["home_goals"], fixture["away_goals"]),
                "actual_advancer": fixture["actual_advancer"],
                "classifier_home_win": classifier_proba["home_win"],
                "classifier_draw": classifier_proba["draw"],
                "classifier_away_win": classifier_proba["away_win"],
                "expected_home_goals": expected_home,
                "expected_away_goals": expected_away,
                **poisson_wdl,
                "top_scorelines": top_scorelines_json(expected_home, expected_away),
            }
        )
    return pd.DataFrame(rows)


def predict_unplayed_matches(fixtures: pd.DataFrame, known_matches: pd.DataFrame, config: ExperimentConfig, params: dict) -> pd.DataFrame:
    base_rows = base_predictions(fixtures, known_matches, config, params)
    rows = base_rows.drop(columns=["home_goals", "away_goals", "actual_result", "actual_advancer"], errors="ignore").copy()
    return rows


def evaluate_strategy(base_rows: pd.DataFrame, alpha: float, conflict_gate: bool) -> tuple[dict, pd.DataFrame]:
    rows = base_rows.copy()
    for idx, row in rows.iterrows():
        final = strategy_probabilities(row, alpha=alpha, conflict_gate=conflict_gate)
        rows.loc[idx, "final_home_win"] = final["home_win"]
        rows.loc[idx, "final_draw"] = final["draw"]
        rows.loc[idx, "final_away_win"] = final["away_win"]
    rows["predicted_result"] = rows[["final_home_win", "final_draw", "final_away_win"]].idxmax(axis=1).str.replace("final_", "")
    rows["predicted_advancer"] = rows.apply(predicted_advancer, axis=1)
    rows["result_correct"] = rows["predicted_result"] == rows["actual_result"]
    rows["advancer_correct"] = rows["predicted_advancer"] == rows["actual_advancer"]
    rows["model_conflict"] = rows.apply(model_conflict, axis=1)
    rows["max_wdl_probability_diff"] = rows.apply(max_wdl_probability_diff, axis=1)
    rows["model_warning"] = ""
    rows.loc[rows["model_conflict"] | (rows["max_wdl_probability_diff"] > 0.20), "model_warning"] = MODEL_WARNING
    metrics = {
        "alpha_classifier": alpha,
        "conflict_gate": conflict_gate,
        "accuracy": float(rows["result_correct"].mean()),
        "advancer_accuracy": float(rows["advancer_correct"].mean()),
        "conflict_count": int(rows["model_conflict"].sum()),
        "warning_count": int((rows["model_warning"] != "").sum()),
        "mean_wdl_probability_diff": float(rows["max_wdl_probability_diff"].mean()),
    }
    return metrics, rows


def apply_strategy_to_predictions(base_rows: pd.DataFrame, alpha: float, conflict_gate: bool) -> pd.DataFrame:
    rows = base_rows.copy()
    for idx, row in rows.iterrows():
        final = strategy_probabilities(row, alpha=alpha, conflict_gate=conflict_gate)
        rows.loc[idx, "final_home_win"] = final["home_win"]
        rows.loc[idx, "final_draw"] = final["draw"]
        rows.loc[idx, "final_away_win"] = final["away_win"]
    rows["predicted_result"] = rows[["final_home_win", "final_draw", "final_away_win"]].idxmax(axis=1).str.replace("final_", "")
    rows["predicted_advancer"] = rows.apply(predicted_advancer, axis=1)
    rows["model_conflict"] = rows.apply(model_conflict, axis=1)
    rows["max_wdl_probability_diff"] = rows.apply(max_wdl_probability_diff, axis=1)
    rows["model_warning"] = ""
    rows.loc[rows["model_conflict"] | (rows["max_wdl_probability_diff"] > 0.20), "model_warning"] = MODEL_WARNING
    return rows


def ensemble_top_strategies(
    top_results: pd.DataFrame,
    upcoming_matches: pd.DataFrame,
    known_matches: pd.DataFrame,
    base_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variant_rows = []
    for rank, strategy in enumerate(top_results.itertuples(index=False), start=1):
        params = dict(base_params)
        params["lambda_decay"] = strategy.lambda_decay
        config = weighted_config(strategy.group_weight, strategy.historical_scale)
        base_rows = predict_unplayed_matches(upcoming_matches, known_matches, config, params)
        predicted = apply_strategy_to_predictions(
            base_rows,
            alpha=strategy.alpha_classifier,
            conflict_gate=bool(strategy.conflict_gate),
        )
        predicted["strategy_rank"] = rank
        predicted["strategy_accuracy"] = strategy.accuracy
        predicted["strategy_advancer_accuracy"] = strategy.advancer_accuracy
        predicted["strategy_group_weight"] = strategy.group_weight
        predicted["strategy_historical_scale"] = strategy.historical_scale
        predicted["strategy_lambda_decay"] = strategy.lambda_decay
        predicted["strategy_alpha_classifier"] = strategy.alpha_classifier
        predicted["strategy_conflict_gate"] = bool(strategy.conflict_gate)
        variant_rows.append(predicted)

    variants = pd.concat(variant_rows, ignore_index=True)
    grouped = variants.groupby(["date", "match_no", "home_team", "away_team"], as_index=False)
    ensemble = grouped.agg(
        final_home_win=("final_home_win", "mean"),
        final_draw=("final_draw", "mean"),
        final_away_win=("final_away_win", "mean"),
        expected_home_goals=("expected_home_goals", "mean"),
        expected_away_goals=("expected_away_goals", "mean"),
        poisson_home_win=("poisson_home_win", "mean"),
        poisson_draw=("poisson_draw", "mean"),
        poisson_away_win=("poisson_away_win", "mean"),
        strategy_conflict_count=("model_conflict", "sum"),
        strategy_warning_count=("model_warning", lambda values: int((values != "").sum())),
    )
    ensemble["predicted_result"] = ensemble[["final_home_win", "final_draw", "final_away_win"]].idxmax(axis=1).str.replace("final_", "")
    ensemble["predicted_advancer"] = ensemble.apply(predicted_advancer, axis=1)
    ensemble["prediction_margin"] = ensemble[["final_home_win", "final_draw", "final_away_win"]].max(axis=1) - ensemble[["final_home_win", "final_draw", "final_away_win"]].apply(lambda row: row.nlargest(2).iloc[-1], axis=1)
    ensemble["top_scorelines"] = ensemble.apply(
        lambda row: top_scorelines_json(row["expected_home_goals"], row["expected_away_goals"]),
        axis=1,
    )
    return ensemble.sort_values(["date", "match_no"]), variants


def main() -> None:
    base_params = load_best_params()
    r32 = pd.read_csv(R32_PATH)
    r32["date"] = pd.to_datetime(r32["date"])
    test_matches = r32[r32["status"] == "completed"].sort_values(["date", "match_no"]).copy()
    upcoming_matches = r32[r32["status"] == "upcoming"].sort_values(["date", "match_no"]).copy()
    known_matches = load_matches()

    experiments = []
    prediction_variants = {}
    for group_weight in GROUP_WEIGHTS:
        for historical_scale in HISTORICAL_SCALES:
            for decay in DECAYS:
                params = dict(base_params)
                params["lambda_decay"] = decay
                config = weighted_config(group_weight, historical_scale)
                base_rows = base_predictions(test_matches, known_matches, config, params)
                for alpha in ALPHAS:
                    for conflict_gate in [False, True]:
                        metrics, rows = evaluate_strategy(base_rows, alpha=alpha, conflict_gate=conflict_gate)
                        metrics.update(
                            {
                                "group_weight": group_weight,
                                "historical_scale": historical_scale,
                                "lambda_decay": decay,
                                "test_matches": len(test_matches),
                            }
                        )
                        experiments.append(metrics)
                        key = (
                            group_weight,
                            historical_scale,
                            decay,
                            alpha,
                            conflict_gate,
                        )
                        prediction_variants[key] = rows

    results = pd.DataFrame(experiments).sort_values(
        ["accuracy", "conflict_count", "warning_count", "advancer_accuracy", "mean_wdl_probability_diff"],
        ascending=[False, True, True, False, True],
    )
    best = results.iloc[0].to_dict()
    top10 = results.head(10).copy()
    best_key = (
        best["group_weight"],
        best["historical_scale"],
        best["lambda_decay"],
        best["alpha_classifier"],
        bool(best["conflict_gate"]),
    )
    best_rows = prediction_variants[best_key].copy()
    ensemble_predictions, ensemble_variants = ensemble_top_strategies(top10, upcoming_matches, known_matches, base_params)
    baseline = results[
        results["group_weight"].eq(1.10)
        & results["historical_scale"].eq(1.00)
        & results["lambda_decay"].eq(base_params["lambda_decay"])
        & results["alpha_classifier"].eq(1.0)
        & ~results["conflict_gate"]
    ].iloc[0].to_dict()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "淘汰赛权重实验结果.csv"
    predictions_path = OUTPUT_DIR / "淘汰赛单一最佳预测.csv"
    ensemble_path = OUTPUT_DIR / "淘汰赛前10策略平均预测.csv"
    ensemble_variants_path = OUTPUT_DIR / "淘汰赛前10策略明细.csv"
    report_path = OUTPUT_DIR / "r32_knockout_weight_experiment_report.md"
    results_path = write_csv(results, results_path)
    predictions_path = write_csv(best_rows, predictions_path)
    ensemble_path = write_csv(ensemble_predictions, ensemble_path)
    ensemble_variants_path = write_csv(ensemble_variants, ensemble_variants_path)

    report = [
        "# R32 Knockout Weight Experiment",
        "",
        "## Goal",
        "",
        "Use all completed Round of 32 matches as the test set, then search higher 2026 group-stage weights and lower historical weights to improve accuracy and reduce model conflicts.",
        "",
        "## Baseline",
        "",
        "```json",
        json.dumps(baseline, indent=2),
        "```",
        "",
        "## Best Strategy",
        "",
        "```json",
        json.dumps(best, indent=2),
        "```",
        "",
        "## Best Predictions",
        "",
        markdown_table(
            best_rows[
                [
                    "date",
                    "home_team",
                    "away_team",
                    "actual_result",
                    "predicted_result",
                    "actual_advancer",
                    "predicted_advancer",
                    "final_home_win",
                    "final_draw",
                    "final_away_win",
                    "expected_home_goals",
                    "expected_away_goals",
                    "poisson_home_win",
                    "poisson_draw",
                    "poisson_away_win",
                    "model_conflict",
                    "model_warning",
                ]
            ]
        ),
        "",
        "## Top 10 Strategy Ensemble For Upcoming Matches",
        "",
        markdown_table(
            ensemble_predictions[
                [
                    "date",
                    "home_team",
                    "away_team",
                    "predicted_advancer",
                    "final_home_win",
                    "final_draw",
                    "final_away_win",
                    "expected_home_goals",
                    "expected_away_goals",
                    "top_scorelines",
                    "strategy_conflict_count",
                    "strategy_warning_count",
                    "prediction_margin",
                ]
            ]
        ),
    ]
    report_path = write_text("\n".join(report), report_path)

    print("Baseline:")
    print(json.dumps(baseline, indent=2))
    print("\nBest:")
    print(json.dumps(best, indent=2))
    print("\nBest predictions:")
    print(
        best_rows[
            [
                "home_team",
                "away_team",
                "actual_result",
                "predicted_result",
                "actual_advancer",
                "predicted_advancer",
                "model_conflict",
                "model_warning",
            ]
        ].to_string(index=False)
    )
    print("\nTop 10 ensemble upcoming predictions:")
    print(
        ensemble_predictions[
            [
                "home_team",
                "away_team",
                "predicted_advancer",
                "final_home_win",
                "final_draw",
                "final_away_win",
                "expected_home_goals",
                "expected_away_goals",
                "strategy_conflict_count",
                "strategy_warning_count",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote {results_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {ensemble_path}")
    print(f"Wrote {ensemble_variants_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
