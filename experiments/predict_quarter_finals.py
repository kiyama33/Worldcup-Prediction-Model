from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, OUTPUT_DIR
from src.poisson_score import get_top_scorelines, scoreline_distribution, scoreline_wdl_probabilities


GROUP_PATH = DATA_DIR / "raw" / "worldcup_2026_group_stage.csv"
R32_PATH = DATA_DIR / "raw" / "worldcup_2026_round_of_32.csv"
R16_PATH = DATA_DIR / "raw" / "worldcup_2026_round_of_16.csv"
QF_PATH = DATA_DIR / "raw" / "worldcup_2026_quarter_finals.csv"

MODEL_NAME = "group_plus_r32_opponent_adjusted"
SHRINKAGE_TO_FIELD_AVERAGE = 0.30
ATTACK_SCALE = 0.82
DEFENSE_SCALE = 0.82
SCHEDULE_STRENGTH_SCALE = 0.22
HOST_GOAL_BONUS = 0.10
DRAW_ADVANCER_STRENGTH_SCALE = 1.25
KNOCKOUT_ADVANCER_POINT_BONUS = 0.35


def read_matches(path: Path) -> pd.DataFrame:
    matches = pd.read_csv(path)
    matches["date"] = pd.to_datetime(matches["date"])
    for column in ["home_goals", "away_goals"]:
        matches[column] = pd.to_numeric(matches[column], errors="coerce")
    return matches


def result_label(home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def result_points(goals_for: float, goals_against: float) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def round_float_columns(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    rounded = df.copy()
    float_columns = rounded.select_dtypes(include=["float", "float64"]).columns
    rounded[float_columns] = rounded[float_columns].round(digits)
    return rounded


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    targets = [
        path,
        path.with_name(f"{path.stem}_updated{path.suffix}"),
        path.with_name(f"{path.stem}_updated_{timestamp}{path.suffix}"),
    ]
    last_error = None
    for target in targets:
        try:
            if target.exists():
                target.unlink()
            round_float_columns(df).to_csv(target, index=False, float_format="%.3f")
            return target
        except PermissionError as exc:
            last_error = exc
    raise last_error


def write_text(text: str, path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    targets = [
        path,
        path.with_name(f"{path.stem}_updated{path.suffix}"),
        path.with_name(f"{path.stem}_updated_{timestamp}{path.suffix}"),
    ]
    last_error = None
    for target in targets:
        try:
            target.write_text(text, encoding="utf-8")
            return target
        except PermissionError as exc:
            last_error = exc
    raise last_error


def add_team_match(
    rows: list[dict],
    date,
    team: str,
    opponent: str,
    goals_for: float,
    goals_against: float,
    advanced: bool,
) -> None:
    points = result_points(goals_for, goals_against)
    if advanced:
        points += KNOCKOUT_ADVANCER_POINT_BONUS
    rows.append(
        {
            "date": date,
            "team": team,
            "opponent": opponent,
            "goals_for": float(goals_for),
            "goals_against": float(goals_against),
            "goal_diff": float(goals_for - goals_against),
            "points": float(points),
        }
    )


def build_training_matches() -> pd.DataFrame:
    group = read_matches(GROUP_PATH).dropna(subset=["home_goals", "away_goals"]).copy()
    group["actual_advancer"] = ""
    r32 = read_matches(R32_PATH)
    r32 = r32[r32["status"].eq("completed")].dropna(subset=["home_goals", "away_goals"]).copy()
    return pd.concat([group, r32], ignore_index=True, sort=False).sort_values(["date", "match_no"], na_position="first")


def build_team_ratings(training_matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, match in training_matches.iterrows():
        actual_advancer = "" if pd.isna(match.get("actual_advancer", "")) else str(match.get("actual_advancer", ""))
        add_team_match(
            rows,
            match["date"],
            match["home_team"],
            match["away_team"],
            match["home_goals"],
            match["away_goals"],
            actual_advancer == match["home_team"],
        )
        add_team_match(
            rows,
            match["date"],
            match["away_team"],
            match["home_team"],
            match["away_goals"],
            match["home_goals"],
            actual_advancer == match["away_team"],
        )

    long = pd.DataFrame(rows)
    base = (
        long.groupby("team", as_index=False)
        .agg(
            played=("team", "size"),
            points=("points", "sum"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            goal_diff=("goal_diff", "sum"),
        )
        .sort_values("team")
    )
    base["points_per_game"] = base["points"] / base["played"]
    base["goals_for_per_game"] = base["goals_for"] / base["played"]
    base["goals_against_per_game"] = base["goals_against"] / base["played"]
    base["goal_diff_per_game"] = base["goal_diff"] / base["played"]

    field_goal_rate = float(long["goals_for"].mean())
    field_ppg = float(base["points_per_game"].mean())
    base["raw_strength"] = (
        (base["points_per_game"] - field_ppg) * 0.45
        + (base["goals_for_per_game"] - base["goals_against_per_game"]) * 0.55
    )
    team_lookup = base.set_index("team")

    opponent_rows = []
    for team, rows_for_team in long.groupby("team"):
        opponents = rows_for_team["opponent"]
        opponent_attack = float(team_lookup.loc[opponents, "goals_for_per_game"].mean())
        opponent_defense = float(team_lookup.loc[opponents, "goals_against_per_game"].mean())
        opponent_strength = float(team_lookup.loc[opponents, "raw_strength"].mean())
        opponent_rows.append(
            {
                "team": team,
                "opponent_attack_per_game": opponent_attack,
                "opponent_goals_allowed_per_game": opponent_defense,
                "schedule_strength": opponent_strength,
            }
        )

    ratings = base.merge(pd.DataFrame(opponent_rows), on="team", how="left")
    attack_raw = ratings["goals_for_per_game"] * (field_goal_rate / ratings["opponent_goals_allowed_per_game"].clip(lower=0.25))
    defense_raw = ratings["goals_against_per_game"] * (field_goal_rate / ratings["opponent_attack_per_game"].clip(lower=0.25))
    ratings["adjusted_attack"] = SHRINKAGE_TO_FIELD_AVERAGE * field_goal_rate + (1 - SHRINKAGE_TO_FIELD_AVERAGE) * attack_raw
    ratings["adjusted_defense"] = SHRINKAGE_TO_FIELD_AVERAGE * field_goal_rate + (1 - SHRINKAGE_TO_FIELD_AVERAGE) * defense_raw
    ratings["team_strength"] = (
        (ratings["adjusted_attack"] - ratings["adjusted_defense"])
        + (ratings["points_per_game"] - field_ppg) * 0.35
        + ratings["schedule_strength"] * SCHEDULE_STRENGTH_SCALE
    )
    ratings["field_goal_rate"] = field_goal_rate
    return ratings.sort_values("team").reset_index(drop=True)


def top_scorelines_json(expected_home_goals: float, expected_away_goals: float, top_n: int = 5) -> str:
    distribution = scoreline_distribution(expected_home_goals, expected_away_goals, max_goals=7)
    rounded = [
        {"score": row["score"], "probability": round(float(row["probability"]), 3)}
        for row in get_top_scorelines(distribution, top_n)
    ]
    return json.dumps(rounded, ensure_ascii=False)


def predict_fixture(match: pd.Series, ratings_by_team: pd.DataFrame, field_goal_rate: float) -> dict:
    home = match["home_team"]
    away = match["away_team"]
    home_rating = ratings_by_team.loc[home]
    away_rating = ratings_by_team.loc[away]
    host_team = "" if pd.isna(match.get("host_team", "")) else str(match.get("host_team", ""))
    host_adjustment = HOST_GOAL_BONUS if host_team == home else -HOST_GOAL_BONUS if host_team == away else 0.0

    home_attack = math.log(float(home_rating["adjusted_attack"]) / field_goal_rate)
    away_attack = math.log(float(away_rating["adjusted_attack"]) / field_goal_rate)
    home_defense_weakness = math.log(float(home_rating["adjusted_defense"]) / field_goal_rate)
    away_defense_weakness = math.log(float(away_rating["adjusted_defense"]) / field_goal_rate)
    schedule_gap = float(home_rating["schedule_strength"] - away_rating["schedule_strength"])

    expected_home_goals = field_goal_rate * math.exp(
        ATTACK_SCALE * home_attack
        + DEFENSE_SCALE * away_defense_weakness
        + SCHEDULE_STRENGTH_SCALE * schedule_gap
        + host_adjustment
    )
    expected_away_goals = field_goal_rate * math.exp(
        ATTACK_SCALE * away_attack
        + DEFENSE_SCALE * home_defense_weakness
        - SCHEDULE_STRENGTH_SCALE * schedule_gap
        - host_adjustment
    )
    expected_home_goals = max(0.05, min(4.50, expected_home_goals))
    expected_away_goals = max(0.05, min(4.50, expected_away_goals))

    distribution = scoreline_distribution(expected_home_goals, expected_away_goals, max_goals=7)
    wdl = scoreline_wdl_probabilities(distribution)
    strength_gap = float(home_rating["team_strength"] - away_rating["team_strength"]) + host_adjustment
    draw_to_home = 1.0 / (1.0 + math.exp(-DRAW_ADVANCER_STRENGTH_SCALE * strength_gap))
    home_advancer_probability = wdl["poisson_home_win"] + wdl["poisson_draw"] * draw_to_home
    away_advancer_probability = wdl["poisson_away_win"] + wdl["poisson_draw"] * (1 - draw_to_home)
    predicted_result = max(
        [
            ("home_win", wdl["poisson_home_win"]),
            ("draw", wdl["poisson_draw"]),
            ("away_win", wdl["poisson_away_win"]),
        ],
        key=lambda item: item[1],
    )[0]

    return {
        "date": match["date"].strftime("%Y-%m-%d"),
        "match_no": int(match["match_no"]),
        "stage": match["stage"],
        "home_team": home,
        "away_team": away,
        "expected_home_goals": expected_home_goals,
        "expected_away_goals": expected_away_goals,
        "home_win": wdl["poisson_home_win"],
        "draw": wdl["poisson_draw"],
        "away_win": wdl["poisson_away_win"],
        "predicted_result": predicted_result,
        "home_advancer_probability": home_advancer_probability,
        "away_advancer_probability": away_advancer_probability,
        "predicted_advancer": home if home_advancer_probability >= away_advancer_probability else away,
        "home_team_strength": float(home_rating["team_strength"]),
        "away_team_strength": float(away_rating["team_strength"]),
        "home_schedule_strength": float(home_rating["schedule_strength"]),
        "away_schedule_strength": float(away_rating["schedule_strength"]),
        "top_scorelines": top_scorelines_json(expected_home_goals, expected_away_goals),
    }


def r16_test_matches() -> pd.DataFrame:
    r16 = read_matches(R16_PATH)
    return r16[r16["status"].eq("completed")].sort_values(["date", "match_no"]).copy()


def quarter_final_fixtures() -> pd.DataFrame:
    qf = read_matches(QF_PATH)
    return qf[qf["status"].eq("upcoming")].sort_values(["date", "match_no"]).copy()


def evaluate_test_set(test_matches: pd.DataFrame, ratings_by_team: pd.DataFrame, field_goal_rate: float) -> pd.DataFrame:
    rows = []
    for _, match in test_matches.iterrows():
        row = predict_fixture(match, ratings_by_team, field_goal_rate)
        row["home_goals"] = int(match["home_goals"])
        row["away_goals"] = int(match["away_goals"])
        row["actual_result"] = result_label(match["home_goals"], match["away_goals"])
        row["actual_advancer"] = match["actual_advancer"]
        row["result_correct"] = row["predicted_result"] == row["actual_result"]
        row["advancer_correct"] = row["predicted_advancer"] == row["actual_advancer"]
        rows.append(row)
    return pd.DataFrame(rows)


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
                value = f"{value:.3f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    training_matches = build_training_matches()
    ratings = build_team_ratings(training_matches)
    ratings_by_team = ratings.set_index("team")
    field_goal_rate = float(ratings["field_goal_rate"].iloc[0])

    test_matches = r16_test_matches()
    test_predictions = evaluate_test_set(test_matches, ratings_by_team, field_goal_rate)
    fixtures = quarter_final_fixtures()
    predictions = pd.DataFrame([predict_fixture(match, ratings_by_team, field_goal_rate) for _, match in fixtures.iterrows()])

    summary = {
        "model": MODEL_NAME,
        "training_matches": int(len(training_matches)),
        "training_scope": "2026 group stage + completed round of 32",
        "test_matches": int(len(test_predictions)),
        "test_scope": "completed round of 16",
        "prediction_scope": "quarter finals",
        "result_accuracy": float(test_predictions["result_correct"].mean()),
        "advancer_accuracy": float(test_predictions["advancer_correct"].mean()),
        "field_goal_rate": field_goal_rate,
        "shrinkage_to_field_average": SHRINKAGE_TO_FIELD_AVERAGE,
        "knockout_advancer_point_bonus": KNOCKOUT_ADVANCER_POINT_BONUS,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = write_csv(predictions, OUTPUT_DIR / "8进4预测.csv")
    test_path = write_csv(test_predictions, OUTPUT_DIR / "16进8测试集结果.csv")
    ratings_path = write_csv(ratings, OUTPUT_DIR / "球队强度评分.csv")
    report_path = OUTPUT_DIR / "quarter_final_prediction_report.md"

    report = [
        "# Quarter-Final Prediction",
        "",
        "## Method",
        "",
        "Training uses only 2026 group-stage matches plus completed Round of 32 matches. Completed Round of 16 matches are used as the holdout test set. Quarter-finals are prediction targets only.",
        "",
        "Team attacking and defensive rates are adjusted by opponent strength, then shrunk toward the tournament average. Round-of-32 advancement adds a small bonus to team points so penalty or extra-time advancement is visible without letting Round of 16 data leak into training.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Quarter-Final Predictions",
        "",
        markdown_table(
            predictions[
                [
                    "date",
                    "match_no",
                    "home_team",
                    "away_team",
                    "predicted_advancer",
                    "home_win",
                    "draw",
                    "away_win",
                    "home_advancer_probability",
                    "away_advancer_probability",
                    "expected_home_goals",
                    "expected_away_goals",
                    "top_scorelines",
                ]
            ]
        ),
        "",
        "## Round of 16 Holdout Test",
        "",
        markdown_table(
            test_predictions[
                [
                    "date",
                    "match_no",
                    "home_team",
                    "away_team",
                    "actual_result",
                    "predicted_result",
                    "actual_advancer",
                    "predicted_advancer",
                    "result_correct",
                    "advancer_correct",
                ]
            ]
        ),
    ]
    report_path = write_text("\n".join(report), report_path)

    print("Summary:")
    print(json.dumps(summary, indent=2))
    print("\nQuarter-final predictions:")
    print(
        predictions[
            [
                "home_team",
                "away_team",
                "predicted_advancer",
                "home_win",
                "draw",
                "away_win",
                "home_advancer_probability",
                "away_advancer_probability",
                "expected_home_goals",
                "expected_away_goals",
                "top_scorelines",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote {predictions_path}")
    print(f"Wrote {test_path}")
    print(f"Wrote {ratings_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
