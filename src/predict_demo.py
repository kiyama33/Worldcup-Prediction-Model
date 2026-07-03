from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

from .config import MODEL_DIR
from .models import predict_match
from .poisson_score import get_top_scorelines, scoreline_distribution


def predict_fixture(team_a: str, team_b: str, match_context: dict | None = None) -> dict:
    bundle = joblib.load(MODEL_DIR / "final_model.joblib")
    features = bundle["features"].copy()
    candidates = features[(features["home_team"] == team_a) & (features["away_team"] == team_b)]
    if candidates.empty:
        candidates = features.tail(1).copy()
        candidates["home_team"] = team_a
        candidates["away_team"] = team_b
    match_features = candidates.tail(1)
    proba, expected_home, expected_away = predict_match(
        bundle["outcome_model"],
        bundle["home_goals_model"],
        bundle["away_goals_model"],
        match_features,
    )
    top_scorelines = get_top_scorelines(scoreline_distribution(expected_home, expected_away), 5)
    return {
        "match": f"{team_a} vs {team_b}",
        "win_draw_loss": {
            f"{team_a}_win": round(proba["home_win"], 4),
            "draw": round(proba["draw"], 4),
            f"{team_b}_win": round(proba["away_win"], 4),
        },
        "expected_goals": {team_a: round(expected_home, 3), team_b: round(expected_away, 3)},
        "top_scorelines": [{"score": row["score"], "probability": round(row["probability"], 4)} for row in top_scorelines],
        "best_params_used": bundle["params"],
    }


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python -m src.predict_demo TeamA TeamB")
    print(json.dumps(predict_fixture(sys.argv[1], sys.argv[2]), indent=2))


if __name__ == "__main__":
    main()
