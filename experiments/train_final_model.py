from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_DIR, OUTPUT_DIR, PARAM_GRID, ExperimentConfig
from src.data_loader import load_coach_data, load_injury_data, load_matches
from src.features import build_match_features
from src.models import train_goals_models, train_outcome_model


def main() -> None:
    best_path = OUTPUT_DIR / "best_params.json"
    if not best_path.exists():
        raise FileNotFoundError("Run experiments/run_weight_search.py before train_final_model.py")
    raw = json.loads(best_path.read_text(encoding="utf-8"))
    params = {key: raw[key] for key in PARAM_GRID}
    config = ExperimentConfig()
    matches = load_matches()
    features = build_match_features(matches, load_coach_data(), load_injury_data(), config, params)
    outcome_model = train_outcome_model(features, features["result"], sample_weight=features["match_weight"])
    home_goals_model, away_goals_model = train_goals_models(
        features,
        features["home_goals"],
        features["away_goals"],
        sample_weight=features["match_weight"],
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "params": params,
            "outcome_model": outcome_model,
            "home_goals_model": home_goals_model,
            "away_goals_model": away_goals_model,
            "features": features,
        },
        MODEL_DIR / "final_model.joblib",
    )
    print(f"Saved {MODEL_DIR / 'final_model.joblib'}")


if __name__ == "__main__":
    main()
