from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"


@dataclass(frozen=True)
class ExperimentConfig:
    initial_elo: float = 1500.0
    form_window: int = 5
    validation_competition: str = "world_cup_2026_group"
    reference_date: str = "2026-07-02"
    competition_weight: dict[str, float] = field(
        default_factory=lambda: {
            "world_cup_2022": 0.60,
            "euro_2024": 1.00,
            "copa_america_2024": 1.00,
            "asian_cup_2023": 0.90,
            "afcon_2023": 0.90,
            "qualifiers_2026": 0.95,
            "world_cup_2026_group": 1.10,
            "world_cup_2026_r32": 1.15,
        }
    )


PARAM_GRID = {
    "elo_weight": [0.5, 0.8, 1.0, 1.2, 1.5],
    "form_weight": [0.0, 0.3, 0.5, 0.8, 1.0],
    "coach_weight": [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
    "injury_weight": [0.0, 0.3, 0.5, 0.8, 1.0, 1.5],
    "lambda_decay": [0.0005, 0.001, 0.002, 0.005],
    "base_k": [20, 30, 40, 50],
}


FEATURE_COLUMNS = [
    "elo_diff",
    "abs_elo_diff",
    "home_elo",
    "away_elo",
    "home_form_points",
    "away_form_points",
    "form_points_diff",
    "home_form_goals_for",
    "away_form_goals_for",
    "form_goals_for_diff",
    "home_form_goals_against",
    "away_form_goals_against",
    "form_goals_against_diff",
    "home_coach_score",
    "away_coach_score",
    "coach_score_diff",
    "home_injury_impact",
    "away_injury_impact",
    "injury_impact_diff",
    "stage_code",
    "is_group_stage",
    "is_knockout",
    "neutral_ground",
    "host_advantage",
    "match_weight",
    "adjusted_strength_diff",
]
