from __future__ import annotations

import pandas as pd


def build_coach_score(coach_data: pd.DataFrame) -> pd.DataFrame:
    if coach_data.empty:
        return pd.DataFrame(columns=["team", "coach_score", "coach_data_missing"])
    data = coach_data.copy()
    for col in ["coach_win_rate", "coach_points_per_game", "coach_tournament_experience", "coach_days_in_charge", "coach_recent_uplift"]:
        data[col] = pd.to_numeric(data.get(col, 0), errors="coerce").fillna(0)
    data["coach_score"] = (
        data["coach_win_rate"]
        + data["coach_points_per_game"] / 3.0
        + data["coach_tournament_experience"] / 10.0
        + data["coach_days_in_charge"].clip(upper=2000) / 2000.0
        + data["coach_recent_uplift"]
    )
    data["coach_data_missing"] = data.get("coach_data_missing", 0)
    return data[["team", "coach_score", "coach_data_missing"]]


def calculate_coach_recent_uplift(matches: pd.DataFrame, coach_data: pd.DataFrame) -> pd.DataFrame:
    return build_coach_score(coach_data)
