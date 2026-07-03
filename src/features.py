from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .coach_features import build_coach_score
from .config import FEATURE_COLUMNS
from .elo import build_elo_history
from .injury_features import calculate_injury_impact


STAGE_CODE = {
    "friendly": 0,
    "qualifier": 1,
    "group": 2,
    "round_of_32": 3,
    "round_of_16": 4,
    "quarter_final": 5,
    "semi_final": 6,
    "final": 7,
}


def _team_form(history: deque[dict]) -> dict[str, float]:
    if not history:
        return {"points": 0.0, "goals_for": 0.0, "goals_against": 0.0, "goal_diff": 0.0}
    rows = list(history)
    return {
        "points": float(np.mean([row["points"] for row in rows])),
        "goals_for": float(np.mean([row["goals_for"] for row in rows])),
        "goals_against": float(np.mean([row["goals_against"] for row in rows])),
        "goal_diff": float(np.mean([row["goals_for"] - row["goals_against"] for row in rows])),
    }


def _points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def build_match_features(matches: pd.DataFrame, coach_data: pd.DataFrame, injury_data: pd.DataFrame, config, params: dict) -> pd.DataFrame:
    matches = matches.sort_values(["date", "competition", "home_team"]).copy()
    elo = build_elo_history(matches, config, params)
    coach = build_coach_score(coach_data).set_index("team")
    injury = calculate_injury_impact(injury_data, config).set_index("team")
    form_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=config.form_window))
    rows = []

    for idx, match in matches.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        home_form = _team_form(form_history[home])
        away_form = _team_form(form_history[away])
        home_coach = float(coach["coach_score"].get(home, 0.0)) if not coach.empty else 0.0
        away_coach = float(coach["coach_score"].get(away, 0.0)) if not coach.empty else 0.0
        home_injury = float(injury["injury_impact"].get(home, 0.0)) if not injury.empty else 0.0
        away_injury = float(injury["injury_impact"].get(away, 0.0)) if not injury.empty else 0.0
        elo_row = elo.loc[idx]
        host_advantage = 1 if match.get("host_team", "") == home else (-1 if match.get("host_team", "") == away else 0)
        stage = str(match["stage"])

        adjusted_home = (
            params["elo_weight"] * (elo_row["home_elo"] / 400.0)
            + params["form_weight"] * home_form["points"]
            + params["coach_weight"] * home_coach
            - params["injury_weight"] * home_injury
        )
        adjusted_away = (
            params["elo_weight"] * (elo_row["away_elo"] / 400.0)
            + params["form_weight"] * away_form["points"]
            + params["coach_weight"] * away_coach
            - params["injury_weight"] * away_injury
        )
        result = "home_win" if match["home_goals"] > match["away_goals"] else "draw" if match["home_goals"] == match["away_goals"] else "away_win"

        rows.append(
            {
                "match_id": idx,
                "date": match["date"],
                "competition": match["competition"],
                "home_team": home,
                "away_team": away,
                "home_goals": match["home_goals"],
                "away_goals": match["away_goals"],
                "result": result,
                "elo_diff": elo_row["elo_diff"],
                "abs_elo_diff": elo_row["abs_elo_diff"],
                "home_elo": elo_row["home_elo"],
                "away_elo": elo_row["away_elo"],
                "home_form_points": home_form["points"],
                "away_form_points": away_form["points"],
                "form_points_diff": home_form["points"] - away_form["points"],
                "home_form_goals_for": home_form["goals_for"],
                "away_form_goals_for": away_form["goals_for"],
                "form_goals_for_diff": home_form["goals_for"] - away_form["goals_for"],
                "home_form_goals_against": home_form["goals_against"],
                "away_form_goals_against": away_form["goals_against"],
                "form_goals_against_diff": home_form["goals_against"] - away_form["goals_against"],
                "home_coach_score": home_coach,
                "away_coach_score": away_coach,
                "coach_score_diff": home_coach - away_coach,
                "home_injury_impact": home_injury,
                "away_injury_impact": away_injury,
                "injury_impact_diff": home_injury - away_injury,
                "stage_code": STAGE_CODE.get(stage, 2),
                "is_group_stage": 1 if stage == "group" else 0,
                "is_knockout": 0 if stage == "group" else 1,
                "neutral_ground": int(match.get("neutral_ground", 1)),
                "host_advantage": host_advantage,
                "match_weight": elo_row["match_weight"],
                "adjusted_strength_diff": adjusted_home - adjusted_away,
            }
        )

        form_history[home].append({"points": _points(match["home_goals"], match["away_goals"]), "goals_for": match["home_goals"], "goals_against": match["away_goals"]})
        form_history[away].append({"points": _points(match["away_goals"], match["home_goals"]), "goals_for": match["away_goals"], "goals_against": match["home_goals"]})

    df = pd.DataFrame(rows).set_index("match_id")
    return normalize_features(df)


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out
