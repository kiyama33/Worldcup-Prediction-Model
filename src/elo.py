from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd


def calculate_expected_result(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_elo(elo_a: float, elo_b: float, result_a: float, k: float) -> tuple[float, float]:
    expected_a = calculate_expected_result(elo_a, elo_b)
    delta = k * (result_a - expected_a)
    return elo_a + delta, elo_b - delta


def actual_result(home_goals: int, away_goals: int) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals == away_goals:
        return 0.5
    return 0.0


def build_elo_history(matches: pd.DataFrame, config, params: dict) -> pd.DataFrame:
    ratings = defaultdict(lambda: config.initial_elo)
    rows = []
    reference_date = pd.Timestamp(config.reference_date)
    for idx, match in matches.sort_values("date").iterrows():
        home = match["home_team"]
        away = match["away_team"]
        home_elo = float(ratings[home])
        away_elo = float(ratings[away])
        days_since = max((reference_date - match["date"]).days, 0)
        comp_weight = config.competition_weight.get(match["competition"], 1.0)
        match_weight = comp_weight * math.exp(-params["lambda_decay"] * days_since)
        k = params["base_k"] * comp_weight

        rows.append(
            {
                "match_id": idx,
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff": home_elo - away_elo,
                "abs_elo_diff": abs(home_elo - away_elo),
                "match_weight": match_weight,
            }
        )
        new_home, new_away = update_elo(
            home_elo,
            away_elo,
            actual_result(match["home_goals"], match["away_goals"]),
            k,
        )
        ratings[home] = new_home
        ratings[away] = new_away
    return pd.DataFrame(rows).set_index("match_id")
