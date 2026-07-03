from __future__ import annotations

from scipy.stats import poisson


def scoreline_distribution(expected_home_goals: float, expected_away_goals: float, max_goals: int = 5) -> list[dict]:
    rows = []
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = poisson.pmf(home_goals, expected_home_goals) * poisson.pmf(away_goals, expected_away_goals)
            rows.append({"score": f"{home_goals}-{away_goals}", "probability": float(probability)})
    total = sum(row["probability"] for row in rows) or 1.0
    for row in rows:
        row["probability"] = row["probability"] / total
    return sorted(rows, key=lambda row: row["probability"], reverse=True)


def get_top_scorelines(distribution: list[dict], top_n: int = 5) -> list[dict]:
    return distribution[:top_n]


def scoreline_wdl_probabilities(distribution: list[dict]) -> dict[str, float]:
    probabilities = {
        "poisson_home_win": 0.0,
        "poisson_draw": 0.0,
        "poisson_away_win": 0.0,
    }
    for row in distribution:
        home_goals, away_goals = [int(part) for part in row["score"].split("-")]
        if home_goals > away_goals:
            probabilities["poisson_home_win"] += row["probability"]
        elif home_goals < away_goals:
            probabilities["poisson_away_win"] += row["probability"]
        else:
            probabilities["poisson_draw"] += row["probability"]
    return probabilities
