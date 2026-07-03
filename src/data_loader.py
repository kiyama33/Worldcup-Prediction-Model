from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_DIR


MATCH_COLUMNS = [
    "date",
    "competition",
    "stage",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "neutral_ground",
    "host_team",
    "source",
]


def load_matches(paths: list[Path] | None = None) -> pd.DataFrame:
    paths = paths or [
        DATA_DIR / "sample" / "historical_matches.csv",
        DATA_DIR / "raw" / "worldcup_2026_group_stage.csv",
    ]
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No match CSV files found.")

    matches = pd.concat(frames, ignore_index=True)
    missing = set(MATCH_COLUMNS) - set(matches.columns)
    if missing:
        raise ValueError(f"Match data missing columns: {sorted(missing)}")

    matches["date"] = pd.to_datetime(matches["date"])
    for col in ["home_goals", "away_goals"]:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")
    matches = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    matches["home_goals"] = matches["home_goals"].astype(int)
    matches["away_goals"] = matches["away_goals"].astype(int)
    matches["neutral_ground"] = matches["neutral_ground"].fillna(1).astype(int)
    return matches.sort_values(["date", "competition", "home_team"]).reset_index(drop=True)


def load_team_table(path: Path, key: str = "team") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[key])
    return pd.read_csv(path)


def load_coach_data() -> pd.DataFrame:
    return load_team_table(DATA_DIR / "sample" / "coach_data.csv")


def load_injury_data() -> pd.DataFrame:
    return load_team_table(DATA_DIR / "sample" / "injury_data.csv")


def split_train_validation(matches: pd.DataFrame, validation_competition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = matches[matches["competition"] == validation_competition].copy()
    train = matches[matches["competition"] != validation_competition].copy()
    if train.empty or validation.empty:
        raise ValueError("Both training and validation sets must contain matches.")
    return train, validation
