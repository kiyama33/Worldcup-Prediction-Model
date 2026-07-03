from __future__ import annotations

import pandas as pd


def calculate_injury_impact(injury_data: pd.DataFrame, config=None) -> pd.DataFrame:
    if injury_data.empty:
        return pd.DataFrame(columns=["team", "injury_impact", "injury_data_missing"])
    data = injury_data.copy()
    for col in [
        "injured_starters_count",
        "missing_market_value",
        "missing_minutes_share",
        "key_player_missing_score",
    ]:
        data[col] = pd.to_numeric(data.get(col, 0), errors="coerce").fillna(0)
    max_value = max(float(data["missing_market_value"].max()), 1.0)
    data["injury_impact"] = (
        0.35 * data["injured_starters_count"]
        + 0.25 * (data["missing_market_value"] / max_value)
        + 0.25 * data["missing_minutes_share"]
        + 0.15 * data["key_player_missing_score"]
    )
    data["injury_data_missing"] = data.get("injury_data_missing", 0)
    return data[["team", "injury_impact", "injury_data_missing"]]
