from __future__ import annotations

import re
import time
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "worldcup_2026_group_stage.csv"
GROUPS = list("ABCDEFGHIJKL")
BASE_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_{}"


def fetch_html(url: str, attempts: int = 4) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_error = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not fetch {url}: {last_error}") from last_error


def parse_score(raw: str) -> tuple[int, int] | None:
    text = str(raw)
    match = re.search(r"(\d+)\D+(\d+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_date(text: str) -> str:
    match = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+2026",
        text,
    )
    if not match:
        return "2026-06-01"
    return pd.to_datetime(match.group(0)).strftime("%Y-%m-%d")


def extract_group(group: str) -> list[dict]:
    url = BASE_URL.format(group)
    rows = []
    soup = BeautifulSoup(fetch_html(url), "html.parser")
    for table_tag in soup.find_all("table", class_="fevent"):
        tables = pd.read_html(StringIO(str(table_tag)))
        if not tables:
            continue
        table = tables[0]
        if table.shape[1] < 3:
            continue
        home, score, away = [str(value).strip() for value in table.columns.tolist()[:3]]
        parsed = parse_score(score)
        if parsed is None:
            continue
        home_goals, away_goals = parsed
        parent_text = table_tag.parent.get_text(" ", strip=True) if table_tag.parent else ""
        rows.append(
            {
                "date": parse_date(parent_text),
                "competition": "world_cup_2026_group",
                "stage": "group",
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "neutral_ground": 1,
                "host_team": "",
                "source": url,
                "group": group,
            }
        )
    return rows


def main() -> None:
    all_rows = []
    for group in GROUPS:
        rows = extract_group(group)
        print(f"Group {group}: {len(rows)} matches")
        all_rows.extend(rows)
    if not all_rows:
        raise RuntimeError("No 2026 group-stage matches found.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(df)} matches to {OUTPUT}")


if __name__ == "__main__":
    main()
