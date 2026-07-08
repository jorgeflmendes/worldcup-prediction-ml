"""Load the 2026 fixtures and materialize the official knockout tree."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from . import pipeline as p
from .tournament_rules import (
    FINAL_SPEC,
    QF_SPECS,
    R16_SPECS,
    R32_SPECS,
    SF_SPECS,
    THIRD_PLACE_SPEC,
)


def load_2026_fixtures() -> pd.DataFrame:
    live = p.EXTERNAL_DIR / "wc2026_group_fixtures_live.csv"
    if live.exists():
        group = pd.read_csv(live)
    else:
        archive = p.RAW_DIR / "world_cup_1930_2026.zip"
        if not archive.exists():
            p.load_2026_teams(False)
        with zipfile.ZipFile(archive) as zipped:
            group = pd.read_csv(io.BytesIO(zipped.read("wc_2026_fixtures.csv")))
        group = group[group.stage == "Group Stage"].copy()
        group["fixture"] = [
            f"Group-{label}-{index + 1}"
            for label, games in group.groupby("group", sort=False)
            for index in range(len(games))
        ]
        group["home_score"] = np.nan
        group["away_score"] = np.nan
        group["completed"] = False

    for column in ("team1", "team2"):
        group[column] = group[column].map(p.clean_team)
    group["stage"] = "Group Stage"

    knockout_rows = [
        {
            "group": np.nan,
            "stage": stage,
            "fixture": match_id,
            "team1": home,
            "team2": away,
        }
        for stage, specs in (
            ("Round of 32", R32_SPECS),
            ("Round of 16", R16_SPECS),
            ("Quarter-final", QF_SPECS),
            ("Semi-final", SF_SPECS),
        )
        for match_id, home, away in specs
    ]
    knockout_rows.extend(
        [
            {
                "group": np.nan,
                "stage": "3rd Place Match",
                "fixture": THIRD_PLACE_SPEC[0],
                "team1": "L101",
                "team2": "L102",
            },
            {
                "group": np.nan,
                "stage": "Final",
                "fixture": FINAL_SPEC[0],
                "team1": FINAL_SPEC[1],
                "team2": FINAL_SPEC[2],
            },
        ]
    )
    return pd.concat(
        [group, pd.DataFrame(knockout_rows)], ignore_index=True, sort=False
    )
