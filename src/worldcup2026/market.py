"""Public bookmaker-consensus probabilities used by the hybrid model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import pipeline as p

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/WorldCup2026.xlsx"
MARKET_WEIGHT = 0.06
GROUP_END = {
    2014: pd.Timestamp("2014-06-26"),
    2018: pd.Timestamp("2018-06-28"),
    2022: pd.Timestamp("2022-12-02"),
    2026: pd.Timestamp("2026-06-27"),
}
TEAM_ALIASES = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Cura\u00e7ao": "Curacao",
    "Czech Republic": "Czechia",
    "D.R. Congo": "DR Congo",
    "Korea Republic": "South Korea",
    "USA": "United States",
}


def _team_name(value: object) -> str:
    text = TEAM_ALIASES.get(str(value), str(value))
    return p.clean_team(text)


def load_consensus_odds(
    year: int,
    *,
    workbook: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load and de-vig average closing 1X2 odds for one World Cup."""
    if workbook is None:
        workbook = p.RAW_DIR / "WorldCup2026.xlsx"
        p.cached_download(FOOTBALL_DATA_URL, workbook, refresh)

    raw = pd.read_excel(workbook, sheet_name=f"WorldCup{year}")
    inverse = 1.0 / raw[["H-Avg", "D-Avg", "A-Avg"]].to_numpy(dtype=float)
    probability = inverse / inverse.sum(axis=1, keepdims=True)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(raw.Date),
            "home": raw.Home.map(_team_name),
            "away": raw.Away.map(_team_name),
            "p_home": probability[:, 0],
            "p_draw": probability[:, 1],
            "p_away": probability[:, 2],
        }
    )


def match_consensus_probabilities(
    frame: pd.DataFrame,
    year: int,
    *,
    workbook: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Match odds by teams and date, reorienting reversed fixtures."""
    output = np.full((len(frame), 3), np.nan)
    try:
        odds = load_consensus_odds(year, workbook=workbook)
    except (FileNotFoundError, ValueError):
        return output, np.zeros(len(frame), dtype=bool)
    group_end = GROUP_END.get(year)

    for index, row in enumerate(frame.itertuples(index=False)):
        if group_end is not None and pd.Timestamp(row.date) > group_end:
            continue
        home, away = _team_name(row.home), _team_name(row.away)
        candidates = odds[
            (
                (odds.home.eq(home) & odds.away.eq(away))
                | (odds.home.eq(away) & odds.away.eq(home))
            )
            & ((odds.date - pd.Timestamp(row.date)).abs() <= pd.Timedelta(days=1))
        ]
        if len(candidates) != 1:
            continue
        match = candidates.iloc[0]
        values = match[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        output[index] = values if match.home == home else values[[2, 1, 0]]

    available = np.isfinite(output).all(axis=1)
    return output, available


def blend_with_consensus(
    structural: np.ndarray,
    frame: pd.DataFrame,
    year: int,
    *,
    workbook: Path | None = None,
    weight: float = MARKET_WEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend structural and market probabilities."""
    market, available = match_consensus_probabilities(
        frame,
        year,
        workbook=workbook,
    )
    blended = np.asarray(structural, dtype=float).copy()
    blended[available] = (1.0 - weight) * blended[available] + weight * market[
        available
    ]
    return blended, available


def tilt_score_cache(
    cache: dict[tuple[str, str], dict[str, np.ndarray | tuple[float, ...]]],
    fixtures: pd.DataFrame,
    year: int,
    *,
    workbook: Path | None = None,
    weight: float = MARKET_WEIGHT,
) -> int:
    """Rescale scoreline regions to the hybrid 1X2 probabilities."""
    group = fixtures[fixtures.stage.eq("Group Stage")].copy()
    frame = group.rename(columns={"team1": "home", "team2": "away"})
    market, available = match_consensus_probabilities(
        frame,
        year,
        workbook=workbook,
    )
    adjusted = 0
    for index, game in enumerate(frame.itertuples(index=False)):
        if not available[index]:
            continue
        entry = cache.get((_team_name(game.home), _team_name(game.away)))
        if entry is None:
            continue
        current = np.asarray(entry["wdl"], dtype=float)
        target = (1.0 - weight) * current + weight * market[index]
        home_scores = np.asarray(entry["home_scores"])
        away_scores = np.asarray(entry["away_scores"])
        regions = (
            home_scores > away_scores,
            home_scores == away_scores,
            home_scores < away_scores,
        )
        scores = np.asarray(entry["scores"], dtype=float).copy()
        for outcome, region in enumerate(regions):
            mass = scores[region].sum()
            if mass > 0:
                scores[region] *= target[outcome] / mass
        scores /= scores.sum()
        entry["scores"] = scores
        entry["wdl"] = tuple(float(value) for value in target)
        adjusted += 1
    return adjusted
