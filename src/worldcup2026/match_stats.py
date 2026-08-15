"""Lagged team-performance features from public final-match statistics."""

from __future__ import annotations

from collections import defaultdict, deque
from functools import lru_cache

import numpy as np
import pandas as pd

from . import pipeline as p


HISTORICAL_STATS_URL = (
    "https://huggingface.co/datasets/adibmed/football-dataset/resolve/"
    "4612e73080f67e00290d9334081cd447f9dff7c6/output/01_matches_all.csv"
)
LIVE_STATS_BASE_URL = (
    "https://huggingface.co/datasets/Mominullptr/fifa-world-cup-2026-dataset/"
    "resolve/cd42d32fcab2608241fa151208bca6f8d23a4401"
)
STAT_METRICS = (
    "xg",
    "shots",
    "shots_on_target",
    "possession",
    "passes_completed",
    "corners",
    "fouls",
    "offsides",
    "yellow_cards",
    "red_cards",
)
STAT_SCALES = {
    "xg": 2.0,
    "shots": 15.0,
    "shots_on_target": 6.0,
    "possession": 1.0,
    "passes_completed": 500.0,
    "corners": 6.0,
    "fouls": 12.0,
    "offsides": 3.0,
    "yellow_cards": 3.0,
    "red_cards": 1.0,
}


def _numeric(series: pd.Series, *, percent: bool = False) -> pd.Series:
    values = pd.to_numeric(
        series.astype(str).str.rstrip("%"),
        errors="coerce",
    )
    return values / 100.0 if percent else values


def _historical_observations(refresh: bool) -> pd.DataFrame:
    path = p.cached_download(
        HISTORICAL_STATS_URL,
        p.RAW_DIR / "hf_matches_all.csv",
        refresh,
    )
    frame = pd.read_csv(path, low_memory=False)
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"]),
            "home": frame["home_team"].map(p.clean_team),
            "away": frame["away_team"].map(p.clean_team),
            "home_xg": frame["home_xg"].fillna(frame["ss_home_expected_goals"]),
            "away_xg": frame["away_xg"].fillna(frame["ss_away_expected_goals"]),
            "home_shots": frame["ss_home_total_shots"],
            "away_shots": frame["ss_away_total_shots"],
            "home_shots_on_target": frame["ss_home_shots_on_target"],
            "away_shots_on_target": frame["ss_away_shots_on_target"],
            "home_possession": _numeric(frame["ss_home_ball_possession"], percent=True),
            "away_possession": _numeric(frame["ss_away_ball_possession"], percent=True),
            "home_passes_completed": frame["ss_home_accurate_passes"],
            "away_passes_completed": frame["ss_away_accurate_passes"],
            "home_corners": frame["ss_home_corner_kicks"],
            "away_corners": frame["ss_away_corner_kicks"],
            "home_fouls": frame["ss_home_fouls"],
            "away_fouls": frame["ss_away_fouls"],
            "home_offsides": frame["ss_home_offsides"],
            "away_offsides": frame["ss_away_offsides"],
            "home_yellow_cards": frame["ss_home_yellow_cards"],
            "away_yellow_cards": frame["ss_away_yellow_cards"],
            "home_red_cards": frame["ss_home_red_cards"],
            "away_red_cards": frame["ss_away_red_cards"],
        }
    )
    stat_columns = [
        f"{side}_{metric}" for metric in STAT_METRICS for side in ("home", "away")
    ]
    return output[output[stat_columns].notna().any(axis=1)]


def _live_observations(refresh: bool) -> pd.DataFrame:
    files = {}
    for name in ("matches.csv", "match_team_stats.csv", "teams.csv"):
        files[name] = p.cached_download(
            f"{LIVE_STATS_BASE_URL}/{name}",
            p.RAW_DIR / f"wc_live_{name}",
            refresh,
        )
    matches = pd.read_csv(files["matches.csv"])
    stats = pd.read_csv(files["match_team_stats.csv"])
    teams = pd.read_csv(files["teams.csv"]).set_index("team_id")["team_name"]
    stats = stats.merge(
        matches[
            [
                "match_id",
                "date",
                "home_team_id",
                "away_team_id",
                "home_xg",
                "away_xg",
                "status",
            ]
        ],
        on="match_id",
        how="inner",
    )
    stats = stats[stats["status"].eq("Completed")]
    home = stats[stats["team_id"].eq(stats["home_team_id"])].set_index("match_id")
    away = stats[stats["team_id"].eq(stats["away_team_id"])].set_index("match_id")
    common = home.index.intersection(away.index)

    def values(side: pd.DataFrame, column: str) -> pd.Series:
        return pd.to_numeric(side.loc[common, column], errors="coerce").reset_index(
            drop=True
        )

    output = pd.DataFrame(
        {
            "date": pd.to_datetime(home.loc[common, "date"]).to_numpy(),
            "home": home.loc[common, "home_team_id"]
            .map(teams)
            .map(p.clean_team)
            .to_numpy(),
            "away": home.loc[common, "away_team_id"]
            .map(teams)
            .map(p.clean_team)
            .to_numpy(),
            "home_xg": values(home, "home_xg"),
            "away_xg": values(home, "away_xg"),
        }
    )
    mapping = {
        "shots": "total_shots",
        "shots_on_target": "shots_on_target",
        "possession": "possession_pct",
        "corners": "corners",
        "fouls": "fouls",
        "offsides": "offsides",
    }
    for metric, column in mapping.items():
        output[f"home_{metric}"] = values(home, column)
        output[f"away_{metric}"] = values(away, column)
    output["home_possession"] /= 100.0
    output["away_possession"] /= 100.0
    for metric in ("passes_completed", "yellow_cards", "red_cards"):
        output[f"home_{metric}"] = np.nan
        output[f"away_{metric}"] = np.nan
    return output


def _read_match_stats(refresh: bool) -> pd.DataFrame:
    frames = [_historical_observations(refresh)]
    try:
        frames.append(_live_observations(refresh))
    except (FileNotFoundError, KeyError, ValueError):
        pass
    output = pd.concat(frames, ignore_index=True)
    output = output.sort_values("date").drop_duplicates(
        ["date", "home", "away"],
        keep="last",
    )
    return output.reset_index(drop=True)


@lru_cache(maxsize=1)
def _cached_match_stats() -> pd.DataFrame:
    return _read_match_stats(False)


def load_match_stats(refresh: bool = False) -> pd.DataFrame:
    """Return one normalized row per match, independent of competition year."""
    if refresh:
        output = _read_match_stats(True)
        _cached_match_stats.cache_clear()
        return output
    return _cached_match_stats().copy()


def rolling_match_stats(
    matches: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    window: int = 10,
) -> pd.DataFrame:
    """Compute rolling team statistics."""
    matches = matches.reset_index(drop=True).copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches["home"] = matches["home"].map(p.clean_team)
    matches["away"] = matches["away"].map(p.clean_team)
    observations = observations.copy()
    observations["date"] = pd.to_datetime(observations["date"])
    observations["home"] = observations["home"].map(p.clean_team)
    observations["away"] = observations["away"].map(p.clean_team)

    history: dict[str, dict[str, deque[tuple[float, float]]]] = defaultdict(
        lambda: {metric: deque(maxlen=window) for metric in STAT_METRICS}
    )
    values: dict[str, np.ndarray] = {}
    for metric in STAT_METRICS:
        for side in ("home", "away"):
            for mode in ("for", "against"):
                values[f"stat_{metric}_{side}_{mode}"] = np.full(len(matches), np.nan)
            values[f"stat_{metric}_{side}_count"] = np.zeros(len(matches))

    targets_by_date = {
        date: group.index.tolist()
        for date, group in matches.groupby("date", sort=False)
    }
    observations_by_date = {
        date: group for date, group in observations.groupby("date", sort=False)
    }
    all_dates = sorted(set(targets_by_date) | set(observations_by_date))
    for date in all_dates:
        for index in targets_by_date.get(date, []):
            row = matches.loc[index]
            for side in ("home", "away"):
                team = row[side]
                for metric in STAT_METRICS:
                    records = history[team][metric]
                    if records:
                        array = np.asarray(records, dtype=float)
                        weights = np.exp(np.arange(len(records)) / (window / 2.0))
                        weights /= weights.sum()
                        values[f"stat_{metric}_{side}_for"][index] = np.average(
                            array[:, 0], weights=weights
                        )
                        values[f"stat_{metric}_{side}_against"][index] = np.average(
                            array[:, 1], weights=weights
                        )
                        values[f"stat_{metric}_{side}_count"][index] = len(records)

        current = observations_by_date.get(date)
        if current is None:
            continue
        for row in current.itertuples(index=False):
            for metric in STAT_METRICS:
                home_value = getattr(row, f"home_{metric}", np.nan)
                away_value = getattr(row, f"away_{metric}", np.nan)
                if pd.isna(home_value) or pd.isna(away_value):
                    continue
                history[row.home][metric].append((float(home_value), float(away_value)))
                history[row.away][metric].append((float(away_value), float(home_value)))
    return pd.concat([matches, pd.DataFrame(values)], axis=1)


def goal_features(frame: pd.DataFrame, *, invert: bool = False) -> pd.DataFrame:
    """Project universal rolling statistics into oriented goal-model features."""
    sign = -1.0 if invert else 1.0
    scoring_side, opponent_side = ("away", "home") if invert else ("home", "away")
    output = pd.DataFrame(index=frame.index)
    for metric in STAT_METRICS:
        scale = STAT_SCALES[metric]
        output[f"stat_{metric}_attack"] = (
            frame[f"stat_{metric}_{scoring_side}_for"] / scale
        )
        output[f"stat_{metric}_opponent_conceded"] = (
            frame[f"stat_{metric}_{opponent_side}_against"] / scale
        )
        home_net = (
            frame[f"stat_{metric}_home_for"] - frame[f"stat_{metric}_home_against"]
        )
        away_net = (
            frame[f"stat_{metric}_away_for"] - frame[f"stat_{metric}_away_against"]
        )
        output[f"stat_{metric}_net_difference"] = sign * (home_net - away_net) / scale
        output[f"stat_{metric}_coverage"] = (
            np.minimum(
                frame[f"stat_{metric}_{scoring_side}_count"],
                frame[f"stat_{metric}_{opponent_side}_count"],
            )
            / 10.0
        )
    return output
