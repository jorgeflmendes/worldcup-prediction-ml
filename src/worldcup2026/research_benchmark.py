from __future__ import annotations

import io
import json
import os
import shutil
import warnings
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import goals_simulation as g
from . import market as market_data
from . import match_stats as match_stat_data
from . import pipeline as p
from .evaluation import (
    apply_beta_calibrator,
    apply_beta_calibrator,
    apply_platt_calibrator,
    apply_temperature,
    clustered_bootstrap_interval,
    clustered_sign_flip_pvalue,
    fit_beta_calibrator,
    fit_beta_calibrator,
    fit_platt_calibrator,
    metric_summary,
    rps_vector,
    safe_probability as _safe_probability,
    tune_temperature,
)
from .fixtures import load_2026_fixtures


warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn.linear_model._logistic"
)

DEVELOPMENT_YEARS = (2006, 2010, 2014)
RETROSPECTIVE_YEARS = (2018, 2022)
EVALUATION_YEARS = DEVELOPMENT_YEARS + RETROSPECTIVE_YEARS
PRIMARY_CANDIDATE = "market_consensus_hybrid"
WALK_FORWARD_YEARS = EVALUATION_YEARS
CALIBRATION_LOOKBACK = 3
OUTCOME_LABELS = np.array([0, 1, 2])
ELO_HISTORY_LAGS_DAYS = (0, 30, 60, 90, 120, 150)
ELO_HISTORY_COLUMNS = [f"elo_diff_lag_{days}d" for days in ELO_HISTORY_LAGS_DAYS]
PAPER_SDR_MODELS = {"paper_sdr_sir_poisson", "paper_sdr_save_poisson"}
MULTI_COMPETITION_TOURNAMENTS = (
    "FIFA World Cup",
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "UEFA Nations League",
    "CONCACAF Nations League",
)
MULTI_COMPETITION_MODELS = (
    "poisson",
    "hist_gradient_tuned_prior",
    "match_stats_hist_gradient_dc",
    "elo_logistic",
    "full_logistic",
    "hist_gb_classifier",
)
FORCED_RAW_MODELS = {
    "hist_gradient_raw",
    "hist_gradient_tuned_prior",
    "squad_hist_gradient_dc_prior",
    "market_squad_hist_gradient_dc_prior",
    "market_consensus_hybrid",
}
SQUAD_DC_RHO = 0.025
MARKET_SQUAD_DC_RHO = 0.075
PERFORMANCE_MARKET_SQUAD_DC_RHO = 0.075
MATCH_STATS_DC_RHO_GRID = (-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12)
TRANSFERMARKT_DATASET = "davidcariboo/player-scores"
TUNED_HIST_GRADIENT_PARAMS = {
    "learning_rate": 0.06,
    "max_leaf_nodes": 16,
    "l2_regularization": 8.0,
    "max_iter": 180,
}
MARKET_SQUAD_HIST_GRADIENT_PARAMS = {
    "learning_rate": 0.05,
    "max_leaf_nodes": 16,
    "l2_regularization": 8.0,
    "max_iter": 300,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 15,
}
PERFORMANCE_MARKET_SQUAD_HIST_GRADIENT_PARAMS = {
    "learning_rate": 0.035,
    "max_leaf_nodes": 12,
    "l2_regularization": 12.0,
    "max_iter": 300,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 15,
}
XGBOOST_POISSON_MARKET_PARAMS = {
    "n_estimators": 160,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_child_weight": 50,
    "reg_lambda": 30.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}


FEATURE_COLUMNS = [
    "elo_diff",
    "neutral",
    "venue_advantage",
    "fifa_rank_diff",
    "fifa_points_diff",
    "fifa_rank_missing",
    "external_elo_diff",
    "external_elo_missing",
    "form_points_diff",
    "form_goal_diff",
    "form_opponent_elo",
    "form_matches_diff",
    "competitive_points_diff",
    "competitive_goal_diff",
]

SQUAD_GOAL_FEATURES = [
    "squad_age_mean",
    "squad_caps_mean",
    "squad_caps_total",
    "squad_caps_p90",
    "squad_caps_50plus_share",
    "club_elo_mean",
    "club_elo_p90",
    "top5_club_share",
    "superelite_club_share",
    "big5_league_share",
    "returning_wc_player_share",
    "coach_prior_wc_appearances",
    "coach_returning_to_same_team",
]

_SQUAD_FEATURE_MAP_CACHE: dict[int, pd.DataFrame] | None = None
_WORLD_CUP_MATCH_KEYS_CACHE: set[tuple[pd.Timestamp, str, str]] | None = None
_MARKET_VALUE_MAP_CACHE: dict[int, pd.DataFrame] | None = None
_PLAYER_PERFORMANCE_MAP_CACHE: dict[int, pd.DataFrame] | None = None
_MATCH_STATS_GOAL_DATA_CACHE: tuple[pd.DataFrame, pd.DataFrame] | None = None
_MATCH_STATS_RHO_CACHE: tuple[pd.DataFrame, float] | None = None
DERIVED_CACHE_DIR = p.RAW_DIR / "derived"


def cuda_enabled() -> bool:
    """Use CUDA when available, with WORLDCUP_USE_CUDA=0 as a reproducible CPU override."""
    return (
        os.environ.get("WORLDCUP_USE_CUDA", "1") != "0"
        and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "-1"
        and shutil.which("nvidia-smi") is not None
    )


def add_competition_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    tournament = frame.get("tournament", frame.get("competition", "")).astype(str)
    output = pd.DataFrame(index=frame.index)
    output["ctx_is_friendly"] = tournament.eq("Friendly").astype(float)
    output["ctx_is_qualifier"] = tournament.str.contains("qualification", case=False).astype(float)
    output["ctx_is_nations_league"] = tournament.str.contains("Nations League", case=False).astype(float)
    output["ctx_is_major_final"] = tournament.isin(MULTI_COMPETITION_TOURNAMENTS[:6]).astype(float)
    output["ctx_is_world_cup"] = tournament.eq("FIFA World Cup").astype(float)
    output["ctx_is_continental_final"] = tournament.isin(
        MULTI_COMPETITION_TOURNAMENTS[1:6]
    ).astype(float)
    tier = np.select(
        [
            tournament.eq("FIFA World Cup"),
            tournament.isin(MULTI_COMPETITION_TOURNAMENTS[1:6]),
            tournament.str.contains("Nations League", case=False),
            tournament.str.contains("qualification", case=False),
            tournament.eq("Friendly"),
        ],
        [5.0, 4.0, 3.0, 2.0, 1.0],
        default=2.5,
    )
    output["ctx_competition_tier"] = tier
    return output


def add_candidate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Generate tabular candidate features."""
    x = frame[FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        if column.endswith("_missing") or column in {"neutral", "venue_advantage"}:
            continue
        values = x[column].astype(float)
        x[f"{column}_abs"] = values.abs()
        x[f"{column}_sq"] = values * values
        x[f"{column}_sign"] = np.sign(values)
    for left, right in [
        ("elo_diff", "fifa_rank_diff"),
        ("elo_diff", "external_elo_diff"),
        ("elo_diff", "form_points_diff"),
        ("fifa_points_diff", "form_goal_diff"),
        ("external_elo_diff", "competitive_points_diff"),
    ]:
        x[f"{left}_x_{right}"] = x[left].astype(float) * x[right].astype(float)
    x = pd.concat([x, add_competition_context_features(frame)], axis=1)
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def world_cup_start(results: pd.DataFrame, year: int) -> pd.Timestamp:
    games = results[
        (results.tournament == "FIFA World Cup") & (results.date.dt.year == year)
    ]
    if games.empty:
        raise ValueError(f"No World Cup matches found for {year}.")
    return pd.Timestamp(games.date.min())


def frozen_match_frame(
    results: pd.DataFrame,
    year: int,
    ratings: dict[str, float],
    fifa: dict[str, tuple[float, float]],
    forms: dict[str, tuple[float, float, float, float, float, float]],
    external: dict[str, float],
) -> pd.DataFrame:
    games = results[
        (results.tournament == "FIFA World Cup")
        & (results.date.dt.year == year)
        & results.home_score.notna()
        & results.away_score.notna()
    ].copy()
    rows = []
    for row in games.sort_values("date").itertuples(index=False):
        home = p.clean_team(row.home_team)
        away = p.clean_team(row.away_team)
        neutral = bool(getattr(row, "neutral", True))
        country = str(getattr(row, "country", ""))
        home_rank, home_points = fifa.get(home, (np.nan, np.nan))
        away_rank, away_points = fifa.get(away, (np.nan, np.nan))
        home_external = external.get(home, np.nan)
        away_external = external.get(away, np.nan)
        home_form = forms.get(home, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))
        away_form = forms.get(away, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))
        rows.append(
            {
                "date": row.date,
                "year": year,
                "home": home,
                "away": away,
                "home_goals": int(row.home_score),
                "away_goals": int(row.away_score),
                "truth": 0
                if row.home_score > row.away_score
                else 1
                if row.home_score == row.away_score
                else 2,
                "elo_diff": ratings.get(home, 1500.0) - ratings.get(away, 1500.0),
                "neutral": int(neutral),
                "venue_advantage": p.venue_advantage(
                    home,
                    away,
                    country,
                    neutral,
                ),
                "fifa_rank_diff": np.nan_to_num(away_rank - home_rank, nan=0.0),
                "fifa_points_diff": np.nan_to_num(home_points - away_points, nan=0.0),
                "fifa_rank_missing": int(np.isnan(home_rank) or np.isnan(away_rank)),
                "external_elo_diff": np.nan_to_num(
                    home_external - away_external, nan=0.0
                ),
                "external_elo_missing": int(
                    np.isnan(home_external) or np.isnan(away_external)
                ),
                "form_points_diff": home_form[0] - away_form[0],
                "form_goal_diff": home_form[1] - away_form[1],
                "form_opponent_elo": home_form[2] - away_form[2],
                "form_matches_diff": home_form[3] - away_form[3],
                "competitive_points_diff": home_form[4] - away_form[4],
                "competitive_goal_diff": home_form[5] - away_form[5],
            }
        )
    return pd.DataFrame(rows)


def frozen_world_cup_frame(results: pd.DataFrame, year: int) -> pd.DataFrame:
    cutoff = world_cup_start(results, year)
    games = results[
        (results.tournament == "FIFA World Cup") & (results.date.dt.year == year)
    ]
    teams = sorted(
        set(games.home_team.map(p.clean_team)).union(games.away_team.map(p.clean_team))
    )
    ratings = p.elo_before(results, cutoff)
    fifa = g.fifa_snapshot(cutoff)
    forms = g.form_snapshot(results, cutoff, ratings, teams)
    external = g.external_elo_snapshot(cutoff)
    return frozen_match_frame(results, year, ratings, fifa, forms, external)


def prematch_competition_frame(
    goal_data: pd.DataFrame,
    tournament: str,
    year: int,
) -> pd.DataFrame:
    frame = goal_data[
        goal_data.tournament.eq(tournament) & goal_data.date.dt.year.eq(year)
    ].copy()
    frame["year"] = year
    frame["competition"] = tournament
    frame["truth"] = np.select(
        [
            frame.home_goals.gt(frame.away_goals),
            frame.home_goals.eq(frame.away_goals),
        ],
        [0, 1],
        default=2,
    ).astype(int)
    return frame.sort_values(["date", "home", "away"]).reset_index(drop=True)


def prematch_world_cup_frame(
    goal_data: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    return prematch_competition_frame(goal_data, "FIFA World Cup", year)


def competition_start(results: pd.DataFrame, tournament: str, year: int) -> pd.Timestamp:
    games = results[
        results.tournament.eq(tournament)
        & results.date.dt.year.eq(year)
        & results.home_score.notna()
        & results.away_score.notna()
    ]
    if games.empty:
        raise ValueError(f"No {tournament} matches found for {year}.")
    return pd.Timestamp(games.date.min())


def competition_specs(
    results: pd.DataFrame,
    tournaments: tuple[str, ...] = MULTI_COMPETITION_TOURNAMENTS,
    minimum_year: int = 2006,
    minimum_matches: int = 8,
) -> list[tuple[str, int, int]]:
    games = results[
        results.tournament.isin(tournaments)
        & results.home_score.notna()
        & results.away_score.notna()
        & results.date.dt.year.ge(minimum_year)
    ]
    specs = []
    for (tournament, year), group in games.groupby(["tournament", games.date.dt.year]):
        if len(group) >= minimum_matches:
            specs.append((str(tournament), int(year), int(len(group))))
    return sorted(specs, key=lambda item: (item[1], item[0]))


def rating_at(
    history: dict[str, list[tuple[pd.Timestamp, float]]], team: str, date: pd.Timestamp
) -> float:
    records = history.get(team)
    if not records:
        return 1500.0
    dates = [record[0] for record in records]
    index = np.searchsorted(dates, date, side="right") - 1
    return 1500.0 if index < 0 else float(records[index][1])


def elo_history_snapshots(
    results: pd.DataFrame,
) -> dict[str, list[tuple[pd.Timestamp, float]]]:
    """Team Elo timeline after each match day, with same update rule as the baseline."""
    cache_path = DERIVED_CACHE_DIR / "elo_history_snapshots.json"
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return {
            team: [(pd.Timestamp(item[0]), float(item[1])) for item in records]
            for team, records in raw.items()
        }
    ratings: dict[str, float] = {}
    history: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for date, daily_games in results.sort_values("date").groupby("date", sort=True):
        pending = []
        for game in daily_games.itertuples(index=False):
            if pd.isna(game.home_score) or pd.isna(game.away_score):
                continue
            home, away = p.clean_team(game.home_team), p.clean_team(game.away_team)
            home_rating, away_rating = (
                ratings.get(home, 1500.0),
                ratings.get(away, 1500.0),
            )
            neutral = bool(getattr(game, "neutral", True))
            venue = p.venue_advantage(
                home,
                away,
                getattr(game, "country", ""),
                neutral,
            )
            pending.append((game, home, away, home_rating, away_rating, venue))
            history.setdefault(home, []).append((pd.Timestamp(date), home_rating))
            history.setdefault(away, []).append((pd.Timestamp(date), away_rating))
        for game, home, away, home_rating, away_rating, venue in pending:
            advantage = 55.0 * venue
            expectation = 1 / (
                1 + 10 ** (-((home_rating + advantage) - away_rating) / 400)
            )
            home_points = (
                1.0
                if game.home_score > game.away_score
                else 0.5
                if game.home_score == game.away_score
                else 0.0
            )
            weight = 1.0 if game.tournament == "Friendly" else 1.35
            change = 22 * weight * (home_points - expectation)
            ratings[home] = home_rating + change
            ratings[away] = away_rating - change
    serializable = {
        team: [(str(date.date()), float(value)) for date, value in records]
        for team, records in history.items()
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(serializable), encoding="utf-8")
    return history


def attach_elo_history_features(
    frame: pd.DataFrame, history: dict[str, list[tuple[pd.Timestamp, float]]]
) -> pd.DataFrame:
    output = frame.copy()
    for days, column in zip(ELO_HISTORY_LAGS_DAYS, ELO_HISTORY_COLUMNS):
        values = []
        for row in output.itertuples(index=False):
            lookup_date = pd.Timestamp(row.date) - pd.Timedelta(days=days)
            values.append(
                rating_at(history, row.home, lookup_date)
                - rating_at(history, row.away, lookup_date)
            )
        output[column] = values
    return output


def elo_history_goal_data(
    results: pd.DataFrame, goal_data: pd.DataFrame
) -> pd.DataFrame:
    cache_path = DERIVED_CACHE_DIR / "elo_history_goal_data.csv"
    if cache_path.exists():
        frame = pd.read_csv(cache_path, parse_dates=["date"])
        if set(ELO_HISTORY_COLUMNS).issubset(frame.columns) and len(frame) == len(
            goal_data
        ):
            return frame
    history = elo_history_snapshots(results)
    frame = attach_elo_history_features(goal_data, history)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def frozen_world_cup_frame_with_history(
    results: pd.DataFrame,
    goal_data: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    return attach_elo_history_features(
        prematch_world_cup_frame(goal_data, year),
        elo_history_snapshots(results),
    )


def world_cup_match_keys(results: pd.DataFrame) -> set[tuple[pd.Timestamp, str, str]]:
    global _WORLD_CUP_MATCH_KEYS_CACHE
    if _WORLD_CUP_MATCH_KEYS_CACHE is None:
        games = results[results.tournament == "FIFA World Cup"]
        _WORLD_CUP_MATCH_KEYS_CACHE = {
            (
                pd.Timestamp(row.date),
                p.clean_team(row.home_team),
                p.clean_team(row.away_team),
            )
            for row in games.itertuples(index=False)
        }
    return _WORLD_CUP_MATCH_KEYS_CACHE


def squad_feature_maps(results: pd.DataFrame) -> dict[int, pd.DataFrame]:
    global _SQUAD_FEATURE_MAP_CACHE
    if _SQUAD_FEATURE_MAP_CACHE is None:
        maps: dict[int, pd.DataFrame] = {}
        for year in g.STANDARD_32_YEARS:
            cutoff = world_cup_start(results, year)
            games = results[
                (results.tournament == "FIFA World Cup")
                & (results.date.dt.year == year)
            ]
            teams = sorted(
                set(games.home_team.map(p.clean_team)).union(
                    games.away_team.map(p.clean_team)
                )
            )
            maps[year] = p.squad_features(cutoff, year, teams, False).set_index("team")
        _SQUAD_FEATURE_MAP_CACHE = maps
    return _SQUAD_FEATURE_MAP_CACHE


def _squad_players_for_year(
    results: pd.DataFrame, year: int, cutoff: pd.Timestamp
) -> pd.DataFrame:
    if year in p.CHAMPIONS:
        players, _ = p.wiki_squad_snapshot(year, False)
        return players
    if year == 2026:
        return p.kaggle_2026_squad_snapshot(cutoff.strftime("%Y-%m-%d"), False)
    games = results[
        (results.tournament == "FIFA World Cup") & (results.date.dt.year == year)
    ]
    teams = sorted(
        set(games.home_team.map(p.clean_team)).union(games.away_team.map(p.clean_team))
    )
    return pd.DataFrame({"team": teams, "player_name": np.nan})


def market_value_maps(results: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Build Transfermarkt squad-value aggregates."""
    global _MARKET_VALUE_MAP_CACHE
    if _MARKET_VALUE_MAP_CACHE is not None:
        return _MARKET_VALUE_MAP_CACHE

    archive = p.cached_download(
        p.kaggle_download_url(TRANSFERMARKT_DATASET),
        p.RAW_DIR / "transfermarkt_player_scores.zip",
        False,
    )
    with zipfile.ZipFile(archive) as zipped:
        players = pd.read_csv(
            io.BytesIO(zipped.read("players.csv")),
            usecols=["player_id", "name", "country_of_citizenship"],
        )
        valuations = pd.read_csv(
            io.BytesIO(zipped.read("player_valuations.csv")),
            usecols=["player_id", "date", "market_value_in_eur"],
            parse_dates=["date"],
        )
    players["name_key"] = players.name.map(p.normalized_text)
    players["country"] = players.country_of_citizenship.map(p.clean_team)
    by_name_country = {
        key: group
        for key, group in players.groupby(["name_key", "country"], dropna=False)
    }
    by_name = {key: group for key, group in players.groupby("name_key")}
    valuations_by_player = {
        player_id: group.sort_values("date")
        for player_id, group in valuations.groupby("player_id", sort=False)
    }

    def candidate_ids(player_name: object, team: object) -> list[int]:
        name_key = p.normalized_text(player_name)
        team_name = p.clean_team(team)
        group = by_name_country.get((name_key, team_name))
        if group is not None and len(group):
            return [int(value) for value in group.player_id]
        group = by_name.get(name_key)
        if group is not None and len(group) == 1:
            return [int(value) for value in group.player_id]
        return []

    def value_before(player_id: int, cutoff: pd.Timestamp) -> float:
        group = valuations_by_player.get(player_id)
        if group is None:
            return np.nan
        index = group.date.searchsorted(cutoff, side="right") - 1
        return np.nan if index < 0 else float(group.iloc[index].market_value_in_eur)

    maps: dict[int, pd.DataFrame] = {}
    years = list(g.STANDARD_32_YEARS) + [2026]
    for year in years:
        cutoff = (
            pd.Timestamp("2026-06-10")
            if year == 2026
            else world_cup_start(results, year)
        )
        squad = _squad_players_for_year(results, year, cutoff)
        rows = []
        for row in squad.itertuples(index=False):
            values = [
                value_before(player_id, cutoff)
                for player_id in candidate_ids(row.player_name, row.team)
            ]
            values = [value for value in values if pd.notna(value)]
            rows.append(
                {
                    "team": p.clean_team(row.team),
                    "market_value_eur": max(values) if values else np.nan,
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            maps[year] = pd.DataFrame(
                columns=[
                    "market_value_sum",
                    "market_value_mean",
                    "market_value_p90",
                    "market_value_count",
                    "market_value_coverage",
                ]
            )
            continue
        grouped = frame.groupby("team").market_value_eur
        aggregate = grouped.agg(
            market_value_sum=lambda values: values.sum(min_count=1),
            market_value_mean="mean",
            market_value_p90=lambda values: values.quantile(0.9),
            market_value_count="count",
        )
        aggregate["market_value_coverage"] = grouped.apply(
            lambda values: float(values.notna().mean())
        )
        maps[year] = aggregate
    _MARKET_VALUE_MAP_CACHE = maps
    return maps


def player_performance_maps(results: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Build annual Transfermarkt player-performance aggregates."""
    global _PLAYER_PERFORMANCE_MAP_CACHE
    if _PLAYER_PERFORMANCE_MAP_CACHE is not None:
        return _PLAYER_PERFORMANCE_MAP_CACHE

    archive = p.cached_download(
        p.kaggle_download_url(TRANSFERMARKT_DATASET),
        p.RAW_DIR / "transfermarkt_player_scores.zip",
        False,
    )
    with zipfile.ZipFile(archive) as zipped:
        players = pd.read_csv(
            io.BytesIO(zipped.read("players.csv")),
            usecols=["player_id", "name", "country_of_citizenship"],
        )
        appearances = pd.read_csv(
            io.BytesIO(zipped.read("appearances.csv")),
            usecols=["player_id", "date", "minutes_played", "goals", "assists"],
            parse_dates=["date"],
        )
    players["name_key"] = players.name.map(p.normalized_text)
    players["country"] = players.country_of_citizenship.map(p.clean_team)
    by_name_country = {
        key: group
        for key, group in players.groupby(["name_key", "country"], dropna=False)
    }
    by_name = {key: group for key, group in players.groupby("name_key")}
    appearances_by_player = {
        player_id: group.sort_values("date")
        for player_id, group in appearances.groupby("player_id", sort=False)
    }

    def candidate_ids(player_name: object, team: object) -> list[int]:
        name_key = p.normalized_text(player_name)
        group = by_name_country.get((name_key, p.clean_team(team)))
        if group is not None and len(group):
            return [int(value) for value in group.player_id]
        group = by_name.get(name_key)
        if group is not None and len(group) == 1:
            return [int(value) for value in group.player_id]
        return []

    def performance_before(player_id: int, cutoff: pd.Timestamp) -> dict[str, float]:
        group = appearances_by_player.get(player_id)
        if group is None:
            recent = group
        else:
            recent = group[
                (group.date < cutoff) & (group.date >= cutoff - pd.Timedelta(days=365))
            ]
        if recent is None or recent.empty:
            return {
                "minutes_365": np.nan,
                "goals_365": np.nan,
                "assists_365": np.nan,
                "appearances_365": np.nan,
            }
        return {
            "minutes_365": float(recent.minutes_played.sum()),
            "goals_365": float(recent.goals.sum()),
            "assists_365": float(recent.assists.sum()),
            "appearances_365": float(len(recent)),
        }

    maps: dict[int, pd.DataFrame] = {}
    for year in list(g.STANDARD_32_YEARS) + [2026]:
        cutoff = (
            pd.Timestamp("2026-06-10")
            if year == 2026
            else world_cup_start(results, year)
        )
        squad = _squad_players_for_year(results, year, cutoff)
        rows = []
        for row in squad.itertuples(index=False):
            candidates = [
                performance_before(player_id, cutoff)
                for player_id in candidate_ids(row.player_name, row.team)
            ]
            candidates = [
                candidate
                for candidate in candidates
                if pd.notna(candidate["minutes_365"])
            ]
            if candidates:
                value = max(candidates, key=lambda candidate: candidate["minutes_365"])
            else:
                value = {
                    "minutes_365": np.nan,
                    "goals_365": np.nan,
                    "assists_365": np.nan,
                    "appearances_365": np.nan,
                }
            rows.append({"team": p.clean_team(row.team), **value})
        frame = pd.DataFrame(rows)
        if frame.empty:
            maps[year] = pd.DataFrame()
            continue
        grouped = frame.groupby("team")
        aggregate = grouped.agg(
            minutes_365_sum=("minutes_365", lambda values: values.sum(min_count=1)),
            minutes_365_mean=("minutes_365", "mean"),
            goals_365_sum=("goals_365", lambda values: values.sum(min_count=1)),
            assists_365_sum=("assists_365", lambda values: values.sum(min_count=1)),
            appearances_365_sum=(
                "appearances_365",
                lambda values: values.sum(min_count=1),
            ),
        )
        aggregate["performance_coverage"] = grouped.minutes_365.apply(
            lambda values: float(values.notna().mean())
        )
        aggregate["goal_contributions_per90"] = (
            90.0
            * (aggregate.goals_365_sum + aggregate.assists_365_sum)
            / aggregate.minutes_365_sum.clip(lower=90.0)
        )
        maps[year] = aggregate
    _PLAYER_PERFORMANCE_MAP_CACHE = maps
    return maps


def squad_goal_features(
    frame: pd.DataFrame,
    results: pd.DataFrame,
    invert_elo: bool = False,
    include_market_value: bool = False,
    include_player_performance: bool = False,
) -> pd.DataFrame:
    base = g.goal_features(frame, invert_elo=invert_elo)
    maps = squad_feature_maps(results)
    keys = world_cup_match_keys(results)
    sign = -1.0 if invert_elo else 1.0
    rows = []
    for row in frame.itertuples(index=False):
        date = pd.Timestamp(row.date)
        year = int(date.year)
        key = (date, p.clean_team(row.home), p.clean_team(row.away))
        squad = maps.get(year)
        values = []
        for column in SQUAD_GOAL_FEATURES:
            value = np.nan
            if key in keys and squad is not None and column in squad.columns:
                home_value = squad[column].get(p.clean_team(row.home), np.nan)
                away_value = squad[column].get(p.clean_team(row.away), np.nan)
                if pd.notna(home_value) and pd.notna(away_value):
                    value = float(home_value) - float(away_value)
            values.append(sign * value if pd.notna(value) else np.nan)
        rows.append(values)
    extra = pd.DataFrame(
        rows,
        columns=[f"{column}_diff" for column in SQUAD_GOAL_FEATURES],
        index=base.index,
    )
    output = pd.concat([base, extra], axis=1)
    if not include_market_value:
        return output

    market_maps = market_value_maps(results)
    market_columns = [
        "market_value_sum",
        "market_value_mean",
        "market_value_p90",
        "market_value_count",
        "market_value_coverage",
    ]
    market_rows = []
    for row in frame.itertuples(index=False):
        date = pd.Timestamp(row.date)
        key = (date, p.clean_team(row.home), p.clean_team(row.away))
        market = market_maps.get(int(date.year))
        values = []
        for column in market_columns:
            value = np.nan
            if key in keys and market is not None and column in market.columns:
                home_value = market[column].get(p.clean_team(row.home), np.nan)
                away_value = market[column].get(p.clean_team(row.away), np.nan)
                if pd.notna(home_value) and pd.notna(away_value):
                    if column in {
                        "market_value_sum",
                        "market_value_mean",
                        "market_value_p90",
                    }:
                        value = np.log1p(float(home_value)) - np.log1p(
                            float(away_value)
                        )
                    else:
                        value = float(home_value) - float(away_value)
            values.append(sign * value if pd.notna(value) else np.nan)
        market_rows.append(values)
    market_extra = pd.DataFrame(
        market_rows,
        columns=[f"{column}_diff" for column in market_columns],
        index=base.index,
    )
    output = pd.concat([output, market_extra], axis=1)
    if not include_player_performance:
        return output

    performance_maps = player_performance_maps(results)
    performance_columns = [
        "minutes_365_sum",
        "minutes_365_mean",
        "goals_365_sum",
        "assists_365_sum",
        "appearances_365_sum",
        "goal_contributions_per90",
        "performance_coverage",
    ]
    performance_rows = []
    for row in frame.itertuples(index=False):
        date = pd.Timestamp(row.date)
        key = (date, p.clean_team(row.home), p.clean_team(row.away))
        performance = performance_maps.get(int(date.year))
        values = []
        for column in performance_columns:
            value = np.nan
            if (
                key in keys
                and performance is not None
                and column in performance.columns
            ):
                home_value = performance[column].get(p.clean_team(row.home), np.nan)
                away_value = performance[column].get(p.clean_team(row.away), np.nan)
                if pd.notna(home_value) and pd.notna(away_value):
                    if column not in {
                        "goal_contributions_per90",
                        "performance_coverage",
                    }:
                        value = np.log1p(float(home_value)) - np.log1p(
                            float(away_value)
                        )
                    else:
                        value = float(home_value) - float(away_value)
            values.append(sign * value if pd.notna(value) else np.nan)
        performance_rows.append(values)
    performance_extra = pd.DataFrame(
        performance_rows,
        columns=[f"{column}_diff" for column in performance_columns],
        index=base.index,
    )
    return pd.concat([output, performance_extra], axis=1)


def market_squad_pair_goal_features(
    frame: pd.DataFrame,
    squad: pd.DataFrame,
    market: pd.DataFrame,
    invert_elo: bool = False,
) -> pd.DataFrame:
    base = squad_pair_goal_features(frame, squad, invert_elo=invert_elo)
    sign = -1.0 if invert_elo else 1.0
    market_columns = [
        "market_value_sum",
        "market_value_mean",
        "market_value_p90",
        "market_value_count",
        "market_value_coverage",
    ]
    rows = []
    for row in frame.itertuples(index=False):
        values = []
        for column in market_columns:
            value = np.nan
            if column in market.columns:
                home_value = market[column].get(p.clean_team(row.home), np.nan)
                away_value = market[column].get(p.clean_team(row.away), np.nan)
                if pd.notna(home_value) and pd.notna(away_value):
                    if column in {
                        "market_value_sum",
                        "market_value_mean",
                        "market_value_p90",
                    }:
                        value = np.log1p(float(home_value)) - np.log1p(
                            float(away_value)
                        )
                    else:
                        value = float(home_value) - float(away_value)
            values.append(sign * value if pd.notna(value) else np.nan)
        rows.append(values)
    extra = pd.DataFrame(
        rows, columns=[f"{column}_diff" for column in market_columns], index=base.index
    )
    return pd.concat([base, extra], axis=1)


def performance_market_squad_pair_goal_features(
    frame: pd.DataFrame,
    squad: pd.DataFrame,
    market: pd.DataFrame,
    performance: pd.DataFrame,
    invert_elo: bool = False,
) -> pd.DataFrame:
    base = market_squad_pair_goal_features(
        frame,
        squad,
        market,
        invert_elo=invert_elo,
    )
    sign = -1.0 if invert_elo else 1.0
    performance_columns = [
        "minutes_365_sum",
        "minutes_365_mean",
        "goals_365_sum",
        "assists_365_sum",
        "appearances_365_sum",
        "goal_contributions_per90",
        "performance_coverage",
    ]
    rows = []
    for row in frame.itertuples(index=False):
        values = []
        for column in performance_columns:
            value = np.nan
            if column in performance.columns:
                home_value = performance[column].get(p.clean_team(row.home), np.nan)
                away_value = performance[column].get(p.clean_team(row.away), np.nan)
                if pd.notna(home_value) and pd.notna(away_value):
                    if column not in {
                        "goal_contributions_per90",
                        "performance_coverage",
                    }:
                        value = np.log1p(float(home_value)) - np.log1p(
                            float(away_value)
                        )
                    else:
                        value = float(home_value) - float(away_value)
            values.append(sign * value if pd.notna(value) else np.nan)
        rows.append(values)
    extra = pd.DataFrame(
        rows,
        columns=[f"{column}_diff" for column in performance_columns],
        index=base.index,
    )
    return pd.concat([base, extra], axis=1)


@dataclass
class SDRPoisson:
    method: str
    mean: np.ndarray
    whitening: np.ndarray
    directions: np.ndarray
    home: PoissonRegressor
    away: PoissonRegressor

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        z = (
            frame[ELO_HISTORY_COLUMNS].to_numpy(dtype=float) - self.mean
        ) @ self.whitening
        scores = z @ self.directions
        data = {f"sdr_{i + 1}": scores[:, i] for i in range(scores.shape[1])}
        data["neutral"] = frame.neutral.to_numpy(dtype=float)
        data["form_goal_diff"] = frame.form_goal_diff.to_numpy(dtype=float)
        data["fifa_points_diff"] = frame.fifa_points_diff.to_numpy(dtype=float) / 1000.0
        return pd.DataFrame(data)


def sdr_directions(
    x: np.ndarray, y: np.ndarray, method: str, dimensions: int = 2
) -> np.ndarray:
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    values = np.clip(values, 1e-8, None)
    whitening = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    z = (x - mean) @ whitening
    classes = np.unique(y)
    if method == "sir":
        matrix = np.zeros((z.shape[1], z.shape[1]))
        for label in classes:
            group = z[y == label]
            if len(group) == 0:
                continue
            mu = group.mean(axis=0)
            matrix += (len(group) / len(z)) * np.outer(mu, mu)
    elif method == "save":
        matrix = np.zeros((z.shape[1], z.shape[1]))
        identity = np.eye(z.shape[1])
        for label in classes:
            group = z[y == label]
            if len(group) < 3:
                continue
            cov = np.cov(group, rowvar=False)
            delta = identity - cov
            matrix += (len(group) / len(z)) * (delta @ delta)
    else:
        raise ValueError(method)
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1][:dimensions]
    directions = vectors[:, order]
    return mean, whitening, directions


def fit_sdr_poisson(
    goal_history: pd.DataFrame, cutoff: pd.Timestamp, method: str
) -> SDRPoisson:
    train = goal_history[goal_history.date < cutoff].copy()
    y = np.where(
        train.home_goals > train.away_goals,
        0,
        np.where(train.home_goals == train.away_goals, 1, 2),
    )
    mean, whitening, directions = sdr_directions(
        train[ELO_HISTORY_COLUMNS].to_numpy(dtype=float), y, method, dimensions=2
    )
    model = SDRPoisson(
        method=method,
        mean=mean,
        whitening=whitening,
        directions=directions,
        home=PoissonRegressor(alpha=0.35, max_iter=700),
        away=PoissonRegressor(alpha=0.35, max_iter=700),
    )
    x_home = model.transform(train)
    mirrored = train.copy()
    for column in ELO_HISTORY_COLUMNS + ["form_goal_diff", "fifa_points_diff"]:
        mirrored[column] = -mirrored[column]
    x_away = model.transform(mirrored)
    model.home.fit(x_home, train.home_goals.clip(upper=10))
    model.away.fit(x_away, train.away_goals.clip(upper=10))
    return model


def sdr_probabilities(model: SDRPoisson, frame: pd.DataFrame) -> np.ndarray:
    x_home = model.transform(frame)
    mirrored = frame.copy()
    for column in ELO_HISTORY_COLUMNS + ["form_goal_diff", "fifa_points_diff"]:
        mirrored[column] = -mirrored[column]
    x_away = model.transform(mirrored)
    home_mean = model.home.predict(x_home).clip(0.05, 6.0)
    away_mean = model.away.predict(x_away).clip(0.05, 6.0)
    rows = []
    goals = np.arange(g.MAX_GOALS + 1)
    home_grid, away_grid = np.meshgrid(goals, goals, indexing="ij")
    for h_mean, a_mean in zip(home_mean, away_mean):
        matrix = np.outer(
            g.poisson_probabilities(float(h_mean)),
            g.poisson_probabilities(float(a_mean)),
        )
        rows.append(
            [
                float(matrix[home_grid > away_grid].sum()),
                float(matrix[home_grid == away_grid].sum()),
                float(matrix[home_grid < away_grid].sum()),
            ]
        )
    return _safe_probability(np.array(rows))


def paper_match_importance(tournament: str) -> float:
    text = str(tournament).lower()
    if text == "fifa world cup":
        return 60.0
    continental = (
        "uefa euro",
        "copa am",
        "african cup",
        "afcon",
        "concacaf championship",
        "gold cup",
        "asian cup",
        "oceania",
    )
    if (
        any(token in text for token in continental)
        and "qualification" not in text
        and "qualifier" not in text
    ):
        return 35.0
    if "qualification" in text or "qualifier" in text:
        return 25.0
    return 20.0


def paper_goal_multiplier(home_goals: float, away_goals: float) -> float:
    margin = abs(float(home_goals) - float(away_goals))
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def rolling_team_values(
    history: dict[str, list[tuple[float, float, float]]], team: str
) -> tuple[float, float, float]:
    recent = history.get(team, [])[-6:]
    if not recent:
        return 1.0, 1.0, 1.0
    scored = np.array([item[0] for item in recent], dtype=float)
    conceded = np.array([item[1] for item in recent], dtype=float)
    points = np.array([item[2] for item in recent], dtype=float)
    return float(scored.mean()), float(conceded.mean()), float(points.mean())


def monthly_rating_at(
    history: dict[str, list[tuple[pd.Timestamp, float]]],
    team: str,
    date: pd.Timestamp,
    lag_months: int,
) -> float:
    month_start = pd.Timestamp(date).replace(day=1).normalize() - pd.DateOffset(
        months=lag_months
    )
    return rating_at(history, team, month_start)


def paper_elo_match_data(results: pd.DataFrame) -> pd.DataFrame:
    cache_path = DERIVED_CACHE_DIR / "paper_elo_match_data.csv"
    if cache_path.exists():
        frame = pd.read_csv(cache_path, parse_dates=["date"])
        required = set(
            ELO_HISTORY_COLUMNS
            + [
                "home_rating",
                "away_rating",
                "post_home_rating",
                "post_away_rating",
                "home_gf6",
                "home_ga6",
                "away_gf6",
                "away_ga6",
                "form_points_diff",
            ]
        )
        if required.issubset(frame.columns):
            return frame
    ratings: dict[str, float] = {}
    rating_history: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    form_history: dict[str, list[tuple[float, float, float]]] = {}
    rows: list[dict[str, object]] = []
    for date, daily_games in results.sort_values("date").groupby("date", sort=True):
        pending = []
        for game in daily_games.itertuples(index=False):
            if pd.isna(game.home_score) or pd.isna(game.away_score):
                continue
            home, away = p.clean_team(game.home_team), p.clean_team(game.away_team)
            home_rating, away_rating = (
                ratings.get(home, 1500.0),
                ratings.get(away, 1500.0),
            )
            home_gf6, home_ga6, home_form = rolling_team_values(form_history, home)
            away_gf6, away_ga6, away_form = rolling_team_values(form_history, away)
            row = {
                "date": pd.Timestamp(date),
                "home": home,
                "away": away,
                "tournament": game.tournament,
                "neutral": int(bool(getattr(game, "neutral", False))),
                "home_goals": int(game.home_score),
                "away_goals": int(game.away_score),
                "elo_diff": home_rating - away_rating,
                "home_gf6": home_gf6,
                "home_ga6": home_ga6,
                "away_gf6": away_gf6,
                "away_ga6": away_ga6,
                "form_points_diff": home_form - away_form,
                "form_goal_diff": (home_gf6 - home_ga6) - (away_gf6 - away_ga6),
                "fifa_points_diff": 0.0,
            }
            for lag, column in enumerate(ELO_HISTORY_COLUMNS):
                row[column] = monthly_rating_at(
                    rating_history, home, pd.Timestamp(date), lag
                ) - monthly_rating_at(rating_history, away, pd.Timestamp(date), lag)
            row["home_rating"] = home_rating
            row["away_rating"] = away_rating
            rows.append(row)
            pending.append((len(rows) - 1, game, home, away, home_rating, away_rating))
            rating_history.setdefault(home, []).append(
                (pd.Timestamp(date), home_rating)
            )
            rating_history.setdefault(away, []).append(
                (pd.Timestamp(date), away_rating)
            )
        for row_index, game, home, away, home_rating, away_rating in pending:
            neutral = int(bool(getattr(game, "neutral", False)))
            advantage = 0.0 if neutral else 100.0
            expected_home = 1.0 / (
                1.0 + 10.0 ** (-((home_rating + advantage) - away_rating) / 400.0)
            )
            observed_home = (
                1.0
                if game.home_score > game.away_score
                else 0.5
                if game.home_score == game.away_score
                else 0.0
            )
            change = (
                paper_match_importance(game.tournament)
                * paper_goal_multiplier(game.home_score, game.away_score)
                * (observed_home - expected_home)
            )
            ratings[home] = home_rating + change
            ratings[away] = away_rating - change
            rows[row_index]["post_home_rating"] = ratings[home]
            rows[row_index]["post_away_rating"] = ratings[away]
            home_points = (
                3.0
                if game.home_score > game.away_score
                else 1.0
                if game.home_score == game.away_score
                else 0.0
            )
            away_points = (
                3.0
                if game.away_score > game.home_score
                else 1.0
                if game.home_score == game.away_score
                else 0.0
            )
            form_history.setdefault(home, []).append(
                (float(game.home_score), float(game.away_score), home_points)
            )
            form_history.setdefault(away, []).append(
                (float(game.away_score), float(game.home_score), away_points)
            )
    frame = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def world_cup_qualified_teams(results: pd.DataFrame, year: int) -> set[str]:
    groups = g.historical_groups(year)
    return {team for teams in groups.values() for team in teams}


def paper_training_frame(
    results: pd.DataFrame, goal_data: pd.DataFrame, year: int
) -> pd.DataFrame:
    cutoff = world_cup_start(results, year)
    teams = world_cup_qualified_teams(results, year)
    frame = paper_elo_match_data(results)
    mask = (
        (frame.date >= pd.Timestamp("2010-01-01"))
        & (frame.date < cutoff)
        & (frame.home.isin(teams) | frame.away.isin(teams))
    )
    return frame.loc[mask].copy()


def paper_frozen_world_cup_frame(
    results: pd.DataFrame,
    goal_data: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    cutoff = world_cup_start(results, year)
    paper_data = paper_elo_match_data(results)
    prior = paper_data[paper_data.date < cutoff]
    form_history: dict[str, list[tuple[float, float, float]]] = {}
    for row in prior.itertuples(index=False):
        home_points = (
            3.0
            if row.home_goals > row.away_goals
            else 1.0
            if row.home_goals == row.away_goals
            else 0.0
        )
        away_points = (
            3.0
            if row.away_goals > row.home_goals
            else 1.0
            if row.home_goals == row.away_goals
            else 0.0
        )
        form_history.setdefault(row.home, []).append(
            (float(row.home_goals), float(row.away_goals), home_points)
        )
        form_history.setdefault(row.away, []).append(
            (float(row.away_goals), float(row.home_goals), away_points)
        )

    rating_events: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for row in prior.itertuples(index=False):
        rating_events.setdefault(row.home, []).append(
            (pd.Timestamp(row.date), float(row.post_home_rating))
        )
        rating_events.setdefault(row.away, []).append(
            (pd.Timestamp(row.date), float(row.post_away_rating))
        )

    def snapshot_rating(team: str, date: pd.Timestamp) -> float:
        events = rating_events.get(team)
        if not events:
            return 1500.0
        dates = [item[0] for item in events]
        index = np.searchsorted(dates, date, side="right") - 1
        return 1500.0 if index < 0 else float(events[index][1])

    base = prematch_world_cup_frame(goal_data, year)
    rows = []
    for row in base.itertuples(index=False):
        home_gf6, home_ga6, home_form = rolling_team_values(form_history, row.home)
        away_gf6, away_ga6, away_form = rolling_team_values(form_history, row.away)
        values = row._asdict()
        values.update(
            {
                "home_gf6": home_gf6,
                "home_ga6": home_ga6,
                "away_gf6": away_gf6,
                "away_ga6": away_ga6,
                "form_points_diff": home_form - away_form,
                "form_goal_diff": (home_gf6 - home_ga6) - (away_gf6 - away_ga6),
                "fifa_points_diff": 0.0,
            }
        )
        for lag, column in enumerate(ELO_HISTORY_COLUMNS):
            lookup_date = (
                cutoff
                if lag == 0
                else cutoff.replace(day=1).normalize() - pd.DateOffset(months=lag)
            )
            values[column] = snapshot_rating(row.home, lookup_date) - snapshot_rating(
                row.away, lookup_date
            )
        rows.append(values)
    return pd.DataFrame(rows)


@dataclass
class PaperSDRPoisson:
    method: str
    mean: np.ndarray
    whitening: np.ndarray
    directions: np.ndarray
    home: PoissonRegressor
    away: PoissonRegressor

    def sdr_scores(self, frame: pd.DataFrame) -> np.ndarray:
        z = (
            frame[ELO_HISTORY_COLUMNS].to_numpy(dtype=float) - self.mean
        ) @ self.whitening
        return z @ self.directions

    def home_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        scores = self.sdr_scores(frame)
        data = {f"sdr_{i + 1}": scores[:, i] for i in range(scores.shape[1])}
        data.update(
            {
                "neutral": frame.neutral.to_numpy(dtype=float),
                "home_gf6": frame.home_gf6.to_numpy(dtype=float),
                "away_ga6": frame.away_ga6.to_numpy(dtype=float),
            }
        )
        return pd.DataFrame(data)

    def away_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        scores = -self.sdr_scores(frame)
        data = {f"sdr_{i + 1}": scores[:, i] for i in range(scores.shape[1])}
        data.update(
            {
                "neutral": frame.neutral.to_numpy(dtype=float),
                "away_gf6": frame.away_gf6.to_numpy(dtype=float),
                "home_ga6": frame.home_ga6.to_numpy(dtype=float),
            }
        )
        return pd.DataFrame(data)


def fit_paper_sdr_poisson(
    train: pd.DataFrame, method: str, dimensions: int = 2
) -> PaperSDRPoisson:
    y = np.where(
        train.home_goals > train.away_goals,
        0,
        np.where(train.home_goals == train.away_goals, 1, 2),
    )
    mean, whitening, directions = sdr_directions(
        train[ELO_HISTORY_COLUMNS].to_numpy(dtype=float),
        y,
        method,
        dimensions=dimensions,
    )
    model = PaperSDRPoisson(
        method=method,
        mean=mean,
        whitening=whitening,
        directions=directions,
        home=PoissonRegressor(alpha=0.0, max_iter=1000),
        away=PoissonRegressor(alpha=0.0, max_iter=1000),
    )
    model.home.fit(model.home_features(train), train.home_goals.clip(upper=8))
    model.away.fit(model.away_features(train), train.away_goals.clip(upper=8))
    return model


def paper_sdr_probabilities(model: PaperSDRPoisson, frame: pd.DataFrame) -> np.ndarray:
    home_mean = model.home.predict(model.home_features(frame)).clip(0.05, 6.0)
    away_mean = model.away.predict(model.away_features(frame)).clip(0.05, 6.0)
    rows = []
    goals = np.arange(9)
    home_grid, away_grid = np.meshgrid(goals, goals, indexing="ij")
    for h_mean, a_mean in zip(home_mean, away_mean):
        matrix = np.outer(
            g.poisson_probabilities(float(h_mean))[:9],
            g.poisson_probabilities(float(a_mean))[:9],
        )
        matrix = matrix / matrix.sum()
        rows.append(
            [
                float(matrix[home_grid > away_grid].sum()),
                float(matrix[home_grid == away_grid].sum()),
                float(matrix[home_grid < away_grid].sum()),
            ]
        )
    return _safe_probability(np.array(rows))


def goal_probabilities(
    model: g.GoalModel, frame: pd.DataFrame, rho: float = 0.0
) -> np.ndarray:
    home_mean = model.home.predict(g.goal_features(frame))
    away_mean = model.away.predict(g.goal_features(frame, invert_elo=True))
    probabilities = []
    goals = np.arange(g.MAX_GOALS + 1)
    home_grid, away_grid = np.meshgrid(goals, goals, indexing="ij")
    for h_mean, a_mean in zip(home_mean, away_mean):
        matrix = np.outer(
            g.poisson_probabilities(float(h_mean)),
            g.poisson_probabilities(float(a_mean)),
        )
        if rho:
            lam = float(h_mean)
            mu = float(a_mean)
            matrix[0, 0] *= max(0.01, 1.0 - lam * mu * rho)
            matrix[0, 1] *= max(0.01, 1.0 + lam * rho)
            matrix[1, 0] *= max(0.01, 1.0 + mu * rho)
            matrix[1, 1] *= max(0.01, 1.0 - rho)
            matrix /= matrix.sum()
        probabilities.append(
            [
                float(matrix[home_grid > away_grid].sum()),
                float(matrix[home_grid == away_grid].sum()),
                float(matrix[home_grid < away_grid].sum()),
            ]
        )
    return _safe_probability(np.array(probabilities))


def fit_tuned_hist_gradient_goal_model(
    data: pd.DataFrame, cutoff: pd.Timestamp
) -> g.GoalModel:
    """Scoreline model configured from 2010 and 2014 validation."""
    train = data[data.date < cutoff]
    home_x = g.goal_features(train)
    away_x = g.goal_features(train, invert_elo=True)
    params = {**TUNED_HIST_GRADIENT_PARAMS, "loss": "poisson", "random_state": 2026}
    home = HistGradientBoostingRegressor(**params).fit(
        home_x, train.home_goals.clip(upper=10)
    )
    away = HistGradientBoostingRegressor(**params).fit(
        away_x, train.away_goals.clip(upper=10)
    )
    return g.GoalModel(home=home, away=away)


def match_stats_goal_features(
    frame: pd.DataFrame,
    *,
    invert_elo: bool = False,
) -> pd.DataFrame:
    return pd.concat(
        [
            g.goal_features(frame, invert_elo=invert_elo),
            match_stat_data.goal_features(frame, invert=invert_elo),
        ],
        axis=1,
    )


def match_stats_goal_data(goal_data: pd.DataFrame) -> pd.DataFrame:
    global _MATCH_STATS_GOAL_DATA_CACHE
    if _MATCH_STATS_GOAL_DATA_CACHE is None or (
        _MATCH_STATS_GOAL_DATA_CACHE[0] is not goal_data
    ):
        enriched = match_stat_data.rolling_match_stats(
            goal_data,
            match_stat_data.load_match_stats(),
        )
        _MATCH_STATS_GOAL_DATA_CACHE = (goal_data, enriched)
    return _MATCH_STATS_GOAL_DATA_CACHE[1]


def fit_match_stats_goal_model(
    goal_data: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> g.GoalModel:
    enriched = match_stats_goal_data(goal_data)
    train = enriched[enriched.date < cutoff]
    params = {
        **MARKET_SQUAD_HIST_GRADIENT_PARAMS,
        "loss": "poisson",
        "random_state": 2026,
    }
    home_features = match_stats_goal_features(train)
    for col in home_features.columns:
        if home_features[col].isna().all():
            home_features[col] = 0.0
    home = HistGradientBoostingRegressor(**params).fit(
        home_features,
        train.home_goals.clip(upper=10),
    )
    away_features = match_stats_goal_features(train, invert_elo=True)
    for col in away_features.columns:
        if away_features[col].isna().all():
            away_features[col] = 0.0
    away = HistGradientBoostingRegressor(**params).fit(
        away_features,
        train.away_goals.clip(upper=10),
    )
    return g.GoalModel(home=home, away=away)


def match_stats_probabilities(
    model: g.GoalModel,
    frame: pd.DataFrame,
    rho: float = 0.0,
) -> np.ndarray:
    enriched = match_stat_data.rolling_match_stats(
        frame,
        match_stat_data.load_match_stats(),
    )
    home_mean = model.home.predict(match_stats_goal_features(enriched))
    away_mean = model.away.predict(match_stats_goal_features(enriched, invert_elo=True))
    rows = []
    goals = np.arange(g.MAX_GOALS + 1)
    home_grid, away_grid = np.meshgrid(goals, goals, indexing="ij")
    for home_value, away_value in zip(home_mean, away_mean):
        lam = float(np.clip(home_value, 0.05, 6.0))
        mu = float(np.clip(away_value, 0.05, 6.0))
        matrix = np.outer(
            g.poisson_probabilities(lam),
            g.poisson_probabilities(mu),
        )
        if rho:
            matrix[0, 0] *= max(0.01, 1.0 - lam * mu * rho)
            matrix[0, 1] *= max(0.01, 1.0 + lam * rho)
            matrix[1, 0] *= max(0.01, 1.0 + mu * rho)
            matrix[1, 1] *= max(0.01, 1.0 - rho)
            matrix /= matrix.sum()
        rows.append(
            [
                float(matrix[home_grid > away_grid].sum()),
                float(matrix[home_grid == away_grid].sum()),
                float(matrix[home_grid < away_grid].sum()),
            ]
        )
    return _safe_probability(np.asarray(rows))


def tune_match_stats_rho(
    goal_data: pd.DataFrame,
    results: pd.DataFrame,
    validation_years: tuple[int, ...] = DEVELOPMENT_YEARS,
    rho_grid: tuple[float, ...] = MATCH_STATS_DC_RHO_GRID,
) -> float:
    """Select Dixon-Coles low-score correction for match-stats models.

    Only the supplied validation years are scored; callers should pass development
    years when tuning for retrospective evaluation.
    """
    records = []
    for rho in rho_grid:
        probabilities = []
        truth = []
        for year in validation_years:
            cutoff = world_cup_start(results, year)
            model = fit_match_stats_goal_model(goal_data, cutoff)
            frame = prematch_world_cup_frame(goal_data, year)
            probabilities.append(match_stats_probabilities(model, frame, rho=rho))
            truth.append(frame.truth.to_numpy(dtype=int))
        summary = metric_summary(np.vstack(probabilities), np.concatenate(truth))
        records.append((summary["accuracy"], -summary["rps"], -abs(rho), rho))
    return float(max(records)[-1])


def development_match_stats_rho(goal_data: pd.DataFrame, results: pd.DataFrame) -> float:
    global _MATCH_STATS_RHO_CACHE
    if _MATCH_STATS_RHO_CACHE is None or _MATCH_STATS_RHO_CACHE[0] is not goal_data:
        _MATCH_STATS_RHO_CACHE = (goal_data, tune_match_stats_rho(goal_data, results))
    return _MATCH_STATS_RHO_CACHE[1]


def fit_squad_hist_gradient_goal_model(
    data: pd.DataFrame,
    results: pd.DataFrame,
    cutoff: pd.Timestamp,
    include_market_value: bool = False,
    include_player_performance: bool = False,
) -> g.GoalModel:
    """Fit the squad histogram-gradient scoreline model."""
    train = data[data.date < cutoff]
    if include_player_performance:
        selected_params = PERFORMANCE_MARKET_SQUAD_HIST_GRADIENT_PARAMS
    elif include_market_value:
        selected_params = MARKET_SQUAD_HIST_GRADIENT_PARAMS
    else:
        selected_params = TUNED_HIST_GRADIENT_PARAMS
    params = {**selected_params, "loss": "poisson", "random_state": 2026}
    home_features = squad_goal_features(
        train,
        results,
        include_market_value=include_market_value,
        include_player_performance=include_player_performance,
    )
    for col in home_features.columns:
        if home_features[col].isna().all():
            home_features[col] = 0.0
    home = HistGradientBoostingRegressor(**params).fit(
        home_features,
        train.home_goals.clip(upper=10),
    )
    away_features = squad_goal_features(
        train,
        results,
        invert_elo=True,
        include_market_value=include_market_value,
        include_player_performance=include_player_performance,
    )
    for col in away_features.columns:
        if away_features[col].isna().all():
            away_features[col] = 0.0
    away = HistGradientBoostingRegressor(**params).fit(
        away_features,
        train.away_goals.clip(upper=10),
    )
    return g.GoalModel(home=home, away=away)


def fit_xgboost_poisson_market_model(
    data: pd.DataFrame,
    results: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> g.GoalModel:
    """Fit CUDA-capable Poisson scoreline heads."""
    from xgboost import XGBRegressor

    train = data[data.date < cutoff]
    params = {
        **XGBOOST_POISSON_MARKET_PARAMS,
        "objective": "count:poisson",
        "eval_metric": "poisson-nloglik",
        "max_delta_step": 0.7,
        "tree_method": "hist",
        "device": "cuda" if cuda_enabled() else "cpu",
        "random_state": 2026,
        "n_jobs": 4,
    }
    home = XGBRegressor(**params).fit(
        squad_goal_features(train, results, include_market_value=True),
        train.home_goals.clip(upper=10),
    )
    away = XGBRegressor(**params).fit(
        squad_goal_features(
            train,
            results,
            invert_elo=True,
            include_market_value=True,
        ),
        train.away_goals.clip(upper=10),
    )
    return g.GoalModel(home=home, away=away)


def squad_goal_probabilities(
    model: g.GoalModel,
    frame: pd.DataFrame,
    results: pd.DataFrame,
    rho: float = SQUAD_DC_RHO,
    include_market_value: bool = False,
    include_player_performance: bool = False,
) -> np.ndarray:
    home_mean = model.home.predict(
        squad_goal_features(
            frame,
            results,
            include_market_value=include_market_value,
            include_player_performance=include_player_performance,
        )
    ).clip(0.05, 6.0)
    away_mean = model.away.predict(
        squad_goal_features(
            frame,
            results,
            invert_elo=True,
            include_market_value=include_market_value,
            include_player_performance=include_player_performance,
        )
    ).clip(0.05, 6.0)
    probabilities = []
    goals = np.arange(g.MAX_GOALS + 1)
    home_grid, away_grid = np.meshgrid(goals, goals, indexing="ij")
    for h_mean, a_mean in zip(home_mean, away_mean):
        matrix = np.outer(
            g.poisson_probabilities(float(h_mean)),
            g.poisson_probabilities(float(a_mean)),
        )
        if rho:
            lam = float(h_mean)
            mu = float(a_mean)
            matrix[0, 0] *= max(0.01, 1.0 - lam * mu * rho)
            matrix[0, 1] *= max(0.01, 1.0 + lam * rho)
            matrix[1, 0] *= max(0.01, 1.0 + mu * rho)
            matrix[1, 1] *= max(0.01, 1.0 - rho)
            matrix /= matrix.sum()
        probabilities.append(
            [
                float(matrix[home_grid > away_grid].sum()),
                float(matrix[home_grid == away_grid].sum()),
                float(matrix[home_grid < away_grid].sum()),
            ]
        )
    return _safe_probability(np.array(probabilities))


def squad_pair_goal_features(
    frame: pd.DataFrame, squad: pd.DataFrame, invert_elo: bool = False
) -> pd.DataFrame:
    base = g.goal_features(frame, invert_elo=invert_elo)
    sign = -1.0 if invert_elo else 1.0
    rows = []
    for row in frame.itertuples(index=False):
        values = []
        for column in SQUAD_GOAL_FEATURES:
            value = np.nan
            if column in squad.columns:
                home_value = squad[column].get(p.clean_team(row.home), np.nan)
                away_value = squad[column].get(p.clean_team(row.away), np.nan)
                if pd.notna(home_value) and pd.notna(away_value):
                    value = float(home_value) - float(away_value)
            values.append(sign * value if pd.notna(value) else np.nan)
        rows.append(values)
    extra = pd.DataFrame(
        rows,
        columns=[f"{column}_diff" for column in SQUAD_GOAL_FEATURES],
        index=base.index,
    )
    return pd.concat([base, extra], axis=1)


def squad_score_cache(
    model: g.GoalModel,
    teams: list[str],
    ratings: dict[str, float],
    fifa: dict[str, tuple[float, float]],
    forms: dict[str, tuple[float, float, float, float, float, float]],
    external_elo: dict[str, float],
    squad: pd.DataFrame,
    market: pd.DataFrame | None = None,
    performance: pd.DataFrame | None = None,
    rho: float = SQUAD_DC_RHO,
    host_teams: set[str] | None = None,
) -> dict[tuple[str, str], dict[str, np.ndarray | tuple[float, float, float]]]:
    pairs = [(home, away) for home in teams for away in teams if home != away]
    home_rank = np.array([fifa.get(home, (np.nan, np.nan))[0] for home, _ in pairs])
    away_rank = np.array([fifa.get(away, (np.nan, np.nan))[0] for _, away in pairs])
    home_points = np.array([fifa.get(home, (np.nan, np.nan))[1] for home, _ in pairs])
    away_points = np.array([fifa.get(away, (np.nan, np.nan))[1] for _, away in pairs])
    home_external = np.array([external_elo.get(home, np.nan) for home, _ in pairs])
    away_external = np.array([external_elo.get(away, np.nan) for _, away in pairs])
    host_teams = host_teams or set()
    venue = np.array(
        [int(home in host_teams) - int(away in host_teams) for home, away in pairs]
    )
    frame = pd.DataFrame(
        {
            "home": [home for home, _ in pairs],
            "away": [away for _, away in pairs],
            "elo_diff": [
                ratings.get(home, 1500.0) - ratings.get(away, 1500.0)
                for home, away in pairs
            ],
            "neutral": (venue == 0).astype(int),
            "venue_advantage": venue,
            "fifa_rank_diff": np.nan_to_num(away_rank - home_rank, nan=0.0),
            "fifa_points_diff": np.nan_to_num(home_points - away_points, nan=0.0),
            "fifa_rank_missing": (np.isnan(home_rank) | np.isnan(away_rank)).astype(
                int
            ),
            "external_elo_diff": np.nan_to_num(home_external - away_external, nan=0.0),
            "external_elo_missing": (
                np.isnan(home_external) | np.isnan(away_external)
            ).astype(int),
            "form_points_diff": [
                forms.get(home, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[0]
                - forms.get(away, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[0]
                for home, away in pairs
            ],
            "form_goal_diff": [
                forms.get(home, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[1]
                - forms.get(away, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[1]
                for home, away in pairs
            ],
            "form_opponent_elo": [
                forms.get(home, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[2]
                - forms.get(away, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[2]
                for home, away in pairs
            ],
            "form_matches_diff": [
                forms.get(home, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[3]
                - forms.get(away, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[3]
                for home, away in pairs
            ],
            "competitive_points_diff": [
                forms.get(home, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[4]
                - forms.get(away, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[4]
                for home, away in pairs
            ],
            "competitive_goal_diff": [
                forms.get(home, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[5]
                - forms.get(away, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[5]
                for home, away in pairs
            ],
            "h2h_win_diff": 0.0,
            "h2h_goal_diff": 0.0,
            "h2h_matches": 0.0,
        }
    )
    if market is not None and performance is not None:
        home_features = performance_market_squad_pair_goal_features(
            frame,
            squad,
            market,
            performance,
        )
        away_features = performance_market_squad_pair_goal_features(
            frame,
            squad,
            market,
            performance,
            invert_elo=True,
        )
    elif market is not None:
        home_features = market_squad_pair_goal_features(frame, squad, market)
        away_features = market_squad_pair_goal_features(
            frame, squad, market, invert_elo=True
        )
    else:
        home_features = squad_pair_goal_features(frame, squad)
        away_features = squad_pair_goal_features(frame, squad, invert_elo=True)
    home_mean = model.home.predict(home_features).clip(0.05, 6.0)
    away_mean = model.away.predict(away_features).clip(0.05, 6.0)
    output = {}
    goals = np.arange(g.MAX_GOALS + 1)
    home_grid, away_grid = np.meshgrid(goals, goals, indexing="ij")
    home_win = home_grid > away_grid
    draw = home_grid == away_grid
    for pair, h_mean, a_mean in zip(pairs, home_mean, away_mean):
        matrix = np.outer(
            g.poisson_probabilities(float(h_mean)),
            g.poisson_probabilities(float(a_mean)),
        )
        if rho:
            lam = float(h_mean)
            mu = float(a_mean)
            matrix[0, 0] *= max(0.01, 1.0 - lam * mu * rho)
            matrix[0, 1] *= max(0.01, 1.0 + lam * rho)
            matrix[1, 0] *= max(0.01, 1.0 + mu * rho)
            matrix[1, 1] *= max(0.01, 1.0 - rho)
            matrix /= matrix.sum()
        output[pair] = {
            "means": (float(h_mean), float(a_mean)),
            "scores": matrix.ravel(),
            "home_scores": home_grid.ravel(),
            "away_scores": away_grid.ravel(),
            "wdl": (
                float(matrix[home_win].sum()),
                float(matrix[draw].sum()),
                float(matrix[~home_win & ~draw].sum()),
            ),
        }
    return output


class TeamPoisson:
    def __init__(self, home: Pipeline, away: Pipeline) -> None:
        self.home = home
        self.away = away

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        features = frame[
            [
                "home",
                "away",
                "neutral",
                "elo_diff",
                "external_elo_diff",
                "fifa_rank_diff",
            ]
        ].copy()
        home_mean = self.home.predict(features).clip(0.05, 6.0)
        away_mean = self.away.predict(features).clip(0.05, 6.0)
        rows = []
        goals = np.arange(g.MAX_GOALS + 1)
        home_grid, away_grid = np.meshgrid(goals, goals, indexing="ij")
        for h_mean, a_mean in zip(home_mean, away_mean):
            matrix = np.outer(
                g.poisson_probabilities(float(h_mean)),
                g.poisson_probabilities(float(a_mean)),
            )
            rows.append(
                [
                    float(matrix[home_grid > away_grid].sum()),
                    float(matrix[home_grid == away_grid].sum()),
                    float(matrix[home_grid < away_grid].sum()),
                ]
            )
        return _safe_probability(np.array(rows))


def fit_team_poisson(goal_data: pd.DataFrame, cutoff: pd.Timestamp) -> TeamPoisson:
    train = goal_data[goal_data.date < cutoff].copy()
    features = train[
        ["home", "away", "neutral", "elo_diff", "external_elo_diff", "fifa_rank_diff"]
    ].copy()
    preprocess = ColumnTransformer(
        [
            (
                "teams",
                OneHotEncoder(handle_unknown="ignore", min_frequency=8),
                ["home", "away"],
            ),
            (
                "num",
                StandardScaler(),
                ["neutral", "elo_diff", "external_elo_diff", "fifa_rank_diff"],
            ),
        ]
    )
    home = Pipeline(
        [("prep", preprocess), ("model", PoissonRegressor(alpha=1.2, max_iter=700))]
    )
    away = Pipeline(
        [("prep", preprocess), ("model", PoissonRegressor(alpha=1.2, max_iter=700))]
    )
    return TeamPoisson(
        home.fit(features, train.home_goals.clip(upper=10)),
        away.fit(features, train.away_goals.clip(upper=10)),
    )


def train_classifier(kind: str, train: pd.DataFrame, y=None) -> ClassifierMixin:
    if kind != "squad_hist_gb_classifier":
        x = add_candidate_features(train)
        y = np.where(
            train.home_goals > train.away_goals,
            0,
            np.where(train.home_goals == train.away_goals, 1, 2),
        )
    else:
        x = train
    if kind == "elo_logistic":
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.08,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=2026,
                    ),
                ),
            ]
        )
        return model.fit(train[["elo_diff", "neutral", "venue_advantage"]], y)
    if kind == "full_logistic":
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.045,
                        max_iter=2500,
                        class_weight="balanced",
                        random_state=2026,
                    ),
                ),
            ]
        )
        return model.fit(x, y)
    if kind == "hist_gb_classifier":
        model = HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=170,
            max_leaf_nodes=12,
            l2_regularization=3.0,
            random_state=2026,
        )
        return model.fit(x, y)
    if kind == "squad_hist_gb_classifier":
        model = HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=300,
            max_leaf_nodes=12,
            l2_regularization=4.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=2026,
        )
        return model.fit(x, y)
    if kind == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=160,
            max_depth=2,
            learning_rate=0.035,
            subsample=0.82,
            colsample_bytree=0.82,
            reg_lambda=5.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=2026,
            n_jobs=1,
            tree_method="hist",
            device="cuda" if cuda_enabled() else "cpu",
        )
        return model.fit(x, y)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.03,
            num_leaves=9,
            reg_lambda=4.0,
            min_child_samples=40,
            objective="multiclass",
            random_state=2026,
            verbose=-1,
            device_type="gpu" if cuda_enabled() else "cpu",
        )
        return model.fit(x, y)
    if kind == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=180,
            depth=3,
            learning_rate=0.035,
            loss_function="MultiClass",
            random_seed=2026,
            verbose=False,
            allow_writing_files=False,
            task_type="GPU" if cuda_enabled() else "CPU",
            devices="0" if cuda_enabled() else None,
        )
        return model.fit(x, y)
    raise ValueError(kind)


def classifier_predict(
    model: ClassifierMixin, kind: str, frame: pd.DataFrame
) -> np.ndarray:
    x = (
        frame[["elo_diff", "neutral", "venue_advantage"]]
        if kind == "elo_logistic"
        else add_candidate_features(frame)
    )
    raw = model.predict_proba(x)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = model.named_steps["clf"].classes_
    out = np.zeros((len(frame), 3))
    for index, label in enumerate(classes):
        out[:, int(label)] = raw[:, index]
    return _safe_probability(out)


@dataclass
class PredictionBlock:
    year: int
    model: str
    probability: np.ndarray
    truth: np.ndarray
    frame: pd.DataFrame


def predict_model(
    kind: str, goal_data: pd.DataFrame, results: pd.DataFrame, year: int
) -> PredictionBlock:
    cutoff = world_cup_start(results, year)
    if kind in PAPER_SDR_MODELS:
        frame = paper_frozen_world_cup_frame(results, goal_data, year)
    elif kind.startswith("sdr_"):
        frame = frozen_world_cup_frame_with_history(results, goal_data, year)
    else:
        frame = prematch_world_cup_frame(goal_data, year)
    if kind == "market_consensus_hybrid":
        structural = predict_model(
            "market_squad_hist_gradient_dc_prior",
            goal_data,
            results,
            year,
        )
        probability, _ = market_data.blend_with_consensus(
            structural.probability,
            structural.frame,
            year,
        )
        return PredictionBlock(
            year,
            kind,
            probability,
            structural.truth,
            structural.frame,
        )
    if kind == "hist_gradient_tuned_prior":
        model = fit_tuned_hist_gradient_goal_model(goal_data, cutoff)
        probability = goal_probabilities(model, frame)
    elif kind == "squad_hist_gradient_dc_prior":
        model = fit_squad_hist_gradient_goal_model(goal_data, results, cutoff)
        probability = squad_goal_probabilities(model, frame, results)
    elif kind == "market_squad_hist_gradient_dc_prior":
        model = fit_squad_hist_gradient_goal_model(
            goal_data, results, cutoff, include_market_value=True
        )
        probability = squad_goal_probabilities(
            model,
            frame,
            results,
            rho=MARKET_SQUAD_DC_RHO,
            include_market_value=True,
        )
    elif kind == "xgboost_poisson_market_prior":
        model = fit_xgboost_poisson_market_model(goal_data, results, cutoff)
        probability = squad_goal_probabilities(
            model,
            frame,
            results,
            rho=0.0,
            include_market_value=True,
        )
    elif kind == "performance_market_squad_hist_gradient_dc_prior":
        model = fit_squad_hist_gradient_goal_model(
            goal_data,
            results,
            cutoff,
            include_market_value=True,
            include_player_performance=True,
        )
        probability = squad_goal_probabilities(
            model,
            frame,
            results,
            rho=PERFORMANCE_MARKET_SQUAD_DC_RHO,
            include_market_value=True,
            include_player_performance=True,
        )
    elif kind in {"poisson", "hist_gradient", "hist_gradient_raw"}:
        model_kind = "hist_gradient" if kind == "hist_gradient_raw" else kind
        model = g.fit_goal_model(goal_data, cutoff, model_kind)
        probability = goal_probabilities(model, frame)
    elif kind == "dixon_coles":
        model = g.fit_goal_model(goal_data, cutoff, "poisson")
        probability = goal_probabilities(model, frame, rho=-0.08)
    elif kind == "team_effect_poisson":
        model = fit_team_poisson(goal_data, cutoff)
        probability = model.predict_proba(frame)
    elif kind == "match_stats_hist_gradient":
        model = fit_match_stats_goal_model(goal_data, cutoff)
        probability = match_stats_probabilities(model, frame)
    elif kind == "match_stats_hist_gradient_dc":
        model = fit_match_stats_goal_model(goal_data, cutoff)
        probability = match_stats_probabilities(
            model,
            frame,
            rho=development_match_stats_rho(goal_data, results),
        )
    elif kind in {"sdr_sir_poisson", "sdr_save_poisson"}:
        goal_history = elo_history_goal_data(results, goal_data)
        method = "sir" if kind == "sdr_sir_poisson" else "save"
        model = fit_sdr_poisson(goal_history, cutoff, method)
        probability = sdr_probabilities(model, frame)
    elif kind in PAPER_SDR_MODELS:
        method = "sir" if kind == "paper_sdr_sir_poisson" else "save"
        train = paper_training_frame(results, goal_data, year)
        model = fit_paper_sdr_poisson(train, method)
        probability = paper_sdr_probabilities(model, frame)
    elif kind == "ensemble_super_blend":
        b1 = predict_model("squad_hist_gb_classifier", goal_data, results, year)
        b2 = predict_model("squad_hist_gradient_dc_prior", goal_data, results, year)
        b3 = predict_model("market_squad_hist_gradient_dc_prior", goal_data, results, year)
        
        prob = (b1.probability * 0.60) + (b2.probability * 0.20) + (b3.probability * 0.20)
        
        return PredictionBlock(
            year, kind, prob, b1.truth, b1.frame
        )
    elif kind == "squad_hist_gb_classifier":
        train = goal_data[goal_data.date < cutoff]
        
        train_features = squad_goal_features(
            train, results, include_market_value=True, include_player_performance=True
        )
        train_match_stats = match_stats_goal_data(train)
        train_features = pd.concat([train_features, match_stats_goal_features(train_match_stats).drop(columns=train_features.columns, errors='ignore')], axis=1)
        for col in train_features.columns:
            if train_features[col].isna().all(): train_features[col] = 0.0
            
        y = np.where(
            train.home_goals > train.away_goals,
            0,
            np.where(train.home_goals == train.away_goals, 1, 2),
        )
        model = train_classifier(kind, train_features, y=y)
        
        test_features = squad_goal_features(
            frame, results, include_market_value=True, include_player_performance=True
        )
        test_match_stats = match_stats_goal_data(frame)
        test_features = pd.concat([test_features, match_stats_goal_features(test_match_stats).drop(columns=test_features.columns, errors='ignore')], axis=1)
        
        for col in test_features.columns:
            if test_features[col].isna().all(): test_features[col] = 0.0
            
        probability = model.predict_proba(test_features)
    else:
        train = goal_data[goal_data.date < cutoff]
        model = train_classifier(kind, train)
        probability = classifier_predict(model, kind, frame)
    return PredictionBlock(
        year, kind, probability, frame.truth.to_numpy(dtype=int), frame
    )


def calibration_predictions(
    kind: str,
    goal_data: pd.DataFrame,
    results: pd.DataFrame,
    test_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum_year = 2014 if kind in PAPER_SDR_MODELS else 2002
    available = [
        year
        for year in g.STANDARD_32_YEARS
        if year < test_year and year >= minimum_year
    ]
    years = available[-CALIBRATION_LOOKBACK:]
    probabilities = []
    truth = []
    groups = []
    for year in years:
        block = predict_model(kind, goal_data, results, year)
        probabilities.append(block.probability)
        truth.append(block.truth)
        groups.append(np.full(len(block.truth), year))
    return np.vstack(probabilities), np.concatenate(truth), np.concatenate(groups)


def best_calibration(
    cal_probability: np.ndarray,
    cal_truth: np.ndarray,
    cal_groups: np.ndarray,
    test_probability: np.ndarray,
) -> tuple[str, np.ndarray, dict[str, float]]:
    methods = ("none", "temperature", "platt", "beta", "isotonic")

    def calibrate(
        method: str,
        fit_probability: np.ndarray,
        fit_truth: np.ndarray,
        target_probability: np.ndarray,
    ) -> np.ndarray:
        if method == "none":
            return _safe_probability(target_probability)
        if method == "temperature":
            temperature = tune_temperature(fit_probability, fit_truth)
            return apply_temperature(target_probability, temperature)
        if method == "platt":
            model = fit_platt_calibrator(fit_probability, fit_truth)
            return apply_platt_calibrator(model, target_probability)
        if method == "beta":
            models = fit_beta_calibrator(fit_probability, fit_truth)
            return apply_beta_calibrator(models, target_probability)
        if method == "isotonic":
            models = fit_beta_calibrator(fit_probability, fit_truth)
            return apply_beta_calibrator(models, target_probability)
        raise ValueError(method)

    groups = np.unique(cal_groups)
    details: dict[str, float] = {}
    best_name = "none"
    best_score = float("inf")
    for method in methods:
        fold_probability = []
        fold_truth = []
        if len(groups) < 2 and method != "none":
            continue
        try:
            for group in groups:
                validation = cal_groups == group
                training = ~validation
                if method == "none":
                    calibrated = _safe_probability(cal_probability[validation])
                else:
                    calibrated = calibrate(
                        method,
                        cal_probability[training],
                        cal_truth[training],
                        cal_probability[validation],
                    )
                fold_probability.append(calibrated)
                fold_truth.append(cal_truth[validation])
        except ValueError:
            continue
        score = metric_summary(
            np.vstack(fold_probability),
            np.concatenate(fold_truth),
        )["rps"]
        details[f"{method}_nested_rps"] = score
        if score < best_score:
            best_name = method
            best_score = score
    calibrated_test = calibrate(
        best_name,
        cal_probability,
        cal_truth,
        test_probability,
    )
    return best_name, calibrated_test, details


def predict_competition_model(
    kind: str,
    goal_data: pd.DataFrame,
    results: pd.DataFrame,
    tournament: str,
    year: int,
) -> PredictionBlock:
    cutoff = competition_start(results, tournament, year)
    frame = prematch_competition_frame(goal_data, tournament, year)
    if kind == "poisson":
        model = g.fit_goal_model(goal_data, cutoff, "poisson")
        probability = goal_probabilities(model, frame)
    elif kind == "hist_gradient_tuned_prior":
        model = fit_tuned_hist_gradient_goal_model(goal_data, cutoff)
        probability = goal_probabilities(model, frame)
    elif kind == "match_stats_hist_gradient_dc":
        model = fit_match_stats_goal_model(goal_data, cutoff)
        probability = match_stats_probabilities(
            model,
            frame,
            rho=development_match_stats_rho(goal_data, results),
        )
    elif kind in {"elo_logistic", "full_logistic", "hist_gb_classifier"}:
        train = goal_data[goal_data.date < cutoff]
        model = train_classifier(kind, train)
        probability = classifier_predict(model, kind, frame)
    else:
        raise ValueError(kind)
    return PredictionBlock(year, kind, probability, frame.truth.to_numpy(dtype=int), frame)


def run_multi_competition_benchmark(
    *,
    minimum_year: int = 2006,
    max_competitions: int | None = None,
) -> pd.DataFrame:
    results = p.load_results(False)
    goal_data = g.sequential_goal_data(results)
    specs = competition_specs(results, minimum_year=minimum_year)
    if max_competitions is not None:
        specs = specs[:max_competitions]
    rows = []
    for tournament, year, matches in specs:
        for model_name in MULTI_COMPETITION_MODELS:
            block = predict_competition_model(model_name, goal_data, results, tournament, year)
            summary = metric_summary(block.probability, block.truth)
            rows.append(
                {
                    "competition": tournament,
                    "year": year,
                    "matches": matches,
                    "model": model_name,
                    **summary,
                }
            )
    metrics = pd.DataFrame(rows)
    pooled_rows = []
    for model_name in MULTI_COMPETITION_MODELS:
        model_blocks = []
        truth_blocks = []
        for tournament, year, _ in specs:
            block = predict_competition_model(model_name, goal_data, results, tournament, year)
            model_blocks.append(block.probability)
            truth_blocks.append(block.truth)
        if model_blocks:
            pooled_rows.append(
                {
                    "competition": "pooled",
                    "year": f">={minimum_year}",
                    "matches": int(sum(len(truth) for truth in truth_blocks)),
                    "model": model_name,
                    **metric_summary(np.vstack(model_blocks), np.concatenate(truth_blocks)),
                }
            )
    output = pd.concat([metrics, pd.DataFrame(pooled_rows)], ignore_index=True)
    print(output[output.competition.eq("pooled")].to_json(orient="records", indent=2))
    return output


def run_benchmark(simulations_2026: int = 100_000, *, write_artifacts: bool = True) -> None:
    results = p.load_results(False)
    goal_data = g.sequential_goal_data(results)
    candidate_models = [
        "elo_logistic",
        "full_logistic",
        "poisson",
        "dixon_coles",
        "hist_gradient",
        "hist_gradient_raw",
        "hist_gradient_tuned_prior",
        "squad_hist_gradient_dc_prior",
        "market_squad_hist_gradient_dc_prior",
        "market_consensus_hybrid",
        "match_stats_hist_gradient",
        "match_stats_hist_gradient_dc",
        "hist_gb_classifier",
        "squad_hist_gb_classifier",
        "xgboost",
        "lightgbm",
        "catboost",
        "team_effect_poisson",
        "sdr_sir_poisson",
        "sdr_save_poisson",
        "ensemble_super_blend",
    ]
    prediction_rows = []
    calibrated_by_year: dict[int, dict[str, np.ndarray]] = {}
    truth_by_year: dict[int, np.ndarray] = {}
    raw_blocks: dict[tuple[int, str], PredictionBlock] = {}

    for test_year in EVALUATION_YEARS:
        calibrated_by_year[test_year] = {}
        for model_name in candidate_models:
            block = predict_model(model_name, goal_data, results, test_year)
            raw_blocks[(test_year, model_name)] = block
            if model_name in FORCED_RAW_MODELS:
                calibrated = block.probability
            else:
                cal_probability, cal_truth, cal_groups = calibration_predictions(
                    model_name, goal_data, results, test_year
                )
                _, calibrated, _ = best_calibration(
                    cal_probability,
                    cal_truth,
                    cal_groups,
                    block.probability,
                )
            calibrated_by_year[test_year][model_name] = calibrated
            truth_by_year[test_year] = block.truth
            for i, row in block.frame.reset_index(drop=True).iterrows():
                prediction_rows.append(
                    {
                        "year": test_year,
                        "date": row.date,
                        "home": row.home,
                        "away": row.away,
                        "truth": int(row.truth),
                        "model": model_name,
                        "p_home": float(calibrated[i, 0]),
                        "p_draw": float(calibrated[i, 1]),
                        "p_away": float(calibrated[i, 2]),
                    }
                )

    top1_selection_records = []
    for model_name in candidate_models:
        probability = np.vstack(
            [calibrated_by_year[year][model_name] for year in DEVELOPMENT_YEARS]
        )
        truth = np.concatenate([truth_by_year[year] for year in DEVELOPMENT_YEARS])
        top1_selection_records.append({"model": model_name, **metric_summary(probability, truth)})
    top1_policy_model = str(
        pd.DataFrame(top1_selection_records)
        .sort_values(["accuracy", "rps", "log_loss"], ascending=[False, True, True])
        .iloc[0]
        .model
    )
    for test_year in EVALUATION_YEARS:
        top1_probability = calibrated_by_year[test_year][top1_policy_model]
        deployed_block = raw_blocks[(test_year, top1_policy_model)]
        for i, row in deployed_block.frame.reset_index(drop=True).iterrows():
            prediction_rows.append(
                {
                    "year": test_year,
                    "date": row.date,
                    "home": row.home,
                    "away": row.away,
                    "truth": int(row.truth),
                    "model": "top1_accuracy_blend",
                    "p_home": float(top1_probability[i, 0]),
                    "p_draw": float(top1_probability[i, 1]),
                    "p_away": float(top1_probability[i, 2]),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    combined_records = []
    evaluated_models = candidate_models + ["top1_accuracy_blend"]
    for split, years in (
        ("retrospective_2018_2022", RETROSPECTIVE_YEARS),
        ("development_2006_2014", DEVELOPMENT_YEARS),
    ):
        for model_name in evaluated_models:
            subset = predictions[
                predictions.model.eq(model_name) & predictions.year.isin(years)
            ].sort_values(["year", "date", "home", "away"])
            probability = subset[["p_home", "p_draw", "p_away"]].to_numpy()
            truth = subset.truth.to_numpy(dtype=int)
            combined_records.append(
                {
                    "year": split,
                    "model": model_name,
                    "variant": "temporally_selected"
                    if split.startswith("retrospective")
                    else "model_selection",
                    **metric_summary(probability, truth),
                }
            )
    combined = pd.DataFrame(combined_records)

    development = combined[
        combined.year.eq("development_2006_2014")
        & combined.model.isin(candidate_models)
    ]
    best_model = str(development.sort_values(["rps", "log_loss"]).iloc[0].model)
    retrospective = combined[combined.year.eq("retrospective_2018_2022")]
    observed_best_model = str(
        retrospective.sort_values(["rps", "log_loss"]).iloc[0].model
    )
    final = predictions[
        predictions.model.eq(best_model) & predictions.year.isin(RETROSPECTIVE_YEARS)
    ].sort_values(["year", "date", "home", "away"])
    reference = predictions[
        predictions.model.eq("poisson") & predictions.year.isin(RETROSPECTIVE_YEARS)
    ].sort_values(["year", "date", "home", "away"])
    final_prob = final[["p_home", "p_draw", "p_away"]].to_numpy()
    ref_prob = reference[["p_home", "p_draw", "p_away"]].to_numpy()
    truth = final.truth.to_numpy(dtype=int)
    test_clusters = final.year.to_numpy()
    delta_rps = rps_vector(ref_prob, truth) - rps_vector(final_prob, truth)
    delta_log = -np.log(ref_prob[np.arange(len(truth)), truth]) + np.log(
        final_prob[np.arange(len(truth)), truth]
    )
    ci_low, ci_high = clustered_bootstrap_interval(delta_rps, test_clusters)
    cluster_p = clustered_sign_flip_pvalue(delta_rps, test_clusters)
    top1_final = predictions[
        predictions.model.eq("top1_accuracy_blend")
        & predictions.year.isin(RETROSPECTIVE_YEARS)
    ].sort_values(["year", "date", "home", "away"])
    top1_probability = top1_final[["p_home", "p_draw", "p_away"]].to_numpy()
    delta_accuracy = (top1_probability.argmax(axis=1) == truth).astype(float) - (
        final_prob.argmax(axis=1) == truth
    ).astype(float)
    accuracy_ci_low, accuracy_ci_high = clustered_bootstrap_interval(
        delta_accuracy,
        test_clusters,
        seed=2029,
    )
    accuracy_cluster_p = clustered_sign_flip_pvalue(
        delta_accuracy,
        test_clusters,
    )
    previous_best_model = (
        "market_squad_hist_gradient_dc_prior"
        if best_model == "market_consensus_hybrid"
        else "hist_gradient_tuned_prior"
    )
    previous_best = predictions[
        predictions.model.eq(previous_best_model)
        & predictions.year.isin(RETROSPECTIVE_YEARS)
    ].sort_values(["year", "date", "home", "away"])
    prev_prob = previous_best[["p_home", "p_draw", "p_away"]].to_numpy()
    delta_prev_rps = rps_vector(prev_prob, truth) - rps_vector(final_prob, truth)
    delta_prev_log = -np.log(prev_prob[np.arange(len(truth)), truth]) + np.log(
        final_prob[np.arange(len(truth)), truth]
    )
    prev_ci_low, prev_ci_high = clustered_bootstrap_interval(
        delta_prev_rps,
        test_clusters,
        seed=2027,
    )
    prev_cluster_p = clustered_sign_flip_pvalue(delta_prev_rps, test_clusters)

    walk_probabilities: dict[str, list[np.ndarray]] = {
        best_model: [],
        "poisson": [],
        "top1_accuracy_blend": [],
    }
    walk_truth: list[np.ndarray] = []
    walk_clusters: list[np.ndarray] = []
    walk_year_summaries = []
    for year in WALK_FORWARD_YEARS:
        deployed_probability = calibrated_by_year[year][best_model]
        poisson_probability = calibrated_by_year[year]["poisson"]
        year_truth = truth_by_year[year]
        walk_probabilities[best_model].append(deployed_probability)
        walk_probabilities["poisson"].append(poisson_probability)
        walk_probabilities["top1_accuracy_blend"].append(
            calibrated_by_year[year][top1_policy_model]
        )
        for model_name, probability in (
            (best_model, deployed_probability),
            ("poisson", poisson_probability),
            (
                "top1_accuracy_blend",
                walk_probabilities["top1_accuracy_blend"][-1],
            ),
        ):
            walk_year_summaries.append(
                {
                    "year": year,
                    "model": model_name,
                    "variant": "walk_forward_diagnostic",
                    **metric_summary(probability, year_truth),
                }
            )
        walk_truth.append(year_truth)
        walk_clusters.append(np.full(len(year_truth), year))

    pooled_walk_truth = np.concatenate(walk_truth)
    pooled_walk_clusters = np.concatenate(walk_clusters)
    walk_summaries = []
    for model_name, probability_blocks in walk_probabilities.items():
        probability = np.vstack(probability_blocks)
        walk_summaries.append(
            {
                "year": "walk_forward_2006_2022",
                "model": model_name,
                "variant": "diagnostic",
                **metric_summary(probability, pooled_walk_truth),
            }
        )
    combined = pd.concat(
        [
            combined,
            pd.DataFrame(walk_year_summaries),
            pd.DataFrame(walk_summaries),
        ],
        ignore_index=True,
        sort=False,
    )
    walk_final_probability = np.vstack(walk_probabilities[best_model])
    walk_reference_probability = np.vstack(walk_probabilities["poisson"])
    walk_delta_rps = rps_vector(
        walk_reference_probability,
        pooled_walk_truth,
    ) - rps_vector(walk_final_probability, pooled_walk_truth)
    walk_ci_low, walk_ci_high = clustered_bootstrap_interval(
        walk_delta_rps,
        pooled_walk_clusters,
        seed=2028,
    )
    walk_cluster_p = clustered_sign_flip_pvalue(
        walk_delta_rps,
        pooled_walk_clusters,
    )
    walk_top1_probability = np.vstack(walk_probabilities["top1_accuracy_blend"])
    walk_delta_accuracy = (
        walk_top1_probability.argmax(axis=1) == pooled_walk_truth
    ).astype(float) - (
        walk_final_probability.argmax(axis=1) == pooled_walk_truth
    ).astype(float)
    walk_accuracy_ci_low, walk_accuracy_ci_high = clustered_bootstrap_interval(
        walk_delta_accuracy,
        pooled_walk_clusters,
        seed=2030,
    )
    walk_accuracy_cluster_p = clustered_sign_flip_pvalue(
        walk_delta_accuracy,
        pooled_walk_clusters,
    )

    forecast_candidates = {"hist_gradient_raw", "hist_gradient_tuned_prior"}
    forecast_model = str(
        development[development.model.isin(forecast_candidates)]
        .sort_values(["rps", "log_loss"])
        .iloc[0]
        .model
    )
    cutoff_2026 = pd.Timestamp("2026-06-10")
    if forecast_model == "hist_gradient_tuned_prior":
        score_model = fit_tuned_hist_gradient_goal_model(goal_data, cutoff_2026)
    else:
        score_model = g.fit_goal_model(goal_data, cutoff_2026, "hist_gradient")
    knockout_model = g.fit_goal_model(goal_data, cutoff_2026, "poisson")
    fixtures = load_2026_fixtures()
    teams_2026 = sorted(
        set(fixtures.loc[fixtures.stage == "Group Stage", "team1"]).union(
            fixtures.loc[fixtures.stage == "Group Stage", "team2"]
        )
    )
    ratings_2026 = p.elo_before(results, cutoff_2026)
    with zipfile.ZipFile(p.RAW_DIR / "world_cup_1930_2026.zip") as zipped:
        teams_frame = pd.read_csv(io.BytesIO(zipped.read("wc_2026_teams.csv")))
    fifa_2026 = {
        p.clean_team(team): (float(rank), np.nan)
        for team, rank in zip(teams_frame.team, teams_frame.fifa_rank)
    }
    forms_2026 = g.form_snapshot(results, cutoff_2026, ratings_2026, teams_2026)
    external_2026 = g.external_elo_snapshot(cutoff_2026)
    hosts_2026 = {"Canada", "Mexico", "United States"}


    train = goal_data[goal_data.date < cutoff_2026].copy()
    train_features = squad_goal_features(
        train, results, include_market_value=True, include_player_performance=True
    )
    train_match_stats = match_stats_goal_data(train)
    train_features = pd.concat(
        [train_features, match_stats_goal_features(train_match_stats).drop(columns=train_features.columns, errors='ignore')], 
        axis=1
    )
    for col in train_features.columns:
        if train_features[col].isna().all(): train_features[col] = 0.0
        
    y_train = np.where(
        train.home_goals > train.away_goals,
        0,
        np.where(train.home_goals == train.away_goals, 1, 2),
    )
    clf = train_classifier("squad_hist_gb_classifier", train_features, y=y_train)

    pairs_2026 = [(home, away) for home in teams_2026 for away in teams_2026 if home != away]
    pair_rows = []
    home_rank = np.array([fifa_2026.get(home, (np.nan, np.nan))[0] for home, _ in pairs_2026])
    away_rank = np.array([fifa_2026.get(away, (np.nan, np.nan))[0] for _, away in pairs_2026])
    home_points = np.array([fifa_2026.get(home, (np.nan, np.nan))[1] for home, _ in pairs_2026])
    away_points = np.array([fifa_2026.get(away, (np.nan, np.nan))[1] for _, away in pairs_2026])
    home_external = np.array([external_2026.get(home, np.nan) for home, _ in pairs_2026])
    away_external = np.array([external_2026.get(away, np.nan) for _, away in pairs_2026])

    for i, (home, away) in enumerate(pairs_2026):
        pair_rows.append({
            "date": cutoff_2026,
            "home": home,
            "away": away,
            "neutral": 0 if home in hosts_2026 or away in hosts_2026 else 1,
            "venue_advantage": 1 if home in hosts_2026 else (-1 if away in hosts_2026 else 0),
            "tournament": "FIFA World Cup",
            "home_goals": 0,
            "away_goals": 0,
            "elo_diff": ratings_2026.get(home, 1500.0) - ratings_2026.get(away, 1500.0),
            "fifa_rank_diff": np.nan_to_num(away_rank[i] - home_rank[i], nan=0.0),
            "fifa_points_diff": np.nan_to_num(home_points[i] - away_points[i], nan=0.0),
            "fifa_rank_missing": int(np.isnan(home_rank[i]) or np.isnan(away_rank[i])),
            "external_elo_diff": np.nan_to_num(home_external[i] - away_external[i], nan=0.0),
            "external_elo_missing": int(np.isnan(home_external[i]) or np.isnan(away_external[i])),
            "form_points_diff": forms_2026.get(home, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[0] - forms_2026.get(away, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[0],
            "form_goal_diff": forms_2026.get(home, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[1] - forms_2026.get(away, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[1],
            "form_opponent_elo": forms_2026.get(home, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[2] - forms_2026.get(away, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[2],
            "form_matches_diff": forms_2026.get(home, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[3] - forms_2026.get(away, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[3],
            "competitive_points_diff": forms_2026.get(home, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[4] - forms_2026.get(away, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[4],
            "competitive_goal_diff": forms_2026.get(home, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[5] - forms_2026.get(away, (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0))[5],
        })
    pairs_df = pd.DataFrame(pair_rows)
    
    test_features = squad_goal_features(
        pairs_df, results, include_market_value=True, include_player_performance=True
    )
    test_match_stats = match_stats_goal_data(pairs_df)
    test_features = pd.concat(
        [test_features, match_stats_goal_features(test_match_stats).drop(columns=test_features.columns, errors='ignore')], 
        axis=1
    )
    for col in test_features.columns:
        if test_features[col].isna().all(): test_features[col] = 0.0
        

    test_features = test_features[train_features.columns]
        
    probs = clf.predict_proba(test_features)
    pair_probs = {pair: tuple(prob) for pair, prob in zip(pairs_2026, probs)}

    
    group_cache = g.score_cache(

        score_model,
        teams_2026,
        ratings_2026,
        fifa_2026,
        forms_2026,
        external_2026,
        hosts_2026,
    )
    knockout_cache = g.score_cache(
        knockout_model,
        teams_2026,
        ratings_2026,
        fifa_2026,
        forms_2026,
        external_2026,
        hosts_2026,
        injected_1x2=pair_probs,
    )
    tie_ranks = {team: fifa_2026.get(team, (999.0, np.nan))[0] for team in teams_2026}
    forecast_2026 = g.simulate_2026_scoreline(
        fixtures,
        group_cache,
        tie_ranks,
        simulations=simulations_2026,
        seed=20260629,
        fair_play=g.load_fair_play_scores(),
        knockout_cache=knockout_cache,
    )

    if write_artifacts:
        p.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(p.ARTIFACT_DIR / "research_model_metrics_combined.csv", index=False)
        forecast_2026.to_csv(
            p.ARTIFACT_DIR / "research_forecast_2026_100k.csv", index=False
        )

    report = {
        "validation_protocol": (
            "Models are selected on 2006-2014 only and evaluated retrospectively "
            "on 2018-2022 using pre-match features and 90-minute 1X2 targets."
        ),
        "estimand": "pre-match probability of the 90-minute home/draw/away result",
        "development_world_cups": list(DEVELOPMENT_YEARS),
        "retrospective_world_cups": list(RETROSPECTIVE_YEARS),
        "models": candidate_models + ["top1_accuracy_blend"],
        "best_combined_model": str(best_model),
        "observed_best_combined_model": observed_best_model,
        "selection_metric": "pooled development RPS",
        "best_combined_metrics": combined[combined.model == best_model]
        .iloc[0]
        .to_dict(),
        "top1_decision": {
            "model": "top1_accuracy_blend",
            "policy": "select_candidate_by_development_accuracy",
            "selected_candidate": top1_policy_model,
            "selection_world_cups": list(DEVELOPMENT_YEARS),
            "metrics": combined[combined.model == "top1_accuracy_blend"]
            .iloc[0]
            .to_dict(),
            "mean_accuracy_improvement": float(delta_accuracy.mean()),
            "clustered_bootstrap_95_ci_accuracy_improvement": [
                float(accuracy_ci_low),
                float(accuracy_ci_high),
            ],
            "tournament_sign_flip_one_sided_p_accuracy": accuracy_cluster_p,
        },
        "poisson_reference_metrics": combined[combined.model == "poisson"]
        .iloc[0]
        .to_dict(),
        "paired_significance_vs_poisson": {
            "mean_rps_improvement": float(delta_rps.mean()),
            "mean_log_loss_improvement": float(delta_log.mean()),
            "clustered_bootstrap_95_ci_rps_improvement": [
                float(ci_low),
                float(ci_high),
            ],
            "tournament_sign_flip_one_sided_p_rps": cluster_p,
        },
        "paired_significance_vs_previous_best": {
            "previous_best_model": previous_best_model,
            "mean_rps_improvement": float(delta_prev_rps.mean()),
            "mean_log_loss_improvement": float(delta_prev_log.mean()),
            "clustered_bootstrap_95_ci_rps_improvement": [
                float(prev_ci_low),
                float(prev_ci_high),
            ],
            "tournament_sign_flip_one_sided_p_rps": prev_cluster_p,
        },
        "walk_forward_diagnostic": {
            "world_cups": list(WALK_FORWARD_YEARS),
            "matches": int(len(pooled_walk_truth)),
            "metrics": {
                row["model"]: {
                    key: value
                    for key, value in row.items()
                    if key not in {"year", "model", "variant"}
                }
                for row in walk_summaries
            },
            "mean_rps_improvement_vs_poisson": float(walk_delta_rps.mean()),
            "clustered_bootstrap_95_ci_rps_improvement": [
                float(walk_ci_low),
                float(walk_ci_high),
            ],
            "tournament_sign_flip_one_sided_p_rps": walk_cluster_p,
            "mean_top1_accuracy_improvement": float(walk_delta_accuracy.mean()),
            "clustered_bootstrap_95_ci_top1_accuracy_improvement": [
                float(walk_accuracy_ci_low),
                float(walk_accuracy_ci_high),
            ],
            "tournament_sign_flip_one_sided_p_top1_accuracy": (walk_accuracy_cluster_p),
        },
        "sdr_literature_benchmark": {
            "source": "Rezaei and Samadi 2026 arXiv:2606.24171 Table 4",
            "best_sdr_combined_rps": 0.127,
            "best_sdr_combined_accuracy": 0.688,
            "best_non_sdr_ensemble_rps": 0.209,
            "best_non_sdr_ensemble_accuracy": 0.547,
            "temporal_audit_status": "failed_independent_reproduction",
            "audit_evidence": {
                "strict_pre_month_save_rps": 0.210475,
                "month_end_save_rps": 0.128166,
                "interpretation": "The reported result is reproduced only when the current calendar month's post-match Elo snapshot is exposed to the target match.",
            },
            "eligible_for_comparison": False,
        },
        "simulation_2026": {
            "simulations": simulations_2026,
            "cutoff": "2026-06-10",
            "forecast_model": forecast_model,
            "estimand": "pre-tournament title probability",
            "uses_closing_odds": False,
            "uses_post_cutoff_squads": False,
            "conditioned_completed_group_games": int(
                fixtures.loc[fixtures.stage == "Group Stage", "completed"]
                .astype(str)
                .str.lower()
                .eq("true")
                .sum()
            ),
            "top_10": forecast_2026.head(10).to_dict(orient="records"),
        },
    }
    sota_rows = []
    for row in retrospective[retrospective.model.isin(candidate_models)].itertuples(
        index=False
    ):
        sota_rows.append(
            {
                "benchmark": row.model,
                "source": "local temporal rerun",
                "target": "90-minute 1X2",
                "information_set": "pre-match",
                "comparison_valid": True,
                "rps": row.rps,
                "log_loss": row.log_loss,
                "brier": row.brier,
                "ece": row.ece,
                "accuracy": row.accuracy,
                "auc_ovr": row.auc_ovr,
                "selected_on_development": row.model == best_model,
                "note": "Same matches, target reconstruction, and scoring code.",
            }
        )
    for name, published_rps, published_accuracy in [
        ("Rezaei-Samadi M1 Elo-logistic", 0.219, 0.508),
        ("Rezaei-Samadi M2 Full logistic", 0.217, 0.523),
        ("Rezaei-Samadi M3 Poisson current Elo", 0.212, 0.516),
        ("Rezaei-Samadi M4 ARIMA-Poisson", 0.213, 0.516),
        ("Rezaei-Samadi M5 NNAR-Poisson", 0.214, 0.516),
        ("Rezaei-Samadi M6 XGBoost", 0.215, 0.531),
        ("Rezaei-Samadi M7 Ensemble", 0.209, 0.547),
        ("Rezaei-Samadi M8 SIR LDA d=1", 0.129, 0.688),
        ("Rezaei-Samadi M9 SIR LDA d=2", 0.127, 0.688),
        ("Rezaei-Samadi M10 SAVE d=1", 0.129, 0.695),
        ("Rezaei-Samadi M11 SAVE d=2", 0.127, 0.680),
    ]:
        sota_rows.append(
            {
                "benchmark": name,
                "source": "Rezaei and Samadi (2026), Table 4",
                "target": "reported 1X2 target",
                "information_set": "not independently matched",
                "comparison_valid": False,
                "rps": published_rps,
                "log_loss": np.nan,
                "brier": np.nan,
                "ece": np.nan,
                "accuracy": published_accuracy,
                "auc_ovr": np.nan,
                "selected_on_development": False,
                "note": (
                    "Descriptive published point estimate only; SDR results fail "
                    "the local temporal audit."
                ),
            }
        )
    external_538_path = p.ARTIFACT_DIR / "research_external_538_2018_2022_metrics.csv"
    if external_538_path.exists():
        external_538 = pd.read_csv(external_538_path)
        combined_538 = external_538[external_538.year.astype(str).eq("2018_2022")]
        if not combined_538.empty:
            row = combined_538.iloc[0]
            sota_rows.append(
                {
                    "benchmark": "FiveThirtyEight SPI recovered Wayback 2018+2022",
                    "source": "recovered public predictions",
                    "target": "legacy final-score evaluation",
                    "information_set": "pre-match",
                    "comparison_valid": False,
                    "rps": float(row.rps),
                    "log_loss": float(row.log_loss),
                    "brier": float(row.brier),
                    "ece": float(row.ece),
                    "accuracy": float(row.accuracy),
                    "auc_ovr": float(row.auc_ovr),
                    "selected_on_development": False,
                    "note": "Must be rescored against the reconstructed 90-minute target.",
                }
            )
    if write_artifacts:
        pd.DataFrame(sota_rows).to_csv(
            p.ARTIFACT_DIR / "research_sota_comparison_audited.csv", index=False
        )
    print(
        json.dumps(
            {
                "best_model": best_model,
                "best_rps": report["best_combined_metrics"]["rps"],
                "best_accuracy": report["best_combined_metrics"]["accuracy"],
                "rps_improvement_vs_poisson": report["paired_significance_vs_poisson"][
                    "mean_rps_improvement"
                ],
                "retrospective_clustered_rps_ci": report[
                    "paired_significance_vs_poisson"
                ]["clustered_bootstrap_95_ci_rps_improvement"],
                "retrospective_tournament_p": report["paired_significance_vs_poisson"][
                    "tournament_sign_flip_one_sided_p_rps"
                ],
                "walk_forward": report["walk_forward_diagnostic"],
                "top_2026": report["simulation_2026"]["top_10"][:3],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run_benchmark()
