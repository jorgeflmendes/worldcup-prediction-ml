"""Scoreline modelling and tournament simulation."""

from __future__ import annotations

import io
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import html
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor

from . import pipeline as p
from .tournament_rules import (
    FINAL_SPEC,
    QF_SPECS,
    R16_SPECS,
    SF_SPECS,
    THIRD_PLACE_SPEC,
    group_table,
    load_annex_c,
    rank_group,
    rank_third_placed,
    resolve_r32,
)

MAX_GOALS = 10
STANDARD_32_YEARS = (1998, 2002, 2006, 2010, 2014, 2018, 2022)
FIFA_RANKINGS_DATASET = "gabipana7/fifa-rankings-and-international-matches-1992-2022"
EXTERNAL_ELO_DATASET = "saifalnimri/international-football-elo-ratings"


@dataclass
class GoalModel:
    home: object
    away: object


class BlendedPredictor:
    """Convex blend of two goal-mean predictors, with a sklearn-like API."""

    def __init__(self, left: object, right: object, right_weight: float) -> None:
        self.left = left
        self.right = right
        self.right_weight = right_weight

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (1.0 - self.right_weight) * self.left.predict(
            features
        ) + self.right_weight * self.right.predict(features)


def goal_features(frame: pd.DataFrame, invert_elo: bool = False) -> pd.DataFrame:
    sign = -1.0 if invert_elo else 1.0
    return pd.DataFrame(
        {
            "elo_diff": sign * frame.elo_diff.to_numpy() / 400.0,
            "neutral": frame.neutral.to_numpy(),
            "venue_advantage": sign * frame.venue_advantage.to_numpy(),
            "fifa_rank_diff": sign * frame.fifa_rank_diff.to_numpy() / 100.0,
            "fifa_points_diff": sign * frame.fifa_points_diff.to_numpy() / 1000.0,
            "fifa_rank_missing": frame.fifa_rank_missing.to_numpy(),
            "external_elo_diff": sign * frame.external_elo_diff.to_numpy() / 400.0,
            "external_elo_missing": frame.external_elo_missing.to_numpy(),
            "form_points_diff": sign * frame.form_points_diff.to_numpy(),
            "form_goal_diff": sign * frame.form_goal_diff.to_numpy(),
            "form_opponent_elo": sign * frame.form_opponent_elo.to_numpy() / 400.0,
            "form_matches_diff": sign * frame.form_matches_diff.to_numpy() / 10.0,
        }
    )


def load_fifa_rankings() -> pd.DataFrame:
    archive = p.cached_download(
        p.kaggle_download_url(FIFA_RANKINGS_DATASET),
        p.RAW_DIR / "fifa_rankings_1992_2022.zip",
        False,
    )
    with zipfile.ZipFile(archive) as zipped:
        frame = pd.read_csv(
            io.BytesIO(zipped.read("fifa_rankings.csv")), parse_dates=["rank_date"]
        )
    frame["team"] = frame.team.map(p.clean_team)
    return frame.sort_values(["team", "rank_date"])


def load_external_elo() -> pd.DataFrame:
    archive = p.cached_download(
        p.kaggle_download_url(EXTERNAL_ELO_DATASET),
        p.RAW_DIR / "international_elo_1872_2025.zip",
        False,
    )
    with zipfile.ZipFile(archive) as zipped:
        frame = pd.read_csv(io.BytesIO(zipped.read("eloratings.csv")))
    frame["date"] = pd.to_datetime(frame.date, format="mixed", errors="coerce")
    frame = frame.dropna(subset=["date"])
    frame["team"] = (
        frame.team.astype(str).str.replace("\u00a0", " ", regex=False).map(p.clean_team)
    )
    return frame.sort_values(["team", "date"])


def attach_fifa_rankings(data: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest available FIFA ratings."""
    lookup = {team: group for team, group in rankings.groupby("team", sort=False)}
    cache: dict[tuple[str, pd.Timestamp], tuple[float, float]] = {}

    def value(team: str, date: pd.Timestamp) -> tuple[float, float]:
        key = (team, date.normalize())
        if key in cache:
            return cache[key]
        group = lookup.get(team)
        if group is None:
            cache[key] = (np.nan, np.nan)
            return cache[key]
        index = group.rank_date.searchsorted(date, side="right") - 1
        cache[key] = (
            (np.nan, np.nan)
            if index < 0
            else (
                float(group.iloc[index]["rank"]),
                float(group.iloc[index].total_points),
            )
        )
        return cache[key]

    home = [value(team, date) for team, date in zip(data.home, data.date)]
    away = [value(team, date) for team, date in zip(data.away, data.date)]
    result = data.copy()
    home_rank, home_points = np.array(home, dtype=float).T
    away_rank, away_points = np.array(away, dtype=float).T
    result["fifa_rank_diff"] = np.nan_to_num(away_rank - home_rank, nan=0.0)
    result["fifa_points_diff"] = np.nan_to_num(home_points - away_points, nan=0.0)
    result["fifa_rank_missing"] = (np.isnan(home_rank) | np.isnan(away_rank)).astype(
        int
    )
    return result


def attach_external_elo(data: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    lookup = {team: group for team, group in ratings.groupby("team", sort=False)}

    def value(team: str, date: pd.Timestamp) -> float:
        group = lookup.get(team)
        if group is None:
            return np.nan
        index = group.date.searchsorted(date, side="left") - 1
        return np.nan if index < 0 else float(group.iloc[index].rating)

    home = np.array([value(team, date) for team, date in zip(data.home, data.date)])
    away = np.array([value(team, date) for team, date in zip(data.away, data.date)])
    result = data.copy()
    result["external_elo_diff"] = np.nan_to_num(home - away, nan=0.0)
    result["external_elo_missing"] = (np.isnan(home) | np.isnan(away)).astype(int)
    return result


def fifa_snapshot(cutoff: pd.Timestamp) -> dict[str, tuple[float, float]]:
    snapshot = {}
    for team, group in load_fifa_rankings().groupby("team", sort=False):
        index = group.rank_date.searchsorted(cutoff, side="right") - 1
        if index >= 0:
            row = group.iloc[index]
            snapshot[team] = (float(row["rank"]), float(row.total_points))
    return snapshot


def external_elo_snapshot(cutoff: pd.Timestamp) -> dict[str, float]:
    snapshot = {}
    for team, group in load_external_elo().groupby("team", sort=False):
        index = group.date.searchsorted(cutoff, side="left") - 1
        if index >= 0:
            snapshot[team] = float(group.iloc[index].rating)
    return snapshot


def form_snapshot(
    results: pd.DataFrame,
    cutoff: pd.Timestamp,
    ratings: dict[str, float],
    teams: list[str],
) -> dict[str, tuple[float, float, float, float, float, float]]:
    output = {}
    for team in teams:
        games = p.team_recent_matches(results, team, cutoff)
        if games.empty:
            output[team] = (0.5, 0.0, 1500.0, 0.0, 0.5, 0.0)
            continue
        weights = np.exp(-(cutoff - games.date).dt.days.to_numpy() / 365.0)
        points = np.where(
            games.goals_for > games.goals_against,
            1.0,
            np.where(games.goals_for == games.goals_against, 0.5, 0.0),
        )
        competitive = games.tournament.ne("Friendly").to_numpy()
        if competitive.any():
            competitive_points = float(
                np.average(points[competitive], weights=weights[competitive])
            )
            competitive_goals = float(
                np.average(
                    (games.goals_for - games.goals_against).to_numpy()[competitive],
                    weights=weights[competitive],
                )
            )
        else:
            competitive_points, competitive_goals = 0.5, 0.0
        output[team] = (
            float(np.average(points, weights=weights)),
            float(
                np.average(
                    (games.goals_for - games.goals_against).to_numpy(), weights=weights
                )
            ),
            float(
                np.average(
                    games.opponent.map(
                        lambda opponent: ratings.get(opponent, 1500.0)
                    ).to_numpy(),
                    weights=weights,
                )
            ),
            float(len(games)),
            competitive_points,
            competitive_goals,
        )
    return output


def sequential_goal_data(results: pd.DataFrame) -> pd.DataFrame:
    scored_results = p.regulation_time_results(results)
    ratings: defaultdict[str, float] = defaultdict(lambda: 1500.0)
    history: defaultdict[str, list[tuple[pd.Timestamp, float, float, float, bool]]] = (
        defaultdict(list)
    )
    rows = []

    def form(
        team: str, date: pd.Timestamp
    ) -> tuple[float, float, float, float, float, float]:
        prior = [record for record in history[team] if (date - record[0]).days <= 730]
        history[team] = prior
        if not prior:
            return 0.5, 0.0, 1500.0, 0.0, 0.5, 0.0
        weights = np.array(
            [np.exp(-(date - record[0]).days / 365.0) for record in prior]
        )
        points = np.array([record[1] for record in prior])
        goals = np.array([record[2] for record in prior])
        opponent = np.array([record[3] for record in prior])
        competitive = np.array([record[4] for record in prior])
        if competitive.any():
            comp_points = float(
                np.average(points[competitive], weights=weights[competitive])
            )
            comp_goals = float(
                np.average(goals[competitive], weights=weights[competitive])
            )
        else:
            comp_points, comp_goals = 0.5, 0.0
        return (
            float(np.average(points, weights=weights)),
            float(np.average(goals, weights=weights)),
            float(np.average(opponent, weights=weights)),
            float(len(prior)),
            comp_points,
            comp_goals,
        )

    for date, daily_games in scored_results.sort_values("date").groupby(
        "date",
        sort=True,
    ):
        pending = []
        for game in daily_games.itertuples(index=False):
            if pd.isna(game.home_score) or pd.isna(game.away_score):
                continue
            home, away = game.home_team, game.away_team
            neutral_value = bool(getattr(game, "neutral", True))
            country = str(getattr(game, "country", ""))
            neutral = int(neutral_value)
            venue_advantage = p.venue_advantage(
                home,
                away,
                country,
                neutral_value,
            )
            home_rating, away_rating = ratings[home], ratings[away]
            home_form, away_form = form(home, date), form(away, date)
            home_score = (
                game.home_score_90 if game.regulation_complete else game.home_score
            )
            away_score = (
                game.away_score_90 if game.regulation_complete else game.away_score
            )
            if game.regulation_complete:
                rows.append(
                    {
                        "date": date,
                        "home": home,
                        "away": away,
                        "tournament": game.tournament,
                        "elo_diff": home_rating - away_rating,
                        "neutral": neutral,
                        "venue_advantage": venue_advantage,
                        "home_goals": home_score,
                        "away_goals": away_score,
                        "form_points_diff": home_form[0] - away_form[0],
                        "form_goal_diff": home_form[1] - away_form[1],
                        "form_opponent_elo": home_form[2] - away_form[2],
                        "form_matches_diff": home_form[3] - away_form[3],
                        "competitive_points_diff": home_form[4] - away_form[4],
                        "competitive_goal_diff": home_form[5] - away_form[5],
                    }
                )
            pending.append(
                (
                    game,
                    home_rating,
                    away_rating,
                    neutral,
                    venue_advantage,
                    home_score,
                    away_score,
                )
            )
        for (
            game,
            home_rating,
            away_rating,
            neutral,
            venue_advantage,
            home_score,
            away_score,
        ) in pending:
            home, away = game.home_team, game.away_team
            advantage = 55.0 * venue_advantage
            expectation = 1 / (
                1 + 10 ** (-((home_rating + advantage) - away_rating) / 400)
            )
            home_points = (
                1.0
                if home_score > away_score
                else 0.5
                if home_score == away_score
                else 0.0
            )
            weight = 1.0 if game.tournament == "Friendly" else 1.35
            change = 22 * weight * (home_points - expectation)
            ratings[home] += change
            ratings[away] -= change
            competitive = game.tournament != "Friendly"
            history[home].append(
                (
                    date,
                    home_points,
                    home_score - away_score,
                    away_rating,
                    competitive,
                )
            )
            history[away].append(
                (
                    date,
                    1.0 - home_points,
                    away_score - home_score,
                    home_rating,
                    competitive,
                )
            )
    data = pd.DataFrame(rows)
    return attach_external_elo(
        attach_fifa_rankings(data, load_fifa_rankings()), load_external_elo()
    )


def fit_goal_model(
    data: pd.DataFrame, cutoff: pd.Timestamp, kind: str = "poisson"
) -> GoalModel:
    train = data[data.date < cutoff]
    if len(train) < 200:
        raise ValueError(f"Not enough matches before {cutoff.date()}.")
    home_x = goal_features(train)
    away_x = goal_features(train, invert_elo=True)
    target_home, target_away = (
        train.home_goals.clip(upper=10),
        train.away_goals.clip(upper=10),
    )
    if kind == "poisson":
        home = PoissonRegressor(alpha=0.4, max_iter=500)
        away = PoissonRegressor(alpha=0.4, max_iter=500)
    elif kind == "hist_gradient":
        for col in home_x.columns:
            if home_x[col].isna().all(): home_x[col] = 0.0
            if away_x[col].isna().all(): away_x[col] = 0.0
        home = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.045,
            max_leaf_nodes=12,
            l2_regularization=4.0,
            max_iter=250,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=2026,
        )
        away = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.045,
            max_leaf_nodes=12,
            l2_regularization=4.0,
            max_iter=250,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=2026,
        )
    elif kind.startswith("blend_"):
        for col in home_x.columns:
            if home_x[col].isna().all(): home_x[col] = 0.0
            if away_x[col].isna().all(): away_x[col] = 0.0
        try:
            hist_weight = float(kind.removeprefix("blend_"))
        except ValueError as error:
            raise ValueError(f"Invalid weight for {kind}.") from error
        if not 0.0 <= hist_weight <= 1.0:
            raise ValueError(f"Invalid weight for {kind}.")
        poisson_home = PoissonRegressor(alpha=0.4, max_iter=500).fit(
            home_x, target_home
        )
        poisson_away = PoissonRegressor(alpha=0.4, max_iter=500).fit(
            away_x, target_away
        )
        hist_home = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.045,
            max_leaf_nodes=12,
            l2_regularization=4.0,
            max_iter=180,
            random_state=2026,
        ).fit(home_x, target_home)
        hist_away = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.045,
            max_leaf_nodes=12,
            l2_regularization=4.0,
            max_iter=180,
            random_state=2026,
        ).fit(away_x, target_away)
        return GoalModel(
            home=BlendedPredictor(poisson_home, hist_home, hist_weight),
            away=BlendedPredictor(poisson_away, hist_away, hist_weight),
        )
    else:
        raise ValueError(f"Unknown goal model: {kind}")
    return GoalModel(
        home=home.fit(home_x, target_home), away=away.fit(away_x, target_away)
    )


def poisson_probabilities(mean: float) -> np.ndarray:
    probabilities = np.empty(MAX_GOALS + 1)
    probabilities[0] = np.exp(-mean)
    for index in range(1, MAX_GOALS + 1):
        probabilities[index] = probabilities[index - 1] * mean / index
    return probabilities / probabilities.sum()


def score_cache(
    model: GoalModel,
    teams: list[str],
    ratings: dict[str, float],
    fifa: dict[str, tuple[float, float]] | None = None,
    forms: dict[str, tuple[float, float, float, float]] | None = None,
    external_elo: dict[str, float] | None = None,
    host_teams: set[str] | None = None,
) -> dict[tuple[str, str], dict[str, np.ndarray | tuple[float, float, float]]]:
    pairs = [(home, away) for home in teams for away in teams if home != away]
    fifa = fifa or {}
    forms = forms or {}
    external_elo = external_elo or {}
    host_teams = host_teams or set()
    venue = np.array(
        [int(home in host_teams) - int(away in host_teams) for home, away in pairs]
    )
    home_rank = np.array([fifa.get(home, (np.nan, np.nan))[0] for home, _ in pairs])
    away_rank = np.array([fifa.get(away, (np.nan, np.nan))[0] for _, away in pairs])
    home_points = np.array([fifa.get(home, (np.nan, np.nan))[1] for home, _ in pairs])
    away_points = np.array([fifa.get(away, (np.nan, np.nan))[1] for _, away in pairs])
    home_external = np.array([external_elo.get(home, np.nan) for home, _ in pairs])
    away_external = np.array([external_elo.get(away, np.nan) for _, away in pairs])
    frame = pd.DataFrame(
        {
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
        }
    )
    home_mean = model.home.predict(goal_features(frame))
    away_mean = model.away.predict(goal_features(frame, invert_elo=True))
    output = {}
    goals = np.arange(MAX_GOALS + 1)
    home_grid, away_grid = np.meshgrid(goals, goals, indexing="ij")
    home_win = home_grid > away_grid
    draw = home_grid == away_grid
    for pair, h_mean, a_mean in zip(pairs, home_mean, away_mean):
        matrix = np.outer(
            poisson_probabilities(float(h_mean)), poisson_probabilities(float(a_mean))
        )
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


def sample_score(
    cache: dict[tuple[str, str], dict[str, np.ndarray | tuple[float, float, float]]],
    home: str,
    away: str,
    rng: np.random.Generator,
) -> tuple[int, int]:
    distribution = cache[(home, away)]
    index = int(rng.choice(len(distribution["scores"]), p=distribution["scores"]))
    return int(distribution["home_scores"][index]), int(
        distribution["away_scores"][index]
    )


def knockout_winner(
    cache: dict[tuple[str, str], dict[str, np.ndarray | tuple[float, float, float]]],
    home: str,
    away: str,
    rng: np.random.Generator,
) -> str:
    home_goals, away_goals = sample_score(cache, home, away, rng)
    if home_goals > away_goals:
        return home
    if away_goals > home_goals:
        return away
    home_mean, away_mean = cache[(home, away)]["means"]
    extra_home = rng.poisson(float(home_mean) / 3.0)
    extra_away = rng.poisson(float(away_mean) / 3.0)
    if extra_home > extra_away:
        return home
    if extra_away > extra_home:
        return away
    return home if rng.random() < 0.5 else away


def historical_groups(year: int) -> dict[str, list[str]]:
    """Read group allocation from the edition roster page."""
    page = p.RAW_DIR / "wikipedia" / f"{year}_squads.html"
    if not page.exists():
        p.wiki_squad_snapshot(year, False)
    tree = html.fromstring(page.read_bytes())
    groups: dict[str, list[str]] = defaultdict(list)
    current = None
    for heading in tree.xpath("//h2|//h3"):
        text = re.sub(r"\[edit\]$", "", heading.text_content().strip())
        group_match = re.fullmatch(r"Group\s+([A-H])", text, flags=re.I)
        if heading.tag == "h2" and group_match:
            current = group_match.group(1).upper()
        elif heading.tag == "h2":
            current = None
        elif heading.tag == "h3" and current:
            team = p.clean_team(text)
            if team not in groups[current]:
                groups[current].append(team)
    valid = {group: teams for group, teams in groups.items() if len(teams) == 4}
    if len(valid) != 8:
        raise RuntimeError(f"Could not obtain all eight groups for {year}: {valid}")
    return valid


def _completed_score(game: object) -> tuple[int, int] | None:
    home_score, away_score = (
        getattr(game, "home_score", np.nan),
        getattr(game, "away_score", np.nan),
    )
    if pd.notna(home_score) and pd.notna(away_score):
        return int(home_score), int(away_score)
    return None


def load_fair_play_scores(path: Path | None = None) -> dict[str, int]:
    """Load optional FIFA fair-play tiebreak scores (higher is better)."""
    path = path or (p.EXTERNAL_DIR / "wc2026_fair_play.csv")
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    required = {"team", "fair_play_score"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must include columns {sorted(required)}.")
    return {
        p.clean_team(team): int(score)
        for team, score in zip(frame.team, frame.fair_play_score)
    }


def _simulate_groups(
    fixtures: pd.DataFrame,
    cache: dict[tuple[str, str], dict[str, np.ndarray | tuple[float, float, float]]],
    fifa_rank: dict[str, float],
    rng: np.random.Generator,
    fair_play: dict[str, int] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Freeze completed group scorelines and sample only the remaining games."""
    standings, third_groups, _, _ = _simulate_groups_detailed(
        fixtures, cache, fifa_rank, rng, fair_play
    )
    return standings, third_groups


def _simulate_groups_detailed(
    fixtures: pd.DataFrame,
    cache: dict[tuple[str, str], dict[str, np.ndarray | tuple[float, float, float]]],
    fifa_rank: dict[str, float],
    rng: np.random.Generator,
    fair_play: dict[str, int] | None = None,
) -> tuple[
    dict[str, list[str]],
    list[str],
    dict[str, list[tuple[str, str, int, int, bool, str]]],
    dict[str, dict[str, dict[str, int]]],
]:
    """Simulate one complete group-stage state."""
    standings: dict[str, list[str]] = {}
    tables: dict[str, dict[str, dict[str, int]]] = {}
    third_placed: dict[str, str] = {}
    rendered_games: dict[str, list[tuple[str, str, int, int, bool, str]]] = {}
    group_games = fixtures[fixtures.stage == "Group Stage"]
    for group, games in group_games.groupby("group", sort=True):
        teams = sorted(set(games.team1).union(games.team2))
        results: list[tuple[str, str, int, int]] = []
        detail: list[tuple[str, str, int, int, bool, str]] = []
        for game in games.itertuples(index=False):
            result = _completed_score(game)
            home_goals, away_goals = (
                result
                if result is not None
                else sample_score(cache, game.team1, game.team2, rng)
            )
            results.append((game.team1, game.team2, home_goals, away_goals))
            detail.append(
                (
                    game.team1,
                    game.team2,
                    home_goals,
                    away_goals,
                    result is not None,
                    game.fixture,
                )
            )
        label = str(group)
        tables[label] = group_table(teams, results)
        standings[label] = rank_group(teams, results, fifa_rank, fair_play)
        third_placed[label] = standings[label][2]
        rendered_games[label] = detail
    return (
        standings,
        rank_third_placed(third_placed, tables, fifa_rank, fair_play)[:8],
        rendered_games,
        tables,
    )


def _simulate_official_knockout(
    standings: dict[str, list[str]],
    third_groups: list[str],
    cache: dict[tuple[str, str], dict[str, np.ndarray | tuple[float, float, float]]],
    rng: np.random.Generator,
    annex_c: dict[frozenset[str], dict[str, str]],
) -> dict[str, tuple[str, str, str]]:
    """Simulate the Article 12.6-12.11 match tree and retain every matchup."""
    matches: dict[str, tuple[str, str, str]] = {}
    for match_id, home, away in resolve_r32(standings, third_groups, annex_c):
        matches[match_id] = (home, away, knockout_winner(cache, home, away, rng))

    for specs in (R16_SPECS, QF_SPECS, SF_SPECS):
        for match_id, source_a, source_b in specs:
            home, away = matches[source_a][2], matches[source_b][2]
            matches[match_id] = (home, away, knockout_winner(cache, home, away, rng))

    final_id, source_a, source_b = FINAL_SPEC
    home, away = matches[source_a][2], matches[source_b][2]
    matches[final_id] = (home, away, knockout_winner(cache, home, away, rng))

    third_id, source_a, source_b = THIRD_PLACE_SPEC
    home = (
        matches[source_a][1]
        if matches[source_a][2] == matches[source_a][0]
        else matches[source_a][0]
    )
    away = (
        matches[source_b][1]
        if matches[source_b][2] == matches[source_b][0]
        else matches[source_b][0]
    )
    matches[third_id] = (home, away, knockout_winner(cache, home, away, rng))
    return matches


def _title_probabilities(
    wins: Counter[str], teams: set[str], simulations: int
) -> pd.DataFrame:
    ordered = sorted(teams)
    probability = np.array([wins[team] / simulations for team in ordered])
    standard_error = np.sqrt(probability * (1.0 - probability) / simulations)
    return (
        pd.DataFrame(
            {
                "team": ordered,
                "win_probability": probability,
                "mc_95_low": np.clip(probability - 1.96 * standard_error, 0.0, 1.0),
                "mc_95_high": np.clip(probability + 1.96 * standard_error, 0.0, 1.0),
            }
        )
        .sort_values(["win_probability", "team"], ascending=[False, True])
        .reset_index(drop=True)
    )


def simulate_2026_scoreline(
    fixtures: pd.DataFrame,
    cache: dict[tuple[str, str], dict[str, np.ndarray | tuple[float, float, float]]],
    fifa_rank: dict[str, float],
    simulations: int = 10_000,
    seed: int = 2026,
    fair_play: dict[str, int] | None = None,
    knockout_cache: dict[
        tuple[str, str], dict[str, np.ndarray | tuple[float, float, float]]
    ]
    | None = None,
) -> pd.DataFrame:
    """Conditioned scoreline simulation using FIFA's official 2026 structure."""
    rng = np.random.default_rng(seed)
    annex_c = load_annex_c()
    group_rows = fixtures[fixtures["group"].notna()]
    teams = set(group_rows["team1"]) | set(group_rows["team2"])
    wins: Counter[str] = Counter()
    for _ in range(simulations):
        standings, third_groups = _simulate_groups(
            fixtures, cache, fifa_rank, rng, fair_play
        )
        matches = _simulate_official_knockout(
            standings, third_groups, knockout_cache or cache, rng, annex_c
        )
        wins[matches[FINAL_SPEC[0]][2]] += 1
    return _title_probabilities(wins, teams, simulations)
