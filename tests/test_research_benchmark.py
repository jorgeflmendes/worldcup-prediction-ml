from __future__ import annotations

import numpy as np
import pandas as pd

from worldcup2026.evaluation import (
    clustered_bootstrap_interval,
    clustered_sign_flip_pvalue,
)
from worldcup2026.match_stats import rolling_match_stats
from worldcup2026.research_benchmark import (
    frozen_match_frame,
    match_stats_goal_data,
    top1_decision_probabilities,
)
from worldcup2026.goals_simulation import sequential_goal_data


def test_frozen_match_frame_ignores_unplayed_fixtures() -> None:
    results = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-06-11"),
                "tournament": "FIFA World Cup",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 2.0,
                "away_score": 1.0,
            },
            {
                "date": pd.Timestamp("2026-06-12"),
                "tournament": "FIFA World Cup",
                "home_team": "Gamma",
                "away_team": "Delta",
                "home_score": float("nan"),
                "away_score": float("nan"),
            },
        ]
    )

    frame = frozen_match_frame(results, 2026, {}, {}, {}, {})

    assert len(frame) == 1
    assert frame.iloc[0][["home", "away", "truth"]].to_dict() == {
        "home": "Alpha",
        "away": "Beta",
        "truth": 0,
    }


def test_sequential_goal_data_preserves_competition(monkeypatch) -> None:
    results = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-01"),
                "tournament": "Friendly",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 1,
                "away_score": 0,
                "neutral": True,
            }
        ]
    )
    monkeypatch.setattr(
        "worldcup2026.goals_simulation.attach_fifa_rankings",
        lambda frame, _: frame.assign(
            fifa_rank_diff=0.0,
            fifa_points_diff=0.0,
            fifa_rank_missing=1,
        ),
    )
    monkeypatch.setattr(
        "worldcup2026.goals_simulation.attach_external_elo",
        lambda frame, _: frame.assign(
            external_elo_diff=0.0,
            external_elo_missing=1,
        ),
    )
    monkeypatch.setattr(
        "worldcup2026.goals_simulation.load_fifa_rankings",
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "worldcup2026.goals_simulation.load_external_elo",
        lambda: pd.DataFrame(),
    )

    frame = sequential_goal_data(results)

    assert frame.loc[0, "tournament"] == "Friendly"


def test_top1_blend_uses_prior_selected_poisson_weight() -> None:
    probabilistic = np.array([[0.60, 0.25, 0.15]])
    poisson = np.array([[0.30, 0.25, 0.45]])

    blended = top1_decision_probabilities(probabilistic, poisson)

    np.testing.assert_allclose(blended, [[0.3975, 0.25, 0.3525]])


def test_rolling_match_stats_shift_each_match_once() -> None:
    matches = pd.DataFrame(
        [
            {"date": "2018-01-01", "home": "Alpha", "away": "Beta"},
            {"date": "2022-01-01", "home": "Alpha", "away": "Gamma"},
        ]
    )
    observations = pd.DataFrame(
        [
            {
                "date": "2018-01-01",
                "home": "Alpha",
                "away": "Beta",
                "home_shots": 12,
                "away_shots": 4,
            }
        ]
    )

    enriched = rolling_match_stats(matches, observations)

    assert np.isnan(enriched.loc[0, "stat_shots_home_for"])
    assert enriched.loc[1, "stat_shots_home_for"] == 12
    assert enriched.loc[1, "stat_shots_home_against"] == 4


def test_rolling_match_stats_has_no_year_specific_reset() -> None:
    matches = pd.DataFrame([{"date": "2026-01-01", "home": "Alpha", "away": "Gamma"}])
    observations = pd.DataFrame(
        [
            {
                "date": "2014-06-01",
                "home": "Alpha",
                "away": "Beta",
                "home_corners": 7,
                "away_corners": 2,
            }
        ]
    )

    enriched = rolling_match_stats(matches, observations)

    assert enriched.loc[0, "stat_corners_home_for"] == 7


def test_match_stats_cache_is_scoped_to_input_frame(monkeypatch) -> None:
    calls = []

    def enrich(frame: pd.DataFrame, _: pd.DataFrame) -> pd.DataFrame:
        calls.append(frame)
        return frame.assign(enriched=len(calls))

    monkeypatch.setattr(
        "worldcup2026.research_benchmark._MATCH_STATS_GOAL_DATA_CACHE",
        None,
    )
    monkeypatch.setattr(
        "worldcup2026.research_benchmark.match_stat_data.load_match_stats",
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "worldcup2026.research_benchmark.match_stat_data.rolling_match_stats",
        enrich,
    )
    first = pd.DataFrame({"value": [1]})
    second = pd.DataFrame({"value": [2]})

    assert match_stats_goal_data(first).enriched.iloc[0] == 1
    assert match_stats_goal_data(first).enriched.iloc[0] == 1
    assert match_stats_goal_data(second).enriched.iloc[0] == 2
    assert len(calls) == 2


def test_clustered_inference_resamples_whole_tournaments() -> None:
    values = np.array([0.1, 0.2, 0.3, 0.4])
    tournaments = np.array([2018, 2018, 2022, 2022])

    low, high = clustered_bootstrap_interval(
        values,
        tournaments,
        iterations=500,
        seed=7,
    )

    assert low < values.mean() < high
    assert clustered_sign_flip_pvalue(values, tournaments) == 0.25
