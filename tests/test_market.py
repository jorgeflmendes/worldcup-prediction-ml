from __future__ import annotations

import numpy as np
import pandas as pd

from worldcup2026 import market


def test_market_matching_reorients_reversed_fixture(monkeypatch) -> None:
    odds = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-06-12"),
                "home": "Away",
                "away": "Home",
                "p_home": 0.2,
                "p_draw": 0.3,
                "p_away": 0.5,
            }
        ]
    )
    monkeypatch.setattr(market, "load_consensus_odds", lambda *args, **kwargs: odds)
    frame = pd.DataFrame(
        [{"date": pd.Timestamp("2026-06-11"), "home": "Home", "away": "Away"}]
    )

    probability, available = market.match_consensus_probabilities(frame, 2026)

    np.testing.assert_allclose(probability[0], [0.5, 0.3, 0.2])
    assert available.tolist() == [True]


def test_market_blend_leaves_unmatched_rows_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(
        market,
        "match_consensus_probabilities",
        lambda *args, **kwargs: (
            np.array([[0.6, 0.2, 0.2], [np.nan, np.nan, np.nan]]),
            np.array([True, False]),
        ),
    )
    structural = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    frame = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-06-11"), "home": "A", "away": "B"},
            {"date": pd.Timestamp("2026-06-12"), "home": "C", "away": "D"},
        ]
    )

    blended, available = market.blend_with_consensus(structural, frame, 2026)

    expected_market = np.array([0.6, 0.2, 0.2])
    np.testing.assert_allclose(
        blended[0],
        0.94 * structural[0] + 0.06 * expected_market,
    )
    np.testing.assert_allclose(blended[1], structural[1])
    assert available.tolist() == [True, False]
