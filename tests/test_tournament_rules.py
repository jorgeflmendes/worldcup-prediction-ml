from __future__ import annotations

import unittest
from collections import Counter

from worldcup2026.goals_simulation import _title_probabilities
from worldcup2026.tournament_rules import load_annex_c, rank_group, resolve_r32


class TournamentRulesTests(unittest.TestCase):
    def test_annex_c_has_every_official_combination(self) -> None:
        annex = load_annex_c()
        self.assertEqual(len(annex), 495)
        self.assertTrue(all(len(groups) == 8 for groups in annex))

    def test_annex_c_never_creates_a_same_group_round_of_32_game(self) -> None:
        standings = {
            group: [f"{group}1", f"{group}2", f"{group}3", f"{group}4"]
            for group in "ABCDEFGHIJKL"
        }
        for groups in load_annex_c():
            games = resolve_r32(standings, groups)
            self.assertEqual(len(games), 16)
            for _, home, away in games:
                self.assertNotEqual(home[0], away[0])

    def test_head_to_head_precedes_overall_goal_difference(self) -> None:
        teams = ["A", "B", "C", "D"]
        matches = [
            ("A", "B", 1, 0),
            ("B", "C", 2, 0),
            ("C", "A", 3, 0),
            ("A", "D", 4, 0),
            ("B", "D", 4, 0),
            ("C", "D", 4, 0),
        ]
        ranking = rank_group(teams, matches, {})
        self.assertEqual(ranking[:3], ["C", "B", "A"])

    def test_title_probabilities_include_teams_with_zero_wins(self) -> None:
        result = _title_probabilities(
            Counter({"Spain": 7, "France": 3}), {"Spain", "France", "Ghana"}, 10
        )

        self.assertEqual(result.team.tolist(), ["Spain", "France", "Ghana"])
        self.assertEqual(result.win_probability.tolist(), [0.7, 0.3, 0.0])


if __name__ == "__main__":
    unittest.main()
