"""FIFA World Cup 2026 competition rules."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .config import EXTERNAL_DIR

ANNEX_C_PATH = EXTERNAL_DIR / "fwc2026_annex_c.csv"

R32_SPECS = (
    ("M73", "2A", "2B"),
    ("M74", "1E", "3"),
    ("M75", "1F", "2C"),
    ("M76", "1C", "2F"),
    ("M77", "1I", "3"),
    ("M78", "2E", "2I"),
    ("M79", "1A", "3"),
    ("M80", "1L", "3"),
    ("M81", "1D", "3"),
    ("M82", "1G", "3"),
    ("M83", "2K", "2L"),
    ("M84", "1H", "2J"),
    ("M85", "1B", "3"),
    ("M86", "1J", "2H"),
    ("M87", "1K", "3"),
    ("M88", "2D", "2G"),
)

R16_SPECS = (
    ("M89", "M74", "M77"),
    ("M90", "M73", "M75"),
    ("M91", "M76", "M78"),
    ("M92", "M79", "M80"),
    ("M93", "M83", "M84"),
    ("M94", "M81", "M82"),
    ("M95", "M86", "M88"),
    ("M96", "M85", "M87"),
)
QF_SPECS = (
    ("M97", "M89", "M90"),
    ("M98", "M93", "M94"),
    ("M99", "M91", "M92"),
    ("M100", "M95", "M96"),
)
SF_SPECS = (("M101", "M97", "M98"), ("M102", "M99", "M100"))
FINAL_SPEC = ("M104", "M101", "M102")
THIRD_PLACE_SPEC = ("M103", "M101", "M102")


def load_annex_c(path: Path = ANNEX_C_PATH) -> dict[frozenset[str], dict[str, str]]:
    """Return {qualifying-third groups: {first-place group: third-place group}}."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing official Annex C table: {path}. Run tools/extract_fifa_annex_c.py."
        )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {"option", "1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"}
    if not rows or set(rows[0]) != expected:
        raise ValueError(f"Invalid Annex C columns in {path}.")
    mapping: dict[frozenset[str], dict[str, str]] = {}
    for row in rows:
        assignment = {
            column[1]: row[column].removeprefix("3")
            for column in expected
            if column.startswith("1")
        }
        key = frozenset(assignment.values())
        if len(key) != 8 or key in mapping:
            raise ValueError(f"Invalid or duplicate Annex C option {row['option']}.")
        mapping[key] = assignment
    if len(mapping) != 495:
        raise ValueError(
            f"Annex C must contain 495 combinations; found {len(mapping)}."
        )
    return mapping


def resolve_r32(
    standings: dict[str, list[str]],
    qualified_third_groups: Iterable[str],
    annex_c: dict[frozenset[str], dict[str, str]] | None = None,
) -> list[tuple[str, str, str]]:
    """Resolve all round-of-32 teams according to Article 12.6 and Annex C."""
    annex_c = annex_c or load_annex_c()
    qualifying = frozenset(qualified_third_groups)
    try:
        assignment = annex_c[qualifying]
    except KeyError as error:
        raise ValueError(
            f"No Annex C allocation for qualifying third-place groups: {sorted(qualifying)}"
        ) from error

    def resolve(slot: str, first_group: str | None = None) -> str:
        if slot == "3":
            if first_group is None:
                raise ValueError(
                    "Third-place slot is missing its paired first-place group."
                )
            return standings[assignment[first_group]][2]
        return standings[slot[1]][int(slot[0]) - 1]

    output = []
    for match_id, home_slot, away_slot in R32_SPECS:
        first_group = home_slot[1] if home_slot.startswith("1") else away_slot[1]
        output.append(
            (match_id, resolve(home_slot, first_group), resolve(away_slot, first_group))
        )
    return output


def _stats_for(
    teams: Iterable[str], matches: Iterable[tuple[str, str, int, int]]
) -> dict[str, dict[str, int]]:
    stats = {team: {"pts": 0, "gd": 0, "gf": 0} for team in teams}
    allowed = set(stats)
    for home, away, home_goals, away_goals in matches:
        if home not in allowed or away not in allowed:
            continue
        stats[home]["gd"] += home_goals - away_goals
        stats[away]["gd"] += away_goals - home_goals
        stats[home]["gf"] += home_goals
        stats[away]["gf"] += away_goals
        if home_goals > away_goals:
            stats[home]["pts"] += 3
        elif away_goals > home_goals:
            stats[away]["pts"] += 3
        else:
            stats[home]["pts"] += 1
            stats[away]["pts"] += 1
    return stats


def group_table(
    teams: Iterable[str], matches: Iterable[tuple[str, str, int, int]]
) -> dict[str, dict[str, int]]:
    """Points, goal difference and goals scored from all group games."""
    return _stats_for(teams, matches)


def rank_group(
    teams: Iterable[str],
    matches: Iterable[tuple[str, str, int, int]],
    fifa_rank: dict[str, float],
    fair_play: dict[str, int] | None = None,
) -> list[str]:
    """Rank a group using the competition tiebreak order."""
    teams = list(teams)
    matches = list(matches)
    fair_play = fair_play or {}
    overall = _stats_for(teams, matches)

    def global_rank(block: list[str]) -> list[str]:
        return sorted(
            block,
            key=lambda team: (
                -overall[team]["gd"],
                -overall[team]["gf"],
                -fair_play.get(team, 0),
                fifa_rank.get(team, float("inf")),
                team,
            ),
        )

    def head_to_head(block: list[str]) -> list[str]:
        mini = _stats_for(block, matches)
        partitions: dict[tuple[int, int, int], list[str]] = defaultdict(list)
        for team in block:
            value = (mini[team]["pts"], mini[team]["gd"], mini[team]["gf"])
            partitions[value].append(team)
        ordered_values = sorted(partitions, reverse=True)
        if len(ordered_values) == 1:
            return global_rank(block)
        ranked: list[str] = []
        for value in ordered_values:
            tied = partitions[value]
            ranked.extend(tied if len(tied) == 1 else head_to_head(tied))
        return ranked

    points_blocks: dict[int, list[str]] = defaultdict(list)
    for team in teams:
        points_blocks[overall[team]["pts"]].append(team)
    ranking: list[str] = []
    for points in sorted(points_blocks, reverse=True):
        block = points_blocks[points]
        ranking.extend(block if len(block) == 1 else head_to_head(block))
    return ranking


def rank_third_placed(
    third_placed: dict[str, str],
    group_tables: dict[str, dict[str, dict[str, int]]],
    fifa_rank: dict[str, float],
    fair_play: dict[str, int] | None = None,
) -> list[str]:
    """Rank third-placed teams under Article 13, selecting the top eight."""
    fair_play = fair_play or {}
    return sorted(
        third_placed,
        key=lambda group: (
            -group_tables[group][third_placed[group]]["pts"],
            -group_tables[group][third_placed[group]]["gd"],
            -group_tables[group][third_placed[group]]["gf"],
            -fair_play.get(third_placed[group], 0),
            fifa_rank.get(third_placed[group], float("inf")),
            group,
        ),
    )
