from __future__ import annotations

import io
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from difflib import SequenceMatcher
from functools import lru_cache
from lxml import etree, html
import unicodedata

import numpy as np
import pandas as pd
import requests
from .config import ARTIFACT_DIR as ARTIFACT_DIR
from .config import EXTERNAL_DIR, PROJECT_ROOT, RAW_DIR

ROOT = PROJECT_ROOT

INTERNATIONAL_RESULTS_COMMIT = "4a0e4ce6d673bf0020e057060b17e230b3c61288"
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    f"{INTERNATIONAL_RESULTS_COMMIT}/results.csv"
)
GOALSCORERS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    f"{INTERNATIONAL_RESULTS_COMMIT}/goalscorers.csv"
)
WC_2026_DATASET = "kulkarniparth09/fifa-world-cup-complete-dataset-19302026"
WC_2026_SQUADS_DATASET = "cabcon/fifa-world-cup-2026-squad-lists-all-48-teams"
KAGGLE_DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/{}"
KAGGLE_DATASET_VERSIONS = {
    "cabcon/fifa-world-cup-2026-squad-lists-all-48-teams": 1,
    "davidcariboo/player-scores": 671,
    "gabipana7/fifa-rankings-and-international-matches-1992-2022": 1,
    "kulkarniparth09/fifa-world-cup-complete-dataset-19302026": 1,
    "saifalnimri/international-football-elo-ratings": 1,
}
WIKIPEDIA_SQUAD_REVISIONS = {
    1998: 1360872349,
    2002: 1361377555,
    2006: 1357069147,
    2010: 1359542814,
    2014: 1359549052,
    2018: 1360846738,
    2022: 1359336946,
}
CLUB_ELO_URL = "http://api.clubelo.com/{}"
CHAMPIONS = {
    1930: "Uruguay",
    1934: "Italy",
    1938: "Italy",
    1950: "Uruguay",
    1954: "Germany",
    1958: "Brazil",
    1962: "Brazil",
    1966: "England",
    1970: "Brazil",
    1974: "Germany",
    1978: "Argentina",
    1982: "Italy",
    1986: "Argentina",
    1990: "Germany",
    1994: "Brazil",
    1998: "France",
    2002: "Brazil",
    2006: "Italy",
    2010: "Spain",
    2014: "Germany",
    2018: "France",
    2022: "Argentina",
}
ALIASES = {
    "Czech Republic": "Czechia",
    "Republic of Ireland": "Ireland",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "USA": "United States",
    "T\u00fcrkiye": "Turkey",
    "T\u00c3\u00bcrkiye": "Turkey",
    "C\u00f4te d'Ivoire": "Ivory Coast",
    "C\u00c3\u00b4te d'Ivoire": "Ivory Coast",
    "Cura\u00e7ao": "Curacao",
    "Cura\u00c3\u00a7ao": "Curacao",
}


def clean_team(value: object) -> str:
    name = re.sub(r"\s+", " ", str(value)).strip()
    return ALIASES.get(name, name)


def venue_advantage(
    home: object,
    away: object,
    country: object,
    neutral: object,
) -> int:
    if bool(neutral):
        return 0
    venue = clean_team(country)
    if venue == clean_team(away):
        return -1
    return 1


def cached_download(url: str, path: Path, refresh: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh:
        return path
    response = requests.get(
        url,
        timeout=90,
        headers={"User-Agent": "worldcup2026-research/1.0 (reproducible research)"},
    )
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def kaggle_download_url(dataset: str) -> str:
    version = KAGGLE_DATASET_VERSIONS[dataset]
    return f"{KAGGLE_DOWNLOAD.format(dataset)}?datasetVersionNumber={version}"


def load_results(refresh: bool) -> pd.DataFrame:
    path = cached_download(RESULTS_URL, RAW_DIR / "international_results.csv", refresh)
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["home_team"] = frame.home_team.map(clean_team)
    frame["away_team"] = frame.away_team.map(clean_team)
    return frame.sort_values("date").reset_index(drop=True)


def load_goal_events(refresh: bool = False) -> pd.DataFrame:
    path = cached_download(
        GOALSCORERS_URL,
        RAW_DIR / "international_goalscorers.csv",
        refresh,
    )
    frame = pd.read_csv(path, parse_dates=["date"])
    for column in ("home_team", "away_team", "team"):
        frame[column] = frame[column].map(clean_team)
    opponent = np.where(
        frame.team.eq(frame.home_team),
        frame.away_team,
        frame.home_team,
    )
    frame["credited_team"] = np.where(frame.own_goal, opponent, frame.team)
    frame["minute"] = pd.to_numeric(frame.minute, errors="coerce")
    return frame


def regulation_time_results(
    results: pd.DataFrame,
    refresh: bool = False,
) -> pd.DataFrame:
    keys = ["date", "home_team", "away_team"]
    events = load_goal_events(refresh)
    event_totals = events.groupby(keys).size().rename("event_goals")
    regulation = (
        events[events.minute.le(90)]
        .groupby(keys + ["credited_team"])
        .size()
        .unstack(fill_value=0)
    )

    output = results.copy()
    output = output.join(event_totals, on=keys)
    output["event_goals"] = output.event_goals.fillna(0).astype(int)
    output["regulation_complete"] = output.event_goals.eq(
        output.home_score + output.away_score
    )
    home_goals = []
    away_goals = []
    for row in output.itertuples(index=False):
        key = (row.date, row.home_team, row.away_team)
        if row.regulation_complete and key in regulation.index:
            counts = regulation.loc[key]
            home_goals.append(float(counts.get(row.home_team, 0)))
            away_goals.append(float(counts.get(row.away_team, 0)))
        elif row.regulation_complete:
            home_goals.append(0.0)
            away_goals.append(0.0)
        else:
            home_goals.append(np.nan)
            away_goals.append(np.nan)
    output["home_score_90"] = home_goals
    output["away_score_90"] = away_goals
    return output


def load_2026_teams(refresh: bool) -> list[str]:
    """Load the participant list from the edition dataset."""
    archive = cached_download(
        kaggle_download_url(WC_2026_DATASET),
        RAW_DIR / "world_cup_1930_2026.zip",
        refresh,
    )
    with zipfile.ZipFile(archive) as zipped:
        candidates = [
            x
            for x in zipped.namelist()
            if x.lower().endswith("teams.csv") and "2026" in x.lower()
        ]
        if not candidates:
            raise RuntimeError("The participant source does not contain a teams CSV.")
        frame = pd.read_csv(io.BytesIO(zipped.read(candidates[0])))
    column = next(
        (
            c
            for c in frame.columns
            if c.lower() in {"team", "team_name", "country", "nation"}
        ),
        None,
    )
    if column is None:
        raise RuntimeError(f"Team column not found: {frame.columns.tolist()}")
    teams = sorted({clean_team(team) for team in frame[column].dropna()})
    if len(teams) < 32:
        raise RuntimeError(f"Incomplete 2026 participant list: found {len(teams)}.")
    return teams


@lru_cache(maxsize=None)
def kaggle_2026_squad_snapshot(cutoff_text: str, refresh: bool) -> pd.DataFrame:
    """Current edition roster metadata, kept separate from historical results and fixtures."""
    archive = cached_download(
        kaggle_download_url(WC_2026_SQUADS_DATASET),
        RAW_DIR / "world_cup_2026_squads.zip",
        refresh,
    )
    with zipfile.ZipFile(archive) as zipped:
        name = next(
            (item for item in zipped.namelist() if item.lower().endswith(".csv")), None
        )
        if name is None:
            raise RuntimeError("The 2026 squad source does not contain a CSV.")
        frame = pd.read_csv(io.BytesIO(zipped.read(name)))
    required = {"Team", "Position", "Player Name", "Club", "Caps", "Goals"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing 2026 squad columns: {sorted(missing)}")
    cutoff = pd.Timestamp(cutoff_text)
    dob = pd.to_datetime(frame.get("DOB"), format="%d/%m/%Y", errors="coerce")
    age = ((cutoff - dob).dt.days / 365.2425).astype(float)
    result = pd.DataFrame(
        {
            "snapshot_date": cutoff,
            "tournament": 2026,
            "team": frame.Team.map(clean_team),
            "player_name": frame["Player Name"],
            "position": frame.Position,
            "club": frame.Club,
            "age": age,
            "international_caps": pd.to_numeric(frame.Caps, errors="coerce"),
            "international_goals": pd.to_numeric(frame.Goals, errors="coerce"),
        }
    )
    result["player_id"] = (
        result.team.astype(str) + "::" + result.player_name.map(normalized_text)
    )
    return result


def normalized_text(value: object) -> str:
    text = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    text = re.sub(r"\b(fc|cf|ac|afc|sc|ssc|cd|de|the)\b", " ", text)
    return re.sub(r"[^a-z0-9]", "", text)


@lru_cache(maxsize=None)
def wiki_squad_snapshot(year: int, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse roster tables and coach names from an edition squad page."""
    revision = WIKIPEDIA_SQUAD_REVISIONS.get(year)
    if revision is None:
        raise ValueError(f"No pinned Wikipedia squad revision for {year}.")
    url = (
        "https://en.wikipedia.org/w/index.php"
        f"?title={year}_FIFA_World_Cup_squads&oldid={revision}"
    )
    page = cached_download(
        url,
        RAW_DIR / "wikipedia" / f"{year}_squads.html",
        refresh,
    )
    tree = html.fromstring(page.read_bytes())
    player_frames: list[pd.DataFrame] = []
    coaches: list[dict[str, object]] = []
    for heading in tree.xpath("//h2|//h3"):
        team = clean_team(re.sub(r"\[edit\]$", "", heading.text_content().strip()))
        section = heading.getparent()
        if section is None:
            continue
        node = section.getnext()
        coach = None
        table = None
        while node is not None and not (
            node.tag == "div" and "mw-heading" in (node.get("class") or "")
        ):
            if not isinstance(node.tag, str):
                node = node.getnext()
                continue
            text = " ".join(node.text_content().split())
            found_coach = re.search(
                r"(?:Head )?(?:coach|manager)\s*:\s*(.+)", text, flags=re.I
            )
            if found_coach:
                coach = re.sub(r"\[.*?\]", "", found_coach.group(1)).strip()
            if node.tag == "table" and "Player" in text[:180]:
                table = node
                break
            node = node.getnext()
        if table is None:
            continue
        try:
            roster = pd.read_html(io.BytesIO(etree.tostring(table)))[0]
        except ValueError:
            continue
        columns = {str(column).lower(): column for column in roster.columns}
        player_column = next(
            (
                original
                for lower, original in columns.items()
                if lower == "player" or lower.startswith("player")
            ),
            None,
        )
        position_column = next(
            (original for lower, original in columns.items() if "pos" in lower), None
        )
        club_column = next(
            (
                original
                for lower, original in columns.items()
                if lower == "club" or lower.startswith("club")
            ),
            None,
        )
        if player_column is None or position_column is None:
            continue
        roster = roster.rename(
            columns={player_column: "player_name", position_column: "position"}
        )
        roster["club"] = roster[club_column] if club_column else np.nan
        roster["team"] = team
        roster["tournament"] = year
        roster["player_id"] = (
            roster.team.astype(str)
            + "::"
            + roster.player_name.astype(str).map(normalized_text)
        )
        age_column = next(
            (
                original
                for lower, original in columns.items()
                if "birth" in lower or lower == "age"
            ),
            None,
        )
        if age_column:
            roster["age"] = pd.to_numeric(
                roster[age_column]
                .astype(str)
                .str.extract(r"(?:aged )?(\d{1,2})(?:\)|$)", expand=False),
                errors="coerce",
            )
        else:
            roster["age"] = np.nan
        for source, target in [
            ("caps", "international_caps"),
            ("goals", "international_goals"),
        ]:
            original = next(
                (value for lower, value in columns.items() if lower == source), None
            )
            roster[target] = (
                pd.to_numeric(roster[original], errors="coerce") if original else np.nan
            )
        player_frames.append(
            roster[
                [
                    "tournament",
                    "team",
                    "player_id",
                    "player_name",
                    "age",
                    "position",
                    "club",
                    "international_caps",
                    "international_goals",
                ]
            ]
        )
        coaches.append({"tournament": year, "team": team, "coach_name": coach})
    if not player_frames:
        raise RuntimeError(f"Could not extract squads from the {year} page.")
    players = pd.concat(player_frames, ignore_index=True)
    return players, pd.DataFrame(coaches)


def club_elo_snapshot(cutoff: pd.Timestamp, refresh: bool) -> pd.DataFrame:
    date = cutoff.strftime("%Y-%m-%d")
    path = cached_download(
        CLUB_ELO_URL.format(date), RAW_DIR / "clubelo" / f"{date}.csv", refresh
    )
    frame = pd.read_csv(path)
    frame["club_key"] = frame.Club.map(normalized_text)
    return frame[["club_key", "Rank", "Country", "Level", "Elo"]].rename(
        columns={
            "Rank": "club_rank",
            "Country": "club_country",
            "Level": "club_level",
            "Elo": "club_elo",
        }
    )


def attach_club_strength(
    players: pd.DataFrame, cutoff: pd.Timestamp, refresh: bool
) -> pd.DataFrame:
    """Attach ClubElo values using normalized club names."""
    clubs = club_elo_snapshot(cutoff, refresh)
    exact = dict(zip(clubs.club_key, clubs.index))
    keys = clubs.club_key.tolist()
    aliases = {
        "manchestercity": "mancity",
        "manchesterunited": "manunited",
        "parissaintgermain": "parissg",
        "bayernmunch": "bayern",
        "bayernmunchen": "bayern",
        "internazionalemilano": "inter",
        "athleticomadrid": "atletico",
        "atleticodemadrid": "atletico",
        "realmadridcf": "realmadrid",
        "fcbarcelona": "barcelona",
    }
    matched = []
    for club in players.club.fillna(""):
        cleaned = re.sub(r"\s*\([A-Z]{3}\)\s*$", "", str(club).strip())
        key = normalized_text(cleaned)
        target = aliases.get(key, key)
        index = exact.get(target)
        if index is None and target:
            scored = sorted(
                (
                    (SequenceMatcher(None, target, candidate).ratio(), i)
                    for i, candidate in enumerate(keys)
                ),
                reverse=True,
            )
            if (
                scored
                and scored[0][0] >= 0.91
                and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.025)
            ):
                index = scored[0][1]
        matched.append(index)
    values = clubs.reindex(matched).reset_index(drop=True)
    base = players.drop(
        columns=["club_elo", "club_rank", "club_level", "club_country"], errors="ignore"
    ).reset_index(drop=True)
    return pd.concat([base, values.drop(columns="club_key")], axis=1)


def coach_features(year: int, teams: Iterable[str], refresh: bool) -> pd.DataFrame:
    historical: list[pd.DataFrame] = []
    coach_years = [wc_year for wc_year in CHAMPIONS if wc_year <= year]
    if year == 2026:
        coach_years.append(2026)
    for wc_year in coach_years:
        try:
            _, coaches = wiki_squad_snapshot(wc_year, refresh)
        except ValueError:
            continue
        historical.append(coaches)
    if not historical:
        base = pd.DataFrame({"team": list(teams)})
        base["coach_prior_wc_appearances"] = np.nan
        base["coach_returning_to_same_team"] = np.nan
        return base
    coaches = pd.concat(historical, ignore_index=True)
    current = coaches[coaches.tournament == year].copy()
    prior = coaches[coaches.tournament < year].copy()
    current["coach_name"] = current.coach_name.fillna("").map(normalized_text)
    prior["coach_name"] = prior.coach_name.fillna("").map(normalized_text)
    current["coach_prior_wc_appearances"] = current.coach_name.map(
        prior.coach_name.value_counts()
    ).fillna(0)
    same_team = set(zip(prior.team, prior.coach_name))
    current["coach_returning_to_same_team"] = [
        int((team, coach) in same_team and bool(coach))
        for team, coach in zip(current.team, current.coach_name)
    ]
    base = pd.DataFrame({"team": list(teams)})
    if current.empty:
        base["coach_prior_wc_appearances"] = np.nan
        base["coach_returning_to_same_team"] = np.nan
        return base
    return base.merge(
        current[["team", "coach_prior_wc_appearances", "coach_returning_to_same_team"]],
        on="team",
        how="left",
    )


def roster_continuity_features(
    players: pd.DataFrame, year: int, refresh: bool
) -> pd.DataFrame:
    """Prior World Cup appearances of the actual players, with no outcome information."""
    prior = []
    for prior_year in CHAMPIONS:
        if prior_year < year:
            try:
                prior_players, _ = wiki_squad_snapshot(prior_year, refresh)
            except ValueError:
                continue
            prior.append(prior_players[["player_id"]])
    if not prior:
        result = players[["team"]].drop_duplicates().copy()
        result["returning_wc_player_count"] = 0.0
        result["returning_wc_player_share"] = 0.0
        return result
    historical_ids = set(pd.concat(prior, ignore_index=True).player_id)
    current = players.assign(
        returning=players.player_id.isin(historical_ids).astype(float)
    )
    return (
        current.groupby("team")
        .agg(
            returning_wc_player_count=("returning", "sum"),
            returning_wc_player_share=("returning", "mean"),
        )
        .reset_index()
    )


def elo_before(results: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, float]:
    """Build an Elo snapshot at a cutoff."""
    ratings: defaultdict[str, float] = defaultdict(lambda: 1500.0)
    for row in results.loc[results.date < cutoff].itertuples(index=False):
        home, away = row.home_team, row.away_team
        advantage = 55.0 * venue_advantage(
            home,
            away,
            row.country,
            row.neutral,
        )
        expected_home = 1.0 / (
            1.0 + 10 ** (-((ratings[home] + advantage) - ratings[away]) / 400.0)
        )
        if row.home_score > row.away_score:
            observed_home = 1.0
        elif row.home_score < row.away_score:
            observed_home = 0.0
        else:
            observed_home = 0.5
        
        diff = abs(row.home_score - row.away_score)
        if diff <= 1:
            g = 1.0
        elif diff == 2:
            g = 1.5
        else:
            g = (11.0 + diff) / 8.0

        importance = 1.0 if row.tournament == "Friendly" else 1.35
        k = 22.0 * importance * g
        delta = k * (observed_home - expected_home)
        ratings[home] += delta
        ratings[away] -= delta
    return dict(ratings)


def team_recent_matches(
    results: pd.DataFrame, team: str, cutoff: pd.Timestamp
) -> pd.DataFrame:
    lower = cutoff - pd.Timedelta(days=730)
    games = results[(results.date < cutoff) & (results.date >= lower)]
    home = games[games.home_team == team].assign(
        goals_for=lambda x: x.home_score,
        goals_against=lambda x: x.away_score,
        opponent=lambda x: x.away_team,
    )
    away = games[games.away_team == team].assign(
        goals_for=lambda x: x.away_score,
        goals_against=lambda x: x.home_score,
        opponent=lambda x: x.home_team,
    )
    return pd.concat([home, away], ignore_index=True)


def squad_features(
    cutoff: pd.Timestamp, year: int, teams: Iterable[str], refresh: bool
) -> pd.DataFrame:
    """Aggregate player, club, and coach metadata."""
    path = EXTERNAL_DIR / "squad_snapshots.csv"
    materialized_panel = EXTERNAL_DIR / "squad_panel_pre_tournament.csv"
    feature_columns = [
        "squad_size",
        "squad_age_mean",
        "squad_value_total",
        "squad_value_median",
        "squad_minutes_total",
        "squad_caps_mean",
        "club_elo_mean",
        "club_rank_mean",
        "club_count",
        "top5_club_share",
        "forward_share",
        "midfielder_share",
        "defender_share",
        "goalkeeper_share",
        "club_elo_p90",
        "tier1_club_share",
        "club_country_count",
        "squad_age_std",
        "squad_caps_p90",
        "squad_goals_mean",
        "squad_age_p10",
        "squad_age_p90",
        "squad_under23_share",
        "squad_over30_share",
        "squad_caps_total",
        "squad_caps_p10",
        "squad_caps_50plus_share",
        "squad_goals_total",
        "club_elo_std",
        "club_rank_p25",
        "club_rank_p75",
        "club_hhi",
        "top_club_share",
        "superelite_club_share",
        "big5_league_share",
        "returning_wc_player_count",
        "returning_wc_player_share",
        "club_domestic_rank_mean",
        "club_league_strength_mean",
        "squad_goals_365_total",
        "squad_assists_365_total",
        "squad_xg_365_total",
        "squad_xa_365_total",
        "squad_injury_days_365_total",
        "squad_champions_league_minutes_365_total",
    ]
    empty = pd.DataFrame({"team": list(teams)})
    for column in feature_columns:
        empty[column] = np.nan
    if year in CHAMPIONS:
        players, _ = wiki_squad_snapshot(year, refresh)
        players["snapshot_date"] = cutoff
    elif year == 2026:
        players = kaggle_2026_squad_snapshot(cutoff.strftime("%Y-%m-%d"), refresh)
    else:
        players = pd.DataFrame(
            columns=[
                "snapshot_date",
                "tournament",
                "team",
                "player_id",
                "age",
                "position",
                "club",
                "international_caps",
                "international_goals",
            ]
        )
    custom_frames = []
    if materialized_panel.exists():
        custom_frames.append(
            pd.read_csv(materialized_panel, parse_dates=["snapshot_date"])
        )
    if path.exists():
        custom_frames.append(pd.read_csv(path, parse_dates=["snapshot_date"]))
    if custom_frames:
        custom = pd.concat(custom_frames, ignore_index=True, sort=False)
        players = pd.concat([players, custom], ignore_index=True, sort=False)
    required = {
        "snapshot_date",
        "tournament",
        "team",
        "player_id",
        "age",
        "position",
        "club",
    }
    missing = required - set(players.columns)
    if missing:
        raise ValueError(
            f"squad_snapshots.csv is missing required columns: {sorted(missing)}"
        )
    players.team = players.team.map(clean_team)
    players = players[
        (players.tournament == year)
        & (players.snapshot_date <= cutoff)
        & players.team.isin(teams)
    ]
    if players.empty:
        return empty.merge(coach_features(year, teams, refresh), on="team", how="left")
    players = (
        players.sort_values("snapshot_date")
        .groupby(["team", "player_id"], as_index=False)
        .tail(1)
    )
    for col in [
        "market_value_eur",
        "minutes_last_365",
        "international_caps",
        "international_goals",
        "club_elo",
        "club_domestic_rank",
        "club_league_strength",
        "goals_last_365",
        "assists_last_365",
        "xg_last_365",
        "xa_last_365",
        "injury_days_last_365",
        "champions_league_minutes_last_365",
    ]:
        if col not in players:
            players[col] = np.nan
    if players.club_elo.isna().all():
        players = attach_club_strength(players, cutoff, refresh)
    else:
        players["club_rank"] = players.get("club_domestic_rank")
        players["club_level"] = np.where(players.club_rank == 1, 1, np.nan)
        players["club_country"] = np.nan
    output = (
        players.groupby("team")
        .agg(
            squad_size=("player_id", "nunique"),
            squad_age_mean=("age", "mean"),
            squad_age_std=("age", "std"),
            squad_age_p10=("age", lambda values: values.quantile(0.1)),
            squad_age_p90=("age", lambda values: values.quantile(0.9)),
            squad_value_total=(
                "market_value_eur",
                lambda values: values.sum(min_count=1),
            ),
            squad_value_median=("market_value_eur", "median"),
            squad_minutes_total=(
                "minutes_last_365",
                lambda values: values.sum(min_count=1),
            ),
            squad_caps_mean=("international_caps", "mean"),
            squad_caps_total=(
                "international_caps",
                lambda values: values.sum(min_count=1),
            ),
            squad_caps_p10=("international_caps", lambda values: values.quantile(0.1)),
            squad_caps_p90=("international_caps", lambda values: values.quantile(0.9)),
            squad_goals_mean=("international_goals", "mean"),
            squad_goals_total=(
                "international_goals",
                lambda values: values.sum(min_count=1),
            ),
            club_elo_mean=("club_elo", "mean"),
            club_elo_p90=("club_elo", lambda values: values.quantile(0.9)),
            club_elo_std=("club_elo", "std"),
            club_rank_mean=("club_rank", "mean"),
            club_rank_p25=("club_rank", lambda values: values.quantile(0.25)),
            club_rank_p75=("club_rank", lambda values: values.quantile(0.75)),
            club_count=("club", "nunique"),
            club_country_count=(
                "club_country",
                lambda values: values.nunique() if values.notna().any() else np.nan,
            ),
            club_domestic_rank_mean=("club_domestic_rank", "mean"),
            club_league_strength_mean=("club_league_strength", "mean"),
            squad_goals_365_total=(
                "goals_last_365",
                lambda values: values.sum(min_count=1),
            ),
            squad_assists_365_total=(
                "assists_last_365",
                lambda values: values.sum(min_count=1),
            ),
            squad_xg_365_total=("xg_last_365", lambda values: values.sum(min_count=1)),
            squad_xa_365_total=("xa_last_365", lambda values: values.sum(min_count=1)),
            squad_injury_days_365_total=(
                "injury_days_last_365",
                lambda values: values.sum(min_count=1),
            ),
            squad_champions_league_minutes_365_total=(
                "champions_league_minutes_last_365",
                lambda values: values.sum(min_count=1),
            ),
        )
        .reset_index()
    )
    position = players.position.astype(str).str.upper().str[:2]
    for label, output_name in [
        ("FW", "forward_share"),
        ("MF", "midfielder_share"),
        ("DF", "defender_share"),
        ("GK", "goalkeeper_share"),
    ]:
        output[output_name] = (
            players.assign(hit=(position == label).astype(int))
            .groupby("team")
            .hit.mean()
            .reindex(output.team)
            .to_numpy()
        )
    output["top5_club_share"] = (
        players.assign(
            hit=np.where(
                players.club_elo.notna(),
                (players.club_elo >= 1750).astype(float),
                np.nan,
            )
        )
        .groupby("team")
        .hit.mean()
        .reindex(output.team)
        .to_numpy()
    )
    output["tier1_club_share"] = (
        players.assign(
            hit=np.where(
                players.club_level.notna(),
                (players.club_level == 1).astype(float),
                np.nan,
            )
        )
        .groupby("team")
        .hit.mean()
        .reindex(output.team)
        .to_numpy()
    )
    output["squad_under23_share"] = (
        players.assign(
            hit=np.where(players.age.notna(), (players.age < 23).astype(float), np.nan)
        )
        .groupby("team")
        .hit.mean()
        .reindex(output.team)
        .to_numpy()
    )
    output["squad_over30_share"] = (
        players.assign(
            hit=np.where(players.age.notna(), (players.age >= 30).astype(float), np.nan)
        )
        .groupby("team")
        .hit.mean()
        .reindex(output.team)
        .to_numpy()
    )
    output["squad_caps_50plus_share"] = (
        players.assign(
            hit=np.where(
                players.international_caps.notna(),
                (players.international_caps >= 50).astype(float),
                np.nan,
            )
        )
        .groupby("team")
        .hit.mean()
        .reindex(output.team)
        .to_numpy()
    )
    output["club_hhi"] = (
        players.groupby("team")
        .club.transform(lambda values: values.value_counts(normalize=True).pow(2).sum())
        .groupby(players.team)
        .first()
        .reindex(output.team)
        .to_numpy()
    )
    output["top_club_share"] = (
        players.groupby("team")
        .club.transform(lambda values: values.value_counts(normalize=True).max())
        .groupby(players.team)
        .first()
        .reindex(output.team)
        .to_numpy()
    )
    output["superelite_club_share"] = (
        players.assign(
            hit=np.where(
                players.club_elo.notna(),
                (players.club_elo >= 1900).astype(float),
                np.nan,
            )
        )
        .groupby("team")
        .hit.mean()
        .reindex(output.team)
        .to_numpy()
    )
    output["big5_league_share"] = (
        players.assign(
            hit=np.where(
                players.club_country.notna(),
                players.club_country.isin(["ENG", "ESP", "GER", "ITA", "FRA"]).astype(
                    float
                ),
                np.nan,
            )
        )
        .groupby("team")
        .hit.mean()
        .reindex(output.team)
        .to_numpy()
    )
    return (
        empty.drop(columns=feature_columns)
        .merge(output, on="team", how="left")
        .merge(
            roster_continuity_features(players, year, refresh), on="team", how="left"
        )
        .merge(coach_features(year, teams, refresh), on="team", how="left")
    )
