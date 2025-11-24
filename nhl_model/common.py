"""Shared datatypes and helper utilities for NHL modeling/publishing."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


TEAM_ABBREV_TO_NAME: Dict[str, str] = {
    'ANA': 'Anaheim Ducks',
    'ARI': 'Arizona Coyotes',
    'BOS': 'Boston Bruins',
    'BUF': 'Buffalo Sabres',
    'CAR': 'Carolina Hurricanes',
    'CBJ': 'Columbus Blue Jackets',
    'CGY': 'Calgary Flames',
    'CHI': 'Chicago Blackhawks',
    'COL': 'Colorado Avalanche',
    'DAL': 'Dallas Stars',
    'DET': 'Detroit Red Wings',
    'EDM': 'Edmonton Oilers',
    'FLA': 'Florida Panthers',
    'LAK': 'Los Angeles Kings',
    'MIN': 'Minnesota Wild',
    'MTL': 'Montreal Canadiens',
    'NJD': 'New Jersey Devils',
    'NSH': 'Nashville Predators',
    'NYI': 'New York Islanders',
    'NYR': 'New York Rangers',
    'OTT': 'Ottawa Senators',
    'PHI': 'Philadelphia Flyers',
    'PIT': 'Pittsburgh Penguins',
    'SEA': 'Seattle Kraken',
    'SJS': 'San Jose Sharks',
    'STL': 'St. Louis Blues',
    'TBL': 'Tampa Bay Lightning',
    'TOR': 'Toronto Maple Leafs',
    'VAN': 'Vancouver Canucks',
    'VGK': 'Vegas Golden Knights',
    'WPG': 'Winnipeg Jets',
    'WSH': 'Washington Capitals'
}

TEAM_NAME_TO_ABBREV: Dict[str, str] = {name.upper(): abbr for abbr, name in TEAM_ABBREV_TO_NAME.items()}


def _normalize_team_abbreviation(team: Optional[str]) -> str:
    raw = str(team or '').strip()
    if not raw:
        return ''
    upper = raw.upper()
    if upper in TEAM_ABBREV_TO_NAME:
        return upper
    return TEAM_NAME_TO_ABBREV.get(upper, '')


def get_team_full_name(team: Optional[str]) -> str:
    abbr = _normalize_team_abbreviation(team)
    if abbr:
        return TEAM_ABBREV_TO_NAME[abbr]
    raw = str(team or '').strip()
    return raw or 'TBD'


def get_team_abbreviation(team: Optional[str]) -> str:
    return _normalize_team_abbreviation(team)


def format_team_display(team: Optional[str]) -> str:
    return get_team_full_name(team)


def format_matchup_display(away_team: Optional[str], home_team: Optional[str]) -> str:
    away_name = get_team_full_name(away_team)
    home_name = get_team_full_name(home_team)
    if away_name and home_name:
        return f"{away_name} @ {home_name}"
    if away_name:
        return away_name
    if home_name:
        return home_name
    return 'TBD'


def format_matchup_code(away_team: Optional[str], home_team: Optional[str]) -> str:
    away_code = get_team_abbreviation(away_team) or str(away_team or '').strip() or 'AWAY'
    home_code = get_team_abbreviation(home_team) or str(home_team or '').strip() or 'HOME'
    return f"{away_code}@{home_code}"


def format_matchup_search_blob(away_team: Optional[str], home_team: Optional[str]) -> str:
    display = format_matchup_display(away_team, home_team)
    code = format_matchup_code(away_team, home_team)
    parts: List[str] = []
    for value in (display, code):
        val = (value or '').strip()
        if val and val not in parts:
            parts.append(val)
    return " | ".join(parts)


@dataclass
class OverUnderPrediction:
    """Structured over/under prediction with confidence metrics"""

    game_id: str
    home_team: str
    away_team: str
    predicted_total: float
    betting_line: float
    over_probability: float
    under_probability: float
    push_probability: float
    confidence: float
    expected_value_over: float
    expected_value_under: float
    recommendation: str  # 'OVER', 'UNDER', or 'No Bet' after thresholds/conflict checks
    edge: float
    kelly_bet_size: float
    # Optional American odds used for EV/Kelly
    over_american_odds: Optional[int] = None
    under_american_odds: Optional[int] = None
    bet_month: Optional[str] = None
    calibration_multiplier: Optional[float] = None
    # Added uncertainty interval (conformal) for display
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    # Market-derived probabilities (with and without vig) and source
    market_over_prob: Optional[float] = None
    market_under_prob: Optional[float] = None
    fair_over_prob: Optional[float] = None
    fair_under_prob: Optional[float] = None
    odds_source: Optional[str] = None
    # No-vig EVs
    ev_over_novig: Optional[float] = None
    ev_under_novig: Optional[float] = None
    # Consensus vs side-specific line info for display
    consensus_total: Optional[float] = None
    best_side_total: Optional[float] = None
    line_diff: Optional[float] = None
    decision_line: Optional[float] = None
    decision_side: Optional[str] = None
    # Best price book names
    best_over_book: Optional[str] = None
    best_under_book: Optional[str] = None
    # Environment / lineup summaries for dashboard
    env_info: Optional[str] = None
    lineup_info: Optional[str] = None
    # Referee enrichment (crew assignments & scoring tendencies)
    ref_goals_gm: Optional[float] = None
    referee_crew: List[str] = field(default_factory=list)
    referee_avg_goals: Optional[float] = None
    referee_home_bias: Optional[float] = None
    referee_info: Optional[str] = None
    referee_source: Optional[str] = None
    ref_goal_adjustment: Optional[float] = None
    # Market velocity (change in total per hour), optional
    market_velocity: Optional[float] = None
    # Why we declined to make a wager (if recommendation == 'No Bet')
    no_bet_reason: Optional[str] = None
    # Whether the pick was forced/overridden after an initial conflict
    forced_recommendation: bool = False
    forced_reason: Optional[str] = None
