"""Data acquisition helpers for NHL modeling."""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests


class NHLDataFetcher:
    """Fetches real NHL data from multiple sources with fallbacks."""

    def __init__(self) -> None:
        # Prefer new NHL web API
        self.base_url = "https://api-web.nhle.com/v1"
        # Legacy stats API kept as fallback reference
        self.legacy_url = "https://statsapi.web.nhl.com/api/v1"
        self.backup_url = "https://api.nhle.com/stats/rest/en"
        self.teams_cache: Optional[Dict[int, Dict]] = None

    def get_teams(self) -> Dict[int, Dict]:
        """Get all NHL teams with multiple fallback sources."""
        if self.teams_cache is None:
            # Try primary NHL API
            try:
                print("📡 Trying NHL statsapi...")
                response = requests.get(f"{self.base_url}/teams", timeout=10)
                response.raise_for_status()
                teams_data = response.json()
                self.teams_cache = {team['id']: team for team in teams_data['teams']}
                print("✅ Successfully fetched teams from NHL statsapi")
                return self.teams_cache
            except Exception as exc:
                print(f"⚠️  NHL statsapi failed: {exc}")

            # Use hardcoded fallback
            print("📋 Using hardcoded team data...")
            self.teams_cache = self._get_fallback_teams()

        return self.teams_cache

    def _get_fallback_teams(self) -> Dict[int, Dict]:
        """Comprehensive fallback team data."""
        return {
            1: {'id': 1, 'name': 'New Jersey Devils', 'abbreviation': 'NJD'},
            2: {'id': 2, 'name': 'New York Islanders', 'abbreviation': 'NYI'},
            3: {'id': 3, 'name': 'New York Rangers', 'abbreviation': 'NYR'},
            4: {'id': 4, 'name': 'Philadelphia Flyers', 'abbreviation': 'PHI'},
            5: {'id': 5, 'name': 'Pittsburgh Penguins', 'abbreviation': 'PIT'},
            6: {'id': 6, 'name': 'Boston Bruins', 'abbreviation': 'BOS'},
            7: {'id': 7, 'name': 'Buffalo Sabres', 'abbreviation': 'BUF'},
            8: {'id': 8, 'name': 'Montreal Canadiens', 'abbreviation': 'MTL'},
            9: {'id': 9, 'name': 'Ottawa Senators', 'abbreviation': 'OTT'},
            10: {'id': 10, 'name': 'Toronto Maple Leafs', 'abbreviation': 'TOR'},
            12: {'id': 12, 'name': 'Carolina Hurricanes', 'abbreviation': 'CAR'},
            13: {'id': 13, 'name': 'Florida Panthers', 'abbreviation': 'FLA'},
            14: {'id': 14, 'name': 'Tampa Bay Lightning', 'abbreviation': 'TBL'},
            15: {'id': 15, 'name': 'Washington Capitals', 'abbreviation': 'WSH'},
            16: {'id': 16, 'name': 'Chicago Blackhawks', 'abbreviation': 'CHI'},
            17: {'id': 17, 'name': 'Detroit Red Wings', 'abbreviation': 'DET'},
            18: {'id': 18, 'name': 'Nashville Predators', 'abbreviation': 'NSH'},
            19: {'id': 19, 'name': 'St. Louis Blues', 'abbreviation': 'STL'},
            20: {'id': 20, 'name': 'Calgary Flames', 'abbreviation': 'CGY'},
            21: {'id': 21, 'name': 'Colorado Avalanche', 'abbreviation': 'COL'},
            22: {'id': 22, 'name': 'Edmonton Oilers', 'abbreviation': 'EDM'},
            23: {'id': 23, 'name': 'Vancouver Canucks', 'abbreviation': 'VAN'},
            24: {'id': 24, 'name': 'Anaheim Ducks', 'abbreviation': 'ANA'},
            25: {'id': 25, 'name': 'Dallas Stars', 'abbreviation': 'DAL'},
            26: {'id': 26, 'name': 'Los Angeles Kings', 'abbreviation': 'LAK'},
            28: {'id': 28, 'name': 'San Jose Sharks', 'abbreviation': 'SJS'},
            29: {'id': 29, 'name': 'Columbus Blue Jackets', 'abbreviation': 'CBJ'},
            30: {'id': 30, 'name': 'Minnesota Wild', 'abbreviation': 'MIN'},
            52: {'id': 52, 'name': 'Winnipeg Jets', 'abbreviation': 'WPG'},
            53: {'id': 53, 'name': 'Utah Mammoth', 'abbreviation': 'UTA'},
            54: {'id': 54, 'name': 'Vegas Golden Knights', 'abbreviation': 'VGK'},
            55: {'id': 55, 'name': 'Seattle Kraken', 'abbreviation': 'SEA'}
        }

    def get_schedule(self, start_date: str, end_date: str) -> List[Dict]:
        """Get NHL schedule using api-web.nhle.com with robust parsing and date iteration."""

        def normalize_game(game: Dict, expected_date: Optional[str] = None) -> Optional[Dict]:
            if not isinstance(game, dict):
                return None
            game_id = game.get('id') or game.get('gameId') or game.get('gamePk')
            start_time = game.get('startTimeUTC') or game.get('gameDate') or game.get('startTime')
            if (start_time is None or start_time == "") and expected_date:
                start_time = f"{expected_date}T00:00:00Z"
            home = game.get('homeTeam') or {}
            away = game.get('awayTeam') or {}
            home_abbr = home.get('abbrev') or home.get('abbreviation') or home.get('triCode') or home.get('name')
            away_abbr = away.get('abbrev') or away.get('abbreviation') or away.get('triCode') or away.get('name')

            def _score(val: Dict) -> Optional[int]:
                try:
                    score_val = val.get('score')
                    return int(score_val) if score_val is not None else None
                except Exception:
                    return None

            home_score = _score(home)
            away_score = _score(away)
            venue_name = game.get('venue') or game.get('venueName') or (game.get('venue', {}) if isinstance(game.get('venue'), str) else None)
            state = game.get('gameState') or game.get('status') or ''
            state_map = {
                'FUT': 'Scheduled', 'PRE': 'Pre-Game', 'LIVE': 'In Progress', 'CRIT': 'In Progress',
                'FINAL': 'Final', 'OFF': 'Final', 'POSTPONED': 'Postponed'
            }
            detailed = state_map.get(str(state).upper(), str(state))
            try:
                return {
                    'gamePk': int(game_id) if game_id is not None else None,
                    'gameDate': start_time,
                    'status': {'detailedState': detailed},
                    'teams': {
                        'home': {'team': {'abbreviation': home_abbr or 'HOME'}, 'score': home_score},
                        'away': {'team': {'abbreviation': away_abbr or 'AWAY'}, 'score': away_score}
                    },
                    'venue': {'name': venue_name if isinstance(venue_name, str) else (venue_name or {}).get('name', '')}
                }
            except Exception:
                return None

        def fetch_date(date_str: str) -> List[Dict]:
            urls = [
                f"{self.base_url}/schedule/{date_str}",
                f"{self.base_url}/scoreboard/{date_str}",
            ]
            for url in urls:
                try:
                    resp = requests.get(url, timeout=15)
                    resp.raise_for_status()
                    data = resp.json()
                    games_raw: List[Dict] = []
                    if isinstance(data, dict):
                        if 'games' in data and isinstance(data['games'], list):
                            games_raw = data['games']
                        elif 'gameWeek' in data and isinstance(data['gameWeek'], list):
                            for day in data['gameWeek']:
                                games_raw.extend(day.get('games', []))
                        elif 'dates' in data and isinstance(data['dates'], list):
                            for entry in data['dates']:
                                games_raw.extend(entry.get('games', []))
                    normalized = []
                    for raw_game in games_raw:
                        normalized_game = normalize_game(raw_game, expected_date=date_str)
                        if normalized_game is not None:
                            normalized.append(normalized_game)
                    if normalized:
                        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        tz_str = os.getenv('SCHEDULE_TZ', 'US/Eastern')
                        filtered = []
                        for ng in normalized:
                            try:
                                gd = pd.to_datetime(ng.get('gameDate'), utc=True, errors='coerce')
                                if pd.isna(gd):
                                    continue
                                try:
                                    local_date = gd.tz_convert(tz_str).date()
                                except Exception:
                                    local_date = gd.tz_convert(None).date()
                                if local_date == target_date:
                                    filtered.append(ng)
                            except Exception:
                                continue
                        if filtered:
                            return filtered
                except Exception:
                    continue
            return []

        print(f"📡 Fetching schedule from {start_date} to {end_date}...")
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
        except Exception:
            start = datetime.now()
            end = start

        all_games: List[Dict] = []
        dt = start
        while dt <= end:
            date_str = dt.strftime('%Y-%m-%d')
            day_games = fetch_date(date_str)
            all_games.extend(day_games)
            dt += timedelta(days=1)

        if all_games:
            print(f"✅ Found {len(all_games)} games from new NHL API")
            return all_games

        print("❌ All NHL API sources failed")
        return []

    def get_game_stats(self, game_id: int) -> Dict:
        """Get detailed game statistics with fallbacks."""
        try:
            url = f"{self.base_url}/gamecenter/{game_id}/boxscore"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception:
            try:
                url = f"{self.legacy_url}/game/{game_id}/boxscore"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            except Exception:
                return {}

    def get_team_powerplay_stats(self, game_id: int) -> Dict[int, Dict]:
        """Fetch team-level power play stats for a single game."""
        try:
            response = requests.get(
                f"{self.backup_url}/team/powerplay",
                params={'cayenneExp': f"gameId={game_id}"},
                timeout=10
            )
            response.raise_for_status()
            entries = response.json().get('data', [])
            results: Dict[int, Dict] = {}
            for entry in entries:
                team_id = entry.get('teamId')
                if team_id is None:
                    continue
                try:
                    results[int(team_id)] = entry
                except (TypeError, ValueError):
                    continue
            return results
        except Exception:
            return {}
