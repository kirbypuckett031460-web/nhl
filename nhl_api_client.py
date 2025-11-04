import time
from typing import Dict, List, Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False


class NHLStatsClient:
    """Thin wrapper around NHL Stats REST API: https://api.nhle.com/stats/rest/en

    This client exposes a couple of convenient methods for shots-related data.
    """

    BASE = "https://api.nhle.com/stats/rest/en"

    def __init__(self, rate_limit_sleep: float = 0.25, timeout: float = 10.0):
        self.enabled = REQUESTS_AVAILABLE
        self.session = requests.Session() if self.enabled else None
        if self.session:
            self.session.headers.update({
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            })
        self.rate_limit_sleep = rate_limit_sleep
        self.timeout = timeout

    def _get(self, path: str, params: Optional[Dict[str, str]] = None) -> Optional[Dict]:
        if not self.enabled:
            return None
        url = f"{self.BASE}/{path.lstrip('/') }"
        try:
            r = self.session.get(url, params=params or {}, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            return None
        finally:
            time.sleep(self.rate_limit_sleep)
        return None

    def team_game_log(self, season: str, team_abbrev: str, is_regular: bool = True) -> Optional[Dict]:
        """Fetch team game logs (shots, goals, etc.). season example: '20242025'"""
        path = "teamgamelog"
        params = {
            "cayenneExp": f"seasonId={season} and teamAbbrev='{team_abbrev}' and gameTypeId={(2 if is_regular else 3)}"
        }
        return self._get(path, params)

    def skater_summary(self, season: str, team_abbrev: Optional[str] = None) -> Optional[Dict]:
        """Basic skater summary; can filter by team via cayenneExp."""
        path = "skatersummary"
        exp = f"seasonId={season} and gameTypeId=2"
        if team_abbrev:
            exp += f" and teamAbbrev='{team_abbrev}'"
        return self._get(path, {"cayenneExp": exp})

    def skater_game_log(self, season: str, player_id: int) -> Optional[Dict]:
        path = "skatergamelog"
        exp = f"seasonId={season} and gameTypeId=2 and playerId={player_id}"
        return self._get(path, {"cayenneExp": exp})

    def find_players_by_team(self, season: str, team_abbrev: str) -> List[Dict]:
        """Return a list of skaters with shots/game and SOG totals for a team."""
        data = self.skaters_by_team(season, team_abbrev)
        return data if data else []

    def skaters_by_team(self, season: str, team_abbrev: str) -> Optional[List[Dict]]:
        res = self.skater_summary(season, team_abbrev)
        if not res or "data" not in res:
            return None
        return res["data"]

