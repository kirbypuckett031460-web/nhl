import os
import time
from typing import Dict, List, Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False


class TheOddsAPIClient:
    """Minimal client for The Odds API: https://the-odds-api.com

    We focus on NHL player SOG markets. If The Odds API's schema for player props
    differs by book, you may need mapping here. This is a best-effort generic client.
    """

    BASE = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0, rate_limit_sleep: float = 0.3):
        self.enabled = REQUESTS_AVAILABLE
        self.session = requests.Session() if self.enabled else None
        self.api_key = api_key or os.environ.get("THE_ODDS_API_KEY")
        self.timeout = timeout
        self.rate_limit_sleep = rate_limit_sleep

    def _get(self, path: str, params: Optional[Dict[str, str]] = None) -> Optional[List[Dict]]:
        if not self.enabled or not self.api_key:
            return None
        url = f"{self.BASE}/{path.lstrip('/') }"
        q = {"apiKey": self.api_key}
        if params:
            q.update(params)
        try:
            r = self.session.get(url, params=q, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            return None
        finally:
            time.sleep(self.rate_limit_sleep)
        return None

    def player_props(self, market: str = "player_shots_on_goal", regions: str = "us", odds_format: str = "american") -> Optional[List[Dict]]:
        """Fetch player prop markets for NHL.

        market: e.g., 'player_shots_on_goal'
        regions: comma-separated region codes
        """
        # Docs use sport key: 'icehockey_nhl'
        path = f"sports/icehockey_nhl/odds"
        params = {
            "regions": regions,
            "oddsFormat": odds_format,
            "markets": market,
            "dateFormat": "iso",
        }
        return self._get(path, params)

    @staticmethod
    def extract_sog_lines(api_response: Optional[List[Dict]]) -> Dict[str, float]:
        """Extract a map: 'Player Name' -> line (float) using median across books if multiple.

        This is heuristic and may need adjustment depending on response formats.
        """
        lines: Dict[str, List[float]] = {}
        if not api_response:
            return {}
        try:
            for event in api_response:
                # Each event includes bookmakers -> markets -> outcomes
                bookmakers = event.get("bookmakers", [])
                for bk in bookmakers:
                    for mk in bk.get("markets", []):
                        # Player SOG market typically has outcomes per player with 'name' and 'point'
                        for oc in mk.get("outcomes", []):
                            name = oc.get("description") or oc.get("name") or ""
                            point = oc.get("point")
                            if name and point is not None:
                                lines.setdefault(name, []).append(float(point))
            # Median per player
            return {name: sorted(vals)[len(vals)//2] for name, vals in lines.items() if vals}
        except Exception:
            return {}

