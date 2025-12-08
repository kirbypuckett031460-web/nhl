import io
import json
import os
import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# Optional imports - will work without these if not available
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("requests library not available. Some data sources will be disabled.")

try:
    from bs4 import BeautifulSoup

    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("BeautifulSoup not available. Web scraping will be disabled.")

try:
    from nhl_model.common import TEAM_ABBREV_TO_NAME, TEAM_NAME_TO_ABBREV
except Exception:
    TEAM_ABBREV_TO_NAME = {}
    TEAM_NAME_TO_ABBREV = {}


def _normalize_team_abbr(value: Optional[str]) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    up = raw.upper()
    if up in TEAM_ABBREV_TO_NAME:
        return up
    return TEAM_NAME_TO_ABBREV.get(up, up)


def _season_from_date(dt: datetime) -> int:
    """Return NHL season end year for a given datetime."""
    return dt.year + 1 if dt.month >= 9 else dt.year


def _toi_to_minutes(value: Optional[str]) -> float:
    """Convert a time-on-ice string (MM:SS) to minutes as float."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).strip().split(":")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        minutes, seconds = parts
        total_seconds = (int(minutes) * 60) + int(seconds)
        return total_seconds / 60.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _serialize_combos(forward_lines: List[List[str]], defense_pairs: List[List[str]]) -> str:
    """Serialize on-ice combinations into a compact string for persistence."""
    parts: List[str] = []
    for idx, line in enumerate(forward_lines, start=1):
        if len(line) == 3:
            parts.append(f"L{idx}:{'-'.join(line)}")
    for idx, pair in enumerate(defense_pairs, start=1):
        if len(pair) == 2:
            parts.append(f"D{idx}:{'-'.join(pair)}")
    return " | ".join(parts)


class GameStatsCache:
    """
    File-backed cache that stores up to a rolling year of NHL game stats so
    repeated model runs only need to download the latest deltas.
    """

    def __init__(self, cache_dir: str = "data/cache", window_days: int = 365):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_csv = self.cache_dir / "nhl_games_cache.csv"
        self.manifest_path = self.cache_dir / "nhl_games_manifest.json"
        self.team_stats_csv = self.cache_dir / "team_stats_per_game.csv"
        self.player_stats_csv = self.cache_dir / "player_stats_per_game.csv"
        self.window_days = window_days

    def load(self) -> pd.DataFrame:
        if self.cache_csv.exists():
            try:
                df = pd.read_csv(self.cache_csv)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                return df
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def _load_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            return pd.DataFrame()

    def load_team_stats(self) -> pd.DataFrame:
        return self._load_csv(self.team_stats_csv)

    def load_player_stats(self) -> pd.DataFrame:
        return self._load_csv(self.player_stats_csv)

    def latest_cached_date(self, df: pd.DataFrame) -> Optional[datetime]:
        if df.empty or "date" not in df.columns:
            return None
        return pd.to_datetime(df["date"]).max()

    def trim_window(self, df: pd.DataFrame, reference_date: datetime) -> pd.DataFrame:
        if df.empty or "date" not in df.columns:
            return df
        cutoff = pd.to_datetime(reference_date) - pd.Timedelta(days=self.window_days)
        trimmed = df[df["date"] >= cutoff].copy()
        return trimmed.reset_index(drop=True)

    def merge(self, cached: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
        if cached.empty:
            combined = fresh.copy()
        elif fresh.empty:
            combined = cached.copy()
        else:
            combined = pd.concat([cached, fresh], ignore_index=True)
        if combined.empty:
            return combined
        # Deduplicate by core identifiers
        subset_cols = [col for col in ["date", "home_team", "away_team"] if col in combined.columns]
        if subset_cols:
            combined = (
                combined.sort_values("date")
                .drop_duplicates(subset=subset_cols, keep="last")
                .reset_index(drop=True)
            )
        return combined

    def persist(self, df: pd.DataFrame) -> None:
        df.to_csv(self.cache_csv, index=False)
        has_dates = not df.empty and "date" in df.columns
        manifest = {
            "last_refresh": datetime.utcnow().isoformat(),
            "rows": int(len(df)),
            "min_date": df["date"].min().strftime("%Y-%m-%d") if has_dates else None,
            "max_date": df["date"].max().strftime("%Y-%m-%d") if has_dates else None,
            "window_days": self.window_days,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    @staticmethod
    def _merge_by_keys(existing: pd.DataFrame, fresh: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        if existing.empty:
            combined = fresh.copy()
        elif fresh.empty:
            combined = existing.copy()
        else:
            combined = pd.concat([existing, fresh], ignore_index=True)
        if combined.empty or not keys:
            return combined
        return combined.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)

    def merge_team_stats(self, existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
        return self._merge_by_keys(existing, fresh, ["game_id", "team"])

    def merge_player_stats(self, existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
        keys = ["game_id", "player_id"] if "player_id" in fresh.columns else ["game_id", "team", "player"]
        return self._merge_by_keys(existing, fresh, keys)

    def _persist_aux(self, df: pd.DataFrame, path: Path) -> None:
        if df.empty:
            return
        to_write = df.copy()
        if "date" in to_write.columns:
            to_write["date"] = pd.to_datetime(to_write["date"]).dt.strftime("%Y-%m-%d")
        to_write.to_csv(path, index=False)

    def persist_team_stats(self, df: pd.DataFrame) -> None:
        self._persist_aux(df, self.team_stats_csv)

    def persist_player_stats(self, df: pd.DataFrame) -> None:
        self._persist_aux(df, self.player_stats_csv)


class MoneyPuckGameFeed:
    """Lazy loader for MoneyPuck game-by-game team and player stats."""

    BASE_URL = "https://moneypuck.com/moneypuck/playerData/gameByGame/{season}/{stage}/{dataset}.csv"
    DATASETS = ("teams", "skaters")

    def __init__(self, cache_dir: Path, stage: str = "regular"):
        self.base_cache = Path(cache_dir) / "moneypuck"
        self.base_cache.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
        if self.session:
            self.session.headers.update(
                {
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "text/csv,application/json",
                }
            )
        self.enabled = self.session is not None
        self._season_payloads: Dict[Tuple[int, str], Dict[str, Dict]] = {}

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(col).strip().lower() for col in df.columns]
        return df

    @staticmethod
    def _find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
        lowered = [col.lower() for col in columns]
        for candidate in candidates:
            cand = candidate.lower()
            if cand in lowered:
                return columns[lowered.index(cand)]
        for candidate in candidates:
            cand = candidate.lower()
            for original, lower_val in zip(columns, lowered):
                if cand in lower_val:
                    return original
        return None

    @staticmethod
    def _safe_numeric(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def ensure_season(self, season: int) -> None:
        if not self.enabled:
            raise RuntimeError("MoneyPuck feed requires requests library.")
        key = (season, self.stage)
        if key in self._season_payloads:
            return
        payload: Dict[str, Dict] = {"team_lookup": {}, "player_lookup": defaultdict(list)}
        for dataset in self.DATASETS:
            df = self._load_dataset(season, dataset)
            if dataset == "teams":
                payload["team_lookup"] = self._build_team_lookup(df)
            elif dataset == "skaters":
                payload["player_lookup"] = self._build_player_lookup(df)
        self._season_payloads[key] = payload

    def _dataset_cache_path(self, season: int, dataset: str) -> Path:
        filename = f"{dataset}_{season}_{self.stage}.csv"
        return self.base_cache / filename

    def _load_dataset(self, season: int, dataset: str) -> pd.DataFrame:
        cache_path = self._dataset_cache_path(season, dataset)
        if cache_path.exists():
            return pd.read_csv(cache_path)
        url = self.BASE_URL.format(season=season, stage=self.stage, dataset=dataset)
        response = self.session.get(url, timeout=25)
        response.raise_for_status()
        cache_path.write_text(response.text, encoding="utf-8")
        return pd.read_csv(io.StringIO(response.text))

    def _build_team_lookup(self, df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, Optional[float]]]:
        if df.empty:
            return {}
        data = self._normalize_columns(df)
        team_col = self._find_column(data.columns.tolist(), ["team", "teamname", "team_name"])
        game_col = self._find_column(data.columns.tolist(), ["game_id", "gameid", "game"])
        date_col = self._find_column(data.columns.tolist(), ["date", "game_date"])
        shots_col = self._find_column(data.columns.tolist(), ["shotsongoalfor", "shots_on_goal_for", "shotsfor"])
        xg_col = self._find_column(data.columns.tolist(), ["xgoalsfor", "xgfor", "expectedgoalsfor"])
        pen_draw_col = self._find_column(data.columns.tolist(), ["penaltiesdrawn", "penalties_drawn"])
        pen_taken_col = self._find_column(data.columns.tolist(), ["penaltiestaken", "penalties_taken"])
        if not all([team_col, game_col, shots_col, xg_col, pen_draw_col, pen_taken_col]):
            raise RuntimeError("MoneyPuck team dataset missing required columns.")
        lookup: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
        for _, row in data.iterrows():
            team_abbr = _normalize_team_abbr(row.get(team_col))
            game_id = str(row.get(game_col))
            if not team_abbr or not game_id:
                continue
            entry = {
                "shots": self._safe_numeric(row.get(shots_col)),
                "xg": self._safe_numeric(row.get(xg_col)),
                "penalties_drawn": self._safe_numeric(row.get(pen_draw_col)),
                "penalties_taken": self._safe_numeric(row.get(pen_taken_col)),
                "date": row.get(date_col),
            }
            lookup[(game_id, team_abbr)] = entry
        return lookup

    def _build_player_lookup(self, df: pd.DataFrame) -> Dict[Tuple[str, str], List[Dict]]:
        if df.empty:
            return defaultdict(list)
        data = self._normalize_columns(df)
        team_col = self._find_column(data.columns.tolist(), ["team", "teamname", "team_name"])
        game_col = self._find_column(data.columns.tolist(), ["game_id", "gameid", "game"])
        player_col = self._find_column(data.columns.tolist(), ["player", "player_name"])
        player_id_col = self._find_column(data.columns.tolist(), ["player_id", "playerid"])
        position_col = self._find_column(data.columns.tolist(), ["position", "positioncode"])
        toi_col = self._find_column(data.columns.tolist(), ["icetime", "time_on_ice"])
        shots_col = self._find_column(data.columns.tolist(), ["shots", "shots_on_goal"])
        xg_col = self._find_column(data.columns.tolist(), ["xgoals", "xg"])
        pen_draw_col = self._find_column(data.columns.tolist(), ["penaltiesdrawn", "penalties_drawn"])
        pen_taken_col = self._find_column(data.columns.tolist(), ["penaltiestaken", "penalties_taken"])
        lookup: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        for _, row in data.iterrows():
            team_abbr = _normalize_team_abbr(row.get(team_col))
            game_id = str(row.get(game_col))
            player_name = str(row.get(player_col) or "").strip()
            if not team_abbr or not game_id or not player_name:
                continue
            lookup[(game_id, team_abbr)].append(
                {
                    "player": player_name,
                    "player_id": row.get(player_id_col) or "",
                    "position": str(row.get(position_col) or "").strip().upper(),
                    "time_on_ice": row.get(toi_col),
                    "shots": self._safe_numeric(row.get(shots_col)),
                    "xg": self._safe_numeric(row.get(xg_col)),
                    "penalties_drawn": self._safe_numeric(row.get(pen_draw_col)),
                    "penalties_taken": self._safe_numeric(row.get(pen_taken_col)),
                }
            )
        return lookup

    def _get_payload(self, season: int) -> Dict[str, Dict]:
        key = (season, self.stage)
        payload = self._season_payloads.get(key)
        if payload is None:
            raise RuntimeError(f"MoneyPuck season {season} not loaded")
        return payload

    def team_entry(self, season: int, game_id: str, team_abbr: str) -> Optional[Dict]:
        payload = self._get_payload(season)
        return payload["team_lookup"].get((str(game_id), _normalize_team_abbr(team_abbr)))

    def player_entries(self, season: int, game_id: str, team_abbr: str) -> List[Dict]:
        payload = self._get_payload(season)
        return payload["player_lookup"].get((str(game_id), _normalize_team_abbr(team_abbr)), [])


def _annotate_line_assignments(player_rows: List[Dict]) -> Tuple[List[Dict], str]:
    """Assign simple line/pair designations based on time on ice."""

    def _copy_rows(rows: List[Dict]) -> List[Dict]:
        return [{**row} for row in rows]

    forwards = []
    defense = []
    others = []
    for row in player_rows:
        pos = row.get("position", "")
        if pos in {"C", "LW", "RW", "F"}:
            forwards.append(row)
        elif pos in {"D", "LD", "RD"}:
            defense.append(row)
        else:
            others.append(row)

    forwards.sort(key=lambda r: _toi_to_minutes(r.get("time_on_ice")), reverse=True)
    defense.sort(key=lambda r: _toi_to_minutes(r.get("time_on_ice")), reverse=True)

    annotated = _copy_rows(forwards + defense + others)
    # Map name to annotated row for easy updates.
    row_map = {(row.get("player"), row.get("position"), row.get("player_id")): row for row in annotated}

    forward_lines: List[List[str]] = []
    defense_pairs: List[List[str]] = []

    def _assign(rows: List[Dict], group_size: int, prefix: str, sink: List[List[str]]) -> None:
        for idx in range(0, len(rows), group_size):
            group = rows[idx : idx + group_size]
            if len(group) < group_size:
                continue
            label = f"{prefix}{(idx // group_size) + 1}"
            names = []
            for player in group:
                key = (player.get("player"), player.get("position"), player.get("player_id"))
                target = row_map.get(key)
                if target is not None:
                    target["line_role"] = label
                names.append(player.get("player", ""))
            sink.append(names)

    _assign(forwards, 3, "L", forward_lines)
    _assign(defense, 2, "D", defense_pairs)

    combo_str = _serialize_combos(forward_lines, defense_pairs)
    return annotated, combo_str

class RealNHLDataCollector:
    def __init__(self):
        self.session = None
        if REQUESTS_AVAILABLE:
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
        self.cache = GameStatsCache()
        self.team_record_buffer: List[Dict] = []
        self.player_record_buffer: List[Dict] = []
        self.money_puck = MoneyPuckGameFeed(self.cache.cache_dir)
    
    def get_real_nhl_data(self, start_date='2024-10-10', end_date='2024-12-15', max_games=500):
        """
        Main function to get real NHL data from multiple sources with on-disk caching.
        """
        print("🏒 NHL Real Data Collector Starting...")
        print(f"📅 Requested Date Range: {start_date} to {end_date}")

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        self.team_record_buffer = []
        self.player_record_buffer = []
        self._ensure_money_puck_coverage(start_dt, end_dt)

        cached_df = self.cache.load()
        cached_df = self.cache.trim_window(cached_df, end_dt)
        latest_cached = self.cache.latest_cached_date(cached_df)

        fetch_start_dt = start_dt
        if latest_cached is not None:
            fetch_start_dt = max(start_dt, latest_cached + timedelta(days=1))

        data_sources = [
            ("NHL Official API", self._get_from_nhl_api),
            ("ESPN API", self._get_from_espn_api),
            ("Local CSV", self._get_from_csv),
            ("Hockey Reference", self._get_from_hockey_reference),
        ]

        new_data = pd.DataFrame()
        if fetch_start_dt <= end_dt:
            effective_start = fetch_start_dt.strftime('%Y-%m-%d')
            print(f"🗂️  Cache latest date: {latest_cached.strftime('%Y-%m-%d') if latest_cached else 'None'}")
            print(f"📤 Fetching deltas starting {effective_start}...")
            for source_name, source_func in data_sources:
                try:
                    print(f"\n🔄 Trying {source_name}...")
                    df = source_func(effective_start, end_date, max_games)
                    if not df.empty and len(df) >= 1:
                        print(f"✅ Retrieved {len(df)} games from {source_name}")
                        new_data = df
                        break
                    else:
                        print(f"❌ {source_name} returned insufficient data")
                except Exception as e:
                    print(f"❌ {source_name} failed: {str(e)[:100]}...")
                    continue
        else:
            print("🆗 Cache already includes requested date range. No download needed.")

        if new_data.empty and cached_df.empty:
            print("\n⚠️  All real data sources failed and cache is empty. Generating enhanced sample data...")
            return self._create_enhanced_sample_data(max_games)

        combined_df = self.cache.merge(cached_df, new_data)
        combined_df = self._clean_and_validate_data(combined_df)
        combined_df = self.cache.trim_window(combined_df, end_dt)

        if not combined_df.empty:
            self.cache.persist(combined_df)
        self._persist_stat_buffers()

        filtered = combined_df[
            (combined_df['date'] >= start_dt) & (combined_df['date'] <= end_dt)
        ].copy() if not combined_df.empty else pd.DataFrame()

        if filtered.empty:
            print("\n⚠️  No games found in the requested window. Falling back to cache contents.")
            filtered = combined_df.copy()

        if filtered.empty:
            print("\n⚠️  Cache contains no usable data. Generating enhanced sample data...")
            return self._create_enhanced_sample_data(max_games)

        filtered = filtered.sort_values('date')
        if max_games and len(filtered) > max_games:
            filtered = (
                filtered.sort_values('date', ascending=False)
                .head(max_games)
                .sort_values('date')
                .reset_index(drop=True)
            )
        else:
            filtered = filtered.reset_index(drop=True)

        print(f"📦 Returning {len(filtered)} games (cache + deltas).")
        return filtered
    
    def _get_from_nhl_api(self, start_date, end_date, max_games):
        """Get data from NHL's official API"""
        if not REQUESTS_AVAILABLE:
            raise Exception("requests library not available")
        
        games_data = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current_date <= end_dt and len(games_data) < max_games:
            date_str = current_date.strftime('%Y-%m-%d')
            
            try:
                # NHL's new API endpoint
                url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    daily_games = self._parse_nhl_api_response(data, date_str)
                    games_data.extend(daily_games)
                    
                    if daily_games:
                        print(f"  📊 {date_str}: {len(daily_games)} games")
                
                time.sleep(0.3)  # Be respectful to API
                
            except Exception as e:
                print(f"  ⚠️  Error for {date_str}: {str(e)[:50]}...")
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(games_data)
    
    def _parse_nhl_api_response(self, data, date_str):
        """Parse NHL API response"""
        games = []
        
        if 'gameWeek' in data:
            for week in data['gameWeek']:
                for day in week.get('games', []):
                    for game in day.get('games', []):
                        if game.get('gameState') in ['OFF', 'FINAL']:  # Completed games
                            try:
                                game_info = self._extract_game_info(game, date_str)
                                if game_info:
                                    games.append(game_info)
                            except Exception as e:
                                continue
        
        return games
    
    def _extract_game_info(self, game, date_str):
        """Extract game information from API response"""
        try:
            home_team = game['homeTeam']['abbrev']
            away_team = game['awayTeam']['abbrev']
            
            # Get basic game info
            game_info = {
                'date': pd.to_datetime(date_str),
                'home_team': home_team,
                'away_team': away_team,
                'venue': game.get('venue', {}).get('default', f"{home_team} Arena"),
                'home_goals': game['homeTeam'].get('score', 0),
                'away_goals': game['awayTeam'].get('score', 0),
            }
            
            game_info['total_goals'] = game_info['home_goals'] + game_info['away_goals']
            
            game_id = game.get('id')
            if not game_id:
                return None

            verified_stats = self._build_verified_stat_block(
                game_id=game_id,
                game_date=game_info['date'],
                home_team=home_team,
                away_team=away_team,
                home_goals=game_info['home_goals'],
                away_goals=game_info['away_goals']
            )
            if not verified_stats:
                return None

            game_info['game_id'] = str(game_id)
            game_info.update(verified_stats)
            return game_info
            
        except Exception as e:
            return None
    
    def _ensure_money_puck_coverage(self, start_dt: datetime, end_dt: datetime) -> None:
        seasons: set[int] = set()
        cursor = start_dt
        while cursor <= end_dt:
            seasons.add(_season_from_date(cursor))
            cursor += timedelta(days=30)
        seasons.add(_season_from_date(end_dt))
        for season in sorted(seasons):
            try:
                self.money_puck.ensure_season(season)
            except RuntimeError as exc:
                raise RuntimeError(f"MoneyPuck stats unavailable for season {season}: {exc}") from exc

    def _record_team_stats(
        self,
        game_id: str,
        game_date: datetime,
        home_team: str,
        away_team: str,
        home_entry: Dict,
        away_entry: Dict,
        box_stats: Dict,
        home_combos: str,
        away_combos: str,
    ) -> None:
        home_team = _normalize_team_abbr(home_team)
        away_team = _normalize_team_abbr(away_team)
        base = {
            "game_id": game_id,
            "date": game_date,
        }
        self.team_record_buffer.append(
            {
                **base,
                "team": home_team,
                "opponent": away_team,
                "is_home": True,
                "shots": box_stats.get("home_shots"),
                "xg": home_entry.get("xg"),
                "penalties_drawn": home_entry.get("penalties_drawn"),
                "penalties_taken": home_entry.get("penalties_taken"),
                "pp_goals": box_stats.get("home_pp_goals"),
                "pp_opps": box_stats.get("home_pp_opps"),
                "on_ice_combos": home_combos,
            }
        )
        self.team_record_buffer.append(
            {
                **base,
                "team": away_team,
                "opponent": home_team,
                "is_home": False,
                "shots": box_stats.get("away_shots"),
                "xg": away_entry.get("xg"),
                "penalties_drawn": away_entry.get("penalties_drawn"),
                "penalties_taken": away_entry.get("penalties_taken"),
                "pp_goals": box_stats.get("away_pp_goals"),
                "pp_opps": box_stats.get("away_pp_opps"),
                "on_ice_combos": away_combos,
            }
        )

    def _record_player_stats(
        self,
        game_id: str,
        game_date: datetime,
        team: str,
        players: List[Dict],
    ) -> None:
        team = _normalize_team_abbr(team)
        for player in players:
            self.player_record_buffer.append(
                {
                    "game_id": game_id,
                    "date": game_date,
                    "team": team,
                    "player": player.get("player"),
                    "player_id": str(player.get("player_id") or "").strip(),
                    "position": player.get("position"),
                    "line_role": player.get("line_role"),
                    "time_on_ice_minutes": _toi_to_minutes(player.get("time_on_ice")),
                    "shots": player.get("shots"),
                    "xg": player.get("xg"),
                    "penalties_drawn": player.get("penalties_drawn"),
                    "penalties_taken": player.get("penalties_taken"),
                }
            )

    def _build_verified_stat_block(
        self,
        game_id: int,
        game_date: datetime,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
    ) -> Optional[Dict]:
        """Combine NHL boxscore + MoneyPuck stats. Returns None if incomplete."""
        box_stats = self._get_boxscore_stats(game_id)
        required_box = ['home_pp_goals', 'away_pp_goals', 'home_pp_opps', 'away_pp_opps']
        if any(key not in box_stats for key in required_box):
            return None

        season = _season_from_date(pd.to_datetime(game_date).to_pydatetime())
        home_entry = self.money_puck.team_entry(season, game_id, home_team)
        away_entry = self.money_puck.team_entry(season, game_id, away_team)
        if not home_entry or not away_entry:
            return None

        home_players = self.money_puck.player_entries(season, game_id, home_team)
        away_players = self.money_puck.player_entries(season, game_id, away_team)
        if not home_players or not away_players:
            return None

        annotated_home, home_combos = _annotate_line_assignments(home_players)
        annotated_away, away_combos = _annotate_line_assignments(away_players)

        for col, entry in [('home_shots', home_entry), ('away_shots', away_entry)]:
            if box_stats.get(col) is None:
                shots_val = entry.get("shots")
                if shots_val is None:
                    return None
                box_stats[col] = int(shots_val)
            else:
                box_stats[col] = int(box_stats[col])

        box_stats['home_xg'] = home_entry.get('xg')
        box_stats['away_xg'] = away_entry.get('xg')
        box_stats['home_penalties_drawn'] = home_entry.get('penalties_drawn')
        box_stats['away_penalties_drawn'] = away_entry.get('penalties_drawn')
        box_stats['home_penalties_taken'] = home_entry.get('penalties_taken')
        box_stats['away_penalties_taken'] = away_entry.get('penalties_taken')
        box_stats['home_on_ice_combos'] = home_combos
        box_stats['away_on_ice_combos'] = away_combos

        numeric_required = [
            'home_shots', 'away_shots', 'home_xg', 'away_xg',
            'home_penalties_drawn', 'away_penalties_drawn',
            'home_penalties_taken', 'away_penalties_taken'
        ]
        if any(pd.isna(box_stats.get(col)) for col in numeric_required):
            return None

        box_stats['home_saves'] = max(0, box_stats['away_shots'] - away_goals)
        box_stats['away_saves'] = max(0, box_stats['home_shots'] - home_goals)

        self._record_team_stats(
            game_id=str(game_id),
            game_date=game_date,
            home_team=home_team,
            away_team=away_team,
            home_entry=home_entry,
            away_entry=away_entry,
            box_stats=box_stats,
            home_combos=home_combos,
            away_combos=away_combos,
        )
        self._record_player_stats(str(game_id), game_date, home_team, annotated_home)
        self._record_player_stats(str(game_id), game_date, away_team, annotated_away)

        return box_stats

    def _get_boxscore_stats(self, game_id):
        """Get detailed stats from game boxscore"""
        try:
            url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
            response = self.session.get(url, timeout=8)
            
            if response.status_code == 200:
                boxscore = response.json()
                return self._parse_boxscore(boxscore)
        except:
            pass
        
        return {}
    
    def _parse_boxscore(self, boxscore):
        """Parse boxscore data"""
        stats = {}
        
        try:
            home_team = boxscore.get('homeTeam', {})
            away_team = boxscore.get('awayTeam', {})
            
            # Shot data
            if home_team.get('sog') is not None:
                stats['home_shots'] = home_team.get('sog')
            if away_team.get('sog') is not None:
                stats['away_shots'] = away_team.get('sog')
            
            # Power play data
            if home_team.get('powerPlayGoals') is not None:
                stats['home_pp_goals'] = home_team.get('powerPlayGoals')
            if away_team.get('powerPlayGoals') is not None:
                stats['away_pp_goals'] = away_team.get('powerPlayGoals')
            if home_team.get('powerPlayOpportunities') is not None:
                stats['home_pp_opps'] = home_team.get('powerPlayOpportunities')
            if away_team.get('powerPlayOpportunities') is not None:
                stats['away_pp_opps'] = away_team.get('powerPlayOpportunities')
            
            # Goaltender info
            stats['home_goalie'] = self._get_starting_goalie(home_team)
            stats['away_goalie'] = self._get_starting_goalie(away_team)
            
        except Exception as e:
            pass
        
        return stats
    
    def _get_starting_goalie(self, team_stats):
        """Extract starting goaltender"""
        try:
            goalies = team_stats.get('goalies', [])
            if goalies:
                # Find goalie with most time played or saves
                starter = max(goalies, key=lambda g: g.get('savesAgainst', 0))
                return starter.get('lastName', 'Unknown')
        except:
            pass
        return 'Unknown'

    def _persist_stat_buffers(self) -> None:
        if self.team_record_buffer:
            team_df = pd.DataFrame(self.team_record_buffer)
            existing = self.cache.load_team_stats()
            merged = self.cache.merge_team_stats(existing, team_df)
            self.cache.persist_team_stats(merged)
            self.team_record_buffer = []
        if self.player_record_buffer:
            player_df = pd.DataFrame(self.player_record_buffer)
            existing_players = self.cache.load_player_stats()
            merged_players = self.cache.merge_player_stats(existing_players, player_df)
            self.cache.persist_player_stats(merged_players)
            self.player_record_buffer = []
    
    def _get_from_espn_api(self, start_date, end_date, max_games):
        """Get data from ESPN API (alternative source)"""
        if not REQUESTS_AVAILABLE:
            raise Exception("requests library not available")
        
        games_data = []
        
        # ESPN API endpoint for NHL
        url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
        
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current_date <= end_dt and len(games_data) < max_games:
            date_str = current_date.strftime('%Y%m%d')
            
            try:
                params = {'dates': date_str}
                response = self.session.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    daily_games = self._parse_espn_response(data)
                    games_data.extend(daily_games)
                    
                    if daily_games:
                        print(f"  📊 {current_date.strftime('%Y-%m-%d')}: {len(daily_games)} games")
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ⚠️  ESPN API error: {str(e)[:50]}...")
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(games_data)
    
    def _parse_espn_response(self, data):
        """Parse ESPN API response"""
        games = []
        
        try:
            for event in data.get('events', []):
                if event.get('status', {}).get('type', {}).get('completed', False):
                    game = self._extract_espn_game(event)
                    if game:
                        games.append(game)
        except Exception as e:
            pass
        
        return games
    
    def _extract_espn_game(self, event):
        """Extract game info from ESPN data"""
        try:
            competitions = event.get('competitions', [{}])[0]
            competitors = competitions.get('competitors', [])
            
            home_team = None
            away_team = None
            
            for comp in competitors:
                if comp.get('homeAway') == 'home':
                    home_team = comp
                elif comp.get('homeAway') == 'away':
                    away_team = comp
            
            if not home_team or not away_team:
                return None
            
            game_info = {
                'date': pd.to_datetime(event.get('date')),
                'home_team': self._convert_espn_team_name(home_team['team']['abbreviation']),
                'away_team': self._convert_espn_team_name(away_team['team']['abbreviation']),
                'venue': competitions.get('venue', {}).get('fullName', 'Unknown Arena'),
                'home_goals': int(home_team.get('score', 0)),
                'away_goals': int(away_team.get('score', 0)),
            }
            
            game_info['total_goals'] = game_info['home_goals'] + game_info['away_goals']
            game_id = event.get('id')
            try:
                parsed_game_id = int(game_id)
            except (TypeError, ValueError):
                parsed_game_id = None
            if not parsed_game_id:
                return None
            verified_stats = self._build_verified_stat_block(
                game_id=parsed_game_id,
                game_date=game_info['date'],
                home_team=game_info['home_team'],
                away_team=game_info['away_team'],
                home_goals=game_info['home_goals'],
                away_goals=game_info['away_goals']
            )
            if not verified_stats:
                return None
            game_info['game_id'] = str(parsed_game_id)
            game_info.update(verified_stats)
            return game_info
            
        except Exception as e:
            return None
    
    def _convert_espn_team_name(self, espn_abbrev):
        """Convert ESPN team abbreviations to standard format"""
        conversion_map = {
            'WSH': 'WSH', 'TB': 'TBL', 'VGK': 'VGK', 'LA': 'LAK',
            'SJ': 'SJS', 'NJ': 'NJD', 'NY': 'NYR', 'MON': 'MTL'
        }
        return conversion_map.get(espn_abbrev, espn_abbrev)
    
    def _get_from_csv(self, start_date, end_date, max_games):
        """Try to load data from local CSV files"""
        csv_files = [
            'nhl_games_2024.csv',
            'nhl_data.csv',
            'hockey_data.csv',
            'games.csv'
        ]
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    print(f"  📁 Found local file: {csv_file}")
                    
                    # Convert date column
                    date_cols = ['date', 'Date', 'game_date', 'DATE']
                    for col in date_cols:
                        if col in df.columns:
                            df['date'] = pd.to_datetime(df[col])
                            break
                    
                    # Filter by date range
                    if 'date' in df.columns:
                        start_dt = pd.to_datetime(start_date)
                        end_dt = pd.to_datetime(end_date)
                        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
                    required_cols = {
                        'game_id', 'home_team', 'away_team', 'home_goals', 'away_goals',
                        'home_shots', 'away_shots', 'home_pp_goals', 'away_pp_goals',
                        'home_pp_opps', 'away_pp_opps', 'home_xg', 'away_xg',
                        'home_penalties_drawn', 'away_penalties_drawn',
                        'home_penalties_taken', 'away_penalties_taken'
                    }
                    if not required_cols.issubset(set(map(str, df.columns))):
                        print(f"  ⚠️  Skipping {csv_file}: missing required stat columns")
                        continue
                    df['home_team'] = df['home_team'].apply(_normalize_team_abbr)
                    df['away_team'] = df['away_team'].apply(_normalize_team_abbr)
                    if 'total_goals' not in df.columns:
                        df['total_goals'] = df['home_goals'] + df['away_goals']
                    if 'home_saves' not in df.columns:
                        df['home_saves'] = df['away_shots'] - df['away_goals']
                    if 'away_saves' not in df.columns:
                        df['away_saves'] = df['home_shots'] - df['home_goals']
                    numeric_cols = [col for col in required_cols if col not in {'home_team', 'away_team', 'game_id'}]
                    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                    if df[numeric_cols].isna().any().any():
                        print(f"  ⚠️  Skipping {csv_file}: numeric columns contain NaN after coercion")
                        continue
                    df['game_id'] = df['game_id'].astype(str)
                    
                    # Limit games
                    if len(df) > max_games:
                        df = df.head(max_games)
                    
                    return df
                    
                except Exception as e:
                    print(f"  ⚠️  Error reading {csv_file}: {e}")
                    continue
        
        raise Exception("No valid CSV files found")
    
    def _get_from_hockey_reference(self, start_date, end_date, max_games):
        """Scrape data from Hockey Reference (simplified)"""
        if not REQUESTS_AVAILABLE or not BEAUTIFULSOUP_AVAILABLE:
            raise Exception("Required libraries not available for scraping")
        
        # This is a simplified version - in practice you'd need to handle
        # Hockey Reference's specific HTML structure and anti-scraping measures
        print("  ⚠️  Hockey Reference scraping not fully implemented")
        raise Exception("Hockey Reference scraping not available")
    
    def _create_enhanced_sample_data(self, max_games):
        """Create enhanced sample data based on real NHL patterns"""
        print("  🎲 Generating realistic sample data...")
        
        # Import the original sample data function
        from nhl_model import create_sample_data
        
        # Create base sample data
        df = create_sample_data()
        
        # Enhance it with more realistic patterns
        df = self._enhance_sample_realism(df)
        
        # Limit to requested number of games
        if len(df) > max_games:
            df = df.head(max_games)
        
        return df
    
    def _enhance_sample_realism(self, df):
        """Make sample data more realistic"""
        
        # Add more realistic date patterns (actual NHL schedule-like)
        start_date = datetime(2024, 10, 10)
        df['date'] = [start_date + timedelta(days=i//8) for i in range(len(df))]
        
        # Add realistic scoring patterns based on actual NHL data
        df['home_goals'] = np.random.poisson(3.1, len(df))
        df['away_goals'] = np.random.poisson(2.9, len(df))
        df['total_goals'] = df['home_goals'] + df['away_goals']
        
        # Add more realistic shot totals
        df['home_shots'] = np.random.normal(31.5, 4.2, len(df)).astype(int)
        df['away_shots'] = np.random.normal(30.8, 4.1, len(df)).astype(int)
        
        # Ensure minimum shots
        df['home_shots'] = df['home_shots'].apply(lambda x: max(x, 15))
        df['away_shots'] = df['away_shots'].apply(lambda x: max(x, 15))
        
        # Recalculate saves
        df['home_saves'] = df['away_shots'] - df['away_goals']
        df['away_saves'] = df['home_shots'] - df['home_goals']
        
        return df
    
    def _clean_and_validate_data(self, df):
        """Clean and validate the collected data"""
        print(f"  🧹 Cleaning data... ({len(df)} games)")
        
        # Remove games with missing critical data
        required_cols = ['date', 'home_team', 'away_team', 'home_goals', 'away_goals']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0 if 'goals' in col else 'Unknown'
        
        # Remove invalid games
        initial_count = len(df)
        df = df[df['home_team'] != df['away_team']]  # Teams can't play themselves
        df = df[df['home_goals'].notna() & df['away_goals'].notna()]  # Need valid scores
        
        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        final_count = len(df)
        if final_count < initial_count:
            print(f"  🗑️  Removed {initial_count - final_count} invalid games")
        
        print(f"  ✅ Final dataset: {final_count} clean games")
        return df

# Updated run_predictions.py to use real data
def run_with_real_data():
    """
    Updated version of run_predictions.py that uses real NHL data
    """
    from nhl_model import NHLOverUnderModel
    
    print("🏒 NHL OVER/UNDER BETTING MODEL - REAL DATA EDITION")
    print("=" * 60)
    
    # Initialize data collector
    collector = RealNHLDataCollector()
    
    # Get real NHL data
    df = collector.get_real_nhl_data(
        start_date='2024-10-10',  # Start of NHL season
        end_date='2024-12-15',    # Current date range
        max_games=300             # Limit for training speed
    )
    
    if df.empty:
        print("❌ No data available. Exiting.")
        return
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Games: {len(df)}")
    print(f"   Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Teams: {len(set(list(df['home_team']) + list(df['away_team'])))} unique teams")
    print(f"   Average Total Goals: {df['total_goals'].mean():.2f}")
    
    # Initialize and train model
    model = NHLOverUnderModel()
    
    print(f"\n🔧 Training Model...")
    print("Creating features...")
    df_features = model.create_features(df)
    
    print("Preparing model data...")
    X, y = model.prepare_model_data(df_features)
    
    print("Training ensemble model...")
    X_test, y_test, predictions = model.train_model(X, y, model_type='ensemble')
    
    # Show results
    print(f"\n🎯 Model Performance on REAL DATA:")
    rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
    mae = np.mean(np.abs(y_test - predictions))
    print(f"   RMSE: {rmse:.3f} goals")
    print(f"   MAE:  {mae:.3f} goals")
    
    # Show feature importance
    print(f"\n⭐ Top 10 Most Important Features:")
    importance = model.get_feature_importance()
    for idx, row in importance.head(10).iterrows():
        print(f"   {idx+1:2d}. {row['feature']:25s} ({row['importance']:.3f})")
    
    # Example predictions
    print(f"\n🎰 SAMPLE BETTING RECOMMENDATIONS:")
    print("-" * 50)
    
    sample_lines = [5.5, 6.0, 6.5, 7.0, 7.5]
    for i, line in enumerate(sample_lines):
        if i < len(predictions):
            actual = y_test.iloc[i]
            predicted = predictions[i]
            
            rec, edge, exp = model.betting_recommendation(predicted, line, confidence_threshold=0.3)
            
            print(f"Game {i+1}: Line {line} | Predicted {predicted:.2f} | Actual {actual}")
            print(f"         Recommendation: {rec} ({edge:+.2f} edge)")
            print()
    
    print(f"🚀 Model trained on REAL NHL data and ready for betting!")
    return model, df

# Function to install required packages
def install_requirements():
    """Install required packages for real data collection"""
    import subprocess
    import sys
    
    packages = ['requests', 'beautifulsoup4', 'lxml']
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    print("Installation complete. Please restart your script.")

if __name__ == "__main__":
    # Check if required packages are available
    if not REQUESTS_AVAILABLE:
        print("⚠️  Missing required packages for real data collection")
        print("Run this command first: python -m pip install requests beautifulsoup4 lxml")
        install_requirements()
    else:
        # Run with real data
        run_with_real_data()