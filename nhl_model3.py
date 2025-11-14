"""
NHL Over/Under Prediction Model with Real Data and Social Media Integration

This script fetches real NHL data, trains a machine learning model, and automatically
posts predictions to X (Twitter) and Discord.

Required packages:
pip install pandas numpy scikit-learn requests scipy tweepy discord.py

Run with: python nhl_model3.py
"""

import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, brier_score_loss, log_loss
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set, Any
import io
import re
from dataclasses import dataclass, field
import webbrowser
import argparse
import os
import time
import json
from urllib.parse import urlparse
import html as html_parser
import errno
import unicodedata
from scipy.stats import norm, poisson, nbinom
from sklearn.isotonic import IsotonicRegression
try:
    import pytz  # timezone handling
except Exception:
    pytz = None
try:
    from statsmodels.discrete.count_model import GeneralizedPoisson
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

# Optional imports for social media
try:
    import tweepy
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False
    print("⚠️  tweepy not installed. Twitter posting disabled.")

try:
    import discord
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    print("⚠️  discord.py not installed. Discord bot posting disabled.")

# Optional tools to render HTML dashboard to an image
try:
    import imgkit  # requires wkhtmltoimage installed on system
    IMGKIT_AVAILABLE = True
except Exception:
    IMGKIT_AVAILABLE = False

try:
    from html2image import Html2Image  # pure-python fallback (will download a browser engine on first run)
    HTML2IMAGE_AVAILABLE = True
except Exception:
    HTML2IMAGE_AVAILABLE = False

# Optional plotting library to build tweet-style image
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# Optional deployment libraries
try:
    import boto3
    BOTO3_AVAILABLE = True
except Exception:
    BOTO3_AVAILABLE = False

try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except Exception:
    PARAMIKO_AVAILABLE = False

try:
    from bs4 import BeautifulSoup  # type: ignore
    BEAUTIFULSOUP_AVAILABLE = True
except Exception:
    BEAUTIFULSOUP_AVAILABLE = False

warnings.filterwarnings('ignore')

def ensure_local_write_path(path: Optional[str]) -> Optional[str]:
    """Ensure the parent directory exists before writing to a local file path.

    Returns the absolute path when it can be safely written. If the path is
    not writable (common on Windows when pointing to protected locations
    such as ``C:\\``), a fallback under the user's home directory is used.
    """

    if path is None:
        return None

    raw = str(path).strip()
    if not raw:
        return None

    # Treat scheme-prefixed strings as remote URLs (skip writing)
    if re.match(r'^[a-z][a-z0-9+.-]*://', raw.lower()):
        return None

    expanded = os.path.expanduser(os.path.expandvars(raw))
    if not expanded:
        return None

    # Handle Windows-style drive paths when running on POSIX environments (e.g., WSL or Linux containers)
    if os.name != 'nt':
        win_match = re.match(r'^([a-zA-Z]):[\\/](.*)$', expanded)
        if win_match:
            drive_letter = win_match.group(1).lower()
            remainder = win_match.group(2).replace('\\', '/').lstrip('/')
            wsl_root = f"/mnt/{drive_letter}"
            if os.path.exists(wsl_root):
                expanded = os.path.join(wsl_root, remainder)
            else:
                fallback_local = os.path.abspath(os.path.join(os.getcwd(), os.path.basename(remainder) or "output.csv"))
                print(f"⚠️  Windows-style path {path} is not accessible on this system. Writing to {fallback_local} instead.")
                expanded = fallback_local

    norm_path = os.path.abspath(expanded)
    basename = os.path.basename(norm_path).strip() or "output.csv"

    def build_fallback_path() -> Optional[str]:
        """Return a writable fallback path for the same filename."""

        candidates = [
            os.path.join(os.path.expanduser("~"), "nhl_model_data"),
            os.path.abspath(os.getcwd()),
        ]

        for candidate in candidates:
            try:
                os.makedirs(candidate, exist_ok=True)
                return os.path.abspath(os.path.join(candidate, basename))
            except Exception:
                continue
        return None

    parent = os.path.dirname(norm_path) or os.getcwd()

    try:
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        if parent and not os.access(parent, os.W_OK):
            fallback_path = build_fallback_path()
            if fallback_path:
                print(f"⚠️  No write access to directory {parent}. Using fallback {fallback_path}")
                return fallback_path
            return None

        return norm_path
    except PermissionError:
        fallback_path = build_fallback_path()
        if fallback_path:
            print(f"⚠️  Permission denied for {norm_path}. Using fallback {fallback_path}")
            return fallback_path
        return None
    except OSError as e:
        if getattr(e, 'errno', None) in (errno.EACCES, errno.EROFS):
            fallback_path = build_fallback_path()
            if fallback_path:
                print(f"⚠️  Permission denied for {norm_path}. Using fallback {fallback_path}")
                return fallback_path
        print(f"⚠️  Could not prepare output path {path}: {e}")
        return None
    except Exception as e:
        print(f"⚠️  Could not prepare output path {path}: {e}")
        return None

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
    recommendation: str  # 'OVER', 'UNDER', 'No Bet'
    edge: float
    kelly_bet_size: float
    # Optional American odds used for EV/Kelly
    over_american_odds: Optional[int] = None
    under_american_odds: Optional[int] = None
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

class NHLDataFetcher:
    """Fetches real NHL data from multiple sources with fallbacks"""
    
    def __init__(self):
        # Prefer new NHL web API
        self.base_url = "https://api-web.nhle.com/v1"
        # Legacy stats API kept as fallback reference
        self.legacy_url = "https://statsapi.web.nhl.com/api/v1"
        self.backup_url = "https://api.nhle.com/stats/rest/en"
        self.teams_cache = None
        
    def get_teams(self) -> Dict[int, Dict]:
        """Get all NHL teams with multiple fallback sources"""
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
            except Exception as e:
                print(f"⚠️  NHL statsapi failed: {e}")
            
            # Use hardcoded fallback
            print("📋 Using hardcoded team data...")
            self.teams_cache = self._get_fallback_teams()
        
        return self.teams_cache
    
    def _get_fallback_teams(self) -> Dict[int, Dict]:
        """Comprehensive fallback team data"""
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
        """Get NHL schedule using api-web.nhle.com with robust parsing and date iteration.

        Returns a list of game dicts normalized to mimic legacy statsapi fields used elsewhere:
        {
          'gamePk': <int>, 'gameDate': <iso>,
          'status': {'detailedState': <str>},
          'teams': {'home': {'team': {'abbreviation': 'XXX'}}, 'away': {'team': {'abbreviation': 'YYY'}}},
          'venue': {'name': <str>}
        }
        """
        def normalize_game(g: Dict, expected_date: Optional[str] = None) -> Optional[Dict]:
            if not isinstance(g, dict):
                return None
            game_id = g.get('id') or g.get('gameId') or g.get('gamePk')
            start_time = g.get('startTimeUTC') or g.get('gameDate') or g.get('startTime')
            if (start_time is None or start_time == "") and expected_date:
                # Fallback to target date at midnight UTC
                start_time = f"{expected_date}T00:00:00Z"
            home = g.get('homeTeam') or {}
            away = g.get('awayTeam') or {}
            home_abbr = home.get('abbrev') or home.get('abbreviation') or home.get('triCode') or home.get('name')
            away_abbr = away.get('abbrev') or away.get('abbreviation') or away.get('triCode') or away.get('name')
            # Try to extract scores if provided by the endpoint
            def _score(val):
                try:
                    s = val.get('score')
                    return int(s) if s is not None else None
                except Exception:
                    return None
            home_score = _score(home)
            away_score = _score(away)
            venue_name = g.get('venue') or g.get('venueName') or (g.get('venue', {}) if isinstance(g.get('venue'), str) else None)
            # Map new gameState -> legacy detailedState
            state = g.get('gameState') or g.get('status') or ''
            state_map = {
                'FUT': 'Scheduled', 'PRE': 'Pre-Game', 'LIVE': 'In Progress', 'CRIT': 'In Progress',
                'FINAL': 'Final', 'OFF': 'Final', 'POSTPONED': 'Postponed'
            }
            detailed = state_map.get(str(state).upper(), str(state))
            try:
                # Normalize
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
            # Try schedule/{date}
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
                            for d in data['dates']:
                                games_raw.extend(d.get('games', []))
                    normalized = []
                    for gr in games_raw:
                        ng = normalize_game(gr, expected_date=date_str)
                        if ng is not None:
                            normalized.append(ng)
                    # Filter strictly to the requested date
                    if normalized:
                        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        tz_str = os.getenv('SCHEDULE_TZ', 'US/Eastern')
                        filtered = []
                        for ng in normalized:
                            try:
                                gd = pd.to_datetime(ng.get('gameDate'), utc=True, errors='coerce')
                                if pd.isna(gd):
                                    continue
                                # Compare by configured local date to capture late games
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
                except Exception as e:
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
        """Get detailed game statistics with fallbacks"""
        # New API endpoint
        try:
            url = f"{self.base_url}/gamecenter/{game_id}/boxscore"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception:
            # Legacy fallback
            try:
                url = f"{self.legacy_url}/game/{game_id}/boxscore"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            except Exception:
                return {}

class SocialMediaPoster:
    """Handles posting predictions to X (Twitter) and Discord"""
    
    def __init__(self):
        self.twitter_api = None
        self.discord_webhook_url = None
        self.streamlit_url = None
        self.streamlit_link_posted = False
        self.discord_verify = True
        self.setup_credentials()
        if TWITTER_AVAILABLE:
            self.setup_twitter()
    
    def setup_credentials(self):
        """Setup API credentials from environment variables"""
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.twitter_consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
        self.twitter_consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
        self.twitter_access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.twitter_access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        # Also support API-style env names used in tweet_predictions_image.py
        api_key = os.getenv('TWITTER_API_KEY')
        api_secret = os.getenv('TWITTER_API_SECRET')
        if not self.twitter_consumer_key and api_key:
            self.twitter_consumer_key = api_key
        if not self.twitter_consumer_secret and api_secret:
            self.twitter_consumer_secret = api_secret
        self.discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        self.streamlit_url = os.getenv('STREAMLIT_URL')
        
        # Always try to read social_config.json to backfill any missing fields
        try:
            with open('social_config.json', 'r') as f:
                config = json.load(f)
            
            twitter_config = config.get('twitter', {})
            if not self.twitter_bearer_token:
                self.twitter_bearer_token = twitter_config.get('bearer_token')
            if not self.twitter_consumer_key:
                self.twitter_consumer_key = twitter_config.get('consumer_key')
            if not self.twitter_consumer_secret:
                self.twitter_consumer_secret = twitter_config.get('consumer_secret')
            if not self.twitter_access_token:
                self.twitter_access_token = twitter_config.get('access_token')
            if not self.twitter_access_token_secret:
                self.twitter_access_token_secret = twitter_config.get('access_token_secret')
            
            discord_config = config.get('discord', {})
            if not self.discord_webhook_url:
                self.discord_webhook_url = discord_config.get('webhook_url')
            
            streamlit_config = config.get('streamlit', {})
            if not self.streamlit_url:
                self.streamlit_url = streamlit_config.get('url')
            # Honor DISCORD_INSECURE from env or config.deploy.discord_insecure
            insecure_env = os.getenv('DISCORD_INSECURE')
            if isinstance(insecure_env, str) and insecure_env.strip().lower() in ('1','true','yes','y'):
                self.discord_verify = False
            deploy_cfg = config.get('deploy') or {}
            insecure_cfg = str((deploy_cfg.get('discord_insecure') or '')).strip().lower()
            if insecure_cfg in ('1','true','yes','y'):
                self.discord_verify = False
        except FileNotFoundError:
            print("⚠️  No social_config.json found. Creating template...")
            self.create_config_template()
        # Debug summary (no secrets printed)
        try:
            dbg = {
                'discord': 'yes' if bool(self.discord_webhook_url) else 'no',
                'streamlit': 'yes' if bool(self.streamlit_url) else 'no',
                'twitter': 'yes' if bool(self.twitter_bearer_token) else 'no'
            }
            print(f"🔧 Social config -> Discord? {dbg['discord']} | Streamlit? {dbg['streamlit']} | Twitter? {dbg['twitter']}")
        except Exception:
            pass

    def post_streamlit_link(self) -> bool:
        """Post the Streamlit URL to Discord exactly once per run."""
        if self.streamlit_link_posted:
            return True
        if not (self.streamlit_url and self.discord_webhook_url):
            return False
        try:
            print("💬 Posting Streamlit link to Discord…")
            requests.post(
                self.discord_webhook_url,
                json={"content": f"🔗 Streamlit Dashboard: {self.streamlit_url}"},
                timeout=10,
                verify=self.discord_verify
            )
            self.streamlit_link_posted = True
            print("✅ Streamlit link posted to Discord")
            return True
        except Exception as e:
            print(f"⚠️  Failed to post Streamlit link: {e}")
            return False
    
    def create_config_template(self):
        """Create a template configuration file"""
        template = {
            "twitter": {
                "bearer_token": "YOUR_TWITTER_BEARER_TOKEN",
                "consumer_key": "YOUR_TWITTER_CONSUMER_KEY",
                "consumer_secret": "YOUR_TWITTER_CONSUMER_SECRET",
                "access_token": "YOUR_TWITTER_ACCESS_TOKEN",
                "access_token_secret": "YOUR_TWITTER_ACCESS_TOKEN_SECRET"
            },
            "discord": {
                "webhook_url": "YOUR_DISCORD_WEBHOOK_URL"
            },
            "odds": {
                "api_key": "YOUR_ODDS_API_KEY"
            },
            "streamlit": {
                "url": "YOUR_STREAMLIT_URL"
            },
            "deploy": {
                "method": "http",  # http | s3 | sftp
                "http": {
                    "url": "https://www.thepointou.com/nhl_real_data_dashboard.html",
                    "http_method": "PUT",
                    "auth": {
                        "type": "bearer",  # bearer | basic | none
                        "token": "YOUR_BEARER_TOKEN",
                        "username": "",
                        "password": ""
                    }
                },
                "s3": {
                    "bucket": "YOUR_BUCKET",
                    "key": "dashboards/nhl_real_data_dashboard.html",
                    "region": "us-east-1",
                    "acl": "public-read"
                },
                "sftp": {
                    "host": "sftp.thepointou.com",
                    "port": 22,
                    "username": "",
                    "password": "",
                    "remote_path": "/var/www/thepointou/nhl_real_data_dashboard.html"
                }
            }
        }
        
        with open('social_config.json', 'w') as f:
            json.dump(template, f, indent=4)
        
        print("📄 Created social_config.json template. Please fill in your API credentials.")

    def post_file_to_discord(self, file_path: str, message: Optional[str] = None) -> bool:
        """Upload a file (e.g., Excel or CSV) to Discord via webhook."""
        if not self.discord_webhook_url:
            print("⚠️  Discord webhook URL not available")
            return False
        try:
            if not os.path.exists(file_path):
                print(f"⚠️  File not found for Discord upload: {file_path}")
                return False
            filename = os.path.basename(file_path)
            # Basic content-type detection
            ctype = 'application/octet-stream'
            fn_l = filename.lower()
            if fn_l.endswith('.xlsx'):
                ctype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            elif fn_l.endswith('.xls'):
                ctype = 'application/vnd.ms-excel'
            elif fn_l.endswith('.csv'):
                ctype = 'text/csv'
            elif fn_l.endswith('.html') or fn_l.endswith('.htm'):
                ctype = 'text/html'
            elif fn_l.endswith('.png'):
                ctype = 'image/png'
            with open(file_path, 'rb') as f:
                files = { 'file': (filename, f, ctype) }
                data = { 'content': message or '📎 Predictions export' }
                resp = requests.post(self.discord_webhook_url, data=data, files=files, timeout=30, verify=self.discord_verify)
                resp.raise_for_status()
                print(f"✅ Uploaded file to Discord: {filename}")
                return True
        except Exception as e:
            print(f"❌ Discord file upload failed: {e}")
            return False

    def post_inline_predictions(self, predictions: List[OverUnderPrediction], top_n: int = 10, title: str = "NHL Predictions (Top)") -> bool:
        """Post a compact inline summary of top predictions to Discord as embeds.

        Splits across multiple messages if needed. Returns True if at least one post succeeds.
        """
        if not self.discord_webhook_url:
            print("⚠️  Discord webhook URL not available")
            return False
        try:
            # Prefer bets first, then by absolute edge desc
            preds = list(predictions or [])
            preds.sort(key=lambda p: (p.recommendation == 'No Bet', -abs(float(getattr(p, 'edge', 0.0) or 0.0))), reverse=False)
            preds = preds[:max(1, int(top_n))]
            # Build embeds in chunks of up to 8 fields per embed (Discord soft limit)
            CHUNK = 8
            posted_any = False
            for i in range(0, len(preds), CHUNK):
                chunk = preds[i:i+CHUNK]
                embed = {
                    "title": f"{title} {i+1}-{i+len(chunk)}",
                    "color": 0x2ecc71,
                    "fields": []
                }
                for p in chunk:
                    name = f"{p.away_team} @ {p.home_team}"
                    val_bits = []
                    try:
                        val_bits.append(f"Line {p.betting_line:.1f} | Pred {p.predicted_total:.2f}")
                    except Exception:
                        val_bits.append(f"Line {p.betting_line} | Pred {p.predicted_total}")
                    val_bits.append(f"Rec {p.recommendation} {p.edge:+.2f}")
                    try:
                        val_bits.append(f"Kelly {p.kelly_bet_size:.1f}%")
                    except Exception:
                        pass
                    try:
                        val_bits.append(f"O {p.over_probability:.0%} / U {p.under_probability:.0%}")
                    except Exception:
                        pass
                    ref_bits = getattr(p, 'referee_info', None)
                    if not ref_bits:
                        crew_list = getattr(p, 'referee_crew', []) or []
                        ref_bits = ", ".join([str(n) for n in crew_list if str(n).strip()])
                    ref_metrics: List[str] = []
                    if isinstance(p.referee_avg_goals, (int, float)) and np.isfinite(p.referee_avg_goals):
                        ref_metrics.append(f"{p.referee_avg_goals:.2f} G/G")
                    if isinstance(p.referee_home_bias, (int, float)) and np.isfinite(p.referee_home_bias):
                        ref_metrics.append(f"HB {p.referee_home_bias:+.2f}")
                    if isinstance(p.ref_goals_gm, (int, float)) and np.isfinite(p.ref_goals_gm) and not ref_metrics:
                        ref_metrics.append(f"Feature {p.ref_goals_gm:.2f} G/G")
                    if ref_bits or ref_metrics:
                        detail = ref_bits or ''
                        if ref_metrics:
                            metric_txt = ", ".join(ref_metrics)
                            detail = f"{detail} ({metric_txt})" if detail else metric_txt
                        val_bits.append(f"Refs {detail}")
                    embed["fields"].append({
                        "name": name,
                        "value": " | ".join(val_bits),
                        "inline": False
                    })
                resp = requests.post(self.discord_webhook_url, json={"embeds": [embed]}, timeout=15, verify=self.discord_verify)
                if 200 <= resp.status_code < 300:
                    posted_any = True
                else:
                    print(f"⚠️  Inline post failed: {resp.status_code} {resp.text[:120]}")
            return posted_any
        except Exception as e:
            print(f"❌ Inline Discord post failed: {e}")
            return False

    def deploy_dashboard_html(self, html_path: str, cli_args: Optional[argparse.Namespace] = None) -> bool:
        """Deploy the dashboard HTML to www.thepointou.com using configured method.

        Supported methods:
          - http (PUT/POST to a URL, optional bearer/basic auth)
          - s3 (upload to S3 bucket/key with public-read)
          - sftp (upload over SFTP to a remote path)
        """
        try:
            # Resolve deploy method and config (CLI overrides file config)
            cfg = getattr(self, 'deploy_config', {}) if hasattr(self, 'deploy_config') else {}
            method = None
            if cli_args and getattr(cli_args, 'deploy_method', None):
                method = str(cli_args.deploy_method).strip().lower()
            elif isinstance(cfg, dict):
                method = str(cfg.get('method', '')).strip().lower() or None
            if not method:
                return False
            print(f"🌐 Deploying dashboard via '{method}'...")

            if method == 'http':
                # Determine URL, method, auth
                url = None
                http_method = 'PUT'
                headers = {'Content-Type': 'text/html'}
                if cli_args and getattr(cli_args, 'deploy_target_url', None):
                    url = cli_args.deploy_target_url
                    http_method = getattr(cli_args, 'deploy_http_method', 'PUT').upper()
                    token = getattr(cli_args, 'deploy_token', None)
                    basic_user = getattr(cli_args, 'deploy_basic_user', None)
                    basic_pass = getattr(cli_args, 'deploy_basic_pass', None)
                else:
                    http_cfg = (cfg.get('http') or {}) if isinstance(cfg, dict) else {}
                    url = http_cfg.get('url')
                    http_method = str(http_cfg.get('http_method', 'PUT')).upper()
                    auth_cfg = http_cfg.get('auth') or {}
                    token = auth_cfg.get('token') if str(auth_cfg.get('type', 'none')).lower() == 'bearer' else None
                    basic_user = auth_cfg.get('username') if str(auth_cfg.get('type', 'none')).lower() == 'basic' else None
                    basic_pass = auth_cfg.get('password') if str(auth_cfg.get('type', 'none')).lower() == 'basic' else None
                if token:
                    headers['Authorization'] = f"Bearer {token}"
                auth = None
                if basic_user and basic_pass:
                    auth = (basic_user, basic_pass)
                with open(html_path, 'rb') as f:
                    data = f.read()
                try:
                    if http_method == 'PUT':
                        resp = requests.put(url, data=data, headers=headers, auth=auth, timeout=30)
                    else:
                        # POST as multipart form if not PUT
                        files = {'file': ('nhl_real_data_dashboard.html', data, 'text/html')}
                        resp = requests.post(url, headers={k: v for k, v in headers.items() if k.lower() != 'content-type'}, files=files, auth=auth, timeout=30)
                    if 200 <= resp.status_code < 300:
                        print("✅ Deployed dashboard via HTTP")
                        return True
                    print(f"⚠️  HTTP deploy failed: {resp.status_code} {resp.text[:200]}")
                except Exception as e:
                    print(f"⚠️  HTTP deploy error: {e}")
                return False

            if method == 's3':
                s3_cfg = (cfg.get('s3') or {}) if isinstance(cfg, dict) else {}
                bucket = getattr(cli_args, 'deploy_s3_bucket', None) or s3_cfg.get('bucket')
                key = getattr(cli_args, 'deploy_s3_key', None) or s3_cfg.get('key')
                region = getattr(cli_args, 'deploy_s3_region', None) or s3_cfg.get('region') or None
                acl = getattr(cli_args, 'deploy_s3_acl', None) or s3_cfg.get('acl') or 'public-read'
                if not BOTO3_AVAILABLE:
                    print("⚠️  boto3 not installed; cannot deploy to S3")
                    return False
                if not (bucket and key):
                    print("⚠️  Missing S3 bucket/key for deploy")
                    return False
                try:
                    s3 = boto3.client('s3', region_name=region) if region else boto3.client('s3')
                    with open(html_path, 'rb') as f:
                        s3.put_object(Bucket=bucket, Key=key, Body=f, ContentType='text/html', ACL=acl)
                    print("✅ Deployed dashboard to S3")
                    return True
                except Exception as e:
                    print(f"⚠️  S3 deploy error: {e}")
                    return False

            if method == 'sftp':
                sftp_cfg = (cfg.get('sftp') or {}) if isinstance(cfg, dict) else {}
                host = getattr(cli_args, 'deploy_sftp_host', None) or sftp_cfg.get('host')
                port = int(getattr(cli_args, 'deploy_sftp_port', 0) or sftp_cfg.get('port') or 22)
                user = getattr(cli_args, 'deploy_sftp_user', None) or sftp_cfg.get('username')
                password = getattr(cli_args, 'deploy_sftp_pass', None) or sftp_cfg.get('password')
                remote_path = getattr(cli_args, 'deploy_sftp_path', None) or sftp_cfg.get('remote_path')
                if not PARAMIKO_AVAILABLE:
                    print("⚠️  paramiko not installed; cannot deploy via SFTP")
                    return False
                if not (host and user and password and remote_path):
                    print("⚠️  Missing SFTP host/user/pass/remote_path for deploy")
                    return False
                try:
                    transport = paramiko.Transport((host, port))
                    transport.connect(username=user, password=password)
                    sftp = paramiko.SFTPClient.from_transport(transport)
                    sftp.put(html_path, remote_path)
                    sftp.close()
                    transport.close()
                    print("✅ Deployed dashboard via SFTP")
                    return True
                except Exception as e:
                    print(f"⚠️  SFTP deploy error: {e}")
                    return False

            print(f"⚠️  Unknown deploy method: {method}")
            return False
        except Exception as e:
            print(f"⚠️  Deploy failed: {e}")
            return False
    
    def setup_twitter(self):
        """Setup Twitter API client"""
        try:
            if all([self.twitter_consumer_key, self.twitter_consumer_secret, 
                   self.twitter_access_token, self.twitter_access_token_secret]):
                
                self.twitter_api = tweepy.Client(
                    bearer_token=self.twitter_bearer_token,
                    consumer_key=self.twitter_consumer_key,
                    consumer_secret=self.twitter_consumer_secret,
                    access_token=self.twitter_access_token,
                    access_token_secret=self.twitter_access_token_secret,
                    wait_on_rate_limit=True
                )
                print("✅ Twitter API initialized successfully")
            else:
                print("⚠️  Twitter credentials not found. Twitter posting disabled.")
                
        except Exception as e:
            print(f"❌ Twitter API setup failed: {e}")
            self.twitter_api = None
    
    def _format_tweet_table(self, predictions: List[OverUnderPrediction], max_rows: int = 4) -> Optional[str]:
        """Return an ASCII table string sized for Twitter using top predictions."""

        preds = [p for p in (predictions or []) if p is not None]
        if not preds:
            return None

        def _fmt_float(val: Optional[float], decimals: int = 1) -> str:
            try:
                return f"{float(val):.{decimals}f}"
            except Exception:
                return "--"

        def _fmt_edge(val: Optional[float]) -> str:
            try:
                return f"{float(val):+.2f}"
            except Exception:
                return "--"

        def _fmt_conf(val: Optional[float]) -> str:
            try:
                num = float(val)
                if 0.0 <= num <= 1.0:
                    num *= 100.0
                return f"{num:.0f}%"
            except Exception:
                return "--"

        def _sort_key(pred: OverUnderPrediction) -> float:
            try:
                return -abs(float(getattr(pred, 'edge', 0.0) or 0.0))
            except Exception:
                return 0.0

        prioritized: List[OverUnderPrediction] = []
        others: List[OverUnderPrediction] = []
        for pred in preds:
            rec = (getattr(pred, 'recommendation', '') or '').strip().upper()
            if rec and rec != 'NO BET':
                prioritized.append(pred)
            else:
                others.append(pred)

        prioritized.sort(key=_sort_key)
        others.sort(key=_sort_key)

        ordered = prioritized + others
        if not ordered:
            return None

        limited = ordered[:max(1, int(max_rows))]
        columns = [
            ("Matchup", "left"),
            ("Line", "right"),
            ("Pred", "right"),
            ("Edge", "right"),
            ("Pick", "left"),
            ("Conf", "right"),
        ]

        rows: List[List[str]] = []
        for pred in limited:
            away = str(getattr(pred, 'away_team', '') or '').strip()
            home = str(getattr(pred, 'home_team', '') or '').strip()
            matchup = f"{away} @ {home}".strip()
            if not matchup or matchup == '@':
                matchup = away or home or 'TBD'

            line_txt = _fmt_float(getattr(pred, 'betting_line', None))
            pred_txt = _fmt_float(getattr(pred, 'predicted_total', None))
            edge_txt = _fmt_edge(getattr(pred, 'edge', None))
            pick_raw = (getattr(pred, 'recommendation', '') or 'No Bet').strip().upper() or 'NO BET'
            if pick_raw == 'NO BET':
                pick_txt = 'NO BET'
            else:
                pick_txt = pick_raw
            conf_txt = _fmt_conf(getattr(pred, 'confidence', None))

            rows.append([matchup, line_txt, pred_txt, edge_txt, pick_txt, conf_txt])

        widths: List[int] = []
        for idx, (header, _) in enumerate(columns):
            width = len(header)
            for row in rows:
                width = max(width, len(row[idx]))
            widths.append(width)

        def _format_cell(value: str, width: int, align: str) -> str:
            return value.rjust(width) if align == 'right' else value.ljust(width)

        separator = []
        for width in widths:
            separator.append('-' * width)

        header_line = '  '.join(
            _format_cell(header, widths[idx], align)
            for idx, (header, align) in enumerate(columns)
        )
        separator_line = '  '.join(separator)
        row_lines = [
            '  '.join(
                _format_cell(row[idx], widths[idx], columns[idx][1])
                for idx in range(len(columns))
            )
            for row in rows
        ]

        table_lines = [header_line, separator_line]
        table_lines.extend(row_lines)
        return '\n'.join(table_lines)

    def post_to_twitter(self, predictions: List[OverUnderPrediction], training_results: Dict) -> bool:
        """Post predictions to Twitter in the same style used by tweet_predictions_image.py.

        We prioritize posting a predictions image with a short caption. If no media is
        available, we fall back to a concise text-only tweet. The image formatting will
        mirror the table style and caption behavior from tweet_predictions_image.py.
        """
        if not self.twitter_api or not TWITTER_AVAILABLE:
            print("⚠️  Twitter API not available")
            return False

        try:
            # Build image similar to tweet_predictions_image.py
            img_path = save_predictions_image(
                predictions,
                training_results=training_results,
                html_path='predictions_table.html',
                image_path='predictions.png'
            )

            # Caption modeled after tweet_predictions_image.py default
            default_caption = "Predictor picks for this week's NFL games."  # will adjust to NHL
            # For NHL context, craft an equivalent concise caption
            tweet_caption = "Predictor picks for tonight's NHL games."

            if img_path and os.path.exists(img_path):
                return self.post_image_to_twitter(img_path, caption=tweet_caption)

            # Fallback: concise text-only tweet (no multiline list)
            betting_preds = [p for p in predictions if p.recommendation != 'No Bet']
            tweet_text: Optional[str] = None

            candidate_groups: List[List[OverUnderPrediction]] = []
            if betting_preds:
                candidate_groups.append(betting_preds)
            if predictions:
                candidate_groups.append(predictions)

            for group in candidate_groups:
                if not group:
                    continue
                max_rows = min(4, len(group))
                for rows in range(max_rows, 0, -1):
                    table_part = self._format_tweet_table(group, max_rows=rows)
                    if not table_part:
                        continue
                    combined_text = f"{tweet_caption}\n{table_part}"
                    if len(combined_text) <= 280:
                        tweet_text = combined_text
                        break
                if tweet_text:
                    break

            if not tweet_text:
                if not betting_preds:
                    tweet_text = tweet_caption
                else:
                    top = betting_preds[0]
                    tweet_text = (
                        f"{tweet_caption} Top: {top.away_team} @ {top.home_team} — "
                        f"{top.recommendation} {top.betting_line} (edge {float(getattr(top, 'edge', 0.0)):+.1f})."
                    )

            try:
                response = self.twitter_api.create_tweet(text=tweet_text)
            except tweepy.TooManyRequests:
                print("⏳ Twitter rate limit hit while posting text tweet. Skipping for now.")
                return False
            print(f"✅ Posted to Twitter: {response.data['id']}")
            return True

        except Exception as e:
            print(f"❌ Twitter posting failed: {e}")
            return False
    
    def post_to_discord_webhook(self, predictions: List[OverUnderPrediction], training_results: Dict) -> bool:
        """Post predictions to Discord via webhook"""
        if not self.discord_webhook_url:
            print("⚠️  Discord webhook URL not available")
            return False
        
        try:
            betting_preds = [p for p in predictions if p.recommendation != 'No Bet']
            
            embed = {
                "title": "🏒 NHL Over/Under Predictions",
                "description": f"Daily predictions powered by machine learning",
                "color": 0x00ff00 if betting_preds else 0xffff00,
                "timestamp": datetime.now().isoformat(),
                "fields": []
            }
            
            summary_value = f"**📊 Model Accuracy:** {training_results.get('over_under_accuracy', 0):.1%}\n"
            summary_value += f"**💰 Opportunities:** {len(betting_preds)} recommended bets"
            
            embed["fields"].append({
                "name": "📈 Daily Summary",
                "value": summary_value,
                "inline": False
            })
            
            for pred in betting_preds[:6]:
                field_name = f"🏒 {pred.away_team} @ {pred.home_team}"
                field_value = f"**Line:** {pred.betting_line} | **Pred:** {pred.predicted_total:.1f}\n"
                field_value += f"**Rec:** {pred.recommendation} ({pred.edge:+.2f})\n"
                field_value += f"**Conf:** {pred.confidence:.0%}"
                ref_bits = getattr(pred, 'referee_info', None)
                if not ref_bits:
                    crew_list = getattr(pred, 'referee_crew', []) or []
                    ref_bits = ", ".join([str(n) for n in crew_list if str(n).strip()])
                if ref_bits:
                    ref_metrics: List[str] = []
                    if isinstance(pred.referee_avg_goals, (int, float)) and np.isfinite(pred.referee_avg_goals):
                        ref_metrics.append(f"{pred.referee_avg_goals:.2f} G/G")
                    if isinstance(pred.referee_home_bias, (int, float)) and np.isfinite(pred.referee_home_bias):
                        ref_metrics.append(f"HB {pred.referee_home_bias:+.2f}")
                    if ref_metrics:
                        ref_bits = f"{ref_bits} ({', '.join(ref_metrics)})" if ref_bits else ", ".join(ref_metrics)
                    field_value += f"\n**Refs:** {ref_bits}"
                
                embed["fields"].append({
                    "name": field_name,
                    "value": field_value,
                    "inline": True
                })
            
            payload = {"embeds": [embed]}
            response = requests.post(self.discord_webhook_url, json=payload, verify=self.discord_verify)
            response.raise_for_status()
            
            print("✅ Posted to Discord via webhook")
            return True
            
        except Exception as e:
            print(f"❌ Discord webhook posting failed: {e}")
            return False
    
    def post_predictions(self, predictions: List[OverUnderPrediction], training_results: Dict) -> Dict[str, bool]:
        """Post predictions to all configured social media platforms"""
        results = {}
        
        print("\n📱 Posting predictions to social media...")
        
        if TWITTER_AVAILABLE:
            print("🐦 Posting to X (Twitter)...")
            results['twitter'] = self.post_to_twitter(predictions, training_results)
        else:
            results['twitter'] = False
        
        # Minimal Discord mode: skip summary embed; only post dashboard image and Streamlit link elsewhere
        print("💬 Skipping Discord summary (minimal mode)")
        results['discord'] = False
        
        # Streamlit link posting removed per request
        
        return results

    def render_dashboard_image(self, html_path: str, image_path: str = 'dashboard.png') -> Optional[str]:
        """Render the local HTML dashboard to an image suitable for posting.

        Tries imgkit (wkhtmltoimage) first, falls back to html2image. Returns image path or None.
        """
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception:
            pass

        # Try imgkit
        if IMGKIT_AVAILABLE:
            try:
                options = {
                    'quality': 85,
                    'format': 'png',
                    'encoding': 'utf-8',
                    'crop-h': '1200',
                    'crop-w': '1400',
                }
                imgkit.from_file(html_path, image_path, options=options)
                if os.path.exists(image_path):
                    return image_path
            except Exception:
                pass

        # Fallback: html2image
        if HTML2IMAGE_AVAILABLE:
            try:
                hti = Html2Image()
                hti.output_path = os.path.dirname(os.path.abspath(image_path)) or '.'
                hti.screenshot(html_file=html_path, save_as=os.path.basename(image_path), size=(1400, 1200))
                if os.path.exists(image_path):
                    return image_path
            except Exception:
                pass

        print("⚠️  Could not render dashboard to image. Install wkhtmltoimage or enable html2image.")
        return None

    def post_dashboard_to_discord(self, html_path: str) -> bool:
        """Post the dashboard as an image attachment to Discord via webhook."""
        if not self.discord_webhook_url:
            print("⚠️  Discord webhook URL not available")
            return False

        try:
            print("💬 Posting dashboard image to Discord…")
            image_path = self.render_dashboard_image(html_path)
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    files = { 'file': (os.path.basename(image_path), f, 'image/png') }
                    data = { 'content': '🏒 NHL Over/Under Dashboard' }
                    resp = requests.post(self.discord_webhook_url, data=data, files=files, verify=self.discord_verify)
                    resp.raise_for_status()
                    print("✅ Dashboard image posted to Discord")
                    # Also post Streamlit link if available (one-time)
                    self.post_streamlit_link()
                    return True
            # Fallback: send a link to the local file path (Discord clients won't open local paths)
            print("ℹ️ Image render unavailable; posting local file path to Discord…")
            payload = { 'content': f"Dashboard saved locally: {os.path.abspath(html_path)}" }
            resp = requests.post(self.discord_webhook_url, json=payload, verify=self.discord_verify)
            resp.raise_for_status()
            print("✅ Dashboard link posted to Discord")
            # Attempt Streamlit link as well (one-time)
            self.post_streamlit_link()
            return True
        except Exception as e:
            print(f"❌ Discord dashboard posting failed: {e}")
            return False

    def post_dashboard_to_twitter(self, html_path: str, caption: str = "🏒 NHL Over/Under Dashboard") -> bool:
        """Post the dashboard as an image to X (Twitter)."""
        if not self.twitter_api or not TWITTER_AVAILABLE:
            print("⚠️  Twitter API not available")
            return False
        try:
            image_path = self.render_dashboard_image(html_path)
            if not image_path or not os.path.exists(image_path):
                print("⚠️  Dashboard image not available for Twitter post")
                return False

            # Upload media then create tweet
            with open(image_path, 'rb') as f:
                media_bytes = f.read()
            # v2 tweepy Client does not support media upload directly; fallback to API v1.1 via OAuth1 if available
            try:
                auth = tweepy.OAuth1UserHandler(
                    self.twitter_consumer_key,
                    self.twitter_consumer_secret,
                    self.twitter_access_token,
                    self.twitter_access_token_secret
                )
                api_v1 = tweepy.API(auth, wait_on_rate_limit=True)
                media = api_v1.media_upload(filename=image_path)
                media_id = media.media_id_string
                self.twitter_api.create_tweet(text=caption, media_ids=[media_id])
                print("✅ Dashboard image posted to Twitter")
                return True
            except tweepy.TooManyRequests:
                print("⏳ Twitter rate limit hit while posting dashboard. Saved for manual posting.")
                print(f"👉 Image: {image_path}")
                print(f"👉 Text:  {caption}")
                return False
            except Exception as e:
                print(f"❌ Twitter media upload failed: {e}")
                return False
        except Exception as e:
            print(f"❌ Twitter dashboard posting failed: {e}")
            return False

    def post_image_to_twitter(self, image_path: str, caption: str = "🏒 NHL Predictions") -> bool:
        """Post an arbitrary image (e.g., predictions.png) to X (Twitter)."""
        if not self.twitter_api or not TWITTER_AVAILABLE:
            print("⚠️  Twitter API not available")
            return False
        try:
            if not os.path.exists(image_path):
                print(f"⚠️  Image not found for Twitter post: {image_path}")
                return False
            try:
                auth = tweepy.OAuth1UserHandler(
                    self.twitter_consumer_key,
                    self.twitter_consumer_secret,
                    self.twitter_access_token,
                    self.twitter_access_token_secret
                )
                api_v1 = tweepy.API(auth, wait_on_rate_limit=True)
                media = api_v1.media_upload(filename=image_path)
                media_id = media.media_id_string
                self.twitter_api.create_tweet(text=caption, media_ids=[media_id])
                print("✅ Predictions image posted to Twitter")
                return True
            except tweepy.TooManyRequests:
                print("⏳ Twitter rate limit hit while posting image. Saved for manual posting.")
                print(f"👉 Image: {image_path}")
                print(f"👉 Text:  {caption}")
                return False
            except Exception as e:
                print(f"❌ Twitter media upload failed: {e}")
                return False
        except Exception as e:
            print(f"❌ Twitter image posting failed: {e}")
            return False

class RealDataNHLModel:
    """NHL Over/Under model using real data"""
    
    def __init__(self):
        self.data_fetcher = NHLDataFetcher()
        self.total_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        # Store conformal quantiles for uncertainty intervals
        self.conformal_q80: Optional[float] = None
        self.conformal_q90: Optional[float] = None
        self.ci_quantile: float = 0.90  # default 90%
        self.conformal_radius: Optional[float] = None
        # Goal models for bivariate Poisson approximation
        self.home_goal_mu_model: Optional[PoissonRegressor] = None
        self.away_goal_mu_model: Optional[PoissonRegressor] = None
        # Optional COM-Poisson/Generalized Poisson fallback control
        self.use_compoisson: bool = False
        # Kelly configuration
        self.kelly_mult: float = float(os.getenv('KELLY_MULT', 0.5))
        self.kelly_cap_pct: float = float(os.getenv('KELLY_CAP_PCT', 2.0))  # percent
        self.daily_exposure_cap_pct: float = float(os.getenv('DAILY_EXPOSURE_CAP_PCT', 6.0))  # percent
        self.kelly_use_fair: bool = False
        self._team_alias_map: Optional[Dict[str, str]] = None
        self.ref_goal_baseline: Optional[float] = None
        try:
            self.ref_goal_weight: float = float(os.getenv('REF_GOAL_WEIGHT', 0.05))
        except Exception:
            self.ref_goal_weight = 0.05
        
    def fetch_historical_games(self, days_back: int = 30) -> pd.DataFrame:
        """Fetch historical games data with robust error handling"""
        print(f"Fetching historical games from last {days_back} days...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        games = self.data_fetcher.get_schedule(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if not games:
            print("⚠️  No games from API, trying extended date range...")
            start_date = end_date - timedelta(days=days_back * 2)
            games = self.data_fetcher.get_schedule(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
        
        if not games:
            print("❌ Still no games from API. Using enhanced sample data...")
            return self.create_realistic_sample_data()
        
        print(f"Found {len(games)} games from API")
        
        games_data = []
        processed_count = 0
        
        for i, game in enumerate(games):
            if game.get('status', {}).get('detailedState') != 'Final':
                continue
                
            try:
                home_team = game['teams']['home']['team']
                away_team = game['teams']['away']['team']
                
                game_data = {
                    'game_id': game['gamePk'],
                    'date': pd.to_datetime(game['gameDate']),
                    'home_team': home_team['abbreviation'],
                    'away_team': away_team['abbreviation'],
                    'venue': game.get('venue', {}).get('name', f"{home_team['abbreviation']} Arena"),
                    'home_goals': game['teams']['home']['score'],
                    'away_goals': game['teams']['away']['score'],
                    'total_goals': game['teams']['home']['score'] + game['teams']['away']['score']
                }
                
                # Add realistic stats
                game_data.update({
                    'home_shots': max(20, int(np.random.normal(32, 4))),
                    'away_shots': max(20, int(np.random.normal(30, 4))),
                    'home_pp_goals': min(game_data['home_goals'], max(0, int(np.random.poisson(0.7)))),
                    'away_pp_goals': min(game_data['away_goals'], max(0, int(np.random.poisson(0.7)))),
                    'home_pp_opps': max(1, int(np.random.poisson(3.0))),
                    'away_pp_opps': max(1, int(np.random.poisson(3.0)))
                })
                
                games_data.append(game_data)
                processed_count += 1
                
                if i % 5 == 0 and i > 0:
                    time.sleep(0.5)
                    print(f"Processed {processed_count} completed games...")
                    
            except Exception as e:
                print(f"⚠️  Error processing game {game.get('gamePk', 'unknown')}: {e}")
                continue
        
        if not games_data:
            print("❌ No valid games processed. Using enhanced sample data...")
            return self.create_realistic_sample_data()
        
        print(f"✅ Successfully processed {len(games_data)} completed games")
        return pd.DataFrame(games_data)
    
    def create_realistic_sample_data(self) -> pd.DataFrame:
        """Create realistic sample data based on current NHL trends"""
        print("📝 Creating realistic NHL sample data...")
        
        np.random.seed(42)
        
        teams_performance = {
            'BOS': {'off': 3.2, 'def': 2.7}, 'TOR': {'off': 3.4, 'def': 3.1},
            'NYR': {'off': 3.0, 'def': 2.8}, 'FLA': {'off': 3.3, 'def': 2.9},
            'CAR': {'off': 3.1, 'def': 2.6}, 'TBL': {'off': 3.2, 'def': 2.9},
            'EDM': {'off': 3.6, 'def': 3.2}, 'COL': {'off': 3.4, 'def': 3.0},
            'VGK': {'off': 3.1, 'def': 2.8}, 'DAL': {'off': 3.0, 'def': 2.7},
            'WPG': {'off': 3.2, 'def': 2.9}, 'MIN': {'off': 2.9, 'def': 2.8},
            'MTL': {'off': 2.7, 'def': 3.3}, 'CBJ': {'off': 2.8, 'def': 3.4},
            'ANA': {'off': 2.6, 'def': 3.1}, 'SJS': {'off': 2.5, 'def': 3.5}
        }
        
        teams = list(teams_performance.keys())
        data = []
        start_date = datetime.now() - timedelta(days=90)
        
        for i in range(150):
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            home_off = teams_performance[home_team]['off']
            away_off = teams_performance[away_team]['off']
            home_def = teams_performance[home_team]['def']
            away_def = teams_performance[away_team]['def']
            
            home_expected = (home_off + (6.0 - away_def)) / 2
            away_expected = (away_off + (6.0 - home_def)) / 2
            
            home_goals = max(0, int(np.random.poisson(home_expected)))
            away_goals = max(0, int(np.random.poisson(away_expected)))
            
            game_date = start_date + timedelta(days=i//3)
            
            game_data = {
                'game_id': f'sample_{i:04d}',
                'date': game_date,
                'home_team': home_team,
                'away_team': away_team,
                'venue': f'{home_team} Arena',
                'home_goals': home_goals,
                'away_goals': away_goals,
                'total_goals': home_goals + away_goals,
                'home_shots': max(20, int(np.random.normal(32, 5))),
                'away_shots': max(20, int(np.random.normal(30, 5))),
                'home_pp_goals': min(home_goals, max(0, int(np.random.poisson(0.8)))),
                'away_pp_goals': min(away_goals, max(0, int(np.random.poisson(0.7)))),
                'home_pp_opps': max(1, int(np.random.poisson(3.2))),
                'away_pp_opps': max(1, int(np.random.poisson(3.0)))
            }
            
            data.append(game_data)
        
        print(f"✅ Created {len(data)} realistic sample games")
        return pd.DataFrame(data)
    
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features from real data"""
        features = df.copy()
        # Normalize date column to tz-naive consistently (convert any tz-aware to naive UTC)
        if 'date' in features.columns:
            try:
                features['date'] = pd.to_datetime(features['date'], utc=True, errors='coerce').dt.tz_convert(None)
            except Exception:
                features['date'] = pd.to_datetime(features['date'], errors='coerce')
        
        print("Creating enhanced features from NHL data...")
        features = features.sort_values(['date']).reset_index(drop=True)

        # Helper utilities for leak-free, vectorized rolling calculations
        def lagged_rolling_mean(group_col: str, value_col: str, window: int, min_periods: int = 2, fill_value: Optional[float] = None) -> pd.Series:
            if value_col not in features.columns:
                return pd.Series(fill_value, index=features.index)
            series = (
                features.groupby(group_col)[value_col]
                .transform(lambda s: s.shift().rolling(window, min_periods=min_periods).mean())
            )
            return series.fillna(fill_value) if fill_value is not None else series

        def lagged_rolling_sum(group_col: str, value_col: str, window: int, min_periods: int = 2, fill_value: Optional[float] = None) -> pd.Series:
            if value_col not in features.columns:
                return pd.Series(fill_value, index=features.index)
            series = (
                features.groupby(group_col)[value_col]
                .transform(lambda s: s.shift().rolling(window, min_periods=min_periods).sum())
            )
            return series.fillna(fill_value) if fill_value is not None else series

        def lagged_ewm(group_col: str, value_col: str, alpha: float, min_periods: int = 2, fill_value: Optional[float] = None) -> pd.Series:
            if value_col not in features.columns:
                return pd.Series(fill_value, index=features.index)
            series = (
                features.groupby(group_col)[value_col]
                .transform(lambda s: s.shift().ewm(alpha=alpha, min_periods=min_periods).mean())
            )
            return series.fillna(fill_value) if fill_value is not None else series
        
        # Team performance metrics (leak-free: shift before rolling)
        for team in ['home', 'away']:
            team_col = f'{team}_team'
            goals_col = f'{team}_goals'
            opp_goals_col = f'{"away" if team == "home" else "home"}_goals'

            for window in [3, 5, 10]:
                features[f'{team}_gpg_l{window}'] = lagged_rolling_mean(team_col, goals_col, window, fill_value=3.0)
                features[f'{team}_gag_l{window}'] = lagged_rolling_mean(team_col, opp_goals_col, window, fill_value=3.0)
        
        # Combined metrics
        features['combined_gpg'] = (features['home_gpg_l5'].fillna(3.0) + features['away_gpg_l5'].fillna(3.0)) / 2
        features['combined_gag'] = (features['home_gag_l5'].fillna(3.0) + features['away_gag_l5'].fillna(3.0)) / 2
        features['expected_pace'] = features['combined_gpg'] + features['combined_gag']
        features['pace_variance'] = abs(features['home_gpg_l5'].fillna(3.0) - features['away_gpg_l5'].fillna(3.0))
        lagged_pace = features['expected_pace'].shift()
        pace_mean = lagged_pace.expanding(min_periods=5).mean()
        pace_std = lagged_pace.expanding(min_periods=5).std().replace(0.0, np.nan)
        features['pace_zscore'] = ((lagged_pace - pace_mean) / pace_std).fillna(0.0)

        # Opponent-adjusted strength-of-schedule approximation using Elo and opponent rates
        try:
            opp_strength = []
            for _, row in features[['home_team','away_team','home_elo','away_elo']].iterrows():
                opp_strength.append(0.5 * float(row['home_elo']) + 0.5 * float(row['away_elo']))
            features['sos_elo'] = pd.Series(opp_strength, index=features.index).fillna(1500.0)
        except Exception:
            features['sos_elo'] = 1500.0
        
        # Venue effects (leak-free: shift then expanding mean)
        features['venue_total_avg'] = (
            features.groupby('venue')['total_goals']
            .apply(lambda s: s.shift().expanding().mean())
            .reset_index(level=0, drop=True)
            .fillna(6.2)
        )
        features['altitude_bonus'] = np.where(
            features['venue'].str.contains('Ball Arena|Pepsi Center', case=False, na=False), 0.3, 0
        )
        # Rink bias adjustments (simplified; positive values inflate shot/xG)
        rink_bias_map = {
            'Rogers Place': 0.08, 'Scotiabank Arena': -0.03, 'Madison Square Garden': -0.02,
            'T-Mobile Arena': 0.02, 'Ball Arena': 0.04, 'Canadian Tire Centre': -0.01
        }
        features['rink_bias'] = features['venue'].map(rink_bias_map).fillna(0.0)
        
        # Back-to-back detection
        # Rest days computed across full team timelines (merge back to wide form)
        long_rows = []
        for _, row in features[['date', 'home_team', 'away_team']].iterrows():
            long_rows.append({'date': row['date'], 'team': row['home_team']})
            long_rows.append({'date': row['date'], 'team': row['away_team']})
        long_df = pd.DataFrame(long_rows).sort_values(['team', 'date'])
        long_df['rest_days'] = long_df.groupby('team')['date'].diff().dt.days
        long_df['rest_days'] = long_df['rest_days'].fillna(3)
        # Schedule density windows (robust on datetimes): count prior games within X days
        def count_in_window(dts: pd.Series, days: int) -> pd.Series:
            arr = pd.to_datetime(dts, errors='coerce').values
            counts = []
            for i in range(len(arr)):
                start = arr[i] - np.timedelta64(days, 'D')
                # find first index j with arr[j] >= start
                j = np.searchsorted(arr, start, side='left')
                counts.append(i - j + 1)
            return pd.Series(counts, index=dts.index)
        long_df['games_last_4d'] = long_df.groupby('team', group_keys=False)['date'].apply(lambda s: count_in_window(s, 4)).astype(int)
        long_df['games_last_6d'] = long_df.groupby('team', group_keys=False)['date'].apply(lambda s: count_in_window(s, 6)).astype(int)
        long_df['is_3in4'] = (long_df['games_last_4d'] >= 3).astype(int)
        long_df['is_4in6'] = (long_df['games_last_6d'] >= 4).astype(int)

        # Home/away streaks and road trip length
        # Reconstruct home/away for each team-date
        ha_rows = []
        for _, row in features[['date','home_team','away_team']].iterrows():
            ha_rows.append({'date': row['date'], 'team': row['home_team'], 'is_home': 1})
            ha_rows.append({'date': row['date'], 'team': row['away_team'], 'is_home': 0})
        ha_df = pd.DataFrame(ha_rows).sort_values(['team','date'])
        def streak(arr):
            out = []
            cur = None
            run = 0
            for v in arr:
                if cur is None or v != cur:
                    cur = v; run = 1
                else:
                    run += 1
                out.append(run if v == 1 else -run)
            return out
        ha_df['home_streak_signed'] = ha_df.groupby('team')['is_home'].transform(lambda s: pd.Series(streak(list(s))))
        # Road trip length is consecutive away games (negative streak length)
        ha_df['road_trip_len'] = ha_df['home_streak_signed'].apply(lambda x: abs(x) if x < 0 else 0)

        # Map back schedule density and streaks
        home_key = features[['date', 'home_team']].rename(columns={'home_team': 'team'})
        away_key = features[['date', 'away_team']].rename(columns={'away_team': 'team'})
        density_cols = ['is_3in4','is_4in6']
        streak_cols = ['home_streak_signed','road_trip_len']
        dens_home = home_key.merge(long_df[['team','date']+density_cols], on=['team','date'], how='left')
        dens_away = away_key.merge(long_df[['team','date']+density_cols], on=['team','date'], how='left')
        streak_home = home_key.merge(ha_df[['team','date']+streak_cols], on=['team','date'], how='left')
        streak_away = away_key.merge(ha_df[['team','date']+streak_cols], on=['team','date'], how='left')
        features['home_3in4'] = dens_home['is_3in4'].fillna(0).astype(int)
        features['away_3in4'] = dens_away['is_3in4'].fillna(0).astype(int)
        features['home_4in6'] = dens_home['is_4in6'].fillna(0).astype(int)
        features['away_4in6'] = dens_away['is_4in6'].fillna(0).astype(int)
        features['home_home_streak'] = streak_home['home_streak_signed'].fillna(0).astype(int)
        features['away_home_streak'] = streak_away['home_streak_signed'].fillna(0).astype(int)
        features['home_road_trip_len'] = streak_home['road_trip_len'].fillna(0).astype(int)
        features['away_road_trip_len'] = streak_away['road_trip_len'].fillna(0).astype(int)

        # Map back to home/away columns for each game date and team
        home_key = features[['date', 'home_team']].rename(columns={'home_team': 'team'})
        away_key = features[['date', 'away_team']].rename(columns={'away_team': 'team'})
        home_rest = home_key.merge(long_df, on=['date', 'team'], how='left')['rest_days'].fillna(3)
        away_rest = away_key.merge(long_df, on=['date', 'team'], how='left')['rest_days'].fillna(3)

        features['home_rest_days'] = home_rest
        features['away_rest_days'] = away_rest
        features['home_b2b'] = (features['home_rest_days'] == 1).astype(int)
        features['away_b2b'] = (features['away_rest_days'] == 1).astype(int)
        features['b2b_penalty'] = (features['home_b2b'] * -0.2) + (features['away_b2b'] * -0.3)
        features['rest_diff'] = (features['home_rest_days'] - features['away_rest_days']).fillna(0.0)
        features['schedule_density_diff'] = (
            features['home_3in4'] - features['away_3in4'] +
            0.5 * (features['home_4in6'] - features['away_4in6'])
        )
        travel_series = features['travel_km'].fillna(0.0) if 'travel_km' in features.columns else pd.Series(0.0, index=features.index)
        features['travel_fatigue_index'] = travel_series / 1000.0 - 0.15 * features['rest_diff']
        
        # Rivalry detection
        rivalry_pairs = [
            ('TOR', 'MTL'), ('BOS', 'MTL'), ('NYR', 'NYI'), ('PHI', 'PIT'),
            ('EDM', 'CGY'), ('VAN', 'CGY'), ('DAL', 'STL'), ('CHI', 'DET')
        ]
        
        features['rivalry_game'] = features.apply(
            lambda x: any((x['home_team'], x['away_team']) in [(r[0], r[1]), (r[1], r[0])] 
                         for r in rivalry_pairs), axis=1
        ).astype(int)
        features['rivalry_boost'] = features['rivalry_game'] * 0.25
        
        # Season timing
        features['season_day'] = (features['date'] - features['date'].min()).dt.days
        features['season_progress'] = features['season_day'] / 200.0
        features['late_season'] = (features['season_progress'] > 0.8).astype(int)
        # Time of day / weekend
        try:
            local_hours = []
            for _, row in features[['date']].iterrows():
                gd = pd.to_datetime(row['date'], utc=True, errors='coerce')
                tz = os.getenv('SCHEDULE_TZ', 'US/Eastern')
                try:
                    local = gd.tz_convert(tz)
                except Exception:
                    local = gd
                local_hours.append((int(local.hour), int(local.weekday())))
            lh = pd.DataFrame(local_hours, columns=['hour_local','weekday'])
            features['hour_local'] = lh['hour_local']
            features['is_weekend'] = lh['weekday'].isin([5,6]).astype(int)
            features['is_early'] = (features['hour_local'] < 13).astype(int)
            features['is_late'] = (features['hour_local'] >= 21).astype(int)
        except Exception:
            features['hour_local'] = 19
            features['is_weekend'] = 0
            features['is_early'] = 0
            features['is_late'] = 0
        
        # Predictive base
        features['base_total_prediction'] = (
            features['combined_gpg'] * 0.35 +
            features['venue_total_avg'] * 0.25 +
            features['expected_pace'] * 0.20 +
            6.0 * 0.20
        )
        
        features['total_adjustments'] = (
            features['rivalry_boost'] +
            features['b2b_penalty'] +
            features['altitude_bonus'] +
            (features['late_season'] * -0.15)
        )
        
        features['final_prediction_base'] = features['base_total_prediction'] + features['total_adjustments']
        # Special teams composite metrics (higher => special teams edge for OVER)
        def _safe_series(col: str, default: float = 2.0) -> pd.Series:
            if col not in features.columns:
                return pd.Series(default, index=features.index)
            return pd.to_numeric(features[col], errors='coerce').fillna(default)
        home_pp = _safe_series('home_pp_xgf60')
        away_pp = _safe_series('away_pp_xgf60')
        home_pk = _safe_series('home_pk_xga60', default=2.0)
        away_pk = _safe_series('away_pk_xga60', default=2.0)
        features['special_teams_index'] = (home_pp - away_pk) + (away_pp - home_pk)
        features['special_teams_diff'] = (home_pp - away_pp)

        # Timezone-based features (coarse mapping by team)
        def team_tz_offset(team: str) -> int:
            # Offsets in hours relative to US/Eastern: Eastern=0, Central=-1, Mountain=-2, Pacific=-3
            eastern = {"BOS","BUF","CAR","CBJ","DET","FLA","MTL","NJD","NYI","NYR","OTT","PHI","PIT","TBL","TOR","WSH"}
            central = {"CHI","DAL","MIN","NSH","STL","WPG"}
            mountain = {"COL","UTA"}
            pacific = {"ANA","LAK","SJS","SEA","VAN","VGK","EDM"}
            t = str(team or '').upper()
            if t in eastern:
                return 0
            if t in central:
                return -1
            if t in mountain:
                return -2
            if t in pacific:
                return -3
            return 0

        features['home_tz_offset'] = features['home_team'].apply(team_tz_offset)
        features['away_tz_offset'] = features['away_team'].apply(team_tz_offset)
        features['timezone_diff'] = features['home_tz_offset'] - features['away_tz_offset']
        # Simple travel penalty: away team on B2B crossing timezones
        features['travel_penalty'] = (features['away_b2b'] * features['timezone_diff'].abs() * -0.05)
        features['total_adjustments'] += features['travel_penalty'] + (features['rink_bias'] * 0.05)
        features['final_prediction_base'] = features['base_total_prediction'] + features['total_adjustments']

        # Optional goalie/injury placeholders (will be populated for today's games if status file provided)
        features['home_goalie_adj'] = 0.0
        features['away_goalie_adj'] = 0.0
        features['injury_penalty_adj'] = 0.0
        features['total_adjustments'] += (features['home_goalie_adj'] + features['away_goalie_adj'] + features['injury_penalty_adj'])
        features['final_prediction_base'] = features['base_total_prediction'] + features['total_adjustments']

        # ---------------- Advanced team stats: proxies and EWMAs ----------------
        # Proxies for 5v5 xGF/60 and HDCF/60 using available shots and goals (fallback when detailed feed not available)
        # Coefficients are heuristic; replace with real feed when available
        for team in ['home', 'away']:
            shots_col = f'{team}_shots'
            opp = 'away' if team == 'home' else 'home'
            opp_shots_col = f'{opp}_shots'
            pp_goals_col = f'{team}_pp_goals'
            pp_opps_col = f'{team}_pp_opps'

            # Rolling means (leak-free)
            features[f'{team}_shots_l5'] = (
                features.groupby(f'{team}_team')[shots_col]
                .apply(lambda s: s.shift().rolling(5, min_periods=2).mean())
                .reset_index(level=0, drop=True)
            )
            features[f'{opp}_shots_l5'] = (
                features.groupby(f'{team}_team')[opp_shots_col]
                .apply(lambda s: s.shift().rolling(5, min_periods=2).mean())
                .reset_index(level=0, drop=True)
            )

            # 5v5 xGF/60 proxy ~ shots_l5 * 0.055
            features[f'{team}_5v5_xgf60'] = (features[f'{team}_shots_l5'].fillna(30.0) * 0.055)
            # 5v5 HDCF/60 proxy ~ shots_l5 * 0.35
            features[f'{team}_5v5_hdcf60'] = (features[f'{team}_shots_l5'].fillna(30.0) * 0.35)

            # PP xGF/60 proxy ~ (pp_goals/opps) * 3.0 (scaled) with guards
            pp_rate = (features[pp_goals_col].fillna(0.0) / features[pp_opps_col].replace(0, np.nan)).fillna(0.0)
            features[f'{team}_pp_xgf60'] = (pp_rate * 3.0)
            # PK xGA/60 proxy ~ opponent PP xGF/60
            features[f'{team}_pk_xga60'] = features[f'{opp}_pp_xgf60'] if f'{opp}_pp_xgf60' in features else 0.0

        # Rolling GSAx proxy: negative deviation of goals against from rolling median baseline
        # Use team conceded (opponent goals) vs rolling median 2.9
        for team in ['home', 'away']:
            opp_goals_col = f'{"away" if team == "home" else "home"}_goals'
            series = features.groupby(f'{team}_team')[opp_goals_col].apply(lambda s: s.shift().ewm(alpha=0.3, min_periods=2).mean()).reset_index(level=0, drop=True)
            features[f'{team}_gsax_ewm'] = (2.9 - series.fillna(2.9))

        # Travel distance (km) between away city and home city as a static mapping
        def team_coords(abbr: str) -> Tuple[float, float]:
            m = {
                'ANA': (33.807, -117.876), 'ARI': (33.531, -112.262), 'BOS': (42.366, -71.062), 'BUF': (42.875, -78.876),
                'CAR': (35.803, -78.721), 'CBJ': (39.969, -83.007), 'CGY': (51.037, -114.052), 'CHI': (41.880, -87.674),
                'COL': (39.748, -105.007), 'DAL': (32.790, -96.810), 'DET': (42.341, -83.055), 'EDM': (53.571, -113.457),
                'FLA': (26.158, -80.325), 'LAK': (33.944, -118.401), 'MIN': (44.944, -93.101), 'MTL': (45.496, -73.569),
                'NJD': (40.733, -74.171), 'NSH': (36.159, -86.778), 'NYI': (40.722, -73.590), 'NYR': (40.750, -73.993),
                'OTT': (45.296, -75.925), 'PHI': (39.901, -75.173), 'PIT': (40.446, -80.005), 'SEA': (47.622, -122.354),
                'SJS': (37.334, -121.901), 'STL': (38.631, -90.200), 'TBL': (27.942, -82.451), 'TOR': (43.643, -79.379),
                'UTA': (40.768, -111.901), 'VAN': (49.277, -123.108), 'VGK': (36.102, -115.178), 'WPG': (49.892, -97.143),
                'WSH': (38.898, -77.020)
            }
            return m.get(str(abbr).upper(), (40.0, -95.0))

        def haversine_km(lat1, lon1, lat2, lon2):
            r = 6371.0088
            p = np.pi/180.0
            dlat = (lat2 - lat1) * p
            dlon = (lon2 - lon1) * p
            a = np.sin(dlat/2) ** 2 + np.cos(lat1*p) * np.cos(lat2*p) * np.sin(dlon/2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            return float(r * c)

        travel_vals = []
        for _, row in features[['home_team','away_team']].iterrows():
            (lat_h, lon_h) = team_coords(row['home_team'])
            (lat_a, lon_a) = team_coords(row['away_team'])
            travel_vals.append(haversine_km(lat_a, lon_a, lat_h, lon_h))
        features['travel_km'] = pd.Series(travel_vals, index=features.index).fillna(0.0)
        # East/West direction (approx): sign of longitude difference for away->home
        try:
            ew_dir = []
            for _, row in features[['home_team','away_team']].iterrows():
                (_, lon_h) = team_coords(row['home_team'])
                (_, lon_a) = team_coords(row['away_team'])
                ew_dir.append(1 if (lon_h - lon_a) < 0 else (-1 if (lon_h - lon_a) > 0 else 0))
            features['travel_direction_ew'] = pd.Series(ew_dir, index=features.index)
        except Exception:
            features['travel_direction_ew'] = 0

        # Shot quality and creation proxies: rush chances, rebounds created, slot attempts
        for team in ['home', 'away']:
            # Use hdcf and shots proxies to derive creation metrics
            shots_l5 = features.get(f'{team}_shots_l5', pd.Series(30.0, index=features.index)).fillna(30.0)
            hdcf60 = features.get(f'{team}_5v5_hdcf60', pd.Series(10.0, index=features.index)).fillna(10.0)
            # Rush chances correlate with transition rate; proxy from shots and hdcf
            features[f'{team}_rush60'] = (0.20 * shots_l5) + (0.15 * hdcf60)
            # Rebounds created scale with shots on target and slot volume
            features[f'{team}_rebounds60'] = (0.12 * shots_l5) + (0.08 * hdcf60)
            # Slot attempts approximate as blend of HDCF and overall volume
            features[f'{team}_slot60'] = (0.60 * hdcf60) + (0.10 * shots_l5)

        # Finishing delta (over/under xG) using EWM goals vs xGF proxy
        for team in ['home', 'away']:
            gpg_ewm = features.get(f'{team}_gpg_ewm', pd.Series(3.0, index=features.index)).fillna(3.0)
            xgf60 = features.get(f'{team}_5v5_xgf60', pd.Series(2.4, index=features.index)).fillna(2.4)
            # Convert xGF/60 to per-game proxy with scale; heuristic factor 0.6
            features[f'{team}_finish_delta'] = gpg_ewm - (0.6 * xgf60)

        # Shooting and save percentage trends (rolling 5 games, leak-free)
        for team in ['home', 'away']:
            team_col = f'{team}_team'
            goals_col = f'{team}_goals'
            shots_col = f'{team}_shots'
            opp_goals_col = f'{"away" if team == "home" else "home"}_goals'
            opp_shots_col = f'{"away" if team == "home" else "home"}_shots'

            goals_roll = lagged_rolling_sum(team_col, goals_col, window=5, min_periods=2, fill_value=0.0)
            shots_roll = lagged_rolling_sum(team_col, shots_col, window=5, min_periods=2, fill_value=0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                shoot_pct = goals_roll / shots_roll.replace(0.0, np.nan)
            features[f'{team}_shoot_pct_l5'] = shoot_pct.clip(0.05, 0.4).fillna(0.1)

            goals_allowed_roll = lagged_rolling_sum(team_col, opp_goals_col, window=5, min_periods=2, fill_value=0.0)
            shots_faced_roll = lagged_rolling_sum(team_col, opp_shots_col, window=5, min_periods=2, fill_value=0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                save_pct = 1.0 - (goals_allowed_roll / shots_faced_roll.replace(0.0, np.nan))
            features[f'{team}_save_pct_l5'] = save_pct.clip(0.75, 0.99).fillna(0.92)

        # EWM variants for stability/recency
        for team in ['home', 'away']:
            team_col = f'{team}_team'
            opp_goals_col = f'{"away" if team=="home" else "home"}_goals'
            features[f'{team}_gpg_ewm'] = lagged_ewm(team_col, f'{team}_goals', alpha=0.3, fill_value=3.0)
            features[f'{team}_gag_ewm'] = lagged_ewm(team_col, opp_goals_col, alpha=0.3, fill_value=3.0)
            # EWM for xGF/60 and HDCF/60 proxies
            for metric in ['5v5_xgf60', '5v5_hdcf60']:
                col = f'{team}_{metric}'
                prefix = metric.split('_')[0]
                base_series = features[col] if col in features.columns else pd.Series(0.0, index=features.index)
                ewm_series = lagged_ewm(team_col, col, alpha=0.3, fill_value=None)
                features[f"{team}_{prefix}_ewm"] = ewm_series.fillna(base_series.fillna(0.0))

        # Score-state pace proxies: use EWM goal differential by team
        try:
            long_gd_rows = []
            for _, row in features[['date','home_team','away_team','home_goals','away_goals']].iterrows():
                gd = float(row['home_goals']) - float(row['away_goals']) if pd.notna(row['home_goals']) and pd.notna(row['away_goals']) else 0.0
                long_gd_rows.append({'date': row['date'], 'team': row['home_team'], 'gd': gd})
                long_gd_rows.append({'date': row['date'], 'team': row['away_team'], 'gd': -gd})
            gd_df = pd.DataFrame(long_gd_rows).sort_values(['team','date'])
            gd_df['gd_ewm'] = gd_df.groupby('team')['gd'].apply(lambda s: s.shift().ewm(alpha=0.25, min_periods=2).mean()).reset_index(level=0, drop=True)
            # Map back to game rows
            features = features.merge(gd_df.rename(columns={'team':'home_team'})[['date','home_team','gd_ewm']].rename(columns={'gd_ewm':'home_gd_ewm'}), on=['date','home_team'], how='left')
            features = features.merge(gd_df.rename(columns={'team':'away_team'})[['date','away_team','gd_ewm']].rename(columns={'gd_ewm':'away_gd_ewm'}), on=['date','away_team'], how='left')
            features['home_gd_ewm'] = features['home_gd_ewm'].fillna(0.0)
            features['away_gd_ewm'] = features['away_gd_ewm'].fillna(0.0)
            # Adjust expected pace: teams with positive GD tend to slow; negative GD accelerate
            pace_adj = (-0.05 * features['home_gd_ewm']) + (0.05 * features['away_gd_ewm'])
            features['expected_pace'] = features['expected_pace'] + pace_adj
        except Exception:
            pass

        # Elo team ratings computed in a leak-free fashion (update after recording current game features)
        try:
            initial_elo = 1500.0
            k_factor = 24.0
            home_adv = 35.0
            team_to_elo: Dict[str, float] = {}
            home_elos: List[float] = []
            away_elos: List[float] = []
            for _, row in features[['home_team','away_team','home_goals','away_goals']].iterrows():
                home = str(row['home_team'])
                away = str(row['away_team'])
                h_elo = team_to_elo.get(home, initial_elo)
                a_elo = team_to_elo.get(away, initial_elo)
                home_elos.append(h_elo)
                away_elos.append(a_elo)
                # Update after using current pre-game ratings; only if scores available
                try:
                    hg = float(row['home_goals'])
                    ag = float(row['away_goals'])
                    if not np.isnan(hg) and not np.isnan(ag) and (hg + ag) >= 0:
                        home_win = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
                        exp_home = 1.0 / (1.0 + 10.0 ** (-(h_elo + home_adv - a_elo) / 400.0))
                        team_to_elo[home] = h_elo + k_factor * (home_win - exp_home)
                        team_to_elo[away] = a_elo + k_factor * ((1.0 - home_win) - (1.0 - exp_home))
                except Exception:
                    pass
            features['home_elo'] = pd.Series(home_elos, index=features.index)
            features['away_elo'] = pd.Series(away_elos, index=features.index)
            features['elo_diff'] = features['home_elo'] - features['away_elo']
        except Exception:
            features['home_elo'] = 0.0
            features['away_elo'] = 0.0
            features['elo_diff'] = 0.0

        # True strength-of-schedule (opponent-adjusted) via simple least-squares team O/D ratings
        try:
            # Build equations: for each game, model goals_for ~ off(team) - def(opp) + H
            # We'll estimate combined rating r_team = off(team) - def(team) for tractability
            teams = pd.Index(sorted(set(features['home_team'].astype(str)) | set(features['away_team'].astype(str))))
            team_to_idx = {t: i for i, t in enumerate(teams)}
            rows_A = []
            rows_b = []
            for _, row in features[['home_team','away_team','home_goals','away_goals']].iterrows():
                ht = str(row['home_team']); at = str(row['away_team'])
                if pd.notna(row['home_goals']):
                    a = np.zeros(len(teams)); a[team_to_idx[ht]] = 1.0; a[team_to_idx[at]] = -1.0
                    rows_A.append(a); rows_b.append(float(row['home_goals']))
                if pd.notna(row['away_goals']):
                    a = np.zeros(len(teams)); a[team_to_idx[at]] = 1.0; a[team_to_idx[ht]] = -1.0
                    rows_A.append(a); rows_b.append(float(row['away_goals']))
            if len(rows_A) >= len(teams):
                A = np.vstack(rows_A)
                b = np.array(rows_b)
                # ridge regularization for stability
                lam = 0.1
                ATA = A.T @ A + lam * np.eye(A.shape[1])
                ATb = A.T @ b
                r = np.linalg.solve(ATA, ATb)
                ratings = {t: float(r[team_to_idx[t]]) for t in teams}
                features['home_sos_true'] = features['home_team'].map(ratings).astype(float).fillna(0.0)
                features['away_sos_true'] = features['away_team'].map(ratings).astype(float).fillna(0.0)
                features['sos_true_diff'] = features['home_sos_true'] - features['away_sos_true']
            else:
                features['home_sos_true'] = 0.0
                features['away_sos_true'] = 0.0
                features['sos_true_diff'] = 0.0
        except Exception:
            features['home_sos_true'] = 0.0
            features['away_sos_true'] = 0.0
            features['sos_true_diff'] = 0.0
        
        # Fill NaNs
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if features[col].isna().any():
                if 'goals' in col.lower() or 'total' in col.lower():
                    features[col] = features[col].fillna(6.2)
                else:
                    features[col] = features[col].fillna(0.0)
        
        return features

    def train_goal_models(self, enhanced_df: pd.DataFrame) -> None:
        """Train Poisson regression models for home and away goals using current feature set."""
        if not self.feature_names:
            print("⚠️  No feature names set; call prepare_model_data first or ensure features are created.")
            # Infer features from enhanced_df
            return
        try:
            df = enhanced_df.dropna(subset=['home_goals','away_goals']).copy()
            Xg = df[self.feature_names].copy()
            Xg_scaled = self.scaler.fit_transform(Xg)
            y_home = df['home_goals'].astype(float)
            y_away = df['away_goals'].astype(float)
            self.home_goal_mu_model = PoissonRegressor(alpha=0.5, max_iter=1000)
            self.away_goal_mu_model = PoissonRegressor(alpha=0.5, max_iter=1000)
            self.home_goal_mu_model.fit(Xg_scaled, y_home)
            self.away_goal_mu_model.fit(Xg_scaled, y_away)
            if self.total_model is None:
                self.total_model = {}
            self.total_model['home_goal_mu_model'] = self.home_goal_mu_model
            self.total_model['away_goal_mu_model'] = self.away_goal_mu_model
            # Learn empirical correlation between home and away goals
            # Shrink correlation towards 0 and bound tighter
            rho = 0.12
            try:
                if len(df) >= 30:
                    corr = float(np.corrcoef(df['home_goals'].values, df['away_goals'].values)[0,1])
                    if np.isfinite(corr):
                        # Empirical shrinkage
                        rho = 0.6 * float(corr)
                        rho = float(max(0.0, min(0.35, rho)))
            except Exception:
                pass
            self.total_model['poisson_rho'] = rho
            print("✅ Trained Poisson goal models (home and away)")
        except Exception as e:
            print(f"⚠️  Failed to train goal models: {e}")

    def load_xg_adjustments(self, todays_games: pd.DataFrame, xg_path: Optional[str], baseline_total: float = 6.2, clamp_abs: float = 2.0) -> Dict[str, float]:
        """Load expected goals per game and convert to total adjustment.

        xg JSON schema (flexible): keyed by game_id or matchup 'AWAY@HOME', values like:
        {"<id>": {"home_xg": 3.1, "away_xg": 2.9}} or {"AWAY@HOME": 6.2}
        Adjustment is (home_xg+away_xg) - market total baseline (6.2 default), clamped.
        """
        if not xg_path or not os.path.exists(xg_path):
            return {}
        try:
            with open(xg_path, 'r') as f:
                raw = json.load(f)
        except Exception:
            return {}

        gid_to_matchup: Dict[str, str] = {}
        for _, g in todays_games.iterrows():
            gid_to_matchup[str(g.get('game_id'))] = f"{g.get('away_team')}@{g.get('home_team')}"

        def clamp(x: float, a: float, b: float) -> float:
            return max(a, min(b, x))

        adj: Dict[str, float] = {}
        if isinstance(raw, dict):
            for key, val in raw.items():
                try:
                    if isinstance(val, dict):
                        hxg = float(val.get('home_xg', 0.0))
                        axg = float(val.get('away_xg', 0.0))
                        total = hxg + axg
                    else:
                        total = float(val)
                    target_gids = []
                    if key in gid_to_matchup:
                        target_gids.append(key)
                    else:
                        # treat key as matchup
                        for gid, mk in gid_to_matchup.items():
                            if mk == key:
                                target_gids.append(gid)
                    for gid in target_gids:
                        # Prefer explicit args; fallback to env
                        baseline = float(baseline_total) if baseline_total is not None else float(os.getenv('XG_BASELINE_TOTAL', 6.2))
                        clamp_v = float(clamp_abs) if clamp_abs is not None else float(os.getenv('XG_CLAMP_ABS', 2.0))
                        adj[gid] = clamp(total - baseline, -clamp_v, clamp_v)
                except Exception:
                    continue
        return adj

    def load_environment(self, todays_games: pd.DataFrame, env_path: Optional[str]) -> Dict[str, Dict[str, float]]:
        """Load environment info for today's games. JSON keyed by game_id or 'AWAY@HOME'.
        Example:
        {
          "2025010101": {"outdoor": true, "start_hour_local": 13, "temp_f": 28, "wind_mph": 5},
          "TOR@BOS": {"outdoor": false, "start_hour_local": 19}
        }
        """
        env: Dict[str, Dict[str, float]] = {}
        if not env_path or not os.path.exists(env_path):
            return env
        try:
            with open(env_path, 'r') as f:
                raw = json.load(f)
        except Exception:
            return env
        gid_to_matchup: Dict[str, str] = {}
        gid_to_local_time: Dict[str, Tuple[int, int]] = {}
        sched_tz = os.getenv('SCHEDULE_TZ', 'US/Eastern')
        for _, g in todays_games.iterrows():
            gid = str(g.get('game_id'))
            gid_to_matchup[gid] = f"{g.get('away_team')}@{g.get('home_team')}"
            # Derive local hour/minute from game date
            try:
                gd = pd.to_datetime(g.get('date'), utc=True, errors='coerce')
                if pd.isna(gd):
                    hour_local, minute_local = 19, 0
                else:
                    try:
                        dt_local = gd.tz_convert(sched_tz)
                    except Exception:
                        # Treat as naive UTC then convert
                        dt_local = gd.tz_localize('UTC', nonexistent='shift_forward', ambiguous='NaT').tz_convert(sched_tz)
                    hour_local = int(dt_local.hour)
                    minute_local = int(dt_local.minute)
            except Exception:
                hour_local, minute_local = 19, 0
            gid_to_local_time[gid] = (hour_local, minute_local)

        for gid, mk in gid_to_matchup.items():
            rec = raw.get(gid, raw.get(mk))
            if isinstance(rec, dict):
                default_hour, default_minute = gid_to_local_time.get(gid, (19, 0))
                env[gid] = {
                    'outdoor': bool(rec.get('outdoor', False)),
                    'start_hour_local': int(rec.get('start_hour_local', default_hour) or default_hour),
                    'start_minute_local': int(rec.get('start_minute_local', default_minute) or default_minute),
                    'temp_f': float(rec.get('temp_f', 70.0) or 70.0),
                    'wind_mph': float(rec.get('wind_mph', 0.0) or 0.0)
                }
        return env

    def load_lineup_strength(self, lineup_path: Optional[str]) -> Dict[str, float]:
        """Load simple team lineup strength ratings from CSV.
        CSV columns: team, lineup_strength (e.g., aggregate RAPM/xGAR scaled ~1.0-3.0)
        """
        team_strength: Dict[str, float] = {}
        if not lineup_path or not os.path.exists(lineup_path):
            return team_strength
        try:
            df = pd.read_csv(lineup_path)
            if 'team' not in df.columns:
                for alt in ['Team','TEAM','abbr','Abbreviation']:
                    if alt in df.columns:
                        df = df.rename(columns={alt: 'team'})
                        break
            if 'lineup_strength' not in df.columns:
                for alt in ['strength','rating','rapm','xgar']:
                    if alt in df.columns:
                        df = df.rename(columns={alt: 'lineup_strength'})
                        break
            if 'team' in df.columns and 'lineup_strength' in df.columns:
                for _, row in df.iterrows():
                    try:
                        t = str(row['team']).upper()
                        v = float(row['lineup_strength'])
                        team_strength[t] = v
                    except Exception:
                        continue
        except Exception:
            return {}
        return team_strength

    def write_environment_template(self, todays_games: pd.DataFrame, out_path: str, overwrite_today: bool = False) -> None:
        """Write/merge an environment.json template for today's games.

        For each game_id, fill defaults:
          outdoor: false, start_hour_local/start_minute_local derived from game date if available else 19:00, temp_f: 70, wind_mph: 0
        """
        try:
            env: Dict[str, Dict[str, float]] = {}
            if os.path.exists(out_path):
                try:
                    with open(out_path, 'r') as f:
                        env = json.load(f)
                    if not isinstance(env, dict):
                        env = {}
                except Exception:
                    env = {}

            for _, g in todays_games.iterrows():
                gid = str(g.get('game_id'))
                if not gid:
                    continue
                # Try to derive local hour/minute from date if present
                start_hour = 19
                start_minute = 0
                try:
                    dt = g.get('date') or g.get('gameDate')
                    if pd.notna(dt):
                        dtp = pd.to_datetime(dt, errors='coerce', utc=True)
                        if pd.notna(dtp):
                            local_tz = os.getenv('SCHEDULE_TZ', 'US/Eastern')
                            ldt = dtp.tz_convert(local_tz)
                            start_hour = int(ldt.hour)
                            start_minute = int(ldt.minute)
                except Exception:
                    pass
                if overwrite_today or gid not in env:
                    env[gid] = {"outdoor": False, "start_hour_local": int(start_hour), "start_minute_local": int(start_minute), "temp_f": 70, "wind_mph": 0}
            with open(out_path, 'w') as f:
                json.dump(env, f, indent=2)
            print(f"✅ Wrote environment template for {len(todays_games)} games to {out_path}")
        except Exception as e:
            print(f"⚠️  Could not write environment template: {e}")

    # ---------------- Real PBP/xG loaders ----------------
    def load_team_rates(self, path_or_url: Optional[str]) -> Optional[pd.DataFrame]:
        """Load team-level rates table with columns like:
        team, xgf60_5v5, xga60_5v5, hdcf60_5v5, hdca60_5v5, pp_xgf60, pk_xga60
        Supports CSV at local path or HTTP(S) URL.
        """
        if not path_or_url:
            return None
        try:
            url_l = str(path_or_url).lower()
            if url_l.startswith(('http://','https://')):
                # If a MoneyPuck HTML page, scrape CSV links
                if url_l.endswith('.htm') or url_l.endswith('.html'):
                    try:
                        # MoneyPuck direct CSV fallback (no HTML parsing required)
                        if 'moneypuck.com' in url_l and 'team' in url_l:
                            from urllib.parse import urljoin
                            base = 'https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/{stage}/teams.csv'
                            now_y = datetime.now().year
                            years = [now_y, now_y-1]
                            stages = ['regular','playoffs']
                            csv_url = None
                            for y in years:
                                for st in stages:
                                    test = base.format(year=y, stage=st)
                                    try:
                                        r = requests.get(test, timeout=20)
                                        if r.ok and len(r.text.splitlines()) > 1:
                                            csv_url = test
                                            break
                                    except Exception:
                                        pass
                                if csv_url:
                                    break
                            if csv_url:
                                r = requests.get(csv_url, timeout=20)
                                r.raise_for_status()
                                df = pd.read_csv(io.StringIO(r.text))
                            else:
                                raise ValueError('MoneyPuck teams.csv not found for recent seasons')
                        else:
                            html = requests.get(path_or_url, timeout=20).text
                        csv_links = re.findall(r'href=["\']([^"\']+\.csv)["\']', html, flags=re.IGNORECASE)
                        csv_url = None
                        for link in csv_links:
                            if 'team' in link.lower() or 'teams' in link.lower():
                                csv_url = link
                                break
                        if csv_url is None and csv_links:
                            csv_url = csv_links[0]
                        if csv_url and not csv_url.lower().startswith(('http://','https://')):
                            from urllib.parse import urljoin
                            csv_url = urljoin(path_or_url, csv_url)
                        if csv_url:
                            r = requests.get(csv_url, timeout=20)
                            r.raise_for_status()
                            df = pd.read_csv(io.StringIO(r.text))
                        else:
                            # Parse HTML tables and pick the first that has a team-like column
                            tables = pd.read_html(io.StringIO(html))
                            picked_df = None
                            for t in tables:
                                # Flatten multiindex columns
                                if isinstance(t.columns, pd.MultiIndex):
                                    t.columns = [' '.join([str(x) for x in c]).strip() for c in t.columns]
                                # Identify a team-like column
                                team_col = None
                                for c in t.columns:
                                    if 'team' in str(c).lower() or str(c).strip().lower() in ['name','team name','abbreviation','abbr']:
                                        team_col = c
                                        break
                                if team_col is None and t.index.name and 'team' in str(t.index.name).lower():
                                    t = t.reset_index()
                                    team_col = t.columns[0]
                                if team_col is not None:
                                    picked_df = t.rename(columns={team_col: 'team'})
                                    break
                            if picked_df is None:
                                raise ValueError('No suitable team table found on page')
                            df = picked_df
                    except Exception:
                        r = requests.get(path_or_url, timeout=20)
                        r.raise_for_status()
                        tables = pd.read_html(io.StringIO(r.text))
                        df = tables[0]
                else:
                    r = requests.get(path_or_url, timeout=20)
                    r.raise_for_status()
                    df = pd.read_csv(io.StringIO(r.text))
            else:
                df = pd.read_csv(path_or_url)
            # Normalize team column (create 'team' as NHL abbreviation if missing)
            # Flatten columns if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join([str(x) for x in c]).strip() for c in df.columns]
            if 'team' not in df.columns:
                alt_cols = ['Team','TEAM','abbr','ABBR','Abbreviation','team_name','Team Name','teamName','name','Name','teamname']
                picked = None
                for alt in alt_cols:
                    if alt in df.columns:
                        picked = alt
                        break
                if picked is not None:
                    df = df.rename(columns={picked: 'team'})
                else:
                    # try any column that contains 'team'
                    for c in df.columns:
                        if 'team' in str(c).lower():
                            df = df.rename(columns={c: 'team'})
                            picked = c
                            break
            # Try index name as team
            if 'team' not in df.columns and df.index.name and 'team' in str(df.index.name).lower():
                df = df.reset_index().rename(columns={df.columns[0]: 'team'})
            if 'team' not in df.columns:
                raise ValueError("No team column found in team rates dataset")
            # Map full team names to abbreviations when needed
            def to_abbr(val: str) -> str:
                s = str(val or '').strip()
                su = s.upper()
                # If already looks like tri-code
                if len(su) <= 4 and su.isalpha():
                    return su
                try:
                    name_map = {}
                    try:
                        # Build once from fallback teams
                        name_map = {v['name'].upper(): v['abbreviation'].upper() for v in NHLDataFetcher()._get_fallback_teams().values()}
                    except Exception:
                        pass
                    return name_map.get(su, su[:3])
                except Exception:
                    return su[:3]
            df['team'] = df['team'].astype(str).apply(to_abbr)
            # Heuristic mapping to expected columns
            lower_cols = {c.lower(): c for c in df.columns}
            def find_col(patterns: List[str]) -> Optional[str]:
                for p in patterns:
                    for lc, orig in lower_cols.items():
                        if re.search(p, lc):
                            return orig
                return None
            # Try to build output columns
            out = pd.DataFrame({'team': df['team']})
            # xGF/60 5v5
            c_xgf60 = find_col([r'xg[f]?[a-z_]*60.*5.*v.*5', r'5.*v.*5.*xg[f]?.*60'])
            if c_xgf60 is None:
                c_xgf = find_col([r'xg[f]?\b.*5.*v.*5', r'5.*v.*5.*xg[f]?'])
                c_toi = find_col([r'toi.*5.*v.*5', r'min.*5.*v.*5'])
                if c_xgf is not None and c_toi is not None:
                    out['xgf60_5v5'] = pd.to_numeric(df[c_xgf], errors='coerce') / (pd.to_numeric(df[c_toi], errors='coerce')/60.0)
            else:
                out['xgf60_5v5'] = pd.to_numeric(df[c_xgf60], errors='coerce')
            # HDCF/60 5v5
            c_hdcf60 = find_col([r'hdcf.*60.*5.*v.*5', r'5.*v.*5.*hdcf.*60'])
            if c_hdcf60 is not None:
                out['hdcf60_5v5'] = pd.to_numeric(df[c_hdcf60], errors='coerce')
            # PP xGF/60
            c_ppxgf60 = find_col([r'pp.*xg[f]?.*60', r'power.*play.*xg[f]?.*60'])
            if c_ppxgf60 is not None:
                out['pp_xgf60'] = pd.to_numeric(df[c_ppxgf60], errors='coerce')
            # PK xGA/60 (short-handed against)
            c_pkxga60 = find_col([r'pk.*xg[a]?.*60', r'sh.*xg[a]?.*60', r'penalty.*kill.*xg[a]?.*60'])
            if c_pkxga60 is not None:
                out['pk_xga60'] = pd.to_numeric(df[c_pkxga60], errors='coerce')
            # Merge back with team and drop duplicates
            out = out.groupby('team', as_index=False).first()
            return out
        except Exception as e:
            print(f"⚠️  Failed to load team rates from {path_or_url}: {e}")
            return None

    def load_goalie_gsax(self, path_or_url: Optional[str]) -> Optional[pd.DataFrame]:
        """Load goalie-level GSAx table with columns like:
        goalie, team, gsax_rolling (or gsax), prob_start (optional)
        """
        if not path_or_url:
            return None
        try:
            url_l = str(path_or_url).lower()
            if url_l.startswith(('http://','https://')):
                if url_l.endswith('.htm') or url_l.endswith('.html'):
                    try:
                        if 'moneypuck.com' in url_l and 'goalie' in url_l:
                            base = 'https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/{stage}/goalies.csv'
                            now_y = datetime.now().year
                            years = [now_y, now_y-1]
                            stages = ['regular','playoffs']
                            csv_url = None
                            for y in years:
                                for st in stages:
                                    test = base.format(year=y, stage=st)
                                    try:
                                        r = requests.get(test, timeout=20)
                                        if r.ok and len(r.text.splitlines()) > 1:
                                            csv_url = test
                                            break
                                    except Exception:
                                        pass
                                if csv_url:
                                    break
                            if csv_url:
                                r = requests.get(csv_url, timeout=20)
                                r.raise_for_status()
                                df = pd.read_csv(io.StringIO(r.text))
                            else:
                                raise ValueError('MoneyPuck goalies.csv not found for recent seasons')
                        else:
                            html = requests.get(path_or_url, timeout=20).text
                            df = pd.read_html(io.StringIO(html))[0]
                    except Exception:
                        r = requests.get(path_or_url, timeout=20)
                        r.raise_for_status()
                        df = pd.read_html(io.StringIO(r.text))[0]
                else:
                    r = requests.get(path_or_url, timeout=20)
                    r.raise_for_status()
                    df = pd.read_csv(io.StringIO(r.text))
            else:
                df = pd.read_csv(path_or_url)
            # Normalize columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join([str(x) for x in c]).strip() for c in df.columns]
            # Team column detection
            if 'team' not in df.columns:
                for alt in ['TEAM','Team','abbr','ABBR','Abbreviation','teamAbbrev','teamCode','team_name','teamName','Team Name']:
                    if alt in df.columns:
                        df = df.rename(columns={alt: 'team'})
                        break
            # Goalie name column (optional)
            if 'goalie' not in df.columns:
                for alt in ['goalieName','name','player','Player']:
                    if alt in df.columns:
                        df = df.rename(columns={alt: 'goalie'})
                        break
            # GSAx column detection (robust to MoneyPuck camelCase)
            if 'gsax_rolling' not in df.columns:
                aliases = [
                    'gsax','GSAx','gsaX','goals_saved_above_expected','goalssavedaboveexpected',
                    'goals saved above expected','goalsSavedAboveExpected','goalsSavedAboveExpectedAll',
                    'goals_saved_above_expected_all'
                ]
                for alt in aliases:
                    if alt in df.columns:
                        df = df.rename(columns={alt: 'gsax_rolling'})
                        break
            # If still missing, try any column containing 'saved' and 'expected'
            if 'gsax_rolling' not in df.columns:
                for c in df.columns:
                    lc = c.lower()
                    if 'saved' in lc and 'expected' in lc:
                        df = df.rename(columns={c: 'gsax_rolling'})
                        break
            # Map team full names to abbreviations
            if 'team' in df.columns:
                name_map = {v['name'].upper(): v['abbreviation'].upper() for v in NHLDataFetcher()._get_fallback_teams().values()}
                def to_team_abbr(val: str) -> str:
                    s = str(val or '').strip().upper()
                    if len(s) <= 4 and s.isalpha():
                        return s
                    return name_map.get(s, s[:3])
                df['team'] = df['team'].astype(str).apply(to_team_abbr)
            # Ensure prob_start exists
            if 'prob_start' not in df.columns:
                df['prob_start'] = 1.0
            # Coerce numeric
            if 'gsax_rolling' in df.columns:
                df['gsax_rolling'] = pd.to_numeric(df['gsax_rolling'], errors='coerce').fillna(0.0)
            df['prob_start'] = pd.to_numeric(df['prob_start'], errors='coerce').fillna(1.0).clip(0.0, 1.0)
            return df
        except Exception as e:
            print(f"⚠️  Failed to load goalie GSAx from {path_or_url}: {e}")
            return None
    
    def prepare_model_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare features, target, and dates for modeling (chronological splits)"""
        
        feature_cols = [
            'home_gpg_l3', 'away_gpg_l3', 'home_gpg_l5', 'away_gpg_l5', 'home_gpg_l10', 'away_gpg_l10',
            'home_gag_l3', 'away_gag_l3', 'home_gag_l5', 'away_gag_l5', 'home_gag_l10', 'away_gag_l10',
            'combined_gpg', 'combined_gag', 'expected_pace', 'pace_variance', 'pace_zscore',
            'venue_total_avg', 'altitude_bonus', 'rivalry_boost', 'b2b_penalty',
            'home_b2b', 'away_b2b', 'season_progress', 'late_season', 'rest_diff', 'schedule_density_diff',
            'base_total_prediction', 'total_adjustments', 'final_prediction_base', 'travel_fatigue_index',
            'timezone_diff', 'travel_penalty', 'home_elo', 'away_elo', 'elo_diff',
            'sos_elo', 'home_sos_true', 'away_sos_true', 'sos_true_diff',
            'home_3in4','away_3in4','home_4in6','away_4in6',
            'home_home_streak','away_home_streak','home_road_trip_len','away_road_trip_len',
            'hour_local','is_weekend','is_early','is_late','travel_direction_ew',
            # Advanced stats and EWM variants
            'home_5v5_xgf60', 'away_5v5_xgf60', 'home_5v5_hdcf60', 'away_5v5_hdcf60',
            'home_pp_xgf60', 'away_pp_xgf60', 'home_pk_xga60', 'away_pk_xga60',
            'home_gsax_ewm', 'away_gsax_ewm', 'travel_km',
            'home_gpg_ewm', 'away_gpg_ewm', 'home_gag_ewm', 'away_gag_ewm',
            'home_5v5_ewm', 'away_5v5_ewm', 'home_shoot_pct_l5', 'away_shoot_pct_l5',
            'home_save_pct_l5', 'away_save_pct_l5',
            # Shot quality/creation
            'home_rush60','away_rush60','home_rebounds60','away_rebounds60','home_slot60','away_slot60',
            'home_finish_delta','away_finish_delta',
            'special_teams_index','special_teams_diff'
        ]
        # Add rink bias proxy and penalties
        extra_optional = [
            'penalties_drawn60', 'penalties_taken60', 'ref_goals_gm', 'rink_bias'
        ]
        for col in extra_optional:
            if col in df.columns:
                feature_cols.append(col)
        
        # Filter to available columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Create feature matrix
        X = df[available_features].copy()
        y = df['total_goals'].copy()
        dates = df['date'].copy()
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        dates = dates[mask]
        
        self.feature_names = available_features

        if 'ref_goals_gm' in df.columns:
            try:
                ref_series = pd.to_numeric(df.loc[mask, 'ref_goals_gm'], errors='coerce')
            except Exception:
                ref_series = None
            if ref_series is not None:
                try:
                    valid_ref = ref_series.dropna()
                    if len(valid_ref):
                        self.ref_goal_baseline = float(valid_ref.mean())
                except Exception:
                    pass
        if self.ref_goal_baseline is None or not np.isfinite(self.ref_goal_baseline):
            try:
                self.ref_goal_baseline = float(os.getenv('REF_GOAL_BASELINE', 6.2))
            except Exception:
                self.ref_goal_baseline = 6.2
        
        print(f"Prepared {len(X)} samples with {len(available_features)} features")
        
        return X, y, dates
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, dates: Optional[pd.Series] = None) -> Dict:
        """Train the over/under prediction model with time-series split and residual calibration"""
        
        if len(X) < 20:
            print("Warning: Very limited training data. Model performance may be poor.")
        
        # Split data (time-series when dates available)
        if dates is not None and len(dates) == len(X):
            ordered_indices = dates.sort_values().index
            X_sorted = X.loc[ordered_indices]
            y_sorted = y.loc[ordered_indices]
            split_index = int(len(X_sorted) * 0.75)
            X_train, X_test = X_sorted.iloc[:split_index], X_sorted.iloc[split_index:]
            y_train, y_test = y_sorted.iloc[:split_index], y_sorted.iloc[split_index:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Time-series cross-validation setup for hyperparameter tuning
        n_splits = 4 if len(X_train) >= 60 else 3
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Hyperparameter spaces (kept compact for speed)
        rf_base = RandomForestRegressor(random_state=42)
        rf_params = {
            'n_estimators': [150, 250, 350],
            'max_depth': [8, 12, 16, None],
            'min_samples_leaf': [1, 2, 3]
        }
        rf_search = RandomizedSearchCV(
            rf_base, rf_params, n_iter=8, cv=tscv, random_state=42,
            scoring='neg_mean_squared_error', n_jobs=-1, verbose=0
        )
        rf_search.fit(X_train_scaled, y_train)

        gb_base = GradientBoostingRegressor(random_state=42)
        gb_params = {
            'n_estimators': [150, 250, 350],
            'learning_rate': [0.03, 0.05, 0.08, 0.1],
            'max_depth': [2, 3, 4]
        }
        gb_search = RandomizedSearchCV(
            gb_base, gb_params, n_iter=8, cv=tscv, random_state=42,
            scoring='neg_mean_squared_error', n_jobs=-1, verbose=0
        )
        gb_search.fit(X_train_scaled, y_train)

        hgb_base = HistGradientBoostingRegressor(
            random_state=42,
            early_stopping=True,
            loss='squared_error'
        )
        hgb_params = {
            'learning_rate': [0.03, 0.05, 0.08, 0.1],
            'max_iter': [225, 275, 325, 375],
            'max_depth': [None, 3, 5, 7],
            'min_samples_leaf': [10, 15, 20, 30],
            'l2_regularization': [0.0, 0.1, 0.3, 0.6]
        }
        hgb_search = RandomizedSearchCV(
            hgb_base, hgb_params, n_iter=8, cv=tscv, random_state=42,
            scoring='neg_mean_squared_error', n_jobs=-1, verbose=0
        )
        hgb_search.fit(X_train_scaled, y_train)

        ridge_base = Ridge()
        ridge_params = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
        ridge_search = RandomizedSearchCV(
            ridge_base, ridge_params, n_iter=min(6, len(ridge_params['alpha'])), cv=tscv, random_state=42,
            scoring='neg_mean_squared_error', n_jobs=-1, verbose=0
        )
        ridge_search.fit(X_train_scaled, y_train)

        # Collect tuned models
        model_order = ['rf', 'gb', 'ridge', 'hgb']
        models = {
            'rf': rf_search.best_estimator_,
            'gb': gb_search.best_estimator_,
            'ridge': ridge_search.best_estimator_,
            'hgb': hgb_search.best_estimator_
        }

        # Learn stacking weights via OOF predictions across time-series splits
        weights_array = np.array([0.32, 0.28, 0.20, 0.20], dtype=float)
        try:
            X_all = X_sorted
            y_all = y_sorted
            if len(X_all) >= 40:
                tscv_w = TimeSeriesSplit(n_splits=min(5, max(3, len(X_all)//20)))
                P = None
                oof = {name: np.full(len(X_all), np.nan) for name in model_order}
                for tr_idx, val_idx in tscv_w.split(X_all):
                    scaler_fold = StandardScaler()
                    X_tr_s = scaler_fold.fit_transform(X_all.iloc[tr_idx])
                    X_val_s = scaler_fold.transform(X_all.iloc[val_idx])
                    for name in model_order:
                        est = models[name]
                        est_fold = type(est)(**est.get_params())
                        est_fold.fit(X_tr_s, y_all.iloc[tr_idx])
                        oof[name][val_idx] = est_fold.predict(X_val_s)
                first_key = model_order[0]
                valid = ~np.isnan(oof[first_key])
                if valid.any():
                    P = np.vstack([oof[name][valid] for name in model_order]).T
                y_oof = y_all.iloc[valid].values
                if P is not None and P.size:
                    coefs, *_ = np.linalg.lstsq(P, y_oof, rcond=None)
                    coefs = np.clip(coefs, 0.0, None)
                    if coefs.sum() > 0:
                        weights_array = coefs / coefs.sum()
        except Exception:
            pass
        if not np.isfinite(weights_array).all() or weights_array.sum() <= 0:
            weights_array = np.ones(len(model_order), dtype=float) / len(model_order)
        else:
            weights_array = weights_array / weights_array.sum()

        # Final train/test fit for reporting and persistence
        trained_models = {}
        test_preds = []
        for name in model_order:
            est = models[name]
            est_final = type(est)(**est.get_params())
            est_final.fit(X_train_scaled, y_train)
            trained_models[name] = est_final
            test_preds.append(est_final.predict(X_test_scaled))
        test_preds = np.vstack(test_preds)  # shape (K, N)
        ensemble_pred = (weights_array.reshape(-1, 1) * test_preds).sum(axis=0)

        # Poisson model for total goals (non-negative)
        poisson_model = PoissonRegressor(alpha=0.5, max_iter=1000)
        # Estimate overdispersion via residual variance; fit NB parameters per training set
        nb_params = None
        gp_model = None
        try:
            poisson_model.fit(X_train_scaled, y_train)
            # Overdispersion estimate: variance/mean > 1 suggests NB
            mu_train = poisson_model.predict(X_train_scaled)
            resid = y_train.values - mu_train
            var_hat = float(np.var(resid)) + float(np.mean(mu_train))
            mean_hat = float(np.mean(mu_train))
            if mean_hat > 0 and var_hat > mean_hat:
                # NB parameterization: variance = mean + mean^2/k => k = mean^2/(var-mean)
                k = (mean_hat ** 2) / max(1e-6, (var_hat - mean_hat))
                nb_params = {'k': float(max(1e-6, k))}
            # Try generalized Poisson (COM-Poisson proxy) if enabled
            if STATSMODELS_AVAILABLE and len(X_train_scaled) >= 100 and bool(getattr(self, 'use_compoisson', False)):
                try:
                    import statsmodels.api as sm
                    Xg = sm.add_constant(X_train_scaled)
                    gp_model = GeneralizedPoisson(y_train.values, Xg).fit(disp=0)
                except Exception:
                    gp_model = None
        except Exception:
            poisson_model = None

        # Poisson expected totals on the test set
        if poisson_model is not None:
            poisson_pred = poisson_model.predict(X_test_scaled)
        else:
            poisson_pred = None
        
        # Store trained model
        self.total_model = {
            'models': trained_models,
            'model_order': model_order,
            'weights': weights_array.tolist(),
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'poisson_model': poisson_model,
            'nb_params': nb_params,
            'gp_model': gp_model
        }
        
        # Calculate metrics and residual-based uncertainty (plus conformal radius)
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        mae = mean_absolute_error(y_test, ensemble_pred)
        residual_std = float(np.std(y_test.values - ensemble_pred)) if len(y_test) > 1 else 0.85

        # Persist residual std for downstream uncertainty
        self.total_model['residual_std'] = residual_std

        # Symmetric conformal interval via absolute residual quantiles (configurable quantile)
        abs_res = np.abs(y_test.values - ensemble_pred)
        if len(abs_res) >= 5:
            self.conformal_q80 = float(np.quantile(abs_res, 0.80))
            self.conformal_q90 = float(np.quantile(abs_res, 0.90))
            q = float(getattr(self, 'ci_quantile', 0.90))
            self.conformal_radius = float(np.quantile(abs_res, max(0.5, min(0.99, q))))
        else:
            self.conformal_q80 = 0.8
            self.conformal_q90 = 1.0
            self.conformal_radius = self.conformal_q90
        self.total_model['conformal_q80'] = self.conformal_q80
        self.total_model['conformal_q90'] = self.conformal_q90
        self.total_model['conformal_radius'] = self.conformal_radius
        
        # Over/under accuracy (using average line of 6.5)
        test_lines = np.full(len(y_test), 6.5)
        actual_over = (y_test.values > test_lines).astype(int)
        predicted_over = (ensemble_pred > test_lines).astype(int)
        ou_accuracy = (actual_over == predicted_over).mean()

        # Quick probability calibration check (Gaussian proxy with residual_std)
        try:
            std_dev = residual_std if residual_std > 0 else 0.85
            raw_over_prob = norm.sf(test_lines, loc=ensemble_pred, scale=std_dev)
            # Brier score (lower is better)
            brier = float(brier_score_loss(actual_over, raw_over_prob))
            # Log loss (guard against 0/1 probs)
            eps = 1e-6
            logloss = float(log_loss(actual_over, np.clip(raw_over_prob, eps, 1-eps)))
        except Exception:
            brier = None
            logloss = None

        # Fit isotonic calibration mapping from raw over-prob proxies (Gaussian) to outcomes
        try:
            # Proxy over-prob using Gaussian on test set for calibration only
            std_dev = residual_std if residual_std > 0 else 0.85
            raw_over_prob = norm.sf(test_lines, loc=ensemble_pred, scale=std_dev)
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(raw_over_prob, actual_over)
            self.total_model['iso_over'] = iso
        except Exception:
            self.total_model['iso_over'] = None
        
        return {
            'rmse': rmse,
            'mae': mae,
            'over_under_accuracy': ou_accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features_used': len(self.feature_names),
            'residual_std': residual_std,
            'brier': brier,
            'logloss': logloss
        }
    
    def predict_game(self, game_features: np.ndarray, betting_line: float = 6.5, over_american_odds: int = -110, under_american_odds: int = -110, odds_source: Optional[str] = None, consensus_total: Optional[float] = None) -> OverUnderPrediction:
        """Predict over/under for a single game"""
        
        if not self.total_model:
            raise ValueError("Model not trained")

        ref_goal_value: Optional[float] = None
        ref_goal_adjustment: float = 0.0
        ref_idx: Optional[int] = None
        try:
            ref_idx = self.feature_names.index('ref_goals_gm')
        except (ValueError, AttributeError):
            ref_idx = None
        if ref_idx is not None and 0 <= ref_idx < len(game_features):
            try:
                candidate_val = float(game_features[ref_idx])
            except Exception:
                candidate_val = np.nan
            if np.isfinite(candidate_val):
                ref_goal_value = candidate_val
        
        # Scale features
        features_scaled = self.scaler.transform(game_features.reshape(1, -1))
        
        # Get predictions from ensemble
        models = self.total_model.get('models', {})
        weights = np.array(self.total_model.get('weights', []), dtype=float)
        model_order = self.total_model.get('model_order', list(models.keys()))

        preds: List[float] = []
        for name in model_order:
            estimator = models.get(name)
            if estimator is None:
                preds.append(0.0)
                continue
            try:
                preds.append(float(estimator.predict(features_scaled)[0]))
            except Exception:
                preds.append(0.0)
        if not preds:
            raise ValueError("No trained ensemble models available for prediction")

        preds_arr = np.array(preds, dtype=float)
        if len(weights) != len(preds_arr):
            if len(preds_arr) == 0:
                weights = np.array([1.0], dtype=float)
            elif len(weights) == 0:
                weights = np.ones(len(preds_arr), dtype=float)
            elif len(weights) > len(preds_arr):
                weights = weights[:len(preds_arr)]
            else:
                pad_len = len(preds_arr) - len(weights)
                pad_values = np.full(pad_len, weights.mean() if weights.size else 1.0)
                weights = np.concatenate([weights, pad_values])
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(len(preds_arr), dtype=float)
        weights = weights / weights.sum()
        ensemble_pred = float(np.dot(weights, preds_arr))

        # Poisson expected total
        poisson_model = self.total_model.get('poisson_model')
        poisson_mu = None
        if poisson_model is not None:
            try:
                poisson_mu = float(poisson_model.predict(features_scaled)[0])
            except Exception:
                poisson_mu = None

        # Blend predictions conservatively if Poisson available
        if poisson_mu is not None:
            predicted_total = 0.6 * ensemble_pred + 0.4 * poisson_mu
        else:
            predicted_total = ensemble_pred
        
        if ref_goal_value is not None:
            baseline_val = getattr(self, 'ref_goal_baseline', None)
            if baseline_val is None or not np.isfinite(baseline_val):
                try:
                    baseline_val = float(os.getenv('REF_GOAL_BASELINE', 6.2))
                except Exception:
                    baseline_val = 6.2
            try:
                weight_val = float(getattr(self, 'ref_goal_weight', 0.05))
            except Exception:
                weight_val = 0.05
            if weight_val != 0:
                ref_goal_adjustment = float(weight_val * (ref_goal_value - baseline_val))
                ref_goal_adjustment = float(max(-0.35, min(0.35, ref_goal_adjustment)))
                predicted_total = float(predicted_total + ref_goal_adjustment)
        else:
            ref_goal_adjustment = 0.0
        
        # Calculate probabilities with calibrated uncertainty and push handling
        edge = predicted_total - betting_line

        def american_to_decimal(american: int) -> float:
            if american >= 100:
                return 1.0 + (american / 100.0)
            else:
                return 1.0 + (100.0 / abs(american))

        def poisson_over_under_probs(mu: float, line_value: float) -> Tuple[float, float, float]:
            # Use exact Poisson tail probabilities
            if mu <= 0:
                return 0.0, 1.0, 0.0
            is_integer_line = abs(line_value - round(line_value)) < 1e-9
            if is_integer_line:
                L = int(round(line_value))
                push_p = float(poisson.pmf(L, mu))
                under_p = float(poisson.cdf(L - 1, mu)) if L > 0 else 0.0
                over_p = float(1.0 - poisson.cdf(L, mu))
                return over_p, under_p, push_p
            else:
                floor_over = int(np.floor(line_value)) + 1
                under_floor = int(np.floor(line_value))
                over_p = float(1.0 - poisson.cdf(over_floor := floor_over - 1, mu))
                under_p = float(poisson.cdf(under_floor, mu))
                return over_p, under_p, 0.0

        # Prefer bivariate Poisson MC; else NB/Poisson totals; else Gaussian
        home_mu_model = (self.total_model or {}).get('home_goal_mu_model')
        away_mu_model = (self.total_model or {}).get('away_goal_mu_model')
        if home_mu_model is not None and away_mu_model is not None:
            try:
                hm = float(home_mu_model.predict(features_scaled)[0])
                am = float(away_mu_model.predict(features_scaled)[0])
                # Empirical correlation if learned, otherwise small positive default
                rho = float((self.total_model or {}).get('poisson_rho', 0.15))
                # Monte Carlo
                sims = 50000
                # Copula-based dependence approximation via Gaussian correlation on uniforms
                z = np.random.multivariate_normal([0,0], [[1,rho],[rho,1]], size=sims)
                u = norm.cdf(z)
                home_goals = poisson.ppf(u[:,0], hm)
                away_goals = poisson.ppf(u[:,1], am)
                totals = home_goals + away_goals
                is_integer_line = abs(betting_line - round(betting_line)) < 1e-9
                if is_integer_line:
                    push_prob = float(np.mean(totals == round(betting_line)))
                    under_prob = float(np.mean(totals < round(betting_line)))
                    over_prob = float(1.0 - under_prob - push_prob)
                else:
                    push_prob = 0.0
                    under_prob = float(np.mean(totals < betting_line))
                    over_prob = float(1.0 - under_prob)
            except Exception:
                home_mu_model = None
                away_mu_model = None

        if home_mu_model is None or away_mu_model is None:
            if poisson_mu is not None and poisson_mu > 0:
                # Negative binomial adjustment for overdispersion if available
                nb = (self.total_model or {}).get('nb_params')
                gp_model = (self.total_model or {}).get('gp_model')
                if nb and isinstance(nb.get('k'), (int, float)) and nb['k'] > 0:
                    k = float(nb['k'])
                    # Approximate NB tail by summing pmf up to floor(line)
                    L = int(np.floor(betting_line))
                    # NB parameter p using mean mu = k*(1-p)/p => p = k/(k+mu)
                    p = k / (k + poisson_mu)
                    under_p = float(nbinom.cdf(L, k, p))
                    if abs(betting_line - round(betting_line)) < 1e-9:
                        push_prob = float(nbinom.pmf(int(round(betting_line)), k, p))
                        over_prob = float(1.0 - under_p - push_prob)
                    else:
                        push_prob = 0.0
                        over_prob = float(1.0 - under_p)
                elif gp_model is not None:
                    try:
                        import statsmodels.api as sm
                        Xg = sm.add_constant(features_scaled)
                        mu_gp = float(gp_model.predict(Xg)[0])
                        # Use Poisson tail with mu_gp as proxy (GP pmf not directly used here)
                        over_prob, under_prob, push_prob = poisson_over_under_probs(max(1e-6, mu_gp), betting_line)
                    except Exception:
                        over_prob, under_prob, push_prob = poisson_over_under_probs(poisson_mu, betting_line)
                else:
                    over_prob, under_prob, push_prob = poisson_over_under_probs(poisson_mu, betting_line)
            else:
                std_dev = float(self.total_model.get('residual_std', 0.85))
                is_integer_line = abs(betting_line - round(betting_line)) < 1e-9
                if is_integer_line:
                    lower = betting_line - 0.25
                    upper = betting_line + 0.25
                    push_prob = float(max(0.0, norm.cdf(upper, predicted_total, std_dev) - norm.cdf(lower, predicted_total, std_dev)))
                    under_prob = float(norm.cdf(lower, predicted_total, std_dev))
                    over_prob = float(1.0 - norm.cdf(upper, predicted_total, std_dev))
                else:
                    push_prob = 0.0
                    under_prob = float(norm.cdf(betting_line, predicted_total, std_dev))
                    over_prob = float(1.0 - under_prob)

        # Normalize to guard against numerical drift
        total_prob = over_prob + under_prob + push_prob
        if total_prob > 0:
            over_prob /= total_prob
            under_prob /= total_prob
            push_prob /= total_prob

        # Market implied and no-vig fair probabilities
        def american_to_decimal(american: int) -> float:
            if american >= 100:
                return 1.0 + (american / 100.0)
            else:
                return 1.0 + (100.0 / abs(american))

        def decimal_to_implied_prob(decimal_odds: float) -> float:
            return 1.0 / max(decimal_odds, 1e-9)

        over_dec = american_to_decimal(over_american_odds)
        under_dec = american_to_decimal(under_american_odds)
        imp_over = decimal_to_implied_prob(over_dec)
        imp_under = decimal_to_implied_prob(under_dec)
        vig = max(1e-9, imp_over + imp_under)
        fair_over = imp_over / vig
        fair_under = imp_under / vig

        # EV using American odds
        over_b = over_dec - 1.0
        under_b = under_dec - 1.0
        ev_over = over_prob * over_b - (1 - over_prob) * 1.0
        ev_under = under_prob * under_b - (1 - under_prob) * 1.0
        # No-vig EV using fair probabilities
        ev_over_novig = fair_over * over_b - (1 - fair_over) * 1.0
        ev_under_novig = fair_under * under_b - (1 - fair_under) * 1.0
        
        # Isotonic calibration of probability towards historical outcomes (if calibration fitted)
        try:
            iso_over = (self.total_model or {}).get('iso_over')
            if isinstance(iso_over, IsotonicRegression):
                over_prob = float(np.clip(iso_over.predict([over_prob])[0], 0.0, 1.0))
                under_prob = float(max(0.0, min(1.0, 1.0 - over_prob - push_prob)))
        except Exception:
            pass

        # Betting recommendation thresholds (tuned slightly post-calibration)
        min_edge = 0.22
        min_prob = 0.56
        
        if abs(edge) < min_edge or max(over_prob, under_prob) < min_prob:
            recommendation = 'No Bet'
            kelly_size = 0.0
        elif edge > min_edge and over_prob > min_prob:
            recommendation = 'OVER'
            # Kelly fraction for decimal odds b (= over_b)
            base_prob = fair_over if getattr(self, 'kelly_use_fair', False) else over_prob
            k = (base_prob * over_dec - 1.0) / over_b if over_b > 0 else 0.0
            # Downscale Kelly by dispersion factor if present
            dispersion_factor = 1.0
            try:
                dispersion_factor = float(max(0.6, min(1.0, 1.0 - 0.5 * float(getattr(self, 'current_dispersion_std', 0.0) / 0.5))))
            except Exception:
                dispersion_factor = 1.0
            stale_factor = 1.0
            try:
                stale_factor = float(max(0.6, min(1.0, float(getattr(self, 'current_stale_factor', 1.0)))))
            except Exception:
                stale_factor = 1.0
            kelly_raw = float(max(0.0, k))
            kelly_scaled = kelly_raw * float(getattr(self, 'kelly_mult', 0.5)) * dispersion_factor * stale_factor
            kelly_size = float(min(float(getattr(self, 'kelly_cap_pct', 2.0))/100.0, kelly_scaled)) * 100.0
        elif edge < -min_edge and under_prob > min_prob:
            recommendation = 'UNDER'
            base_prob = fair_under if getattr(self, 'kelly_use_fair', False) else under_prob
            k = (base_prob * under_dec - 1.0) / under_b if under_b > 0 else 0.0
            dispersion_factor = 1.0
            try:
                dispersion_factor = float(max(0.6, min(1.0, 1.0 - 0.5 * float(getattr(self, 'current_dispersion_std', 0.0) / 0.5))))
            except Exception:
                dispersion_factor = 1.0
            stale_factor = 1.0
            try:
                stale_factor = float(max(0.6, min(1.0, float(getattr(self, 'current_stale_factor', 1.0)))))
            except Exception:
                stale_factor = 1.0
            kelly_raw = float(max(0.0, k))
            kelly_scaled = kelly_raw * float(getattr(self, 'kelly_mult', 0.5)) * dispersion_factor * stale_factor
            kelly_size = float(min(float(getattr(self, 'kelly_cap_pct', 2.0))/100.0, kelly_scaled)) * 100.0
        else:
            recommendation = 'No Bet'
            kelly_size = 0.0
        
        # Confidence score
        conf_base = 0.5 + abs(edge) * 0.15 + (max(over_prob, under_prob) - 0.5) * 0.8
        # Conformal interval width penalty (wider intervals => lower confidence)
        interval_penalty = 0.0
        if self.conformal_q90 is not None:
            interval_penalty = min(0.15, self.conformal_q90 / 10.0)
        confidence = float(min(0.95, max(0.05, conf_base - interval_penalty)))
        
        # Conformal confidence interval around predicted total
        qrad = self.total_model.get('conformal_radius') or self.total_model.get('conformal_q90')
        ci_lower = float(max(0.0, predicted_total - qrad)) if qrad is not None else None
        ci_upper = float(predicted_total + qrad) if qrad is not None else None

        # Line deviation (proxy for CLV if our line differs from consensus)
        line_diff = None
        try:
            if consensus_total is not None:
                line_diff = float(betting_line - float(consensus_total))
        except Exception:
            line_diff = None

        return OverUnderPrediction(
            game_id="", home_team="", away_team="",
            predicted_total=predicted_total, betting_line=betting_line,
            over_american_odds=over_american_odds, under_american_odds=under_american_odds,
            over_probability=over_prob, under_probability=under_prob, push_probability=push_prob,
            confidence=confidence, expected_value_over=ev_over, expected_value_under=ev_under,
            recommendation=recommendation, edge=edge, kelly_bet_size=kelly_size,
            ci_lower=ci_lower, ci_upper=ci_upper,
            market_over_prob=imp_over, market_under_prob=imp_under,
            fair_over_prob=fair_over, fair_under_prob=fair_under,
            odds_source=odds_source,
            ev_over_novig=ev_over_novig, ev_under_novig=ev_under_novig,
            consensus_total=consensus_total, best_side_total=betting_line, line_diff=line_diff,
            ref_goals_gm=ref_goal_value, ref_goal_adjustment=ref_goal_adjustment
        )

    def get_betting_lines(self, todays_games: pd.DataFrame) -> Dict[str, float]:
        """[Backward compatible] Load only totals line as float."""
        lines: Dict[str, float] = {}
        odds = self.get_betting_odds(todays_games)
        for gid, rec in odds.items():
            lines[gid] = float(rec.get('total', 6.5))
        return lines

    def get_betting_odds(self, todays_games: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Load betting odds including total and American prices; supports multi-book consensus.

        JSON schema can be one of:
        - Single book per game: {"<game_id>": {"total":6.5, "over":-115, "under":-105}}
        - By matchup string: {"VGK@FLA": {"total":6.0, "over":-110, "under":-110}}
        - Number shorthand: {"TOR@BOS": 6.0}
        - Multi-book array: {"<game_id>": [{"book":"PINN","total":6.5,"over":-112,"under":+100}, {"book":"DK","total":6.5,"over":-110,"under":-105}]}
        We choose consensus total as median of totals and best prices for over/under across books. Also return 'consensus_total' and 'source'.
        """
        odds: Dict[str, Dict[str, float]] = {}
        odds_path = os.getenv('ODDS_JSON_PATH', 'odds.json')
        odds_data = None
        try:
            if os.path.exists(odds_path):
                with open(odds_path, 'r') as f:
                    odds_data = json.load(f)
                print(f"✅ Loaded betting odds from {odds_path}")
        except Exception as e:
            print(f"⚠️  Failed to load odds from {odds_path}: {e}")

        for _, game in todays_games.iterrows():
            game_id = str(game.get('game_id'))
            home = str(game.get('home_team', 'HOME'))
            away = str(game.get('away_team', 'AWAY'))
            matchup_key = f"{away}@{home}"
            rec: Dict[str, float] = {}

            if isinstance(odds_data, dict):
                raw = odds_data.get(game_id, odds_data.get(matchup_key))
            else:
                raw = None

            if isinstance(raw, list):
                # Multi-book: aggregate
                totals = []
                over_prices = []
                under_prices = []
                best_over = None
                best_under = None
                best_over_book = None
                best_under_book = None
                books_list = []
                for entry in raw:
                    try:
                        t = float(entry.get('total', np.nan))
                        if not np.isnan(t):
                            totals.append(t)
                    except Exception:
                        pass
                    try:
                        o = int(entry.get('over', -110))
                        u = int(entry.get('under', -110))
                        over_prices.append(o)
                        under_prices.append(u)
                        # Select best price for bettor: highest decimal payout for the side
                        # For over: prefer higher decimal odds
                        def american_to_decimal(american: int) -> float:
                            return 1.0 + (american / 100.0) if american >= 100 else 1.0 + (100.0 / abs(american))
                        dec_o = american_to_decimal(o)
                        dec_u = american_to_decimal(u)
                        if best_over is None or dec_o > american_to_decimal(best_over):
                            best_over = o
                            best_over_book = entry.get('book')
                        if best_under is None or dec_u > american_to_decimal(best_under):
                            best_under = u
                            best_under_book = entry.get('book')
                        books_list.append({'book': entry.get('book'), 'book_key': entry.get('book'), 'total': entry.get('total'), 'over': o, 'under': u})
                    except Exception:
                        pass
                consensus_total = float(np.median(totals)) if totals else 6.5
                rec['consensus_total'] = consensus_total
                rec['total'] = consensus_total
                rec['over'] = int(best_over if best_over is not None else (-110))
                rec['under'] = int(best_under if best_under is not None else (-110))
                if totals:
                    try:
                        rec['dispersion_total_std'] = float(np.std(totals))
                        rec['dispersion_total_range'] = float(max(totals) - min(totals))
                    except Exception:
                        pass
                if best_over_book or best_under_book:
                    rec['source'] = f"best_over:{best_over_book or 'n/a'},best_under:{best_under_book or 'n/a'}"
                    rec['best_over_book'] = best_over_book
                    rec['best_under_book'] = best_under_book
                rec['books'] = books_list
            elif isinstance(raw, dict):
                try:
                    rec['total'] = float(raw.get('total', 6.5))
                except Exception:
                    rec['total'] = 6.5
                try:
                    rec['over'] = int(raw.get('over', -110))
                except Exception:
                    rec['over'] = -110
                try:
                    rec['under'] = int(raw.get('under', -110))
                except Exception:
                    rec['under'] = -110
            elif raw is not None:
                # Probably a number-like total
                try:
                    rec['total'] = float(raw)
                except Exception:
                    rec['total'] = 6.5
                rec['over'] = -110
                rec['under'] = -110
            else:
                # Fallback heuristic if not provided
                base_line = float(np.random.uniform(5.5, 7.5))
                rec['total'] = round(base_line * 2.0) / 2.0
                rec['over'] = -110
                rec['under'] = -110

            odds[game_id] = rec

        return odds

    def get_betting_odds_realtime(self, todays_games: pd.DataFrame, api_key_env: str = 'ODDS_API_KEY', regions: str = 'us,us2,eu,uk,au', timeout_s: int = 25, retries: int = 3, dispersion_all: bool = False) -> Dict[str, Dict[str, float]]:
        """Fetch realtime totals odds from The Odds API (or compatible) and map to our schema, aggregating every available sportsbook region by default.

        Requires an API key in environment variable specified by api_key_env. This example uses
        The Odds API v4 (https://the-odds-api.com/) for demonstration and may need adjustments to your provider.
        """
        api_key = os.getenv(api_key_env)
        if not api_key:
            # Fallback to social_config.json
            try:
                with open('social_config.json', 'r') as f:
                    cfg = json.load(f)
                api_key = (cfg.get('odds') or {}).get('api_key')
            except Exception:
                api_key = None
        if not api_key:
            print("⚠️  No API key set for realtime odds. Set ODDS_API_KEY or add to social_config.json (odds.api_key).")
            return self.get_betting_odds(todays_games)

        # Build a set of matchup keys we care about
        matchups = set()
        gid_to_matchup = {}
        for _, g in todays_games.iterrows():
            home = str(g.get('home_team', 'HOME'))
            away = str(g.get('away_team', 'AWAY'))
            mk = f"{away}@{home}"
            matchups.add(mk)
            gid_to_matchup[str(g.get('game_id'))] = mk

        results: Dict[str, Dict[str, float]] = {}
        default_regions = 'us,us2,eu,uk,au'
        region_tokens = [seg.strip() for seg in str(regions or '').split(',') if seg.strip()]
        if region_tokens and any(seg.lower() == 'all' for seg in region_tokens):
            regions_clean = default_regions
        else:
            regions_clean = ",".join(region_tokens)
        if not regions_clean:
            regions_clean = default_regions

        # The Odds API parameters
        params = {
            'apiKey': api_key,
            'regions': regions_clean,
            'markets': 'totals',
            'oddsFormat': 'american'
        }
        url = 'https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds'
        data = None
        last_err = None
        for attempt in range(1, max(1, retries) + 1):
            try:
                r = requests.get(url, params=params, timeout=max(5, timeout_s))
                r.raise_for_status()
                data = r.json()
                remaining = r.headers.get('x-requests-remaining')
                used = r.headers.get('x-requests-used')
                print(f"✅ Realtime odds fetched: {len(data) if isinstance(data, list) else 0} events (remaining={remaining}, used={used})")
                break
            except Exception as e:
                last_err = e
                backoff = min(5 * attempt, 10)
                print(f"⚠️  Realtime odds fetch failed (attempt {attempt}/{retries}): {e}. Retrying in {backoff}s...")
                try:
                    time.sleep(backoff)
                except Exception:
                    pass
        if data is None:
            print(f"⚠️  Realtime odds ultimately failed: {last_err}")
            return self.get_betting_odds(todays_games)

        # Map provider team names to abbreviations (best-effort fallback)
        def normalize_team(name: str) -> str:
            """Map provider full team names to NHL abbreviations (best effort)."""
            name_u = (name or '').upper()
            mapping = {
                'ANAHEIM DUCKS': 'ANA','ARIZONA COYOTES': 'ARI','BOSTON BRUINS': 'BOS','BUFFALO SABRES': 'BUF',
                'CALGARY FLAMES': 'CGY','CAROLINA HURRICANES': 'CAR','CHICAGO BLACKHAWKS': 'CHI','COLORADO AVALANCHE': 'COL',
                'COLUMBUS BLUE JACKETS': 'CBJ','DALLAS STARS': 'DAL','DETROIT RED WINGS': 'DET','EDMONTON OILERS': 'EDM',
                'FLORIDA PANTHERS': 'FLA','LOS ANGELES KINGS': 'LAK','MINNESOTA WILD': 'MIN','MONTREAL CANADIENS': 'MTL',
                'NASHVILLE PREDATORS': 'NSH','NEW JERSEY DEVILS': 'NJD','NEW YORK ISLANDERS': 'NYI','NEW YORK RANGERS': 'NYR',
                'OTTAWA SENATORS': 'OTT','PHILADELPHIA FLYERS': 'PHI','PITTSBURGH PENGUINS': 'PIT','SAN JOSE SHARKS': 'SJS',
                'SEATTLE KRAKEN': 'SEA','ST. LOUIS BLUES': 'STL','TAMPA BAY LIGHTNING': 'TBL','TORONTO MAPLE LEAFS': 'TOR',
                'UTAH MAMMOTH': 'UTA','VANCOUVER CANUCKS': 'VAN','VEGAS GOLDEN KNIGHTS': 'VGK','WASHINGTON CAPITALS': 'WSH',
                'WINNIPEG JETS': 'WPG'
            }
            return mapping.get(name_u, name_u[:3])

        def american_to_decimal(a: int) -> float:
            try:
                if a >= 100:
                    return 1.0 + (a / 100.0)
                return 1.0 + (100.0 / max(1, abs(a)))
            except Exception:
                return 1.0

        # Aggregate odds for each matchup across all available books
        tmp: Dict[str, List[Dict[str, Any]]] = {}
        # Collect totals from all books (used for consensus/dispersion metrics)
        totals_all: Dict[str, List[float]] = {}
        matched_events = 0
        for ev in data if isinstance(data, list) else []:
            try:
                home = normalize_team(ev.get('home_team'))
                away = normalize_team(ev.get('away_team'))
                mk = f"{away}@{home}"
                if mk not in matchups:
                    continue
                matched_events += 1
                ev_id = ev.get('id') or ev.get('event_id') or ev.get('commence_time')
                for bk in ev.get('bookmakers', []) or []:
                    book_key_raw = (bk.get('key') or '').strip()
                    book_title_raw = (bk.get('title') or '').strip()
                    book_key = book_key_raw or book_title_raw or 'unknown'
                    book_title = book_title_raw or book_key_raw or 'Unknown'
                    last_update = bk.get('last_update')
                    for mkts in bk.get('markets', []) or []:
                        if mkts.get('key') != 'totals':
                            continue
                        lines_by_point: Dict[float, Dict[str, int]] = {}
                        points_recorded: Set[float] = set()
                        for out in mkts.get('outcomes', []) or []:
                            nm = (out.get('name') or out.get('description') or '').strip().lower()
                            pt = out.get('point')
                            pr = out.get('price')
                            try:
                                pt_val = float(pt) if pt is not None else None
                            except Exception:
                                pt_val = None
                            if pt_val is not None:
                                if dispersion_all and pt_val not in points_recorded:
                                    totals_all.setdefault(mk, []).append(pt_val)
                                    points_recorded.add(pt_val)
                                line_entry = lines_by_point.setdefault(pt_val, {})
                            else:
                                line_entry = None
                            try:
                                pr_int = int(pr)
                            except Exception:
                                pr_int = None
                            if pr_int is None or line_entry is None:
                                continue
                            if 'over' in nm:
                                line_entry['over'] = pr_int
                            elif 'under' in nm:
                                line_entry['under'] = pr_int
                        best_line = None
                        best_hold = None
                        for point, prices in lines_by_point.items():
                            if 'over' not in prices or 'under' not in prices:
                                continue
                            hold = abs((1.0 / american_to_decimal(prices['over'])) + (1.0 / american_to_decimal(prices['under'])) - 1.0)
                            if best_line is None or hold < best_hold:
                                best_line = (point, prices['over'], prices['under'])
                                best_hold = hold
                        if best_line:
                            total_point, over_price, under_price = best_line
                            entry = {
                                'total': total_point,
                                'over': over_price,
                                'under': under_price,
                                'book_key': book_key,
                                'book_title': book_title,
                                'event_id': str(ev_id),
                                'last_update': last_update
                            }
                            if dispersion_all:
                                if total_point not in points_recorded:
                                    totals_all.setdefault(mk, []).append(float(total_point))
                            else:
                                totals_all.setdefault(mk, []).append(float(total_point))
                            tmp.setdefault(mk, []).append(entry)
                            break
            except Exception:
                continue

        print(f"✅ Matched realtime odds to {matched_events} of {len(matchups)} matchups")
        # Build result per game id using consensus and best prices across all books
        for _, g in todays_games.iterrows():
            gid = str(g.get('game_id'))
            mk = gid_to_matchup.get(gid)
            if not mk:
                continue
            arr = tmp.get(mk, [])
            if not arr:
                # No FD odds available for this matchup; skip returning odds (no local fallback)
                continue
            totals = [t['total'] for t in arr if t.get('total') is not None]
            overs = [t['over'] for t in arr if t.get('over') is not None]
            unders = [t['under'] for t in arr if t.get('under') is not None]
            # consensus median
            try:
                cons = float(np.median(totals))
            except Exception:
                cons = float(totals[0])
            # best price by highest decimal payout
            best_over = max(overs, key=lambda a: american_to_decimal(a)) if overs else -110
            best_under = max(unders, key=lambda a: american_to_decimal(a)) if unders else -110
            consensus_over = None
            consensus_under = None
            if overs:
                try:
                    consensus_over = int(np.median(overs))
                except Exception:
                    consensus_over = int(overs[0])
            if unders:
                try:
                    consensus_under = int(np.median(unders))
                except Exception:
                    consensus_under = int(unders[0])
            # Determine best books
            best_over_book = None
            best_under_book = None
            best_over_val = -1.0
            best_under_val = -1.0
            for entry in arr:
                o = entry.get('over')
                u = entry.get('under')
                bk_name = entry.get('book_title') or entry.get('book_key')
                if o is not None:
                    od = american_to_decimal(o)
                    if od > best_over_val:
                        best_over_val = od
                        best_over_book = bk_name
                if u is not None:
                    ud = american_to_decimal(u)
                    if ud > best_under_val:
                        best_under_val = ud
                        best_under_book = bk_name
            # include per-book list
            books = []
            for entry in arr:
                books.append({
                    'book': entry.get('book_title') or entry.get('book_key'),
                    'book_key': entry.get('book_key'),
                    'event_id': entry.get('event_id'),
                    'total': entry.get('total'),
                    'over': entry.get('over'),
                    'under': entry.get('under'),
                    'last_update': entry.get('last_update')
                })
            rec_out = {
                'total': cons,
                'over': int(best_over),
                'under': int(best_under),
                'consensus_total': cons,
                'books': books,
                'book_count': len(books),
                'consensus_over': consensus_over,
                'consensus_under': consensus_under,
                'best_over_book': str(best_over_book) if best_over_book is not None else None,
                'best_under_book': str(best_under_book) if best_under_book is not None else None,
                'odds_source': 'the-odds-api:aggregate'
            }
            # Add dispersion metrics from all books if requested
            arr_all = totals_all.get(mk, []) if dispersion_all else totals
            if arr_all:
                try:
                    rec_out['dispersion_total_std'] = float(np.std(arr_all))
                    rec_out['dispersion_total_range'] = float(max(arr_all) - min(arr_all))
                except Exception:
                    pass
            results[gid] = rec_out

        # No local fallback: only return real-time odds per request

        return results

    def get_status_adjustments(self, todays_games: pd.DataFrame, status_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Load goalie/injury status adjustments from JSON and map to game_id.

        Schema (flexible):
        - Object keyed by game_id or matchup ("AWAY@HOME"), or an array of entries with fields:
          {
            "game_id": "..." | "matchup": "AWAY@HOME",
            "home_goalie_adj": 0.1, "away_goalie_adj": -0.05,
            "home_goalie_gsax": 0.2, "away_goalie_gsax": -0.1,
            "home_injuries": ["PLY1", "PLY2"], "away_injuries": ["PLY3"],
            "injury_penalty_adj": -0.06
          }
        Priority: explicit *_adj values override derived values.
        """
        spath = status_path or os.getenv('STATUS_JSON_PATH', 'status.json')
        if not spath or not os.path.exists(spath):
            return {}
        try:
            with open(spath, 'r') as f:
                raw = json.load(f)
        except Exception:
            return {}

        # Build matchup keys
        gid_to_matchup: Dict[str, str] = {}
        for _, g in todays_games.iterrows():
            home = str(g.get('home_team', 'HOME'))
            away = str(g.get('away_team', 'AWAY'))
            gid_to_matchup[str(g.get('game_id'))] = f"{away}@{home}"

        def clamp(x: float, a: float, b: float) -> float:
            return max(a, min(b, x))

        def derive(entry: Dict) -> Dict[str, float]:
            out = {'home_goalie_adj': 0.0, 'away_goalie_adj': 0.0, 'injury_penalty_adj': 0.0}
            # Direct adj wins
            for k in out.keys():
                if isinstance(entry.get(k), (int, float)):
                    out[k] = float(entry.get(k))
            # Derive from GSAX if adj not given
            if not isinstance(entry.get('home_goalie_adj'), (int, float)) and isinstance(entry.get('home_goalie_gsax'), (int, float)):
                out['home_goalie_adj'] = clamp(0.1 * float(entry.get('home_goalie_gsax')), -0.5, 0.5)
            if not isinstance(entry.get('away_goalie_adj'), (int, float)) and isinstance(entry.get('away_goalie_gsax'), (int, float)):
                out['away_goalie_adj'] = clamp(0.1 * float(entry.get('away_goalie_gsax')), -0.5, 0.5)
            # Injuries: small per-player penalty if not given
            if not isinstance(entry.get('injury_penalty_adj'), (int, float)):
                h_inj = entry.get('home_injuries') or []
                a_inj = entry.get('away_injuries') or []
                try:
                    count = len(h_inj) + len(a_inj)
                except Exception:
                    count = 0
                out['injury_penalty_adj'] = -0.03 * float(count)
            return out

        adjustments: Dict[str, Dict[str, float]] = {}
        if isinstance(raw, list):
            for entry in raw:
                try:
                    gid = str(entry.get('game_id')) if entry.get('game_id') is not None else None
                    matchup = str(entry.get('matchup')) if entry.get('matchup') is not None else None
                    target_keys = []
                    if gid is not None:
                        target_keys.append(gid)
                    if matchup is not None:
                        # Map matchup to matching game_ids for today
                        for gk, mk in gid_to_matchup.items():
                            if mk == matchup:
                                target_keys.append(gk)
                    if not target_keys and matchup is None and gid is None:
                        continue
                    adj = derive(entry)
                    for k in target_keys:
                        adjustments[k] = adj
                except Exception:
                    continue
        elif isinstance(raw, dict):
            for key, entry in raw.items():
                try:
                    # key can be game_id or matchup
                    target_keys = []
                    if key in gid_to_matchup:
                        target_keys.append(key)
                    else:
                        # assume matchup
                        for gk, mk in gid_to_matchup.items():
                            if mk == key:
                                target_keys.append(gk)
                    adj = derive(entry if isinstance(entry, dict) else {})
                    for k in target_keys:
                        adjustments[k] = adj
                except Exception:
                    continue
        return adjustments

    # ---------------- Referee auto-mapping helpers ----------------
    @staticmethod
    def _clean_ref_name(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        try:
            txt = html_parser.unescape(str(raw))
            txt = txt.replace('\u2019', "'")
            txt = re.sub(r"<[^>]+>", " ", txt)
            txt = txt.replace('\xa0', ' ').replace('\u2013', ' ').replace('\u2014', ' ')
            txt = re.sub(r"\s*\(#?\d+\)\s*", " ", txt)
            txt = re.sub(r"\s*#\d+\b", " ", txt)
            txt = re.sub(r"[|/&]", " ", txt)
            txt = re.sub(r"\s+", " ", txt).strip(" ,;:-")
            if len(txt) < 3 or not any(ch.isalpha() for ch in txt):
                return None
            if any(ch.isdigit() for ch in txt):
                return None
            words = [w.strip(" .") for w in txt.split() if w.strip(" .")]
            if len(words) < 2 or len(words) > 5:
                return None
            suffixes = {'jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv'}
            if words and words[-1].lower() in suffixes:
                words = words[:-1]
            if not words:
                return None
            if any(word and word[0].islower() for word in words if word and word[0].isalpha()):
                return None
            lowered = " ".join(words).lower()
            if ' at ' in lowered or ' vs ' in lowered:
                return None
            banned_words = {
                'game', 'games', 'birthplace', 'career', 'goals', 'goal', 'penl', 'pim',
                'records', 'win', 'loss', 'notes', 'lines', 'official', 'penalty', 'percent',
                'season', 'preview', 'matchup', 'team', 'teams', 'working', 'liney',
                'tonight', 'home', 'away', 'pm', 'pp', 'ppg', 'ppp', 'shootout'
            }
            if any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in banned_words):
                return None
            return " ".join(words)
        except Exception:
            return None

    @staticmethod
    def _normalize_team_key(name: Optional[str]) -> str:
        if not name:
            return ""
        normalized = unicodedata.normalize('NFKD', str(name))
        normalized = ''.join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = normalized.replace('&', 'AND')
        return re.sub(r'[^A-Z0-9]', '', normalized.upper())

    def _get_team_alias_map(self) -> Dict[str, str]:
        if not isinstance(self._team_alias_map, dict) or not self._team_alias_map:
            alias_map: Dict[str, str] = {}
            try:
                teams = NHLDataFetcher()._get_fallback_teams()
            except Exception:
                teams = {}
            for info in teams.values():
                name = str(info.get('name', ''))
                abbr = str(info.get('abbreviation', '')).upper()
                if not abbr:
                    continue
                variants = {
                    name,
                    abbr,
                    name.replace('St.', 'St'),
                    name.replace('St', 'Saint'),
                    name.replace('Saint', 'St'),
                    name.replace('.', ''),
                    name.replace('-', ' '),
                }
                for variant in variants:
                    key = self._normalize_team_key(variant)
                    if key and key not in alias_map:
                        alias_map[key] = abbr
                alias_map.setdefault(self._normalize_team_key(abbr), abbr)
            # Additional manual aliases for recent rebrands/alternates
            manual_aliases = {
                'UTAHHOCKEYCLUB': 'UTA',
                'UTAHNHL': 'UTA',
                'ARIZONACOYOTES': 'ARI',
                'QUEBEKNORDIQUES': 'QUE',
            }
            for key, value in manual_aliases.items():
                alias_map.setdefault(key, value)
            self._team_alias_map = alias_map
        return self._team_alias_map or {}

    def _extract_matchup_from_text(self, text: str, alias_map: Dict[str, str]) -> Optional[Dict[str, Optional[str]]]:
        if not text:
            return None
        normalized = html_parser.unescape(text)
        normalized = normalized.replace('\u2013', ' - ').replace('\u2014', ' - ')
        match = re.search(r'([A-Za-z0-9 .\'’&-]+?)\s+at\s+([A-Za-z0-9 .\'’&-]+)', normalized, flags=re.IGNORECASE)
        if not match:
            return None
        away_raw = match.group(1).strip()
        home_raw = match.group(2).strip()

        def clean_team(val: str) -> str:
            val = re.sub(r'\s+[-–—]\s+.*$', '', val)
            val = re.sub(r'\s+\d{1,2}:\d{2}\s*(?:[AP]M)?\s*(?:ET|CT|MT|PT)?\b.*$', '', val, flags=re.IGNORECASE)
            val = re.sub(r'\s+\d{4}\b.*$', '', val)
            val = re.sub(r'\s*\(.*?\)\s*$', '', val)
            val = re.sub(r'\s+\d+\s*$', '', val)
            return " ".join(val.split())

        away_name = clean_team(away_raw)
        home_name = clean_team(home_raw)
        away_abbr = alias_map.get(self._normalize_team_key(away_name))
        home_abbr = alias_map.get(self._normalize_team_key(home_name))
        matchup = None
        if away_abbr and home_abbr:
            matchup = f"{away_abbr}@{home_abbr}"
        elif away_abbr or home_abbr:
            matchup = f"{(away_abbr or away_name).upper()}@{(home_abbr or home_name).upper()}"

        return {
            'away_name': away_name,
            'home_name': home_name,
            'away_abbr': away_abbr,
            'home_abbr': home_abbr,
            'matchup': matchup
        }

    def _parse_scoutingtherefs_tables(self, html_blob: str) -> List[Dict[str, Any]]:
        assignments: List[Dict[str, Any]] = []
        if not html_blob:
            return assignments
        try:
            alias_map = self._get_team_alias_map()
            soup = BeautifulSoup(html_blob, 'html.parser') if BEAUTIFULSOUP_AVAILABLE else None
            if soup:
                entry = soup.find('div', class_='entry-content') or soup
                current_matchup: Optional[Dict[str, Optional[str]]] = None
                for node in entry.children:
                    if not getattr(node, 'name', None):
                        continue
                    tag = node.name.lower()
                    if tag in ('h1', 'h2', 'h3', 'p', 'div'):
                        text = node.get_text(' ', strip=True)
                        match_info = self._extract_matchup_from_text(text, alias_map)
                        if match_info:
                            current_matchup = match_info
                            continue
                    if tag == 'table':
                        classes = [str(c) for c in (node.get('class') or [])]
                        if not any(cls.lower() == 'totable' for cls in classes):
                            continue

                        def extract_first_float(cell) -> Optional[float]:
                            try:
                                strong = cell.find('strong')
                                candidates: List[str] = []
                                if strong:
                                    candidates.append(strong.get_text(' ', strip=True))
                                candidates.append(cell.get_text(' ', strip=True))
                                for candidate in candidates:
                                    match = re.search(r'[-+]?\d+(?:\.\d+)?', candidate)
                                    if match:
                                        try:
                                            return float(match.group(0))
                                        except Exception:
                                            continue
                            except Exception:
                                return None
                            return None

                        def normalize_label(label: str) -> Optional[str]:
                            lbl = (label or '').strip().lower()
                            if not lbl:
                                return None
                            if 'goals' in lbl and ('/g' in lbl or 'per game' in lbl or 'gm' in lbl or 'goalspg' in lbl or 'g/g' in lbl):
                                return 'goals_gm'
                            if re.search(r'\bgoals?\s*per\s*game\b', lbl):
                                return 'goals_gm'
                            return None

                        ref_entries: List[Dict[str, Any]] = []
                        expect_ref_names = False
                        for tr in node.find_all('tr'):
                            header = tr.find('h3')
                            if header:
                                header_txt = header.get_text(' ', strip=True)
                                if header_txt:
                                    header_upper = header_txt.upper()
                                    if 'REFEREE' in header_upper:
                                        expect_ref_names = True
                                        continue
                                    if 'LINES' in header_upper:
                                        break

                            if expect_ref_names:
                                ref_entries = []
                                for cell in tr.find_all('td'):
                                    strong = cell.find('strong')
                                    if not strong:
                                        continue
                                    nm = self._clean_ref_name(strong.get_text(' ', strip=True))
                                    if nm:
                                        ref_entries.append({'name': nm, 'stats': {}})
                                expect_ref_names = False
                                continue

                            if not ref_entries:
                                continue

                            cells = tr.find_all('td')
                            if not cells:
                                continue
                            label_text = cells[0].get_text(' ', strip=True) if cells else ''
                            stat_key = normalize_label(label_text)
                            if not stat_key:
                                continue
                            value_cells = cells[1:]
                            if not value_cells:
                                continue
                            for idx, entry in enumerate(ref_entries):
                                if idx >= len(value_cells):
                                    break
                                val = extract_first_float(value_cells[idx])
                                if val is not None:
                                    entry_stats = entry.setdefault('stats', {})
                                    entry_stats[stat_key] = val

                        refs: List[str] = [entry['name'] for entry in ref_entries if entry.get('name')]
                        if refs:
                            assignment: Dict[str, Any] = {'referees': refs}
                            if current_matchup:
                                assignment.update(current_matchup)
                            stats_map: Dict[str, Dict[str, Any]] = {}
                            goal_values: List[float] = []
                            for entry in ref_entries:
                                nm = entry.get('name')
                                stats = entry.get('stats') if isinstance(entry, dict) else None
                                if nm and isinstance(stats, dict) and stats:
                                    stats_map[nm] = stats
                                    val = stats.get('goals_gm')
                                    if isinstance(val, (int, float)) and np.isfinite(val):
                                        goal_values.append(float(val))
                                elif nm:
                                    stats_map.setdefault(nm, {})
                            if stats_map:
                                assignment['referee_stats'] = stats_map
                            if goal_values:
                                try:
                                    trimmed = goal_values[:2] if len(goal_values) >= 2 else goal_values
                                    avg_val = float(np.mean(trimmed))
                                    assignment['crew_goals_gm'] = round(avg_val, 3)
                                except Exception:
                                    pass
                            assignments.append(assignment)
            if assignments:
                return assignments
            # Fallback regex-based extraction if BeautifulSoup unavailable
            header_table_pattern = re.compile(
                r'(<h[12][^>]*>.*?</h[12]>)[\s\S]*?(<table[^>]*class="[^"]*TOTable[^"]*"[^>]*>[\s\S]*?</table>)',
                flags=re.IGNORECASE
            )
            for head_html, table_html in header_table_pattern.findall(html_blob):
                header_text = re.sub(r'<[^>]+>', ' ', head_html)
                match_info = self._extract_matchup_from_text(header_text, alias_map)
                refs_block_match = re.search(
                    r'<h3[^>]*>\s*REFEREES\s*</h3>(.*?)(?:<h3[^>]*>\s*LINES|$)',
                    table_html,
                    flags=re.IGNORECASE | re.DOTALL
                )
                refs_block = refs_block_match.group(1) if refs_block_match else table_html
                refs: List[str] = []
                for strong_match in re.finditer(r'<strong[^>]*>(.*?)</strong>', refs_block, flags=re.IGNORECASE | re.DOTALL):
                    nm = self._clean_ref_name(strong_match.group(1))
                    if nm:
                        refs.append(nm)
                    if len(refs) >= 2:
                        # Referees are always two names; stop to avoid picking stats strong tags
                        break
                if refs:
                    assignment: Dict[str, Any] = {'referees': refs}
                    if match_info:
                        assignment.update(match_info)
                    assignments.append(assignment)
            return assignments
        except Exception:
            return []

    def build_referee_crew_map(self, referees_url: Optional[str], todays_games: pd.DataFrame) -> Dict[str, List[str]]:
        """Best-effort mapping from matchup 'AWAY@HOME' to list of referee names.

        If a daily ScoutingTheRefs URL is provided, parse nearby text around 'Referees:'
        to infer the two teams and crew. Returns empty dict on failure.
        """
        crew_map: Dict[str, List[str]] = {}
        if not referees_url:
            return crew_map
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            r = requests.get(referees_url, timeout=20, headers=headers, allow_redirects=True)
            r.raise_for_status()
            html = r.text
            assignments = self._parse_scoutingtherefs_tables(html)
            for assignment in assignments:
                refs = [self._clean_ref_name(nm) for nm in assignment.get('referees', []) if nm]
                refs = [nm for nm in refs if nm]
                away = assignment.get('away_abbr')
                home = assignment.get('home_abbr')
                if refs and away and home:
                    mk1 = f"{away}@{home}"
                    mk2 = f"{home}@{away}"
                    crew_map[mk1] = refs
                    crew_map[mk2] = refs
            # Fallback to legacy regex approach if modern parser produced nothing
            if not crew_map:
                name_map = {v['name'].upper(): v['abbreviation'].upper() for v in NHLDataFetcher()._get_fallback_teams().values()}
                team_mentions: List[Tuple[int, str]] = []
                for full in name_map.keys():
                    for m in re.finditer(re.escape(full), html, flags=re.IGNORECASE):
                        team_mentions.append((m.start(), name_map[full]))
                team_mentions.sort()
                for m in re.finditer(r"Referees?\s*:\s*([^\n<]+)", html, flags=re.IGNORECASE):
                    idx = m.start()
                    ref_str = m.group(1)
                    abbrs: List[str] = []
                    for pos, ab in reversed(team_mentions):
                        if pos < idx and ab not in abbrs:
                            abbrs.append(ab)
                        if len(abbrs) >= 2:
                            break
                    refs: List[str] = []
                    for p in re.split(r"\s+and\s+|,", ref_str):
                        nm = re.sub(r"\s*\(#?\d+\)\s*", "", p).strip()
                        nm = re.sub(r"\s+/.*$", "", nm).strip()
                        if len(nm) >= 3:
                            refs.append(nm)
                    if len(abbrs) >= 2 and refs:
                        mk1 = f"{abbrs[0]}@{abbrs[1]}"
                        mk2 = f"{abbrs[1]}@{abbrs[0]}"
                        crew_map[mk1] = refs
                        crew_map[mk2] = refs
        except Exception:
            return {}
        # Filter to only today's matchups to avoid ambiguity
        valid_mks = {f"{str(g.get('away_team'))}@{str(g.get('home_team'))}" for _, g in todays_games.iterrows()}
        crew_map = {k: v for k, v in crew_map.items() if k in valid_mks}
        return crew_map

    def crew_features(self, crew_names: List[str], referee_rates: Optional[pd.DataFrame]) -> Tuple[Optional[float], Optional[float]]:
        """Compute crew goals/game average and a naive home-bias proxy if available.

        Expects `referee_rates` to optionally have columns like: ref, Goals/Gm (or similar), home_bias.
        Returns (avg_goals_gm, avg_home_bias).
        """
        if not crew_names or referee_rates is None or not isinstance(referee_rates, pd.DataFrame):
            return None, None
        try:
            df = referee_rates.copy()
            cols = {str(c).strip().lower(): c for c in df.columns}
            ref_col = cols.get('ref') or cols.get('name') or cols.get('official')
            if not ref_col:
                return None, None

            goal_col: Optional[str] = None
            goal_aliases = ['goals_gm', 'goals/gm', 'goals per game', 'goals_per_game', 'goals gm', 'goalspg', 'gpg', 'goalsgm']
            for key in goal_aliases:
                if key in cols:
                    goal_col = cols[key]
                    break
            if goal_col is None:
                for lc, orig in cols.items():
                    normalized = re.sub(r'[_-]+', ' ', lc)
                    if (
                        re.search(r'goals\s*/?\s*g(m|ame)', normalized)
                        or re.search(r'goals\s*per\s*game', normalized)
                        or re.search(r'\bgpg\b', normalized)
                        or 'goalspg' in normalized
                    ):
                        goal_col = orig
                        break

            bias_col = cols.get('home_bias') or cols.get('homebias')
            order_map = {str(n).upper(): idx for idx, n in enumerate(crew_names)}
            sub = df[df[ref_col].astype(str).str.upper().isin(order_map.keys())].copy()

            avg_goals: Optional[float] = None
            if goal_col and goal_col in sub.columns:
                sub['__order'] = sub[ref_col].astype(str).str.upper().map(order_map)
                sub = sub.dropna(subset=['__order']).sort_values('__order')
                goal_series = pd.to_numeric(sub[goal_col], errors='coerce').dropna()
                if not goal_series.empty:
                    vals = [float(v) for v in goal_series.tolist() if np.isfinite(v)]
                    if vals:
                        trimmed = vals[:2] if len(vals) >= 2 else vals
                        avg_goals = float(np.mean(trimmed))
                sub = sub.drop(columns=['__order'], errors='ignore')

            avg_b: Optional[float] = None
            if bias_col and bias_col in sub.columns:
                bias_series = pd.to_numeric(sub[bias_col], errors='coerce').dropna()
                if not bias_series.empty:
                    avg_b = float(bias_series.mean())

            return avg_goals, avg_b
        except Exception:
            return None, None

    # ---------------- Referee data loaders/scrapers ----------------
    def load_referee_rates(self, path_or_url: Optional[str]) -> Optional[pd.DataFrame]:
        """Load referee rates from a CSV path/URL or attempt to parse an HTML page.

        Returns a DataFrame with at least column 'ref' and optionally 'goals_gm', 'home_bias'.
        """
        if not path_or_url:
            return None
        try:
            url_l = str(path_or_url).lower().strip()
            # Fetch remote
            if url_l.startswith(('http://', 'https://')):
                r = requests.get(path_or_url, timeout=20)
                r.raise_for_status()
                text = r.text
                # Try CSV first
                try:
                    df = pd.read_csv(io.StringIO(text))
                except Exception:
                    # Try HTML tables
                    try:
                        tables = pd.read_html(io.StringIO(text))
                        df = tables[0] if tables else None
                    except Exception:
                        # As a last resort, scrape names from content
                        return self.scrape_referees_scoutingtherefs(path_or_url)
            else:
                df = pd.read_csv(path_or_url)

            if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
                return None

            # Normalize columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join([str(x) for x in c]).strip() for c in df.columns]

            lower_to_orig = {str(c).strip().lower(): c for c in df.columns}

            # Find referee name column
            ref_col: Optional[str] = None
            for key in ['ref', 'name', 'official', 'official_name', 'officials', 'referee', 'referees']:
                for lc, orig in lower_to_orig.items():
                    if lc == key or re.search(rf'\b{re.escape(key)}\b', lc):
                        ref_col = orig
                        break
                if ref_col:
                    break
            # Combine first/last name if available
            if ref_col is None:
                fn_col = None
                ln_col = None
                for lc, orig in lower_to_orig.items():
                    if 'first' in lc and 'name' in lc:
                        fn_col = orig
                    if 'last' in lc and 'name' in lc:
                        ln_col = orig
                if fn_col and ln_col:
                    df['ref'] = (df[fn_col].astype(str).str.strip() + ' ' + df[ln_col].astype(str).str.strip()).str.strip()
                    ref_col = 'ref'

            if ref_col is None:
                # Fallback to the first column
                ref_col = df.columns[0]

            # Optional numeric columns
            pens_col: Optional[str] = None
            goal_col: Optional[str] = None
            bias_col: Optional[str] = None
            goal_aliases = ['goals_gm', 'goals/gm', 'goals per game', 'goals_per_game', 'goals gm', 'goalspg', 'gpg', 'goalsgm']
            for alias in goal_aliases:
                if alias in lower_to_orig:
                    goal_col = lower_to_orig[alias]
                    break
            for lc, orig in lower_to_orig.items():
                if goal_col is None:
                    normalized = re.sub(r'[_-]+', ' ', lc)
                    if (
                        normalized == 'goals gm'
                        or 'goals per game' in normalized
                        or re.search(r'goals\s*/?\s*g(m|ame)', normalized)
                        or re.search(r'\bgpg\b', normalized)
                        or 'goalspg' in normalized
                    ):
                        goal_col = orig
                if pens_col is None and (lc == 'penalties60' or 'pens60' in lc or 'penalties/60' in lc or re.search(r'pen[a-z]*\s*/?\s*60', lc)):
                    pens_col = orig
                if bias_col is None and ('home_bias' in lc or 'home bias' in lc or lc == 'homebias'):
                    bias_col = orig

            out = pd.DataFrame({'ref': df[ref_col].astype(str).str.strip()})
            if goal_col and goal_col in df:
                out['goals_gm'] = pd.to_numeric(df[goal_col], errors='coerce')
            if pens_col and pens_col in df:
                out['penalties60'] = pd.to_numeric(df[pens_col], errors='coerce')
            if bias_col and bias_col in df:
                out['home_bias'] = pd.to_numeric(df[bias_col], errors='coerce')

            out = out[out['ref'].str.len() > 0].drop_duplicates(subset=['ref']).reset_index(drop=True)
            return out if len(out) else None
        except Exception as e:
            print(f"⚠️  Failed to load referee rates from {path_or_url}: {e}")
            return None

    def scrape_referees_scoutingtherefs(self, url: Optional[str]) -> Optional[pd.DataFrame]:
        """Scrape an assignments page for referee names; returns DataFrame with 'ref'.

        Attempts to use the WordPress API (`/wp-json/wp/v2/posts/{id}`) for a cleaner
        content payload before falling back to scraping the rendered HTML. The parser
        is designed to be resilient to markup changes by focusing on the text surrounding
        "Referees" headings and jersey numbers.
        """
        if not url:
            return None
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9'
            }

            def extract_post_id(target_url: str) -> Optional[str]:
                try:
                    parts = [p for p in (urlparse(target_url).path or '').split('/') if p]
                    for part in reversed(parts):
                        if part.isdigit() and len(part) >= 5 and int(part) > 1900:
                            return part
                except Exception:
                    return None
                return None

            def fetch_wp_content(post_id: str) -> Optional[str]:
                try:
                    api_url = f"https://scoutingtherefs.com/wp-json/wp/v2/posts/{post_id}"
                    resp = requests.get(api_url, params={'_fields': 'content.rendered'}, headers=headers, timeout=20)
                    resp.raise_for_status()
                    if not resp.headers.get('content-type', '').startswith('application/json'):
                        return None
                    data = resp.json()
                    if isinstance(data, dict):
                        content = data.get('content')
                        if isinstance(content, dict):
                            return content.get('rendered')
                        if isinstance(content, str):
                            return content
                except Exception:
                    return None
                return None

            def fetch_html(target_url: str) -> Optional[str]:
                try:
                    resp = requests.get(target_url, timeout=20, headers=headers, allow_redirects=True)
                    resp.raise_for_status()
                    return resp.text
                except Exception:
                    return None

            def iter_referee_chunks(html_blob: str):
                if not html_blob:
                    return
                normalized = html_parser.unescape(html_blob)
                normalized = normalized.replace('\u2019', "'")
                table_pat = re.compile(r"<table[^>]*>.*?</table>", flags=re.IGNORECASE | re.DOTALL)
                tables = [m.group(0) for m in table_pat.finditer(normalized) if re.search(r"Referees?", m.group(0), flags=re.IGNORECASE)]
                search_targets = tables if tables else [normalized]
                section_pat = re.compile(
                    r"Referees?\s*:?\s*(.*?)(?=Lines|Linespersons|Linesmen|Officials|Referee Assignments|$)",
                    flags=re.IGNORECASE
                )
                for target in search_targets:
                    plain = html_parser.unescape(target)
                    plain = re.sub(r"<[^>]+>", " ", plain)
                    plain = plain.replace('\xa0', ' ').replace('\u2013', ' ').replace('\u2014', ' ')
                    plain = re.sub(r"\s+", " ", plain)
                    for match in section_pat.finditer(plain):
                        chunk = match.group(1)
                        if chunk:
                            yield chunk[:600]

            def collect_names(html_blob: str) -> Set[str]:
                found: Set[str] = set()
                for chunk in iter_referee_chunks(html_blob):
                    chunk = chunk.replace('\u2019', "'").replace('\xa0', ' ')
                    chunk = chunk.replace('\u2013', ' ').replace('\u2014', ' ')
                    chunk = re.sub(r"\s+", " ", chunk)
                    exact = False
                    for nm_match in re.finditer(r"([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'\-]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'\-]+)+)\s*#\s*\d{1,3}", chunk):
                        nm = self._clean_ref_name(nm_match.group(1))
                        if nm:
                            found.add(nm)
                            exact = True
                    if exact:
                        continue
                    for part in re.split(r"\s+(?:and|&|/)\s+|,|;", chunk):
                        nm = self._clean_ref_name(part)
                        if nm:
                            found.add(nm)
                return found

            assignments: List[Dict[str, Any]] = []
            names: Set[str] = set()
            wp_html: Optional[str] = None
            page_html: Optional[str] = None

            post_id = extract_post_id(url)
            if post_id:
                wp_html = fetch_wp_content(post_id)
                if wp_html:
                    assignments = self._parse_scoutingtherefs_tables(wp_html)
                    for assignment in assignments:
                        for nm in assignment.get('referees', []):
                            nm_clean = self._clean_ref_name(nm)
                            if nm_clean:
                                names.add(nm_clean)

            if not names:
                page_html = fetch_html(url)
                if page_html:
                    if not assignments:
                        assignments = self._parse_scoutingtherefs_tables(page_html)
                    for assignment in assignments:
                        for nm in assignment.get('referees', []):
                            nm_clean = self._clean_ref_name(nm)
                            if nm_clean:
                                names.add(nm_clean)

            if not names:
                if wp_html:
                    names.update({nm for nm in collect_names(wp_html) if nm})
                if not names and page_html:
                    names.update({nm for nm in collect_names(page_html) if nm})

            if not names:
                return None

            assignment_records: List[Dict[str, Optional[Any]]] = []
            if assignments:
                for assignment in assignments:
                    refs = [self._clean_ref_name(nm) for nm in assignment.get('referees', []) if nm]
                    refs = [nm for nm in refs if nm]
                    if not refs:
                        continue
                    away = assignment.get('away_abbr')
                    home = assignment.get('home_abbr')
                    away_name = assignment.get('away_name')
                    home_name = assignment.get('home_name')
                    matchup = assignment.get('matchup')
                    stats_map_raw = assignment.get('referee_stats') or {}
                    stats_lookup: Dict[str, Dict[str, Any]] = {}
                    for key, val in stats_map_raw.items():
                        nm_clean = self._clean_ref_name(key)
                        if nm_clean:
                            if isinstance(val, dict):
                                stats_lookup[nm_clean] = val
                            else:
                                stats_lookup[nm_clean] = {}

                    crew_goal_val = assignment.get('crew_goals_gm')

                    def coerce_float(val: Any) -> Optional[float]:
                        try:
                            if val is None:
                                return None
                            if isinstance(val, (int, float)):
                                if np.isfinite(val):
                                    return float(val)
                                return None
                            if isinstance(val, str):
                                stripped = val.strip()
                                if not stripped:
                                    return None
                                num = float(stripped)
                                if np.isfinite(num):
                                    return float(num)
                                return None
                        except Exception:
                            return None
                        return None

                    crew_goal_float = coerce_float(crew_goal_val)

                    for nm in refs:
                        stats_for_ref = stats_lookup.get(nm, {})
                        goals_val = coerce_float(stats_for_ref.get('goals_gm'))
                        assignment_records.append({
                            'ref': nm,
                            'matchup': matchup,
                            'away_team': away,
                            'home_team': home,
                            'away_name': away_name,
                            'home_name': home_name,
                            'goals_gm': goals_val,
                            'crew_goals_gm': crew_goal_float
                        })

            if assignment_records:
                df = pd.DataFrame(assignment_records).drop_duplicates(subset=['matchup', 'ref']).reset_index(drop=True)
                df['ref'] = df['ref'].astype(str)
                return df

            sorted_names = sorted(nm for nm in names if nm)
            return pd.DataFrame({'ref': sorted_names})
        except Exception as e:
            print(f"⚠️  Failed to scrape referees from {url}: {e}")
            return None

    def find_todays_scoutingtherefs_url(self) -> Optional[str]:
        """Best-effort to locate today's NHL officiating assignments post on scoutingtherefs.com.

        Strategy: scan the current month's archive and homepage for links that look like daily
        officiating assignments and infer their publish dates from nearby metadata or the slug.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            today = datetime.now().date()
            candidates: List[Tuple[datetime, str]] = []
            seen_links: Set[str] = set()

            def parse_iso_date(raw: str) -> Optional[datetime]:
                val = raw.strip()
                if not val:
                    return None
                try:
                    cleaned = val.replace('Z', '+00:00')
                    return datetime.fromisoformat(cleaned)
                except Exception:
                    pass
                try:
                    return datetime.strptime(val[:10], '%Y-%m-%d')
                except Exception:
                    return None

            def register_candidate(dt_val: Optional[datetime], link: Optional[str]) -> None:
                if dt_val is None or link is None:
                    return
                if link in seen_links:
                    return
                seen_links.add(link)
                candidates.append((dt_val.date(), link))

            def is_assignment_link(link: Optional[str], title: Optional[str] = None) -> bool:
                ll = (link or '').lower()
                tt = (title or '').lower()
                if not ll and not tt:
                    return False
                keywords = ('referee', 'referees', 'official', 'assign', 'linesperson', 'linespersons', 'linesmen', 'linespeople')
                if not any(k in ll for k in keywords) and not any(k in tt for k in keywords):
                    return False
                if 'nhl' not in ll and 'nhl' not in tt:
                    return False
                return True

            def harvest_wp_posts() -> None:
                api_url = "https://scoutingtherefs.com/wp-json/wp/v2/posts"
                queries = [
                    {'search': "Today's NHL Referees"},
                    {'search': "NHL Referees and Linespersons"},
                    {'search': 'NHL Referees'}
                ]
                base_params = {
                    'per_page': 20,
                    '_fields': 'date,date_gmt,link,title.rendered',
                    'orderby': 'date',
                    'order': 'desc'
                }
                for query in queries + [{}]:
                    try:
                        params = {**base_params, **query}
                        resp = requests.get(api_url, params=params, headers=headers, timeout=15)
                        resp.raise_for_status()
                        posts = resp.json()
                        if isinstance(posts, dict):
                            posts = [posts]
                        if not isinstance(posts, list):
                            continue
                        for post in posts:
                            if not isinstance(post, dict):
                                continue
                            link = post.get('link')
                            title_obj = post.get('title')
                            title = title_obj.get('rendered') if isinstance(title_obj, dict) else title_obj if isinstance(title_obj, str) else ''
                            if not is_assignment_link(link, title):
                                continue
                            dt_val = None
                            for key in ('date', 'date_gmt'):
                                raw = post.get(key)
                                if isinstance(raw, str):
                                    dt_val = parse_iso_date(raw)
                                    if dt_val:
                                        break
                            if dt_val:
                                register_candidate(dt_val, link)
                    except Exception:
                        continue

            def extract_date_from_url(link: str) -> Optional[datetime]:
                try:
                    m_full = re.search(r'scoutingtherefs\.com/(\d{4})/(\d{2})/(\d{2})/', link, flags=re.IGNORECASE)
                    if m_full:
                        y, mo, d = int(m_full.group(1)), int(m_full.group(2)), int(m_full.group(3))
                        return datetime(y, mo, d)
                    m_month = re.search(r'scoutingtherefs\.com/(\d{4})/(\d{2})/', link, flags=re.IGNORECASE)
                    if not m_month:
                        return None
                    y, mo = int(m_month.group(1)), int(m_month.group(2))
                    path = urlparse(link).path or ''
                    slug = path.rstrip('/').split('/')[-1]
                    nums = [int(n) for n in re.findall(r'\d+', slug)] if slug else []
                    if len(nums) >= 3:
                        maybe_month, maybe_day, maybe_year = nums[-3], nums[-2], nums[-1]
                        year_val = maybe_year + 2000 if maybe_year < 100 else maybe_year
                        if 1 <= maybe_month <= 12 and 1 <= maybe_day <= 31:
                            try:
                                return datetime(year_val, maybe_month, maybe_day)
                            except Exception:
                                pass
                    for day_candidate in reversed(nums):
                        if 1 <= day_candidate <= 31:
                            try:
                                return datetime(y, mo, day_candidate)
                            except Exception:
                                continue
                    return datetime(y, mo, 1)
                except Exception:
                    return None

            def harvest(page_url: str) -> None:
                try:
                    resp = requests.get(page_url, timeout=20, headers=headers, allow_redirects=True)
                    resp.raise_for_status()
                    html = resp.text
                except Exception:
                    return

                for m in re.finditer(r'href=["\'](https?://scoutingtherefs\.com/[^"\']+)["\']', html, flags=re.IGNORECASE):
                    link = m.group(1)
                    ll = link.lower()
                    if 'nhl' not in ll or not any(k in ll for k in ('assign', 'referee', 'official')):
                        continue
                    if link in seen_links:
                        continue
                    seen_links.add(link)

                    snippet = html[max(0, m.start() - 500):min(len(html), m.end() + 500)]
                    dt_match = re.search(r'datetime\s*=\s*["\']([^"\']+)["\']', snippet, flags=re.IGNORECASE)
                    dt_val: Optional[datetime]
                    if dt_match:
                        iso_dt = parse_iso_date(dt_match.group(1))
                        dt_val = iso_dt if iso_dt else None
                    else:
                        dt_val = None
                    if dt_val is None:
                        dt_val = extract_date_from_url(link)
                    if dt_val is None:
                        continue
                    register_candidate(dt_val, link)

            harvest_wp_posts()
            month_url = f"https://scoutingtherefs.com/{today.year}/{today.month:02d}/"
            harvest(month_url)
            harvest("https://scoutingtherefs.com/")

            if not candidates:
                return None

            candidates.sort(key=lambda x: (abs((x[0] - today).days), -int(datetime.combine(x[0], datetime.min.time()).timestamp())))
            best_dt, best_link = candidates[0]
            if abs((best_dt - today).days) <= 1:
                return best_link
            return None
        except Exception:
            return None

    def load_penalty_rates(self, path_or_url: Optional[str]) -> Optional[pd.DataFrame]:
        """Minimal loader for league penalty rates; accepts CSV path/URL and returns DataFrame.
        This is intentionally flexible; downstream code only needs a DataFrame-like object.
        """
        if not path_or_url:
            return None
        try:
            url_l = str(path_or_url).lower().strip()
            if url_l.startswith(('http://', 'https://')):
                r = requests.get(path_or_url, timeout=20)
                r.raise_for_status()
                return pd.read_csv(io.StringIO(r.text))
            return pd.read_csv(path_or_url)
        except Exception as e:
            print(f"⚠️  Failed to load penalty rates from {path_or_url}: {e}")
            return None
    
    def get_todays_games(self, target_date: Optional[str] = None, offline_path: Optional[str] = None, offline_only: bool = False) -> pd.DataFrame:
        """Get games for a target date with API and/or offline fallbacks.

        - target_date: YYYY-MM-DD (defaults to today)
        - offline_path: JSON file with array of games or object keyed by matchup
          Example array entries: {"game_id":"custom_1","home_team":"BOS","away_team":"TOR","date":"2025-09-29"}
        - offline_only: if True, skip API and rely on offline_path
        """
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        try:
            # normalize to tz-naive string
            _ = datetime.strptime(target_date, '%Y-%m-%d')
        except Exception:
            target_date = datetime.now().strftime('%Y-%m-%d')
        today = target_date
        tomorrow = (datetime.strptime(today, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"🏒 Looking for games on {today}...")

        # Offline path first if demanded
        if offline_only and offline_path and os.path.exists(offline_path):
            try:
                with open(offline_path, 'r') as f:
                    data = json.load(f)
                entries = data if isinstance(data, list) else list(data.values())
                rows = []
                for i, g in enumerate(entries):
                    rows.append({
                        'game_id': g.get('game_id', f'offline_{i}'),
                        'date': pd.to_datetime(g.get('date', today)),
                        'home_team': g.get('home_team', 'HOME'),
                        'away_team': g.get('away_team', 'AWAY'),
                        'venue': g.get('venue', f"{g.get('home_team','HOME')} Arena"),
                        'home_goals': 0, 'away_goals': 0, 'total_goals': 0
                    })
                print(f"✅ Loaded {len(rows)} offline games from {offline_path}")
                return pd.DataFrame(rows)
            except Exception as e:
                print(f"⚠️  Failed to load offline games from {offline_path}: {e}")
        
        games = [] if offline_only else self.data_fetcher.get_schedule(today, tomorrow)
        
        if not games:
            print("ℹ️  No NHL games found today from API")
            for days_ahead in range(1, 4):
                future_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
                future_games = self.data_fetcher.get_schedule(future_date, future_date)
                if future_games:
                    print(f"📅 Found games {days_ahead} days ahead on {future_date}")
                    games = future_games
                    break
        
        todays_games = []
        qualifying_statuses = {"Scheduled", "Pre-Game", "Preview", "In Progress", "Final"}
        
        if games:
            total_returned = len(games)
            kept = 0
            for game in games:
                status = game.get('status', {}).get('detailedState')
                # Filter to target_date strictly
                try:
                    gd = pd.to_datetime(game.get('gameDate'), utc=True, errors='coerce')
                    if pd.isna(gd):
                        gdate = None
                    else:
                        try:
                            gdate = gd.tz_convert(os.getenv('SCHEDULE_TZ', 'US/Eastern')).date()
                        except Exception:
                            gdate = gd.tz_convert(None).date()
                except Exception:
                    gdate = None
                # Keep all games on the date regardless of status to avoid missing late-added statuses
                if gdate == datetime.strptime(today, '%Y-%m-%d').date():
                    try:
                        home_team = game['teams']['home']['team']
                        away_team = game['teams']['away']['team']
                        game_data = {
                            'game_id': game['gamePk'],
                            'date': pd.to_datetime(game['gameDate']),
                            'home_team': home_team['abbreviation'],
                            'away_team': away_team['abbreviation'],
                            'venue': game.get('venue', {}).get('name', f"{home_team['abbreviation']} Arena"),
                            'home_goals': 0,
                            'away_goals': 0,
                            'total_goals': 0
                        }
                        todays_games.append(game_data)
                        kept += 1
                    except Exception as e:
                        print(f"⚠️  Error processing upcoming game: {e}")
                        continue
            if kept == 0:
                print(f"ℹ️  API returned {total_returned} games but none in qualifying statuses: {sorted(list(qualifying_statuses))}")
        
        if not todays_games and offline_path and os.path.exists(offline_path):
            # Secondary offline fallback even if offline_only was false
            try:
                with open(offline_path, 'r') as f:
                    data = json.load(f)
                entries = data if isinstance(data, list) else list(data.values())
                for i, g in enumerate(entries):
                    todays_games.append({
                        'game_id': g.get('game_id', f'offline_{i}'),
                        'date': pd.to_datetime(g.get('date', today)),
                        'home_team': g.get('home_team', 'HOME'),
                        'away_team': g.get('away_team', 'AWAY'),
                        'venue': g.get('venue', f"{g.get('home_team','HOME')} Arena"),
                        'home_goals': 0, 'away_goals': 0, 'total_goals': 0
                    })
                if todays_games:
                    print(f"✅ Loaded {len(todays_games)} offline games from {offline_path}")
            except Exception as e:
                print(f"⚠️  Failed to load offline games fallback from {offline_path}: {e}")

        if not todays_games:
            print("📝 No upcoming games found. Creating sample matchups for demonstration...")
            sample_matchups = [
                ('TOR', 'BOS'), ('EDM', 'COL'), ('VGK', 'FLA'), ('NYR', 'CAR'),
                ('TBL', 'DAL'), ('WPG', 'MIN'), ('MTL', 'CBJ'), ('ANA', 'SJS')
            ]
            
            for i, (away, home) in enumerate(sample_matchups[:4]):
                todays_games.append({
                    'game_id': f'demo_{i}',
                    'date': pd.to_datetime(today),
                    'home_team': home,
                    'away_team': away,
                    'venue': f'{home} Arena',
                    'home_goals': 0,
                    'away_goals': 0,
                    'total_goals': 0
                })
        
        print(f"✅ Prepared {len(todays_games)} games for prediction")
        return pd.DataFrame(todays_games)

def log_bets(predictions: List[OverUnderPrediction], logfile: str = 'bets_log.csv', closing_odds_path: Optional[str] = None) -> None:
    """Append recommended bets to a CSV and compute simple CLV if closing totals provided.

    closing_odds_path JSON per game id or matchup may include:
    {"<game_id>": {"closing_total": 6.0, "closing_over": -115, "closing_under": -105}}
    """
    recs = [p for p in predictions if p.recommendation != 'No Bet']
    if not recs:
        print("ℹ️  No bets to log today.")
        return

    closing = None
    if closing_odds_path and os.path.exists(closing_odds_path):
        try:
            with open(closing_odds_path, 'r') as f:
                closing = json.load(f)
        except Exception:
            closing = None

    rows = []
    for p in recs:
        gid = p.game_id
        matchup = f"{p.away_team}@{p.home_team}"
        rec_side = p.recommendation
        my_line = p.betting_line
        close_total = None
        clv = None
        if isinstance(closing, dict):
            rec = closing.get(gid, closing.get(matchup))
            if isinstance(rec, dict):
                try:
                    close_total = float(rec.get('closing_total'))
                except Exception:
                    close_total = None
        if close_total is not None:
            try:
                clv = float((my_line - close_total) if rec_side == 'OVER' else (close_total - my_line))
            except Exception:
                clv = None
        rows.append({
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'game_id': gid,
            'matchup': matchup,
            'side': rec_side,
            'line': my_line,
            'price': p.over_american_odds if rec_side == 'OVER' else p.under_american_odds,
            'pred_total': p.predicted_total,
            'edge': p.edge,
            'confidence': p.confidence,
            'kelly_pct': p.kelly_bet_size,
            'ev_novig': p.ev_over_novig if rec_side == 'OVER' else p.ev_under_novig,
            'consensus_total': p.consensus_total,
            'line_diff_vs_consensus': p.line_diff,
            'best_book': p.best_over_book if rec_side == 'OVER' else p.best_under_book,
            'referee_info': p.referee_info,
            'referee_avg_goals': p.referee_avg_goals,
            'referee_home_bias': p.referee_home_bias,
            'closing_total': close_total,
            'clv_vs_closing': clv
        })

    df = pd.DataFrame(rows)
    header = not os.path.exists(logfile)
    df.to_csv(logfile, mode='a', header=header, index=False)
    print(f"✅ Logged {len(rows)} bets to {logfile}")

    # Daily summary
    try:
        recent = pd.read_csv(logfile)
        today = datetime.now().strftime('%Y-%m-%d')
        today_df = recent[recent['date'].str.startswith(today)]
        if len(today_df) > 0:
            avg_clv = today_df['clv_vs_closing'].dropna().mean() if 'clv_vs_closing' in today_df else None
            avg_ev = today_df['ev_novig'].dropna().mean() if 'ev_novig' in today_df else None
            print(f"📈 Daily log: {len(today_df)} bets | avg no-vig EV {avg_ev:+.2f} | avg CLV {avg_clv:+.2f} (goals)")
    except Exception:
        pass

def save_predictions_excel(predictions: List[OverUnderPrediction], out_path: str = 'predictions.xlsx') -> Optional[str]:
    """Save predictions to an Excel file with a flat schema for sharing."""
    try:
        rows = []
        for p in predictions:
            rows.append({
                'game_id': p.game_id,
                'matchup': f"{p.away_team}@{p.home_team}",
                'home_team': p.home_team,
                'away_team': p.away_team,
                'line': p.betting_line,
                'predicted_total': p.predicted_total,
                'over_prob': p.over_probability,
                'under_prob': p.under_probability,
                'push_prob': p.push_probability,
                'recommendation': p.recommendation,
                'edge': p.edge,
                'confidence': p.confidence,
                'kelly_pct': p.kelly_bet_size,
                'over_american_odds': p.over_american_odds,
                'under_american_odds': p.under_american_odds,
                'ev_over_novig': p.ev_over_novig,
                'ev_under_novig': p.ev_under_novig,
                'consensus_total': p.consensus_total,
                'line_diff_vs_consensus': p.line_diff,
                'best_over_book': p.best_over_book,
                'best_under_book': p.best_under_book,
                'ref_goals_gm': p.ref_goals_gm,
                'referee_crew': ", ".join(p.referee_crew) if getattr(p, 'referee_crew', None) else None,
                'referee_avg_goals': p.referee_avg_goals,
                'referee_home_bias': p.referee_home_bias,
                'referee_info': p.referee_info,
                'referee_source': p.referee_source,
                'env_info': p.env_info,
                'lineup_info': p.lineup_info
            })
        df = pd.DataFrame(rows)
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
        # Choose engine by extension
        suffix = os.path.splitext(out_path)[1].lower()
        engine = 'xlsxwriter'
        if suffix == '.xls':
            try:
                import xlwt  # noqa: F401
                engine = 'xlwt'
            except Exception:
                # Fallback to xlsx
                base = os.path.splitext(out_path)[0]
                out_path = base + '.xlsx'
                engine = 'xlsxwriter'
        with pd.ExcelWriter(out_path, engine=engine) as writer:
            df.to_excel(writer, index=False, sheet_name='predictions')
            # Apply formatting for xlsxwriter engine: color Recommendation cell
            if engine == 'xlsxwriter':
                try:
                    workbook = writer.book
                    worksheet = writer.sheets.get('predictions')
                    if worksheet is not None:
                        # Locate the recommendation column
                        rec_idx = None
                        for i, c in enumerate(df.columns):
                            if str(c).strip().lower() == 'recommendation':
                                rec_idx = i
                                break
                        if rec_idx is not None:
                            nrows = len(df) + 1  # header row at 0
                            # Formats
                            fmt_over = workbook.add_format({'bold': True, 'bg_color': '#D5F4E6', 'font_color': '#1E8449'})
                            fmt_under = workbook.add_format({'bold': True, 'bg_color': '#FADBD8', 'font_color': '#C0392B'})
                            fmt_nobet = workbook.add_format({'bold': True, 'bg_color': '#F9E79F', 'font_color': '#7D6608'})
                            # Apply text contains rules on the recommendation column
                            col = rec_idx
                            worksheet.conditional_format(1, col, nrows, col, {
                                'type': 'text', 'criteria': 'containing', 'value': 'OVER', 'format': fmt_over
                            })
                            worksheet.conditional_format(1, col, nrows, col, {
                                'type': 'text', 'criteria': 'containing', 'value': 'UNDER', 'format': fmt_under
                            })
                            worksheet.conditional_format(1, col, nrows, col, {
                                'type': 'text', 'criteria': 'containing', 'value': 'No Bet', 'format': fmt_nobet
                            })
                except Exception:
                    pass
        print(f"✅ Saved predictions to Excel: {out_path}")
        return out_path
    except Exception as e:
        print(f"⚠️  Failed to save predictions to Excel: {e}")
        return None

def build_predictions_table_html(predictions: List[OverUnderPrediction]) -> str:
    """Return a minimal HTML table of predictions with colored Recommendation cells."""
    rows_html = []
    for p in predictions:
        rec = (p.recommendation or 'No Bet')
        cls = 'rec-no-bet'
        if rec.upper().startswith('OVER'):
            cls = 'rec-over'
        elif rec.upper().startswith('UNDER'):
            cls = 'rec-under'
        try:
            over_txt = f"{p.over_probability:.0%}"
        except Exception:
            over_txt = ''
        try:
            under_txt = f"{p.under_probability:.0%}"
        except Exception:
            under_txt = ''
        pred_val = getattr(p, 'predicted_total', None)
        pred_txt = f"{pred_val:.2f}" if isinstance(pred_val, (int, float)) else ''
        edge_val = getattr(p, 'edge', None)
        edge_txt = f"{edge_val:+.2f}" if isinstance(edge_val, (int, float)) else ''
        conf_val = getattr(p, 'confidence', None)
        conf_txt = f"{conf_val:.0%}" if isinstance(conf_val, (int, float)) else ''
        env_txt = getattr(p, 'env_info', '') or ''
        lineup_txt = getattr(p, 'lineup_info', '') or ''
        ref_txt = getattr(p, 'referee_info', None)
        if not ref_txt:
            crew_list = getattr(p, 'referee_crew', []) or []
            ref_txt = ", ".join([str(n) for n in crew_list if str(n).strip()])
        ref_txt = ref_txt or ''
        kelly_val = getattr(p, 'kelly_bet_size', None)
        kelly_txt = f"{kelly_val:.1f}%" if isinstance(kelly_val, (int, float)) else ''
        rows_html.append(
            f"<tr>"
            f"<td>{p.away_team}@{p.home_team}</td>"
            f"<td>{p.betting_line}</td>"
            f"<td>{pred_txt}</td>"
            f"<td>{edge_txt}</td>"
            f"<td>{over_txt}</td>"
            f"<td>{under_txt}</td>"
            f"<td>{conf_txt}</td>"
            f"<td>{ref_txt}</td>"
            f"<td>{env_txt}</td>"
            f"<td>{lineup_txt}</td>"
            f"<td class='{cls}'>{rec}</td>"
            f"<td>{kelly_txt}</td>"
            f"</tr>"
        )
    head = (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset=\"utf-8\"><style>\n"
        "body { font-family: Segoe UI, Arial, sans-serif; background:#ffffff; color:#2c3e50; }\n"
        "table { border-collapse: collapse; width: 100%; background:#ffffff; }\n"
        "th, td { border: 1px solid #ddd; padding: 8px; font-size: 14px; }\n"
        "th { background:#2c3e50; color:#fff; text-align:left; }\n"
        ".rec-over { background:#27ae60; color:#ffffff; font-weight:700; }\n"
        ".rec-under { background:#c0392b; color:#ffffff; font-weight:700; }\n"
        ".rec-no-bet { background:#f9e79f; color:#7d6608; font-weight:700; }\n"
        "tr:nth-child(even) { background:#fafafa; }\n"
        "</style></head><body>\n<table>\n<thead><tr>\n"
        "<th>Matchup</th><th>Line</th><th>Predicted</th><th>Edge</th>\n"
        "<th>Over%</th><th>Under%</th><th>Confidence</th><th>Referees</th><th>Env</th><th>Lineup</th><th>Recommendation</th><th>Kelly%</th>\n"
        "</tr></thead>\n<tbody>\n"
    )
    tail = "\n</tbody></table>\n</body></html>\n"
    return head + "\n".join(rows_html) + tail

def save_predictions_image(
    predictions: List[OverUnderPrediction],
    training_results: Optional[Dict] = None,
    html_path: str = 'predictions_table.html',
    image_path: str = 'predictions.png'
) -> Optional[str]:
    """Render a social image styled like the NFL predictions graphic."""

    # Persist HTML table for reference even though the image is matplotlib-based
    try:
        html = build_predictions_table_html(predictions)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
    except Exception:
        pass

    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Matplotlib not available; cannot render predictions image.")
        return None

    if not predictions:
        print("ℹ️  No predictions available to render.")
        return None

    def compute_accuracy_strings() -> Tuple[str, str]:
        ytd_default = "YTD: 0.0% (0/0)"
        last_week_default = "Last Week: 0.0% (0/0)"
        log_candidates = [
            os.getenv('NHL_ACCURACY_FILE'),
            os.getenv('NHL_BETS_LOG'),
            'bets_log.csv'
        ]

        for path in log_candidates:
            if not path:
                continue
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                continue
            try:
                df_log = pd.read_csv(path)
            except Exception:
                continue
            if df_log.empty:
                continue

            result_col = next(
                (c for c in df_log.columns if str(c).lower() in {'result', 'outcome', 'grade', 'bet_result', 'graded_result'}),
                None
            )
            if not result_col:
                continue

            df_log['_result_norm'] = (
                df_log[result_col]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            win_tokens = {'WIN', 'W', 'HIT', 'SUCCESS', 'CASH', 'TRUE'}
            loss_tokens = {'LOSS', 'L', 'LOSE', 'FAILED', 'MISS', 'FALSE'}
            win_mask = df_log['_result_norm'].isin(win_tokens)
            loss_mask = df_log['_result_norm'].isin(loss_tokens)

            total_scored = int(win_mask.sum() + loss_mask.sum())
            if total_scored > 0:
                total_wins = int(win_mask.sum())
                accuracy_pct = (total_wins / total_scored) * 100.0
                ytd_str = f"YTD: {accuracy_pct:.1f}% ({total_wins}/{total_scored})"
            else:
                ytd_str = ytd_default

            date_col = next(
                (c for c in df_log.columns if str(c).lower() in {'date', 'datetime', 'timestamp', 'created_at'}),
                None
            )
            last_week_str = last_week_default
            if date_col:
                dates_utc = pd.to_datetime(df_log[date_col], utc=True, errors='coerce')
                if dates_utc.notna().any():
                    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=7)
                    lw_mask = dates_utc >= cutoff
                    lw_wins = int((win_mask & lw_mask).sum())
                    lw_losses = int((loss_mask & lw_mask).sum())
                    lw_total = lw_wins + lw_losses
                    if lw_total > 0:
                        lw_pct = (lw_wins / lw_total) * 100.0
                        last_week_str = f"Last Week: {lw_pct:.1f}% ({lw_wins}/{lw_total})"

            return ytd_str, last_week_str

        if training_results:
            try:
                acc_pct = float(training_results.get('over_under_accuracy', 0.0)) * 100.0
            except Exception:
                acc_pct = 0.0
            total_games = int(training_results.get('test_size') or training_results.get('train_size') or 0)
            if total_games > 0:
                wins = int(round((acc_pct / 100.0) * total_games))
                ytd_str = f"YTD: {acc_pct:.1f}% ({wins}/{total_games})"
            else:
                ytd_str = f"YTD: {acc_pct:.1f}% (0/0)"
            return ytd_str, last_week_default

        return ytd_default, last_week_default

    ytd_str, last_week_str = compute_accuracy_strings()

    rows: List[Dict[str, str]] = []
    for pred in predictions:
        # Determine scheduled time (Eastern) when available
        display_time = getattr(pred, 'start_time_display', '')
        raw_dt = getattr(pred, 'game_datetime', None) or getattr(pred, 'game_datetime_utc', None)
        start_dt = None
        if raw_dt is not None:
            try:
                start_dt = pd.Timestamp(raw_dt)
            except Exception:
                start_dt = None
        if start_dt is None:
            raw_str = getattr(pred, 'game_date', None)
            if raw_str:
                try:
                    start_dt = pd.to_datetime(raw_str, utc=True, errors='coerce')
                except Exception:
                    start_dt = None
        sort_key = float('inf')
        if start_dt is not None and not pd.isna(start_dt):
            try:
                if start_dt.tzinfo is None or start_dt.tzinfo.utcoffset(start_dt) is None:
                    start_dt = start_dt.tz_localize('UTC')
            except (TypeError, AttributeError):
                start_dt = start_dt.tz_localize('UTC')
            start_dt_et = start_dt.tz_convert('US/Eastern')
            sort_key = float(start_dt_et.timestamp())
            if not display_time:
                display_time = start_dt_et.strftime('%I:%M %p ET')

        def _fmt_float(value: Optional[float]) -> str:
            try:
                return f"{float(value):.1f}"
            except Exception:
                return '—'

        def _fmt_conf(value: Optional[float]) -> str:
            try:
                val = float(value)
                if val <= 1.0:
                    val *= 100.0
                return f"{val:.1f}%"
            except Exception:
                return '—'

        recommendation = getattr(pred, 'recommendation', '') or ''
        if isinstance(recommendation, str):
            pick_txt = recommendation.title()
        else:
            pick_txt = str(recommendation)

        ref_goal_value = getattr(pred, 'ref_goals_gm', None)
        if ref_goal_value is None:
            ref_goal_value = getattr(pred, 'referee_avg_goals', None)
        ref_goal_display = _fmt_float(ref_goal_value) if ref_goal_value is not None else ''

        rows.append({
            'Time': display_time,
            'Away': str(getattr(pred, 'away_team', '')),
            'Home': str(getattr(pred, 'home_team', '')),
            'Line': _fmt_float(getattr(pred, 'betting_line', None)),
            'Predicted': _fmt_float(getattr(pred, 'predicted_total', None)),
            'Pick': pick_txt,
            'Conf%': _fmt_conf(getattr(pred, 'confidence', None)),
            'Ref G/G': ref_goal_display,
            '_sort_key': sort_key
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("ℹ️  No rows constructed for predictions table.")
        return None

    df = df.sort_values(by=['_sort_key', 'Time']).drop(columns=['_sort_key'])
    df = df.replace({None: '—', np.nan: '—'})

    fig_height = 0.27 * max(1, len(df)) + 1.0
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    fig.patch.set_facecolor('white')
    ax.axis('off')

    title_text = (
        f"NHL Predictions\n"
        # f"{ytd_str} | {last_week_str}\n"
        "Confidence is the model's probability the prediction is accurate.\n"
        "Odds aggregated from multiple books."
    )
    ax.set_title(title_text, fontsize=16, fontweight='bold', loc='center', pad=6)

    # Compute column widths dynamically so the layout stays stable if columns change
    default_col_widths = {
        'Time': 0.12,
        'Away': 0.18,
        'Home': 0.18,
        'Line': 0.08,
        'Predicted': 0.12,
        'Pick': 0.1,
        'Conf%': 0.1,
        'Ref G/G': 0.08
    }
    fallback_width = max(0.08, 1.0 / max(1, len(df.columns)))
    col_widths = [default_col_widths.get(str(col), fallback_width) for col in df.columns]
    total_width = sum(col_widths)
    if total_width > 0:
        col_widths = [w / total_width for w in col_widths]
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        colWidths=col_widths,
        loc='upper center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.2)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor('#bdc3c7')
        if row_idx == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', weight='bold', fontsize=13)
        else:
            if col_idx == 5:  # Pick column
                pick_val = str(df.iloc[row_idx - 1, col_idx]).upper()
                if pick_val == 'OVER':
                    cell.set_facecolor('#27ae60')
                    cell.set_text_props(color='white', weight='bold')
                elif pick_val == 'UNDER':
                    cell.set_facecolor('#c0392b')
                    cell.set_text_props(color='white', weight='bold')
                else:
                    cell.set_facecolor('#ecf0f1')
            elif row_idx % 2 == 0:
                cell.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig(image_path, bbox_inches='tight', dpi=200)
    plt.close(fig)

    if os.path.exists(image_path):
        print(f"✅ Saved predictions image to {image_path}")
        return image_path

    return None

def create_dashboard_html(predictions: List[OverUnderPrediction], training_results: Dict, betting_odds: Optional[Dict] = None) -> str:
    """Create enhanced dashboard HTML"""
    
    if not predictions:
        return """
        <html><body>
        <h1>NHL Over/Under Dashboard</h1>
        <p>No predictions available. Please check if there are games today or if data fetching was successful.</p>
        </body></html>
        """
    
    betting_preds = [p for p in predictions if p.recommendation != 'No Bet']
    avg_confidence = np.mean([p.confidence for p in betting_preds]) if betting_preds else 0
    
    def attr_escape(value: Optional[str]) -> str:
        if value is None:
            return ''
        return html_parser.escape(str(value), quote=True)

    prediction_rows = []
    for pred in predictions:
        conf_color = '#27ae60' if pred.confidence > 0.75 else '#f39c12' if pred.confidence > 0.6 else '#e74c3c'
        edge_color = '#27ae60' if pred.edge > 0.2 else '#e74c3c' if pred.edge < -0.2 else '#95a5a6'
        
        crew_names = [str(n).strip() for n in (getattr(pred, 'referee_crew', []) or []) if str(n).strip()]
        referee_display = getattr(pred, 'referee_info', None)
        if not referee_display:
            referee_display = ", ".join(crew_names)
        referee_display = referee_display or ''
        referee_metric_bits: List[str] = []
        if isinstance(pred.referee_avg_goals, (int, float)) and np.isfinite(pred.referee_avg_goals):
            referee_metric_bits.append(f"{pred.referee_avg_goals:.2f} G/G")
        if isinstance(pred.referee_home_bias, (int, float)) and np.isfinite(pred.referee_home_bias):
            referee_metric_bits.append(f"HB {pred.referee_home_bias:+.2f}")
        if isinstance(pred.ref_goals_gm, (int, float)) and np.isfinite(pred.ref_goals_gm) and not referee_metric_bits:
            referee_metric_bits.append(f"Feature {pred.ref_goals_gm:.2f} G/G")
        referee_cell = referee_display
        if referee_metric_bits:
            metrics_txt = ", ".join(referee_metric_bits)
            metric_html = f"<div style=\"font-size: 0.8rem; color:#7f8c8d;\">{metrics_txt}</div>"
            referee_cell = f"{referee_display}{metric_html}" if referee_display else metric_html

        ref_crew_attr = ", ".join(crew_names)
        ref_avg_attr = (
            f"{pred.referee_avg_goals:.3f}"
            if isinstance(pred.referee_avg_goals, (int, float)) and np.isfinite(pred.referee_avg_goals)
            else ''
        )
        ref_bias_attr = (
            f"{pred.referee_home_bias:.3f}"
            if isinstance(pred.referee_home_bias, (int, float)) and np.isfinite(pred.referee_home_bias)
            else ''
        )
        ref_source_attr = (str(pred.referee_source).strip() if getattr(pred, 'referee_source', None) else '')
        ref_goal_adjust_val = getattr(pred, 'ref_goal_adjustment', None)
        ref_goal_adjust_attr = (
            f"{ref_goal_adjust_val:.3f}"
            if isinstance(ref_goal_adjust_val, (int, float)) and np.isfinite(ref_goal_adjust_val)
            else ''
        )

        row = f"""
        <tr data-gid="{pred.game_id}" data-rec="{pred.recommendation}" data-matchup="{pred.away_team} @ {pred.home_team}"
            data-ref-crew="{attr_escape(ref_crew_attr)}"
            data-ref-avg="{attr_escape(ref_avg_attr)}"
            data-ref-bias="{attr_escape(ref_bias_attr)}"
            data-ref-source="{attr_escape(ref_source_attr)}"
            data-ref-adjust="{attr_escape(ref_goal_adjust_attr)}">
            <td><strong>{pred.away_team} @ {pred.home_team}</strong></td>
            <td>{pred.betting_line:.1f}</td>
            <td><strong style="color: #2c3e50;">{pred.predicted_total:.2f}</strong>
                <div style="font-size: 0.8rem; color:#7f8c8d;">CI90: {'' if pred.ci_lower is None else f'{pred.ci_lower:.1f}–{pred.ci_upper:.1f}'} </div>
                <div style="font-size: 0.8rem; color:#7f8c8d;">Odds: O {'' if pred.over_american_odds is None else pred.over_american_odds} {'' if pred.best_over_book is None else f'({pred.best_over_book})'} / U {'' if pred.under_american_odds is None else pred.under_american_odds} {'' if pred.best_under_book is None else f'({pred.best_under_book})'} </div>
            </td>
            <td style="color: {edge_color}; font-weight: bold;">{pred.edge:+.2f}
                <div style="font-size: 0.8rem; color:#7f8c8d;">No-vig EV: O {'' if pred.ev_over_novig is None else f'{pred.ev_over_novig:+.2f}'} / U {'' if pred.ev_under_novig is None else f'{pred.ev_under_novig:+.2f}'}</div>
                <div style="font-size: 0.8rem; color:#7f8c8d;">Line Δ vs consensus: {'' if pred.line_diff is None else f'{pred.line_diff:+.1f}'}</div>
            </td>
            <td>{pred.over_probability:.0%}
                <div style="font-size: 0.8rem; color:#7f8c8d;">Fair: {'' if pred.fair_over_prob is None else f'{pred.fair_over_prob:.0%}'}
                </div>
            </td>
            <td>{pred.under_probability:.0%}
                <div style="font-size: 0.8rem; color:#7f8c8d;">Fair: {'' if pred.fair_under_prob is None else f'{pred.fair_under_prob:.0%}'}
                </div>
            </td>
            <td>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 60px; height: 8px; background: #eee; border-radius: 4px;">
                        <div style="width: {pred.confidence*100}%; height: 100%; background: {conf_color}; border-radius: 4px;"></div>
                    </div>
                    <span>{pred.confidence:.0%}</span>
                </div>
            </td>
            <td>{referee_cell}</td>
            <td>{pred.env_info or ''}</td>
            <td>{pred.lineup_info or ''}</td>
            <td>
                <span class="rec-{pred.recommendation.lower().replace(' ', '-')}">{pred.recommendation}</span>
            </td>
            <td>{pred.kelly_bet_size:.1f}%</td>
        </tr>"""
        prediction_rows.append(row)
    
    # Embed per-book odds map for modal (if provided)
    per_book_map = {}
    try:
        if isinstance(betting_odds, dict):
            for pred in predictions:
                gid = str(getattr(pred, 'game_id', ''))
                if not gid:
                    continue
                rec = betting_odds.get(gid, {})
                books = rec.get('books') if isinstance(rec, dict) else None
                if isinstance(books, list) and books:
                    simple_list = []
                    for b in books:
                        try:
                            simple_list.append({'book': b.get('book'), 'book_key': b.get('book_key'), 'event_id': b.get('event_id'), 'total': b.get('total'), 'over': b.get('over'), 'under': b.get('under')})
                        except Exception:
                            continue
                    if simple_list:
                        per_book_map[gid] = simple_list
    except Exception:
        per_book_map = {}
    per_book_json = json.dumps(per_book_map, ensure_ascii=False).replace('\\', '\\\\').replace("'", "\\'")
    inline_js_resilient = """
            
    """
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NHL Over/Under Analytics Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 15px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .content {{
            padding: 30px;
        }}

        .toolbar {{
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 10px;
        }}

        .toolbar input, .toolbar select, .toolbar button {{
            padding: 8px 10px;
            border: 1px solid #ccc;
            border-radius: 6px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 25px;
            border-radius: 10px;
            border-left: 5px solid #3498db;
        }}
        
        .metric-card h3 {{
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 1.1rem;
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #27ae60;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        th {{
            background: linear-gradient(135deg, #34495e, #2c3e50);
            color: white;
            padding: 15px 10px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .rec-over {{
            background: linear-gradient(135deg, #d5f4e6, #b8e6cc);
            color: #27ae60;
            padding: 6px 12px;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        
        .rec-under {{
            background: linear-gradient(135deg, #fadbd8, #f1948a);
            color: #e74c3c;
            padding: 6px 12px;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        
        .rec-no-bet {{
            background: linear-gradient(135deg, #fdeaa7, #f9e79f);
            color: #f39c12;
            padding: 6px 12px;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        
        .status-bar {{
            background: linear-gradient(135deg, #e8f5e8, #d5f4e6);
            border: 1px solid #27ae60;
            border-radius: 8px;
            padding: 20px;
            margin-top: 30px;
            text-align: center;
            color: #27ae60;
            font-weight: 500;
        }}
        .legend {{
            margin: 12px 0 6px 0;
            font-size: 0.9rem;
            color: #2c3e50;
            background: #f9fbfd;
            border: 1px solid #e1e8f0;
            border-radius: 8px;
            padding: 10px 14px;
        }}
        .legend strong {{ display: inline-block; margin-bottom: 6px; }}
        .legend ul {{ margin-left: 18px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏒 NHL Over/Under Analytics Dashboard</h1>
            <p>Real-time predictions powered by machine learning</p>
        </div>
        
        <div class="content">
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>📊 Model Performance</h3>
                    <div class="metric-value">{training_results.get('rmse', 0):.3f}</div>
                </div>
                
                <div class="metric-card">
                    <h3>🎯 Over/Under Accuracy</h3>
                    <div class="metric-value">{training_results.get('over_under_accuracy', 0):.1%}</div>
                </div>
                
                <div class="metric-card">
                    <h3>💰 Betting Opportunities</h3>
                    <div class="metric-value">{len(betting_preds)}</div>
                </div>
                
                <div class="metric-card">
                    <h3>🔥 Average Confidence</h3>
                    <div class="metric-value">{avg_confidence:.0%}</div>
                </div>
            </div>
            
            <h2>🎯 Today's Predictions</h2>

            <div class="legend">
                <strong>Legend</strong>
                <ul>
                    <li><b>Line</b>: Market total for the game.</li>
                    <li><b>Prediction</b>: Model total with 90% CI.</li>
                    <li><b>Edge</b>: Predicted total minus line (positive favors OVER).</li>
                    <li><b>Over% / Under%</b>: Probability estimates for each side (Fair shows no‑vig).</li>
                    <li><b>Confidence</b>: Composite score (edge, calibration, dispersion, movement).</li>
                    <li><b>Refs</b>: Assigned crew with scoring and bias tendencies.</li>
                    <li><b>Env</b>: Outdoor flag, local start hour, temperature.</li>
                    <li><b>Lineup</b>: Aggregate lineup strength (Home/Away).</li>
                    <li><b>Recommendation</b>: OVER/UNDER/No Bet based on thresholds.</li>
                    <li><b>Kelly%</b>: Suggested stake (scaled for risk/dispersion).</li>
                </ul>
            </div>

            <div class="toolbar">
                <select id="filterRec" onchange="window.applyFilters && window.applyFilters()">
                    <option value="All">All</option>
                    <option value="OVER">OVER</option>
                    <option value="UNDER">UNDER</option>
                    <option value="No Bet">No Bet</option>
                </select>
                <button id="exportCsv" type="button" onclick="window.toCsv && window.toCsv()">Export CSV</button>
                <button id="copyBest" type="button" onclick="window.copyBestBets && window.copyBestBets()">Copy Best Bets</button>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>🏒 Matchup</th>
                        <th>📊 Line</th>
                        <th>🔮 Prediction</th>
                        <th>⚡ Edge</th>
                        <th>📈 Over %</th>
                        <th>📉 Under %</th>
                        <th>🔥 Confidence</th>
                        <th>🧑‍⚖️ Refs</th>
                        <th>🌤️ Env</th>
                        <th>👥 Lineup</th>
                        <th>💡 Recommendation</th>
                        <th>💰 Kelly %</th>
                    </tr>
                </thead>
                <tbody>
{''.join(prediction_rows)}
                </tbody>
            </table>

           <script>
// Bind resiliently to window to survive re-renders
(function(){{
  const table = document.querySelector('table'); if (!table) return;
  const tbody = table.querySelector('tbody');
  const getRows = () => Array.from(tbody.querySelectorAll('tr'));
  const norm = s => (s || '').replace(/\u00A0/g,' ').trim();
  try {{
    let sn0 = document.getElementById('status-note');
    if (!sn0) {{
      const sb = document.querySelector('.status-bar');
      if (sb) {{ sn0 = document.createElement('span'); sn0.id = 'status-note'; sb.appendChild(document.createTextNode(' ')); sb.appendChild(sn0); }}
    }}
    if (sn0) {{ sn0.textContent = 'Initializing UI…'; }}
  }} catch(e) {{}}

  // Client-side filter by Recommendation (works locally and attempts to in Streamlit iframe)
  window.applyFilters = function(){{
    try {{
      const recEl = document.getElementById('filterRec');
      const recUpper = norm(recEl && recEl.value).toUpperCase();
      getRows().forEach(tr => {{
        // Fallbacks: use cell text if data-attrs missing
        let recAttr = norm(tr.getAttribute('data-rec'));
        if (!recAttr) {{
          const recCell = Array.from(tr.children).find(td => td && td.querySelector && td.querySelector('.rec-over, .rec-under, .rec-no-bet'));
          if (recCell) {{ recAttr = norm(recCell.innerText); }}
        }}
        const recUpperRow = (recAttr || '').toUpperCase();
        const matchRec = !recUpper || recUpper === 'ALL' || recUpperRow === recUpper;
        tr.style.display = matchRec ? 'table-row' : 'none';
      }});
    }} catch(e) {{ console.log('applyFilters error', e); }}
  }};

    window.toCsv = function(){{
      const headers = Array.from(table.querySelectorAll('thead th')).map(th => norm(th.innerText));
      const extraHeaders = ['Referee Crew','Referee Avg Goals','Referee Home Bias','Referee Source','Ref Goal Adjustment'];
      const quote = value => '"' + value.replace(/"/g,'""') + '"';
      const lines = [headers.concat(extraHeaders).join(',')];
      getRows().forEach(tr => {{
        if (tr.style.display === 'none') return;
        const baseCols = Array.from(tr.children).map(td => quote(norm(td.innerText)));
        const extraCols = [
          quote(norm(tr.getAttribute('data-ref-crew') || tr.dataset.refCrew || '')),
          quote(norm(tr.getAttribute('data-ref-avg') || tr.dataset.refAvg || '')),
          quote(norm(tr.getAttribute('data-ref-bias') || tr.dataset.refBias || '')),
          quote(norm(tr.getAttribute('data-ref-source') || tr.dataset.refSource || '')),
          quote(norm(tr.getAttribute('data-ref-adjust') || tr.dataset.refAdjust || ''))
        ];
        lines.push(baseCols.concat(extraCols).join(','));
      }});
    const csv = '\ufeff' + lines.join('\n');
    const blob = new Blob([csv], {{type:'text/csv;charset=utf-8;'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display='none'; a.href=url; a.download='predictions.csv';
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }};

  window.copyBestBets = function(){{
    const lines = [];
    getRows().forEach(tr => {{
      if (tr.style.display === 'none') return;
      const recAttr = norm(tr.getAttribute('data-rec'));
      if (recAttr === 'No Bet') return;
      const matchup = norm(tr.getAttribute('data-matchup')) || norm(tr.children[0] && tr.children[0].innerText);
      const line = norm(tr.children[1] && tr.children[1].innerText);
      const pred = norm((tr.children[2] && tr.children[2].innerText) || '').replace('\\n',' ');
      const edge = norm(tr.children[3] && tr.children[3].innerText);
      lines.push(matchup + ': ' + recAttr + ' ' + line + ' (Pred ' + pred + ', ' + edge + ')');
    }});
    const text = lines.join('\n');
    function fallbackCopy(){{
      const ta = document.createElement('textarea');
      ta.value = text; ta.style.position='fixed'; ta.style.top='-1000px';
      document.body.appendChild(ta); ta.focus(); ta.select();
      try {{ document.execCommand('copy'); alert('Best bets copied to clipboard'); }} catch(e) {{}}
      document.body.removeChild(ta);
    }}
    if (navigator.clipboard && navigator.clipboard.writeText) {{
      navigator.clipboard.writeText(text).then(() => {{ alert('Best bets copied to clipboard'); }}).catch(fallbackCopy);
    }} else {{
      fallbackCopy();
    }}
  }};

  Array.from(table.querySelectorAll('thead th')).forEach((th, idx) => {{
    th.style.cursor = 'pointer';
    th.onclick = function(){{
      const dir = th.getAttribute('data-sort') === 'asc' ? -1 : 1;
      const rows = getRows();
      rows.sort((a,b) => {{
        const av = norm(a.children[idx] && a.children[idx].innerText);
        const bv = norm(b.children[idx] && b.children[idx].innerText);
        const an = parseFloat(av.replace(/[^0-9.-]/g,'')); const bn = parseFloat(bv.replace(/[^0-9.-]/g,''));
        if (!isNaN(an) && !isNaN(bn)) return dir*(an-bn);
        return dir*av.localeCompare(bv);
      }});
      tbody.innerHTML = '';
      rows.forEach(tr => tbody.appendChild(tr));
      th.setAttribute('data-sort', dir===1 ? 'asc' : 'desc');
    }};
  }});

  // Resilient bindings for Streamlit re-renders
  const bindControls = () => {{
    // Ensure toolbar exists (do not overwrite static content)
    let bar = document.querySelector('.toolbar');
    if (!bar) {{
      bar = document.createElement('div'); bar.className = 'toolbar';
      const container = table.closest('.content') || document.body;
      container.insertBefore(bar, table);
      // Hydrate defaults if we had to create it
      bar.innerHTML = '<select id="filterRec"><option value="All">All</option><option value="OVER">OVER</option><option value="UNDER">UNDER</option><option value="No Bet">No Bet</option></select> <button id="exportCsv" type="button">Export CSV</button> <button id="copyBest" type="button">Copy Best Bets</button>';
    }}
    const recEl = document.getElementById('filterRec');
    const expEl = document.getElementById('exportCsv');
    const copyEl = document.getElementById('copyBest');
    if (recEl && !recEl._bound) {{ recEl.addEventListener('change', () => {{ try {{ window.applyFilters && window.applyFilters(); }} catch(e) {{ console.log('applyFilters error', e); }} }}); recEl._bound = true; }}
    if (expEl && !expEl._bound) {{ expEl.addEventListener('click', () => {{ try {{ window.toCsv && window.toCsv(); }} catch(e) {{ console.log('toCsv error', e); }} }}); expEl._bound = true; }}
    if (copyEl && !copyEl._bound) {{ copyEl.addEventListener('click', () => {{ try {{ window.copyBestBets && window.copyBestBets(); }} catch(e) {{ console.log('copyBestBets error', e); }} }}); copyEl._bound = true; }}
  }};
  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', () => {{ bindControls(); window.applyFilters && window.applyFilters(); }});
  }} else {{
    bindControls();
  }}
  try {{
    const mo = new MutationObserver(() => {{ bindControls(); }});
    mo.observe(document.body, {{ childList: true, subtree: true }});
  }} catch(e) {{}}
  // Periodic safety rebinder for Streamlit iframe reflows
  try {{
    let tries = 0; const timer = setInterval(() => {{
      tries += 1; bindControls();
      if (tries >= 20) clearInterval(timer);
    }}, 500);
  }} catch(e) {{}}

  try {{
    window.applyFilters();
    let sn = document.getElementById('status-note');
    if (!sn) {{
      const sb = document.querySelector('.status-bar');
      if (sb) {{ sn = document.createElement('span'); sn.id = 'status-note'; sb.appendChild(document.createTextNode(' ')); sb.appendChild(sn); }}
    }}
    if (sn) {{ sn.textContent = 'UI ready — filters active (' + (typeof window.applyFilters) + ', ' + (typeof window.copyBestBets) + ')'; }}
  }} catch(e) {{
    let sn = document.getElementById('status-note');
    if (!sn) {{
      const sb = document.querySelector('.status-bar');
      if (sb) {{ sn = document.createElement('span'); sn.id = 'status-note'; sb.appendChild(document.createTextNode(' ')); sb.appendChild(sn); }}
    }}
    if (sn) {{ sn.textContent = 'UI init failed'; }}
  }}
}})();
</script>
            

           
            
            <div class="status-bar">
                <strong>🟢 Model Status:</strong> ACTIVE |
                <strong>📊 Training Data:</strong> {training_results.get('train_size', 0)} games |
                <strong>🕐 Last Updated:</strong> {(datetime.now(pytz.timezone(os.getenv('SCHEDULE_TZ','US/Eastern'))) if pytz else datetime.now()).strftime('%Y-%m-%d %I:%M:%S %p %Z' if pytz else '%Y-%m-%d %I:%M:%S %p')} |
                <span id="status-note"></span>
            </div>
        </div>
    </div>
</body>
</html>"""
    
    return html

def main(cli_args: Optional[argparse.Namespace] = None):
    """Main execution function"""
    print("🏒 NHL Over/Under Model with Real Data")
    print("=" * 50)
    
    model = None
    predictions = []
    dashboard_file = None
    
    try:
        model = RealDataNHLModel()
        
        print("\n📊 Step 1: Fetching historical NHL data...")
        print("🔄 Trying multiple data sources...")
        
        historical_data = model.fetch_historical_games(days_back=35)
        
        if len(historical_data) < 20:
            print(f"⚠️  Limited data ({len(historical_data)} games). Trying extended range...")
            historical_data = model.fetch_historical_games(days_back=90)
        
        if len(historical_data) < 10:
            print("⚠️  Still limited data. Using enhanced sample data...")
            historical_data = model.create_realistic_sample_data()
        
        print(f"✅ Using {len(historical_data)} games for training")
        
        if historical_data['total_goals'].mean() < 4 or historical_data['total_goals'].mean() > 8:
            print("⚠️  Data seems unusual. Using sample data...")
            historical_data = model.create_realistic_sample_data()
            print(f"✅ Using sample dataset: {len(historical_data)} games")
        
        print("\n🔧 Step 2: Engineering features...")
        enhanced_data = model.create_enhanced_features(historical_data)
        X, y, dates = model.prepare_model_data(enhanced_data)
        
        print(f"✅ Created {len(model.feature_names)} features from {len(X)} games")
        
        print("\n🎯 Step 3: Training prediction model...")
        training_results = model.train_model(X, y, dates)
        # Train goal models for bivariate Poisson MC
        try:
            model.train_goal_models(enhanced_data)
        except Exception as e:
            print(f"⚠️  Skipped goal model training: {e}")
        
        print(f"✅ Model trained successfully!")
        print(f"   📊 RMSE: {training_results['rmse']:.3f}")
        print(f"   🎯 O/U Accuracy: {training_results['over_under_accuracy']:.1%}")
        print(f"   📈 Training samples: {training_results['train_size']}")
        
        print("\n🏒 Step 4: Fetching today's games...")
        todays_games = model.get_todays_games(
            target_date=getattr(cli_args, 'date', None) if cli_args else None,
            offline_path=getattr(cli_args, 'today_games_path', None) if cli_args else None,
            offline_only=getattr(cli_args, 'offline', False) if cli_args else False
        )
        
        print(f"✅ Found {len(todays_games)} games to predict")
        
        detected_referees_url: Optional[str] = None
        referee_rates_fetched: Optional[pd.DataFrame] = None
        cached_referee_assignments_df: Optional[pd.DataFrame] = None
        referee_default_path = os.getenv('REFEREE_RATES_PATH') or 'referees.csv'
        if cli_args:
            cli_ref_path = getattr(cli_args, 'referee_rates_path', None)
            if not cli_ref_path:
                setattr(cli_args, 'referee_rates_path', referee_default_path)
        
        # Optional auto-populate CSVs from URLs
        if cli_args and getattr(cli_args, 'auto_populate', False):
            print("\n🌐 Auto-populating rate CSVs from provided URLs...")
            try:
                if cli_args.team_rates_url and cli_args.team_rates_path:
                    tr = None
                    try:
                        tr = model.load_team_rates(cli_args.team_rates_url)
                    except Exception:
                        time.sleep(1.0)
                        try:
                            tr = model.load_team_rates(cli_args.team_rates_url)
                        except Exception:
                            tr = None
                    if tr is not None:
                        target_path = ensure_local_write_path(cli_args.team_rates_path)
                        if target_path:
                            try:
                                tr.to_csv(target_path, index=False)
                                print(f"✅ Wrote team rates to {target_path}")
                            except Exception as e:
                                print(f"⚠️  Team rates fetched but could not write to {target_path}: {e}")
                        else:
                            print(f"⚠️  Team rates fetched but could not determine local path for {cli_args.team_rates_path}")
                if cli_args.goalie_gsax_url and cli_args.goalie_gsax_path:
                    gg = None
                    try:
                        gg = model.load_goalie_gsax(cli_args.goalie_gsax_url)
                    except Exception:
                        time.sleep(1.0)
                        try:
                            gg = model.load_goalie_gsax(cli_args.goalie_gsax_url)
                        except Exception:
                            gg = None
                    if gg is not None:
                        target_path = ensure_local_write_path(cli_args.goalie_gsax_path)
                        if target_path:
                            try:
                                gg.to_csv(target_path, index=False)
                                print(f"✅ Wrote goalie GSAx to {target_path}")
                            except Exception as e:
                                print(f"⚠️  Goalie GSAx fetched but could not write to {target_path}: {e}")
                        else:
                            print(f"⚠️  Goalie GSAx fetched but could not determine local path for {cli_args.goalie_gsax_path}")
                if cli_args.penalties_url and cli_args.penalty_rates_path:
                    pr = model.load_penalty_rates(cli_args.penalties_url)
                    if pr is not None:
                        target_path = ensure_local_write_path(cli_args.penalty_rates_path)
                        if target_path:
                            try:
                                pr.to_csv(target_path, index=False)
                                print(f"✅ Wrote penalties to {target_path}")
                            except Exception as e:
                                print(f"⚠️  Penalty rates fetched but could not write to {target_path}: {e}")
                        else:
                            print(f"⚠️  Penalty rates fetched but could not determine local path for {cli_args.penalty_rates_path}")
                if (cli_args.referees_url or True) and cli_args.referee_rates_path:
                    rr = None
                    try:
                        src_url = cli_args.referees_url
                        if not src_url:
                            src_url = model.find_todays_scoutingtherefs_url()
                        if src_url:
                            detected_referees_url = src_url
                            url_l = str(src_url).lower()
                            if url_l.endswith('.csv'):
                                rr = model.load_referee_rates(src_url)
                            else:
                                rr = model.scrape_referees_scoutingtherefs(src_url)
                    except Exception:
                        rr = None
                    if rr is not None and not rr.empty:
                        # Ensure Goals/Gm present with a neutral baseline if missing
                        if 'goals_gm' not in rr.columns:
                            rr['goals_gm'] = 6.2
                        referee_rates_fetched = rr.copy()
                        if 'matchup' in rr.columns:
                            cached_referee_assignments_df = rr.copy()
                        target_path = ensure_local_write_path(cli_args.referee_rates_path)
                        if target_path:
                            try:
                                rr.to_csv(target_path, index=False)
                                print(f"✅ Wrote referees to {target_path}")
                            except Exception as e:
                                print(f"⚠️  Referee data fetched but could not write to {target_path}: {e}")
                        else:
                            print(f"⚠️  Referee data fetched but could not write to {cli_args.referee_rates_path}")
            except Exception as e:
                print(f"⚠️  Auto-populate failed: {e}")

        referee_rates_path = getattr(cli_args, 'referee_rates_path', None) if cli_args else referee_default_path
        if not referee_rates_path:
            referee_rates_path = referee_default_path
        referees_url_config: Optional[str] = getattr(cli_args, 'referees_url', None) if cli_args else None
        if not referees_url_config:
            env_ref_url = os.getenv('REFEREES_URL')
            if env_ref_url:
                referees_url_config = env_ref_url
        if not referees_url_config and detected_referees_url:
            referees_url_config = detected_referees_url

        # Independently refresh referees.csv if a path is provided, even without --auto-populate
        try:
            if referee_rates_path:
                have_df = isinstance(referee_rates_fetched, pd.DataFrame) and not referee_rates_fetched.empty
                rr: Optional[pd.DataFrame] = referee_rates_fetched if have_df else None
                src_url = referees_url_config
                if not have_df:
                    if not src_url:
                        try:
                            rr_url = model.find_todays_scoutingtherefs_url()
                            src_url = rr_url
                        except Exception:
                            src_url = None
                    if src_url:
                        try:
                            url_l = str(src_url).lower()
                            if url_l.endswith('.csv'):
                                rr = model.load_referee_rates(src_url)
                            else:
                                rr = model.scrape_referees_scoutingtherefs(src_url)
                        except Exception:
                            rr = None
                    if rr is not None and not rr.empty:
                        referee_rates_fetched = rr.copy()
                        if 'matchup' in rr.columns:
                            cached_referee_assignments_df = rr.copy()
                        if src_url:
                            detected_referees_url = src_url
                            if not referees_url_config:
                                referees_url_config = src_url
                        have_df = True
                if have_df and isinstance(referee_rates_fetched, pd.DataFrame):
                    if 'goals_gm' not in referee_rates_fetched.columns:
                        referee_rates_fetched['goals_gm'] = 6.2
                    target_path = ensure_local_write_path(referee_rates_path)
                    if target_path:
                        try:
                            referee_rates_fetched.to_csv(target_path, index=False)
                            print(f"✅ Wrote referees to {target_path}")
                            referee_rates_path = target_path
                        except Exception as e:
                            print(f"⚠️  Referee data fetched but could not write to {target_path}: {e}")
                    else:
                        print(f"⚠️  Referee data fetched but could not write to {referee_rates_path}")
                else:
                    print("⚠️  Referee data was not updated (empty result)")
        except Exception as e:
            print(f"⚠️  Referee update failed: {e}")

        print("\n🔮 Step 5: Generating predictions...")
        predictions = []
        
        if len(todays_games) > 0:
            # Apply configurable CI quantile if provided
            try:
                if cli_args and getattr(cli_args, 'ci_quantile', None) is not None:
                    model.ci_quantile = float(cli_args.ci_quantile)
            except Exception:
                pass
            combined_data = pd.concat([historical_data, todays_games], ignore_index=True)
            combined_features = model.create_enhanced_features(combined_data)
            todays_features = combined_features.tail(len(todays_games)).copy()

            # Load full odds including American prices if available
            if cli_args and cli_args.odds_path:
                os.environ['ODDS_JSON_PATH'] = cli_args.odds_path
            if cli_args and getattr(cli_args, 'realtime_odds', False):
                betting_odds = model.get_betting_odds_realtime(
                    todays_games,
                    regions=getattr(cli_args, 'odds_regions', 'us,us2,eu,uk,au'),
                    timeout_s=getattr(cli_args, 'odds_timeout', 25),
                    retries=getattr(cli_args, 'odds_retries', 3),
                    dispersion_all=bool(getattr(cli_args, 'odds_dispersion_all', False))
                )
            else:
                betting_odds = model.get_betting_odds(todays_games)

            # Auto-generate environment template if requested
            try:
                if cli_args and getattr(cli_args, 'environment_path', None):
                    refresh = bool(getattr(cli_args, 'env_refresh', False))
                    need_create = not os.path.exists(cli_args.environment_path)
                    if need_create or refresh:
                        model.write_environment_template(todays_games, cli_args.environment_path, overwrite_today=refresh)
            except Exception as e:
                print(f"⚠️  Environment template not written: {e}")

            # Optional goalie/injury adjustments
            status_path = getattr(cli_args, 'status_path', None) if cli_args else os.getenv('STATUS_JSON_PATH', 'status.json')
            status_adj = model.get_status_adjustments(todays_games, status_path=status_path)

            # Optional xG adjustments
            xg_path = getattr(cli_args, 'xg_path', None) if cli_args else os.getenv('XG_JSON_PATH', None)
            xg_adj = model.load_xg_adjustments(
                todays_games,
                xg_path,
                baseline_total=getattr(cli_args, 'xg_baseline_total', float(os.getenv('XG_BASELINE_TOTAL', 6.2))) if cli_args else float(os.getenv('XG_BASELINE_TOTAL', 6.2)),
                clamp_abs=getattr(cli_args, 'xg_clamp_abs', float(os.getenv('XG_CLAMP_ABS', 2.0))) if cli_args else float(os.getenv('XG_CLAMP_ABS', 2.0))
            )

            # Apply Kelly CLI settings
            try:
                if cli_args:
                    model.kelly_mult = float(getattr(cli_args, 'kelly_mult', model.kelly_mult))
                    model.kelly_cap_pct = float(getattr(cli_args, 'kelly_cap', model.kelly_cap_pct))
                    model.daily_exposure_cap_pct = float(getattr(cli_args, 'daily_exposure_cap', model.daily_exposure_cap_pct))
                    model.kelly_use_fair = bool(getattr(cli_args, 'kelly_use_fair', False))
                    model.use_compoisson = bool(getattr(cli_args, 'use_compoisson', False))
            except Exception:
                pass

            # Optional real team rates overlay (replaces proxies when present)
            team_rates = None
            try:
                tr_path = getattr(cli_args, 'team_rates_path', None) if cli_args else os.getenv('TEAM_RATES_PATH')
                team_rates = model.load_team_rates(tr_path)
            except Exception:
                team_rates = None

            # Optional goalie GSAx
            goalie_rates = None
            try:
                gg_path = getattr(cli_args, 'goalie_gsax_path', None) if cli_args else os.getenv('GOALIE_GSAX_PATH')
                goalie_rates = model.load_goalie_gsax(gg_path)
            except Exception:
                goalie_rates = None

            # Optional penalties and referees
            penalty_rates = None
            try:
                pr_path = getattr(cli_args, 'penalty_rates_path', None) if cli_args else os.getenv('PENALTY_RATES_PATH')
                penalty_rates = model.load_penalty_rates(pr_path)
            except Exception:
                penalty_rates = None

            rr_path = referee_rates_path
            referee_rates = None
            referee_rates_source: Optional[str] = None

            if referee_rates_fetched is not None and isinstance(referee_rates_fetched, pd.DataFrame) and not referee_rates_fetched.empty:
                referee_rates = referee_rates_fetched.copy()
                referee_rates_source = referees_url_config or str(rr_path)

            if referee_rates is None:
                try:
                    referee_rates = model.load_referee_rates(rr_path)
                    if isinstance(referee_rates, pd.DataFrame) and not referee_rates.empty and referee_rates_source is None:
                        referee_rates_source = str(rr_path) if rr_path else None
                except Exception:
                    referee_rates = None

            referee_default_goal: Optional[float] = None
            if isinstance(referee_rates, pd.DataFrame) and 'goals_gm' in referee_rates.columns:
                try:
                    referee_default_goal = float(pd.to_numeric(referee_rates['goals_gm'], errors='coerce').dropna().mean())
                    if not np.isfinite(referee_default_goal):
                        referee_default_goal = None
                except Exception:
                    referee_default_goal = None

            referee_goal_map: Dict[str, float] = {}
            referee_bias_map: Dict[str, float] = {}
            referee_assignments_df: Optional[pd.DataFrame] = None

            def normalize_referee_assignments(df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                try:
                    if 'ref' in out.columns:
                        out['ref'] = out['ref'].astype(str).str.strip()
                    if 'matchup' in out.columns:
                        out['matchup'] = out['matchup'].astype(str).str.upper().str.strip()
                except Exception:
                    pass
                return out

            referee_assignments_source: Optional[str] = None
            referee_map: Dict[str, List[str]] = {}

            if (
                cached_referee_assignments_df is not None
                and isinstance(cached_referee_assignments_df, pd.DataFrame)
                and not cached_referee_assignments_df.empty
                and 'matchup' in cached_referee_assignments_df.columns
            ):
                referee_assignments_df = normalize_referee_assignments(cached_referee_assignments_df)
                fallback_source = rr_path if rr_path else None
                referee_assignments_source = referees_url_config or detected_referees_url or fallback_source

            auto_url_for_map: Optional[str] = referees_url_config or detected_referees_url
            try:
                need_assignments = referee_assignments_df is None or referee_assignments_df.empty
                auto_url = auto_url_for_map
                if need_assignments and not auto_url:
                    auto_url = model.find_todays_scoutingtherefs_url()
                if need_assignments and auto_url:
                    referee_assignments_source = str(auto_url)
                    scraped_assignments = model.scrape_referees_scoutingtherefs(auto_url)
                    if isinstance(scraped_assignments, pd.DataFrame) and not scraped_assignments.empty:
                        normalized = normalize_referee_assignments(scraped_assignments)
                        referee_assignments_df = normalized
                        cached_referee_assignments_df = normalized.copy()
                        detected_referees_url = auto_url
                        if not referees_url_config:
                            referees_url_config = auto_url
                    auto_url_for_map = auto_url
            except Exception:
                referee_map = {}
                referee_assignments_df = None
                referee_assignments_source = None
                auto_url_for_map = referees_url_config or detected_referees_url

            if isinstance(referee_assignments_df, pd.DataFrame) and not referee_assignments_df.empty:
                if 'matchup' in referee_assignments_df.columns:
                    for mk, grp in referee_assignments_df.groupby('matchup'):
                        mk_str = str(mk or '').strip().upper()
                        if not mk_str or mk_str == 'NAN':
                            continue
                        crew_series = grp.get('ref', pd.Series(dtype=object))
                        crew_list = [str(nm).strip() for nm in crew_series.dropna().unique() if str(nm).strip()]
                        parts = mk_str.split('@')
                        if len(parts) == 2:
                            away_code, home_code = parts[0], parts[1]
                            key_primary = f"{away_code}@{home_code}"
                            key_secondary = f"{home_code}@{away_code}"
                        else:
                            key_primary = mk_str
                            key_secondary = None
                        if crew_list:
                            referee_map[key_primary] = crew_list
                            if key_secondary:
                                referee_map[key_secondary] = crew_list
                        crew_vals = pd.to_numeric(grp.get('crew_goals_gm', pd.Series(dtype=float)), errors='coerce')
                        crew_vals = crew_vals.dropna() if isinstance(crew_vals, pd.Series) else pd.Series(dtype=float)
                        value: Optional[float] = None
                        if isinstance(crew_vals, pd.Series) and len(crew_vals):
                            value = float(crew_vals.mean())
                        if value is None or not np.isfinite(value):
                            indiv_vals = pd.to_numeric(grp.get('goals_gm', pd.Series(dtype=float)), errors='coerce')
                            if isinstance(indiv_vals, pd.Series):
                                indiv_vals = indiv_vals.dropna()
                                if len(indiv_vals):
                                    value = float(indiv_vals.mean())
                        if isinstance(value, (int, float)) and np.isfinite(value):
                            referee_goal_map[key_primary] = float(value)
                            if key_secondary:
                                referee_goal_map[key_secondary] = float(value)
                if cached_referee_assignments_df is None:
                    cached_referee_assignments_df = referee_assignments_df.copy()

            if not referee_map and auto_url_for_map:
                referee_map = model.build_referee_crew_map(auto_url_for_map, todays_games)

            # Merge scraped goals/gm into referee rates so downstream averaging works
            if isinstance(referee_assignments_df, pd.DataFrame) and not referee_assignments_df.empty:
                try:
                    assign_df = referee_assignments_df.copy()
                    for col in ('goals_gm', 'crew_goals_gm'):
                        if col in assign_df.columns:
                            assign_df[col] = pd.to_numeric(assign_df[col], errors='coerce')
                    simple_rates = None
                    if 'goals_gm' in assign_df.columns and assign_df['goals_gm'].notna().any():
                        simple_rates = assign_df[['ref', 'goals_gm']].dropna(subset=['ref', 'goals_gm'])
                    elif 'crew_goals_gm' in assign_df.columns and assign_df['crew_goals_gm'].notna().any():
                        simple_rates = assign_df[['ref', 'crew_goals_gm']].dropna(subset=['ref', 'crew_goals_gm']).rename(columns={'crew_goals_gm': 'goals_gm'})
                    if simple_rates is not None and len(simple_rates):
                        simple_rates = simple_rates.groupby('ref', as_index=False)['goals_gm'].mean()
                        if referee_rates is None or not isinstance(referee_rates, pd.DataFrame) or referee_rates.empty:
                            referee_rates = simple_rates.copy()
                            referee_rates_source = referee_rates_source or referee_assignments_source
                        else:
                            merged_base = referee_rates.copy()
                            try:
                                merged_base['ref'] = merged_base['ref'].astype(str).str.strip()
                                merged_base = merged_base.set_index('ref')
                                simple_idx = simple_rates.copy()
                                simple_idx['ref'] = simple_idx['ref'].astype(str).str.strip()
                                simple_idx = simple_idx.set_index('ref')
                                for col in simple_idx.columns:
                                    if col not in merged_base.columns:
                                        merged_base[col] = np.nan
                                missing_refs = simple_idx.index.difference(merged_base.index)
                                merged_base.update(simple_idx)
                                if len(missing_refs):
                                    merged_base = pd.concat([merged_base, simple_idx.loc[missing_refs]], axis=0)
                                referee_rates = merged_base.reset_index()
                            except Exception:
                                pass
                except Exception:
                    pass

            if (referee_default_goal is None or not np.isfinite(referee_default_goal)) and isinstance(referee_rates, pd.DataFrame) and 'goals_gm' in referee_rates.columns:
                try:
                    referee_default_goal = float(pd.to_numeric(referee_rates['goals_gm'], errors='coerce').dropna().mean())
                    if not np.isfinite(referee_default_goal):
                        referee_default_goal = None
                except Exception:
                    referee_default_goal = None

            # Pre-compute referee Goals/Gm feature for today's matchups
            try:
                if isinstance(todays_features, pd.DataFrame) and 'ref_goals_gm' not in todays_features.columns:
                    todays_features['ref_goals_gm'] = np.nan
                if isinstance(referee_rates, pd.DataFrame):
                    if isinstance(referee_map, dict) and referee_map:
                        for matchup_key, crew in referee_map.items():
                            avg_goals, avg_bias = model.crew_features(crew, referee_rates)
                            if isinstance(avg_goals, (int, float)) and np.isfinite(avg_goals):
                                if matchup_key not in referee_goal_map:
                                    referee_goal_map[matchup_key] = float(avg_goals)
                            if isinstance(avg_bias, (int, float)) and np.isfinite(avg_bias):
                                if matchup_key not in referee_bias_map:
                                    referee_bias_map[matchup_key] = float(avg_bias)
                    if isinstance(todays_features, pd.DataFrame):
                        goal_values = []
                        for _, row in todays_features[['away_team', 'home_team']].iterrows():
                            mk = f"{row['away_team']}@{row['home_team']}"
                            val = referee_goal_map.get(mk, referee_default_goal)
                            goal_values.append(float(val) if isinstance(val, (int, float)) and np.isfinite(val) else np.nan)
                        todays_features['ref_goals_gm'] = goal_values
                        baseline_fill = getattr(model, 'ref_goal_baseline', None)
                        if baseline_fill is None or not np.isfinite(baseline_fill):
                            if isinstance(referee_default_goal, (int, float)) and np.isfinite(referee_default_goal):
                                baseline_fill = float(referee_default_goal)
                            else:
                                try:
                                    baseline_fill = float(os.getenv('REF_GOAL_BASELINE', 6.2))
                                except Exception:
                                    baseline_fill = 6.2
                        todays_features['ref_goals_gm'] = pd.to_numeric(todays_features['ref_goals_gm'], errors='coerce').fillna(baseline_fill)
            except Exception:
                pass

            # Optional odds history logging
            if cli_args and getattr(cli_args, 'log_odds_history', False):
                try:
                    rows = []
                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    for _, g in todays_games.iterrows():
                        gid = str(g.get('game_id'))
                        rec = betting_odds.get(gid, {})
                        base_row = {
                            'timestamp': ts,
                            'game_id': gid,
                            'matchup': f"{g.get('away_team')}@{g.get('home_team')}",
                            'consensus_total': rec.get('consensus_total'),
                            'best_over': rec.get('over'),
                            'best_under': rec.get('under'),
                            'best_over_book': rec.get('best_over_book'),
                            'best_under_book': rec.get('best_under_book')
                        }
                        # One row per book when available; otherwise one aggregate row
                        if isinstance(rec.get('books'), list) and rec.get('books'):
                            for b in rec['books']:
                                row = dict(base_row)
                                row.update({'book': b.get('book'), 'book_key': b.get('book_key'), 'event_id': b.get('event_id'), 'book_total': b.get('total'), 'book_over': b.get('over'), 'book_under': b.get('under')})
                                rows.append(row)
                        else:
                            row = dict(base_row)
                            row.update({'book': rec.get('source'), 'book_key': rec.get('source'), 'event_id': None, 'book_total': rec.get('total'), 'book_over': rec.get('over'), 'book_under': rec.get('under')})
                            rows.append(row)
                    if rows:
                        dfh = pd.DataFrame(rows)
                        header = not os.path.exists(cli_args.odds_history_path)
                        dfh.to_csv(cli_args.odds_history_path, mode='a', header=header, index=False)
                        print(f"✅ Logged {len(rows)} odds snapshots to {cli_args.odds_history_path}")
                except Exception as e:
                    print(f"⚠️  Could not log odds history: {e}")
            
            # Optional closing odds data for open→close calibration
            closing_data = None
            try:
                closing_path = getattr(cli_args, 'closing_odds_path', None) if cli_args else None
                if closing_path and os.path.exists(closing_path):
                    with open(closing_path, 'r') as f:
                        closing_data = json.load(f)
            except Exception:
                closing_data = None

            # Optional odds velocity from odds_history.csv
            velocity_map = {}
            # Environment and lineup data
            env_data = {}
            try:
                env_path = getattr(cli_args, 'environment_path', None) if cli_args else os.getenv('ENVIRONMENT_JSON')
                env_data = model.load_environment(todays_games, env_path)
            except Exception:
                env_data = {}
            lineup_strength = {}
            try:
                lineup_path = getattr(cli_args, 'lineup_path', None) if cli_args else os.getenv('LINEUP_STRENGTH_CSV')
                lineup_strength = model.load_lineup_strength(lineup_path)
                try:
                    print(f"🧩 Lineup strength loaded: {len(lineup_strength)} teams from {lineup_path}")
                except Exception:
                    pass
            except Exception:
                lineup_strength = {}
            try:
                hist_path = getattr(cli_args, 'odds_history_path', None) if cli_args else 'odds_history.csv'
                if hist_path and os.path.exists(hist_path):
                    oh = pd.read_csv(hist_path)
                    if {'timestamp','game_id','book_total'}.issubset(set(oh.columns)):
                        oh['ts'] = pd.to_datetime(oh['timestamp'], errors='coerce')
                        for gid, grp in oh.groupby('game_id'):
                            g = grp.sort_values('ts')
                            if len(g) >= 2:
                                try:
                                    dt = (g['ts'].iloc[-1] - g['ts'].iloc[0]).total_seconds() / 3600.0
                                    if dt > 0:
                                        vel = (float(g['book_total'].iloc[-1]) - float(g['book_total'].iloc[0])) / dt
                                        velocity_map[str(gid)] = vel
                                except Exception:
                                    continue
            except Exception:
                velocity_map = {}
            
            # Build optional team rate map for quick lookups
            team_rate_map = None
            if team_rates is not None and isinstance(team_rates, pd.DataFrame) and 'team' in team_rates.columns:
                try:
                    needed = {
                        'xgf60_5v5': ['xgf60_5v5','xgf60','xgf_5v5'],
                        'hdcf60_5v5': ['hdcf60_5v5','hdcf60','hdcf_5v5'],
                        'pp_xgf60': ['pp_xgf60','pp_xgf'],
                        'pk_xga60': ['pk_xga60','pk_xga'],
                    }
                    cols = {}
                    for k, aliases in needed.items():
                        for a in aliases:
                            if a in team_rates.columns:
                                cols[k] = a
                                break
                    team_rate_map = team_rates.rename(columns={v:k for k,v in cols.items()}).set_index('team')
                except Exception:
                    team_rate_map = None

            for idx, game in todays_features.iterrows():
                try:
                    feature_values = []
                    for feature_name in model.feature_names:
                        if feature_name in game.index and pd.notna(game[feature_name]):
                            feature_values.append(float(game[feature_name]))
                        else:
                            defaults = {
                                'home_gpg_l5': 3.0, 'away_gpg_l5': 3.0,
                                'combined_gpg': 3.0, 'venue_total_avg': 6.2,
                                'base_total_prediction': 6.2, 'final_prediction_base': 6.2,
                                'pace_zscore': 0.0, 'rest_diff': 0.0, 'schedule_density_diff': 0.0,
                                'travel_fatigue_index': 0.0,
                                'home_shoot_pct_l5': 0.10, 'away_shoot_pct_l5': 0.10,
                                'home_save_pct_l5': 0.92, 'away_save_pct_l5': 0.92,
                                'special_teams_index': 0.0, 'special_teams_diff': 0.0
                            }
                            feature_values.append(defaults.get(feature_name, 0.0))
                    
                    game_id = str(game.get('game_id', f'game_{idx}'))
                    odds_rec = betting_odds.get(game_id, {'total': 6.5, 'over': -110, 'under': -110})
                    betting_line = float(odds_rec.get('total', 6.5))
                    over_price = int(odds_rec.get('over', -110))
                    under_price = int(odds_rec.get('under', -110))
                    odds_source = str(odds_rec.get('source')) if 'source' in odds_rec else None
                    consensus_total = float(odds_rec.get('consensus_total', betting_line))
                    best_over_book = odds_rec.get('best_over_book')
                    best_under_book = odds_rec.get('best_under_book')

                    # Pass through current market dispersion and stale-book factor for Kelly scaling
                    if isinstance(odds_rec.get('dispersion_total_std'), (int, float)):
                        try:
                            model.current_dispersion_std = float(odds_rec.get('dispersion_total_std'))
                        except Exception:
                            pass
                    try:
                        # Stale-book: if chosen best price deviates >0.15 from median decimal, downscale
                        books = odds_rec.get('books') if isinstance(odds_rec, dict) else None
                        stale_factor = 1.0
                        if isinstance(books, list) and books:
                            decs = []
                            for b in books:
                                try:
                                    if isinstance(b.get('total'), (int, float)) and abs(float(b['total']) - betting_line) < 1e-6:
                                        if isinstance(b.get('over'), (int, float)):
                                            oo = 1.0 + (float(b['over'])/100.0) if b['over'] >= 100 else 1.0 + (100.0/abs(float(b['over'])))
                                            decs.append(oo)
                                        if isinstance(b.get('under'), (int, float)):
                                            uu = 1.0 + (float(b['under'])/100.0) if b['under'] >= 100 else 1.0 + (100.0/abs(float(b['under'])))
                                            decs.append(uu)
                                except Exception:
                                    continue
                            if len(decs) >= 3:
                                med = float(np.median(decs))
                                best_dec = 1.0 + (float(over_price)/100.0) if over_price >= 100 else 1.0 + (100.0/abs(float(over_price)))
                                # if best price is too far from median, scale down
                                if abs(best_dec - med) / max(1e-6, med) > 0.15:
                                    stale_factor = 0.75
                        model.current_stale_factor = stale_factor
                    except Exception:
                        pass
                    pred = model.predict_game(
                        np.array(feature_values), betting_line,
                        over_american_odds=over_price, under_american_odds=under_price,
                        odds_source=odds_source, consensus_total=consensus_total
                    )
                    pred.game_id = game_id
                    pred.home_team = game.get('home_team', 'HOME')
                    pred.away_team = game.get('away_team', 'AWAY')
                    pred.best_over_book = best_over_book
                    pred.best_under_book = best_under_book

                    ref_goal_val = game.get('ref_goals_gm')
                    if isinstance(ref_goal_val, (int, float)) and np.isfinite(ref_goal_val):
                        pred.ref_goals_gm = float(ref_goal_val)
                    else:
                        pred.ref_goals_gm = None

                    matchup_key = f"{pred.away_team}@{pred.home_team}"
                    crew_names: List[str] = []
                    if isinstance(referee_map, dict):
                        raw_crew = referee_map.get(matchup_key)
                        if isinstance(raw_crew, (list, tuple, set)):
                            crew_names = [str(n).strip() for n in raw_crew if str(n).strip()]
                        elif isinstance(raw_crew, str) and raw_crew.strip():
                            crew_names = [n.strip() for n in re.split(r",|;|/|\band\b", raw_crew, flags=re.IGNORECASE) if n.strip()]
                    pred.referee_crew = crew_names

                    crew_goal_val = referee_goal_map.get(matchup_key, referee_default_goal)
                    if isinstance(crew_goal_val, (int, float)) and np.isfinite(crew_goal_val):
                        pred.referee_avg_goals = float(crew_goal_val)
                        if pred.ref_goals_gm is None:
                            pred.ref_goals_gm = float(crew_goal_val)

                    bias_val = referee_bias_map.get(matchup_key)
                    if isinstance(bias_val, (int, float)) and np.isfinite(bias_val):
                        pred.referee_home_bias = float(bias_val)

                    source_val = referee_assignments_source or referee_rates_source
                    if source_val:
                        pred.referee_source = source_val

                    info_parts: List[str] = []
                    if crew_names:
                        info_parts.append(", ".join(crew_names))
                    metric_parts: List[str] = []
                    if isinstance(pred.referee_avg_goals, (int, float)) and np.isfinite(pred.referee_avg_goals):
                        metric_parts.append(f"{pred.referee_avg_goals:.2f} G/G")
                    if isinstance(pred.referee_home_bias, (int, float)) and np.isfinite(pred.referee_home_bias):
                        metric_parts.append(f"HB {pred.referee_home_bias:+.2f}")
                    if metric_parts:
                        info_parts.append(f"({', '.join(metric_parts)})")
                    pred.referee_info = " ".join(info_parts) if info_parts else None
                    # Capture scheduled start time (Eastern) for social media presentation
                    try:
                        game_date_raw = game.get('date')
                        start_ts = None
                        if isinstance(game_date_raw, pd.Timestamp):
                            start_ts = game_date_raw
                        elif isinstance(game_date_raw, datetime):
                            start_ts = pd.Timestamp(game_date_raw)
                        elif isinstance(game_date_raw, str) and game_date_raw:
                            start_ts = pd.to_datetime(game_date_raw, errors='coerce', utc=True)
                        if start_ts is not None and not pd.isna(start_ts):
                            if start_ts.tzinfo is None or start_ts.tzinfo.utcoffset(start_ts) is None:
                                start_ts = start_ts.tz_localize('UTC')
                            start_ts_et = start_ts.tz_convert('US/Eastern')
                            pred.game_datetime = start_ts_et.to_pydatetime()
                            pred.game_datetime_utc = start_ts.tz_convert('UTC').to_pydatetime()
                            pred.start_time_display = start_ts_et.strftime('%I:%M %p ET')
                        else:
                            pred.game_datetime = None
                            pred.game_datetime_utc = None
                            pred.start_time_display = ''
                    except Exception:
                        pred.game_datetime = None
                        pred.game_datetime_utc = None
                        pred.start_time_display = ''

                    # Apply status and xG adjustments to predicted total and edge
                    total_adj = 0.0
                    if game_id in status_adj:
                        s = status_adj[game_id]
                        total_adj += float(s.get('home_goalie_adj', 0.0) + s.get('away_goalie_adj', 0.0) + s.get('injury_penalty_adj', 0.0))
                    if game_id in xg_adj:
                        total_adj += float(xg_adj[game_id])
                    # Environment: outdoor/start-time/weather small capped adjustments
                    try:
                        e = env_data.get(game_id)
                        if isinstance(e, dict):
                            if e.get('outdoor'):
                                total_adj += 0.03  # slight variance boost
                            hr = int(e.get('start_hour_local', 19))
                            if hr < 13:
                                total_adj -= 0.05  # early starts trend slightly under
                            elif hr >= 21:
                                total_adj += 0.03
                            # Weather effects (very small; mostly for outdoors)
                            temp_f = float(e.get('temp_f', 70.0))
                            wind = float(e.get('wind_mph', 0.0))
                            total_adj += float(max(-0.08, min(0.08, (70.0 - temp_f) * 0.001 - wind * 0.002)))
                    except Exception:
                        pass
                    # Lineup strength: favor higher combined offensive strength slightly
                    try:
                        ht = str(pred.home_team).upper(); at = str(pred.away_team).upper()
                        hs = float(lineup_strength.get(ht, 2.0))
                        as_ = float(lineup_strength.get(at, 2.0))
                        total_adj += float(max(-0.2, min(0.2, 0.03 * (hs + as_ - 4.0))))
                    except Exception:
                        pass
                    # PP vs PK matchup blend using team_rates when available
                    if team_rate_map is not None:
                        try:
                            ht = str(pred.home_team).upper(); at = str(pred.away_team).upper()
                            if ht in team_rate_map.index and at in team_rate_map.index:
                                pp_home = float(team_rate_map.loc[ht].get('pp_xgf60', np.nan))
                                pp_away = float(team_rate_map.loc[at].get('pp_xgf60', np.nan))
                                pk_home = float(team_rate_map.loc[ht].get('pk_xga60', np.nan))
                                pk_away = float(team_rate_map.loc[at].get('pk_xga60', np.nan))
                                comp = [v for v in [pp_home, pp_away, pk_home, pk_away] if np.isfinite(v)]
                                if comp:
                                    # If both teams have strong PP relative to opponent PK, nudge total up
                                    pp_adv = 0.0
                                    if np.isfinite(pp_home) and np.isfinite(pk_away):
                                        pp_adv += (pp_home - pk_away)
                                    if np.isfinite(pp_away) and np.isfinite(pk_home):
                                        pp_adv += (pp_away - pk_home)
                                    total_adj += float(max(-0.4, min(0.4, 0.02 * pp_adv)))
                        except Exception:
                            pass
                    # Blend real team rates if available (small effect)
                    if team_rate_map is not None:
                        try:
                            ht = str(pred.home_team).upper(); at = str(pred.away_team).upper()
                            if ht in team_rate_map.index and at in team_rate_map.index:
                                hx = float(team_rate_map.loc[ht].get('xgf60_5v5', np.nan))
                                ax = float(team_rate_map.loc[at].get('xgf60_5v5', np.nan))
                                hh = float(team_rate_map.loc[ht].get('hdcf60_5v5', np.nan))
                                ah = float(team_rate_map.loc[at].get('hdcf60_5v5', np.nan))
                                ppx = float(team_rate_map.loc[ht].get('pp_xgf60', np.nan)) + float(team_rate_map.loc[at].get('pp_xgf60', np.nan))
                                pkx = float(team_rate_map.loc[ht].get('pk_xga60', np.nan)) + float(team_rate_map.loc[at].get('pk_xga60', np.nan))
                                comp = [v for v in [hx,ax,hh,ah,ppx,pkx] if np.isfinite(v)]
                                if comp:
                                    blend = 0.02 * (np.nanmean(comp) - 2.0)
                                    total_adj += float(max(-0.6, min(0.6, blend)))
                        except Exception:
                            pass
                    # Penalty/referee impact: estimate special-teams minutes inflation
                    if penalty_rates is not None and isinstance(penalty_rates, pd.DataFrame) and 'team' in penalty_rates.columns:
                        try:
                            ht = str(pred.home_team).upper(); at = str(pred.away_team).upper()
                            pr_map = penalty_rates.set_index('team')
                            draw = pr_map.loc[ht]['penalties_drawn60'] + pr_map.loc[at]['penalties_drawn60'] if ht in pr_map.index and at in pr_map.index else np.nan
                            take = pr_map.loc[ht]['penalties_taken60'] + pr_map.loc[at]['penalties_taken60'] if ht in pr_map.index and at in pr_map.index else np.nan
                            pens = np.nanmean([draw, take]) if np.isfinite(draw) or np.isfinite(take) else np.nan
                            if np.isfinite(pens):
                                total_adj += float(max(-0.4, min(0.4, 0.015 * (pens - 7.0))))
                        except Exception:
                            pass
                    # Referee Goals/Gm: small global nudge based on crew scoring profile (if available)
                    try:
                        baseline_g = 6.2
                        if isinstance(referee_default_goal, (int, float)) and np.isfinite(referee_default_goal):
                            baseline_g = float(referee_default_goal)
                        elif referee_rates is not None and isinstance(referee_rates, pd.DataFrame) and 'goals_gm' in referee_rates.columns:
                            ref_mean = float(pd.to_numeric(referee_rates['goals_gm'], errors='coerce').dropna().mean()) if len(referee_rates) else np.nan
                            if np.isfinite(ref_mean):
                                baseline_g = float(ref_mean)

                        ref_goal_feature: Optional[float] = None
                        ref_bias_feature: Optional[float] = None

                        if isinstance(pred.ref_goals_gm, (int, float)) and np.isfinite(pred.ref_goals_gm):
                            ref_goal_feature = float(pred.ref_goals_gm)
                        elif isinstance(pred.referee_avg_goals, (int, float)) and np.isfinite(pred.referee_avg_goals):
                            ref_goal_feature = float(pred.referee_avg_goals)

                        if isinstance(pred.referee_home_bias, (int, float)) and np.isfinite(pred.referee_home_bias):
                            ref_bias_feature = float(pred.referee_home_bias)

                        if (ref_goal_feature is None or ref_bias_feature is None) and isinstance(referee_map, dict) and referee_map:
                            mk = f"{pred.away_team}@{pred.home_team}"
                            crew = referee_map.get(mk)
                            avg_goals, avg_b = model.crew_features(crew, referee_rates)
                            if ref_goal_feature is None and isinstance(avg_goals, (int, float)) and np.isfinite(avg_goals):
                                ref_goal_feature = float(avg_goals)
                                if pred.referee_avg_goals is None:
                                    pred.referee_avg_goals = ref_goal_feature
                                if pred.ref_goals_gm is None:
                                    pred.ref_goals_gm = ref_goal_feature
                            if ref_bias_feature is None and isinstance(avg_b, (int, float)) and np.isfinite(avg_b):
                                ref_bias_feature = float(avg_b)
                                if pred.referee_home_bias is None:
                                    pred.referee_home_bias = ref_bias_feature

                        if ref_goal_feature is not None:
                            try:
                                residual_coeff = float(max(0.0, 0.05 - float(getattr(model, 'ref_goal_weight', 0.0))))
                            except Exception:
                                residual_coeff = 0.0
                            if residual_coeff > 0:
                                adj_val = residual_coeff * (ref_goal_feature - baseline_g)
                                adj_val = float(max(-0.25, min(0.25, adj_val)))
                                total_adj += adj_val
                                pred.ref_goal_adjustment = float((pred.ref_goal_adjustment or 0.0) + adj_val)

                        if ref_bias_feature is not None:
                            total_adj += float(max(-0.05, min(0.05, 0.005 * ref_bias_feature)))
                    except Exception:
                        pass

                    # Empty-net expectation proxy: uplift totals when projected close game and late ENG likelihood high
                    try:
                        closeness = float(max(0.0, 1.0 - abs(pred.predicted_total - betting_line) / 2.5))
                        total_adj += 0.05 * closeness
                    except Exception:
                        pass

                    if abs(total_adj) > 1e-6:
                        pred.predicted_total = float(pred.predicted_total + total_adj)
                        pred.edge = float(pred.predicted_total - pred.betting_line)
                    # Populate env/lineup info strings for dashboard
                    try:
                        e = env_data.get(game_id, {}) if isinstance(env_data, dict) else {}
                        ht = str(pred.home_team).upper(); at = str(pred.away_team).upper()
                        hs = float(lineup_strength.get(ht, 0.0)); as_ = float(lineup_strength.get(at, 0.0))
                        # Format start time as 12-hour with AM/PM and temperature with °F
                        try:
                            hr = int(e.get('start_hour_local', 0))
                        except Exception:
                            hr = 0
                        try:
                            mn = int(e.get('start_minute_local', 0))
                        except Exception:
                            mn = 0
                        suffix = 'AM' if hr < 12 else 'PM'
                        hr12 = hr % 12
                        if hr12 == 0:
                            hr12 = 12
                        tempf = int(e.get('temp_f', 0))
                        env_bits = []
                        if e.get('outdoor'):
                            env_bits.append('OD')
                        env_bits.append(f"{hr12}:{mn:02d}{suffix}")
                        env_bits.append(f"{tempf}°F")
                        pred.env_info = ' '.join(env_bits)
                        pred.lineup_info = f"H:{hs:.2f}/A:{as_:.2f}"
                    except Exception:
                        pass
                    
                    # Open→close calibration: nudge confidence toward markets that moved in same direction
                    try:
                        if isinstance(closing_data, dict):
                            matchup_key = f"{pred.away_team}@{pred.home_team}"
                            c = closing_data.get(game_id, closing_data.get(matchup_key))
                            if isinstance(c, dict) and 'closing_total' in c and isinstance(c['closing_total'], (int, float)):
                                closing_total = float(c['closing_total'])
                                moved_up = closing_total > betting_line
                                aligned = (pred.edge > 0 and moved_up) or (pred.edge < 0 and not moved_up)
                                conf_delta = 0.03 if aligned else -0.03
                                pred.confidence = float(min(0.95, max(0.05, pred.confidence + conf_delta)))
                        # Velocity-based nudge
                        vel = velocity_map.get(game_id)
                        if isinstance(vel, (int, float)) and abs(vel) > 0:
                            vel_aligned = (vel > 0 and pred.edge > 0) or (vel < 0 and pred.edge < 0)
                            pred.confidence = float(min(0.95, max(0.05, pred.confidence + (0.02 if vel_aligned else -0.02))))
                            pred.market_velocity = float(vel)
                    except Exception:
                        pass

                    predictions.append(pred)
                    
                except Exception as e:
                    print(f"⚠️  Error predicting game {idx}: {e}")
                    continue
            
            print(f"✅ Generated {len(predictions)} predictions")
            
            betting_preds = [p for p in predictions if p.recommendation != 'No Bet']
            if betting_preds:
                print(f"\n💰 Recommended bets ({len(betting_preds)}):")
                for pred in betting_preds:
                    price = pred.over_american_odds if pred.recommendation == 'OVER' else pred.under_american_odds
                    print(
                        f"   {pred.away_team} @ {pred.home_team}: {pred.recommendation} {pred.betting_line} @ {price} "
                        f"(Pred: {pred.predicted_total:.1f}, Edge: {pred.edge:+.2f}, Conf: {pred.confidence:.0%}, Kelly: {pred.kelly_bet_size:.1f}%)"
                    )
            else:
                print("\n💡 No betting opportunities identified today")
        
        print("\n📱 Step 6: Creating dashboard...")
        dashboard_html = create_dashboard_html(predictions, training_results, betting_odds=betting_odds)
        
        dashboard_file = "nhl_real_data_dashboard.html"
        tmp_path = dashboard_file + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        try:
            os.replace(tmp_path, dashboard_file)
        except Exception:
            try:
                import shutil
                shutil.move(tmp_path, dashboard_file)
            except Exception:
                pass
        
        print(f"✅ Dashboard saved: {dashboard_file}")

        # Optional: deploy to www.thepointou.com (HTTP/S3/SFTP)
        try:
            if not cli_args or getattr(cli_args, 'deploy', False):
                smp = SocialMediaPoster()
                smp.deploy_dashboard_html(dashboard_file, cli_args=cli_args)
        except Exception as e:
            print(f"⚠️  Deployment step skipped/failed: {e}")

        # Optional: export predictions to Excel in repo root and post to Discord
        try:
            if not cli_args or getattr(cli_args, 'export_excel', False):
                excel_path = getattr(cli_args, 'excel_path', 'predictions.xls') if cli_args else 'predictions.xls'
                saved = save_predictions_excel(predictions, out_path=excel_path)
                if saved and (not cli_args or getattr(cli_args, 'post_excel', False)):
                    smp = SocialMediaPoster()
                    # Also render a clean image and attach alongside Excel
                    img = save_predictions_image(
                        predictions,
                        training_results=training_results,
                        html_path='predictions_table.html',
                        image_path='predictions.png'
                    )
                    if img and os.path.exists(img):
                        smp.post_file_to_discord(img, message='🧾 NHL Predictions')
                    # Skipping Excel upload per request
        except Exception as e:
            print(f"⚠️  Excel export/upload skipped/failed: {e}")
        
        if not cli_args or cli_args.log_bets:
            print("\n🧾 Step 7: Logging recommended bets...")
            try:
                closing_odds_path = getattr(cli_args, 'closing_odds_path', None) if cli_args else None
                log_bets(predictions, logfile=getattr(cli_args, 'log_path', 'bets_log.csv'), closing_odds_path=closing_odds_path)
            except Exception as e:
                print(f"⚠️  Could not log bets: {e}")

        if not cli_args or cli_args.post_social:
            print("\n📲 Step 8: Posting to social media...")
            social_poster = SocialMediaPoster()
            social_results = {'twitter': False, 'discord': False}
            try:
                # Only post predictions image to Twitter (no text summary)
                try:
                    img_path = save_predictions_image(
                        predictions,
                        training_results=training_results,
                        html_path='predictions_table.html',
                        image_path='predictions.png'
                    )
                    if img_path and os.path.exists(img_path):
                        social_poster.post_image_to_twitter(img_path, caption='🏒 NHL Predictions')
                except Exception:
                    pass
                # Optional inline predictions to Discord
                topn = 10
                try:
                    if cli_args is not None:
                        topn = int(getattr(cli_args, 'post_inline_top', 10))
                except Exception:
                    topn = 10
                if not cli_args or getattr(cli_args, 'post_inline', False):
                    social_poster.post_inline_predictions(predictions, top_n=topn, title='NHL Predictions (Top)')
            except Exception as e:
                print(f"⚠️  Social posting skipped/failed: {e}")
        else:
            social_results = {'twitter': False, 'discord': False}
        
        try:
            full_path = os.path.abspath(dashboard_file)
            webbrowser.open(f'file://{full_path}')
            print(f"🌐 Dashboard opened: {full_path}")
        except Exception as e:
            print(f"⚠️  Could not auto-open browser: {e}")
            print(f"📂 Please manually open: {dashboard_file}")
        
        print(f"\n🎉 SUCCESS!")
        print(f"📊 Model Performance: RMSE {training_results['rmse']:.3f}, Accuracy {training_results['over_under_accuracy']:.1%}")
        
        betting_preds = [p for p in predictions if p.recommendation != 'No Bet']
        print(f"🎯 Generated {len(predictions)} predictions with {len(betting_preds)} recommended bets")
        print(f"💻 Dashboard: {dashboard_file}")
        
        successful_posts = sum(social_results.values())
        if successful_posts > 0:
            print(f"📱 Posted to {successful_posts}/{len(social_results)} social platforms")
        else:
            print(f"📱 Social media posting disabled - configure social_config.json to enable")
        
        return model, predictions, dashboard_file
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        
        return model, predictions, dashboard_file

if __name__ == "__main__":
    print("🚀 Starting NHL Over/Under System with Real Data...")
    parser = argparse.ArgumentParser(description="NHL Over/Under model")
    parser.add_argument('--odds-path', type=str, default=os.getenv('ODDS_JSON_PATH', 'odds.json'), help='Path to odds JSON file')
    parser.add_argument('--closing-odds-path', type=str, default=os.getenv('CLOSING_ODDS_JSON', 'closing_odds.json'), help='Path to closing odds JSON for CLV logging')
    parser.add_argument('--log-bets', action='store_true', help='Enable logging bets to CSV')
    parser.add_argument('--log-path', type=str, default='bets_log.csv', help='Path to bets log CSV')
    parser.add_argument('--post-social', action='store_true', help='Enable social media posting')
    parser.add_argument('--date', type=str, default=None, help='ISO date YYYY-MM-DD for which to fetch/predict games (default: today)')
    parser.add_argument('--today-games-path', type=str, default=None, help='Path to offline today games JSON to bypass API')
    parser.add_argument('--offline', action='store_true', help='Use offline today games if provided and skip API calls')
    parser.add_argument('--realtime-odds', action='store_true', help='Fetch realtime totals odds from an external API (requires ODDS_API_KEY)')
    parser.add_argument('--odds-regions', type=str, default='us,us2,eu,uk,au', help='Comma-separated odds regions (use "all" for every Odds API region)')
    parser.add_argument('--odds-timeout', type=int, default=25, help='Realtime odds request timeout in seconds')
    parser.add_argument('--odds-retries', type=int, default=3, help='Realtime odds fetch retries on failure')
    parser.add_argument('--odds-dispersion-all', action='store_true', help='Collect totals from all books for dispersion metrics (still pick prices from FD)')
    parser.add_argument('--log-odds-history', action='store_true', help='Append odds snapshots to odds_history.csv')
    parser.add_argument('--odds-history-path', type=str, default='odds_history.csv', help='Path to odds history CSV')
    parser.add_argument('--xg-path', type=str, default=None, help='Path to expected goals JSON for today\'s games')
    parser.add_argument('--xg-baseline-total', type=float, default=float(os.getenv('XG_BASELINE_TOTAL', 6.2)), help='xG baseline total used to compute adjustments')
    parser.add_argument('--xg-clamp-abs', type=float, default=float(os.getenv('XG_CLAMP_ABS', 2.0)), help='Absolute clamp for xG total adjustment')
    parser.add_argument('--ci-quantile', type=float, default=float(os.getenv('CI_QUANTILE', 0.90)), help='Conformal CI quantile (e.g., 0.90 for 90% radius)')
    parser.add_argument('--kelly-mult', type=float, default=float(os.getenv('KELLY_MULT', 0.5)), help='Kelly multiplier (e.g., 0.5 for half Kelly)')
    parser.add_argument('--kelly-cap', type=float, default=float(os.getenv('KELLY_CAP_PCT', 2.0)), help='Max Kelly stake per bet in percent')
    parser.add_argument('--daily-exposure-cap', type=float, default=float(os.getenv('DAILY_EXPOSURE_CAP_PCT', 6.0)), help='Max total daily exposure percent')
    parser.add_argument('--kelly-use-fair', action='store_true', help='Use no-vig fair probabilities for Kelly sizing')
    parser.add_argument('--team-rates-path', type=str, default=os.getenv('TEAM_RATES_PATH', None), help='CSV or URL with team xG/CF/HDCF/PP/PK rates')
    parser.add_argument('--goalie-gsax-path', type=str, default=os.getenv('GOALIE_GSAX_PATH', None), help='CSV or URL with goalie rolling GSAx and prob_start')
    parser.add_argument('--penalty-rates-path', type=str, default=os.getenv('PENALTY_RATES_PATH', None), help='CSV or URL with team penalties drawn/taken per 60')
    parser.add_argument('--referee-rates-path', type=str, default=os.getenv('REFEREE_RATES_PATH', 'referees.csv'), help='CSV or URL with referee penalties per 60 (optional)')
    parser.add_argument('--environment-path', type=str, default=os.getenv('ENVIRONMENT_JSON', None), help='Path to environment JSON (outdoor/start time/weather)')
    parser.add_argument('--env-refresh', action='store_true', help='Refresh/overwrite today entries in environment.json')
    parser.add_argument('--lineup-path', type=str, default=os.getenv('LINEUP_STRENGTH_CSV', None), help='Path to lineup strength CSV (team,lineup_strength)')
    parser.add_argument('--auto-populate', action='store_true', help='Auto-fetch team rates/goalie GSAx/penalties/referees from URLs and write CSVs')
    parser.add_argument('--team-rates-url', type=str, default=os.getenv('TEAM_RATES_URL', None), help='URL to fetch team rates CSV')
    parser.add_argument('--goalie-gsax-url', type=str, default=os.getenv('GOALIE_GSAX_URL', None), help='URL to fetch goalie GSAx CSV')
    parser.add_argument('--penalties-url', type=str, default=os.getenv('PENALTIES_URL', None), help='URL to fetch team penalties CSV')
    parser.add_argument('--referees-url', type=str, default=os.getenv('REFEREES_URL', None), help='URL to fetch referee rates CSV')
    # Deployment options
    parser.add_argument('--deploy', action='store_true', help='Deploy dashboard HTML to www.thepointou.com')
    parser.add_argument('--deploy-method', type=str, default=os.getenv('DEPLOY_METHOD', None), help='Deploy method: http|s3|sftp')
    parser.add_argument('--deploy-target-url', type=str, default=os.getenv('DEPLOY_TARGET_URL', None), help='HTTP target URL for PUT/POST (e.g., https://www.thepointou.com/nhl_real_data_dashboard.html)')
    parser.add_argument('--deploy-http-method', type=str, default=os.getenv('DEPLOY_HTTP_METHOD', 'PUT'), help='HTTP method to use for --deploy-method=http (PUT or POST)')
    parser.add_argument('--deploy-token', type=str, default=os.getenv('DEPLOY_BEARER_TOKEN', None), help='Bearer token for HTTP deploy (if required)')
    parser.add_argument('--deploy-basic-user', type=str, default=os.getenv('DEPLOY_BASIC_USER', None), help='Basic auth username for HTTP deploy (optional)')
    parser.add_argument('--deploy-basic-pass', type=str, default=os.getenv('DEPLOY_BASIC_PASS', None), help='Basic auth password for HTTP deploy (optional)')
    parser.add_argument('--deploy-s3-bucket', type=str, default=os.getenv('DEPLOY_S3_BUCKET', None), help='S3 bucket for --deploy-method=s3')
    parser.add_argument('--deploy-s3-key', type=str, default=os.getenv('DEPLOY_S3_KEY', None), help='S3 key for --deploy-method=s3')
    parser.add_argument('--deploy-s3-region', type=str, default=os.getenv('DEPLOY_S3_REGION', None), help='S3 region (optional)')
    parser.add_argument('--deploy-s3-acl', type=str, default=os.getenv('DEPLOY_S3_ACL', 'public-read'), help='S3 ACL (default public-read)')
    parser.add_argument('--deploy-sftp-host', type=str, default=os.getenv('DEPLOY_SFTP_HOST', None), help='SFTP host for --deploy-method=sftp')
    parser.add_argument('--deploy-sftp-port', type=int, default=int(os.getenv('DEPLOY_SFTP_PORT', '22')), help='SFTP port for --deploy-method=sftp')
    parser.add_argument('--deploy-sftp-user', type=str, default=os.getenv('DEPLOY_SFTP_USER', None), help='SFTP username')
    parser.add_argument('--deploy-sftp-pass', type=str, default=os.getenv('DEPLOY_SFTP_PASS', None), help='SFTP password')
    parser.add_argument('--deploy-sftp-path', type=str, default=os.getenv('DEPLOY_SFTP_PATH', None), help='SFTP remote path to write HTML')
    # Excel export options
    parser.add_argument('--export-excel', action='store_true', help='Export predictions to an Excel file')
    parser.add_argument('--excel-path', type=str, default=os.getenv('EXCEL_PATH', 'predictions.xlsx'), help='Path to save Excel predictions')
    parser.add_argument('--post-excel', action='store_true', help='Upload the Excel file to Discord via webhook')
    # Inline Discord options
    parser.add_argument('--post-inline', action='store_true', help='Post a compact inline summary of predictions to Discord')
    parser.add_argument('--post-inline-top', type=int, default=int(os.getenv('POST_INLINE_TOP', '10')), help='How many rows to include in inline Discord posts')
    parser.add_argument('--use-compoisson', action='store_true', help='Enable Generalized Poisson (COM-Poisson proxy) calibration fallback')
    args = parser.parse_args()

    try:
        model, predictions, dashboard_file = main(args)
        
        if model is not None:
            print(f"\n🎉 System initialized successfully!")
            if predictions:
                betting_opportunities = len([p for p in predictions if p.recommendation != 'No Bet'])
                print(f"💰 Found {betting_opportunities} betting opportunities")
            if dashboard_file:
                print(f"📱 Dashboard created: {dashboard_file}")
        else:
            print(f"\n⚠️  System encountered issues but continued with available data")
            
    except Exception as e:
        print(f"\n❌ Critical error: {e}")

        print("Please check your internet connection and try again.")

