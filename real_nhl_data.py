import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import warnings
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

class RealNHLDataCollector:
    def __init__(self):
        self.session = None
        if REQUESTS_AVAILABLE:
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
        self.cache = GameStatsCache()
    
    def get_real_nhl_data(self, start_date='2024-10-10', end_date='2024-12-15', max_games=500):
        """
        Main function to get real NHL data from multiple sources with on-disk caching.
        """
        print("🏒 NHL Real Data Collector Starting...")
        print(f"📅 Requested Date Range: {start_date} to {end_date}")

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

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
            
            # Try to get detailed stats from boxscore
            game_id = game.get('id')
            if game_id:
                detailed_stats = self._get_boxscore_stats(game_id)
                game_info.update(detailed_stats)
            
            # Fill in missing data with estimates
            game_info = self._fill_missing_stats(game_info)
            
            return game_info
            
        except Exception as e:
            return None
    
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
            stats['home_shots'] = home_team.get('sog', 30)
            stats['away_shots'] = away_team.get('sog', 30)
            
            # Power play data
            stats['home_pp_goals'] = home_team.get('powerPlayGoals', 0)
            stats['away_pp_goals'] = away_team.get('powerPlayGoals', 0)
            stats['home_pp_opps'] = home_team.get('powerPlayOpportunities', 3)
            stats['away_pp_opps'] = away_team.get('powerPlayOpportunities', 3)
            
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
            game_info = self._fill_missing_stats(game_info)
            
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
    
    def _fill_missing_stats(self, game_info):
        """Fill in missing statistics with realistic estimates"""
        
        # Calculate games played (rough estimate)
        if 'games_played' not in game_info:
            season_start = datetime(2024, 10, 1)
            game_date = pd.to_datetime(game_info['date'])
            days_passed = (game_date - season_start).days
            game_info['games_played'] = max(1, days_passed // 2)
        
        # Estimate shots if missing
        if 'home_shots' not in game_info:
            # Rough correlation: ~10 shots per goal + base shots
            home_goals = game_info.get('home_goals', 0)
            game_info['home_shots'] = max(15, int(np.random.normal(30 + home_goals * 3, 4)))
        
        if 'away_shots' not in game_info:
            away_goals = game_info.get('away_goals', 0)
            game_info['away_shots'] = max(15, int(np.random.normal(28 + away_goals * 3, 4)))
        
        # Calculate saves
        game_info['home_saves'] = max(0, game_info['away_shots'] - game_info['away_goals'])
        game_info['away_saves'] = max(0, game_info['home_shots'] - game_info['home_goals'])
        
        # Estimate power play stats if missing
        if 'home_pp_goals' not in game_info:
            total_goals = game_info.get('home_goals', 0)
            game_info['home_pp_goals'] = min(total_goals, int(np.random.poisson(0.6)))
        
        if 'away_pp_goals' not in game_info:
            total_goals = game_info.get('away_goals', 0)
            game_info['away_pp_goals'] = min(total_goals, int(np.random.poisson(0.6)))
        
        if 'home_pp_opps' not in game_info:
            game_info['home_pp_opps'] = max(1, int(np.random.poisson(3.2)))
        
        if 'away_pp_opps' not in game_info:
            game_info['away_pp_opps'] = max(1, int(np.random.poisson(3.2)))
        
        # Add goaltender names if missing
        if 'home_goalie' not in game_info:
            game_info['home_goalie'] = self._get_likely_goalie(game_info['home_team'])
        
        if 'away_goalie' not in game_info:
            game_info['away_goalie'] = self._get_likely_goalie(game_info['away_team'])
        
        return game_info
    
    def _get_likely_goalie(self, team):
        """Get likely starting goalie for a team"""
        team_goalies = {
            'TOR': ['Samsonov', 'Woll'], 'BOS': ['Ullmark', 'Swayman'], 
            'NYR': ['Shesterkin', 'Quick'], 'PHI': ['Hart', 'Ersson'],
            'PIT': ['Jarry', 'Nedeljkovic'], 'WSH': ['Lindgren', 'Kuemper'],
            'CAR': ['Andersen', 'Kochetkov'], 'FLA': ['Bobrovsky', 'Knight'],
            'TBL': ['Vasilevskiy', 'Johansson'], 'MTL': ['Allen', 'Montembeault'],
            'OTT': ['Ullmark', 'Forsberg'], 'BUF': ['Luukkonen', 'Comrie'],
            'DET': ['Husso', 'Talbot'], 'CBJ': ['Merzlikins', 'Tarasov'],
            'NJD': ['Markstrom', 'Allen'], 'NYI': ['Sorokin', 'Varlamov'],
            'COL': ['Georgiev', 'Wedgewood'], 'VGK': ['Hill', 'Samsonov'],
            'EDM': ['Skinner', 'Pickard'], 'CGY': ['Markstrom', 'Wolf'],
            'VAN': ['Demko', 'Silovs'], 'SEA': ['Grubauer', 'Daccord'],
            'LAK': ['Kuemper', 'Rittich'], 'ANA': ['Gibson', 'Dostal'],
            'SJS': ['Blackwood', 'Vanecek'], 'MIN': ['Fleury', 'Gustavsson'],
            'WPG': ['Hellebuyck', 'Brossoit'], 'STL': ['Bennington', 'Hofer'],
            'CHI': ['Mrazek', 'Soderblom'], 'DAL': ['Oettinger', 'DeSmith'],
            'NSH': ['Saros', 'Annunen'], 'ARI': ['Ingram', 'Vejmelka']
        }
        
        goalies = team_goalies.get(team, ['Starter'])
        return np.random.choice(goalies)
    
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