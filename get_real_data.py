import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Try to import requests, fallback gracefully if not available
try:
    import requests
    REQUESTS_AVAILABLE = True
    print("✅ requests library available - real data enabled")
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  requests library not available - will use sample data")

def get_real_nhl_data_simple(days_back=30, max_games=200):
    """
    Simple function to get real NHL data
    Falls back to sample data if API fails
    """
    if not REQUESTS_AVAILABLE:
        print("📦 Install requests first: python -m pip install requests")
        from nhl_model import create_sample_data
        return create_sample_data()
    
    print(f"🔄 Fetching real NHL data (last {days_back} days)...")
    
    games_data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    current_date = start_date
    while current_date <= end_date and len(games_data) < max_games:
        date_str = current_date.strftime('%Y-%m-%d')
        
        try:
            # NHL API call
            url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                daily_games = parse_nhl_day(data, date_str)
                games_data.extend(daily_games)
                
                if daily_games:
                    print(f"  📅 {date_str}: {len(daily_games)} games")
            
            time.sleep(0.2)  # Be nice to the API
            
        except Exception as e:
            print(f"  ⚠️  {date_str}: {str(e)[:40]}...")
        
        current_date += timedelta(days=1)
    
    if games_data:
        df = pd.DataFrame(games_data)
        print(f"✅ Got {len(df)} real NHL games!")
        return clean_real_data(df)
    else:
        print("❌ No real data available, using sample data...")
        from nhl_model import create_sample_data
        return create_sample_data()

def parse_nhl_day(data, date_str):
    """Parse NHL API response for one day"""
    games = []
    
    try:
        if 'gameWeek' in data:
            for week in data['gameWeek']:
                for day in week.get('games', []):
                    for game in day.get('games', []):
                        if game.get('gameState') in ['OFF', 'FINAL']:
                            game_info = extract_basic_game_info(game, date_str)
                            if game_info:
                                games.append(game_info)
    except Exception as e:
        pass
    
    return games

def extract_basic_game_info(game, date_str):
    """Extract basic game info from NHL API"""
    try:
        home_team = game['homeTeam']['abbrev']
        away_team = game['awayTeam']['abbrev']
        home_score = game['homeTeam'].get('score', 0)
        away_score = game['awayTeam'].get('score', 0)
        
        game_info = {
            'date': pd.to_datetime(date_str),
            'home_team': home_team,
            'away_team': away_team,
            'venue': game.get('venue', {}).get('default', f"{home_team} Arena"),
            'home_goals': home_score,
            'away_goals': away_score,
            'total_goals': home_score + away_score,
        }
        
        # Add realistic estimates for missing data
        game_info = add_estimated_stats(game_info)
        
        return game_info
        
    except Exception as e:
        return None

def add_estimated_stats(game_info):
    """Add estimated stats for missing data"""
    
    # Estimate shots (roughly 30 shots per team + variation based on goals)
    home_goals = game_info['home_goals']
    away_goals = game_info['away_goals']
    
    game_info['home_shots'] = max(15, int(np.random.normal(30 + home_goals * 2, 4)))
    game_info['away_shots'] = max(15, int(np.random.normal(30 + away_goals * 2, 4)))
    
    # Calculate saves
    game_info['home_saves'] = max(0, game_info['away_shots'] - away_goals)
    game_info['away_saves'] = max(0, game_info['home_shots'] - home_goals)
    
    # Estimate power play stats
    game_info['home_pp_goals'] = min(home_goals, int(np.random.poisson(0.6)))
    game_info['away_pp_goals'] = min(away_goals, int(np.random.poisson(0.6)))
    game_info['home_pp_opps'] = max(1, int(np.random.poisson(3)))
    game_info['away_pp_opps'] = max(1, int(np.random.poisson(3)))
    
    # Add goaltender names
    game_info['home_goalie'] = get_team_goalie(game_info['home_team'])
    game_info['away_goalie'] = get_team_goalie(game_info['away_team'])
    
    # Estimate games played
    season_start = datetime(2024, 10, 1)
    game_date = pd.to_datetime(game_info['date'])
    days_passed = (game_date - season_start).days
    game_info['games_played'] = max(1, days_passed // 2)
    
    return game_info

def get_team_goalie(team):
    """Get a realistic goalie name for the team"""
    team_goalies = {
        'TOR': ['Samsonov', 'Woll'], 'BOS': ['Ullmark', 'Swayman'], 
        'NYR': ['Shesterkin', 'Quick'], 'EDM': ['Skinner', 'Pickard'],
        'COL': ['Georgiev', 'Wedgewood'], 'FLA': ['Bobrovsky', 'Knight'],
        'VGK': ['Hill', 'Samsonov'], 'DAL': ['Oettinger', 'DeSmith'],
        'CAR': ['Andersen', 'Kochetkov'], 'NJD': ['Markstrom', 'Allen'],
        'NYI': ['Sorokin', 'Varlamov'], 'WSH': ['Lindgren', 'Kuemper'],
        'TBL': ['Vasilevskiy', 'Johansson'], 'PIT': ['Jarry', 'Nedeljkovic'],
        'PHI': ['Hart', 'Ersson'], 'MTL': ['Allen', 'Montembeault'],
        'OTT': ['Ullmark', 'Forsberg'], 'BUF': ['Luukkonen', 'Comrie'],
        'DET': ['Husso', 'Talbot'], 'CBJ': ['Merzlikins', 'Tarasov'],
        'VAN': ['Demko', 'Silovs'], 'SEA': ['Grubauer', 'Daccord'],
        'LAK': ['Kuemper', 'Rittich'], 'ANA': ['Gibson', 'Dostal'],
        'SJS': ['Blackwood', 'Vanecek'], 'MIN': ['Fleury', 'Gustavsson'],
        'WPG': ['Hellebuyck', 'Brossoit'], 'STL': ['Bennington', 'Hofer'],
        'CHI': ['Mrazek', 'Soderblom'], 'NSH': ['Saros', 'Annunen'],
        'ARI': ['Ingram', 'Vejmelka']
    }
    
    goalies = team_goalies.get(team, ['Starter'])
    return np.random.choice(goalies)

def clean_real_data(df):
    """Clean and validate the real data"""
    print("🧹 Cleaning real data...")
    
    # Remove invalid games
    initial_count = len(df)
    df = df[df['home_team'] != df['away_team']]
    df = df[df['total_goals'] >= 0]
    df = df.sort_values('date')
    df = df.reset_index(drop=True)
    
    final_count = len(df)
    if final_count < initial_count:
        print(f"  🗑️  Removed {initial_count - final_count} invalid games")
    
    return df