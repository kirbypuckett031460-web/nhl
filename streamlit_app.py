import pathlib, streamlit as st
import html
import pandas as pd
import io, requests, time
from nhl_model3 import main, argparse

st.set_page_config(page_title="NHL O/U Dashboard", layout="wide")
root = pathlib.Path(__file__).parent
html_path = root / "nhl_real_data_dashboard.html"
lineup_csv = root / "lineup_strength.csv"
team_rates_csv = root / "team_rates.csv"

@st.cache_data(ttl=21600)
def _fetch_url_text(url: str, timeout: int = 30) -> str:
    resp = requests.get(url, timeout=timeout, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    })
    resp.raise_for_status()
    return resp.text

def ensure_team_rates_csv(out_path: pathlib.Path) -> None:
    # Always refresh to keep repo copy up-to-date
    try:
        url = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2024/regular/teams.csv"
        text = _fetch_url_text(url, timeout=30)
        pd.read_csv(io.StringIO(text)).to_csv(out_path, index=False)
    except Exception:
        # If remote fails but a local copy exists, keep it
        if not out_path.exists():
            # create a minimal placeholder so downstream code doesn't break
            pd.DataFrame({"team": []}).to_csv(out_path, index=False)

def ensure_lineup_strength_csv(out_path: pathlib.Path) -> None:
    # Build from repo team_rates.csv (kept updated by ensure_team_rates_csv)
    team_rates = team_rates_csv
    df = None
    if team_rates.exists():
        try:
            df = pd.read_csv(team_rates)
        except Exception:
            df = None
    if df is None:
        return
    if df is None or 'team' not in df.columns:
        return
    # Compute a simple lineup strength index from available columns
    def col(name_opts):
        for c in df.columns:
            cu = str(c).lower()
            for n in name_opts:
                if n in cu:
                    return c
        return None
    xgf_col = col(["scorevenueadjustedxgoalsfor","xgoalsfor","xgf"]) or 'scoreVenueAdjustedxGoalsFor'
    xga_col = col(["scorevenueadjustedxgoalsagainst","xgoalsagainst","xga"]) or 'scoreVenueAdjustedxGoalsAgainst'
    sogf_col = col(["shotsongoalfor","sog for"]) or 'shotsOnGoalFor'
    soga_col = col(["shotsongoalagainst","sog against"]) or 'shotsOnGoalAgainst'
    toi_col = col(["icetime"]) or 'iceTime'
    try:
        xgf = pd.to_numeric(df.get(xgf_col), errors='coerce').fillna(0.0)
        xga = pd.to_numeric(df.get(xga_col), errors='coerce').fillna(0.0)
        sogf = pd.to_numeric(df.get(sogf_col), errors='coerce').fillna(0.0)
        soga = pd.to_numeric(df.get(soga_col), errors='coerce').fillna(0.0)
        toi = pd.to_numeric(df.get(toi_col), errors='coerce').fillna(1.0)
        # if seconds, convert to minutes heuristic
        toi = (toi/60.0) if toi.median() > 5000 else toi
        denom = (toi/60.0).replace(0, 1.0)
        xgf60 = xgf / denom
        xga60 = xga / denom
        sogf60 = sogf / denom
        soga60 = soga / denom
        raw = 0.6*(xgf60 - xga60) + 0.2*(sogf60 - soga60)
        std = raw.std(ddof=0)
        z = (raw - raw.mean()) / (std if std else 1.0)
        strength = (z.clip(-2, 2) + 2.0).round(2)
        # Map to NHL abbreviations
        name_col = col(["name"]) or 'name'
        def to_abbr(s: str) -> str:
            m = {
                'ANAHEIM DUCKS':'ANA','ARIZONA COYOTES':'ARI','BOSTON BRUINS':'BOS','BUFFALO SABRES':'BUF',
                'CALGARY FLAMES':'CGY','CAROLINA HURRICANES':'CAR','CHICAGO BLACKHAWKS':'CHI','COLORADO AVALANCHE':'COL',
                'COLUMBUS BLUE JACKETS':'CBJ','DALLAS STARS':'DAL','DETROIT RED WINGS':'DET','EDMONTON OILERS':'EDM',
                'FLORIDA PANTHERS':'FLA','LOS ANGELES KINGS':'LAK','MINNESOTA WILD':'MIN','MONTREAL CANADIENS':'MTL',
                'NASHVILLE PREDATORS':'NSH','NEW JERSEY DEVILS':'NJD','NEW YORK ISLANDERS':'NYI','NEW YORK RANGERS':'NYR',
                'OTTAWA SENATORS':'OTT','PHILADELPHIA FLYERS':'PHI','PITTSBURGH PENGUINS':'PIT','SAN JOSE SHARKS':'SJS',
                'SEATTLE KRAKEN':'SEA','ST. LOUIS BLUES':'STL','TAMPA BAY LIGHTNING':'TBL','TORONTO MAPLE LEAFS':'TOR',
                'UTAH MAMMOTH':'UTA','VANCOUVER CANUCKS':'VAN','VEGAS GOLDEN KNIGHTS':'VGK','WASHINGTON CAPITALS':'WSH',
                'WINNIPEG JETS':'WPG'
            }
            u = str(s or '').upper().strip()
            if u in m:
                return m[u]
            # If already looks like abbr, keep it
            if len(u) <= 4 and u.isalpha():
                return u
            return u[:3]
        if name_col in df.columns:
            teams_series = df[name_col].astype(str).map(to_abbr)
        else:
            teams_series = df['team'].astype(str).map(to_abbr)
        pd.DataFrame({'team': teams_series, 'lineup_strength': strength}).to_csv(out_path, index=False)
    except Exception:
        return

def run_model():
    # Always refresh repo CSVs so Streamlit and local use the same inputs
    ensure_team_rates_csv(team_rates_csv)
    ensure_lineup_strength_csv(lineup_csv)
    args = argparse.Namespace(
        odds_path=str(root / "odds.json"),
        closing_odds_path=None,
        log_bets=False,
        post_social=False,
        date=None,
        today_games_path=None,
        offline=False,
        realtime_odds=True,
        odds_regions="us",
        odds_timeout=25,
        odds_retries=3,
        log_odds_history=False,
        odds_history_path=str(root / "odds_history.csv"),
        xg_path=None, xg_baseline_total=6.2, xg_clamp_abs=2.0,
        kelly_mult=0.5, kelly_cap=2.0, daily_exposure_cap=6.0,
        team_rates_path=str(team_rates_csv), goalie_gsax_path=None,
        penalty_rates_path=None, referee_rates_path=str(root / "referees.csv"),
        environment_path=str(root / "environment.json"), env_refresh=True,
        lineup_path=str(lineup_csv),
        auto_populate=False,
        team_rates_url=None, goalie_gsax_url=None,
        penalties_url=None, referees_url=None,
    )
    main(args)

left, right = st.columns([3, 1])
with right:
    regen = st.button("Regenerate dashboard")

if regen or not html_path.exists():
    # Cooldown to avoid fair-use triggers on Streamlit Cloud
    last_ts = st.session_state.get('last_run_ts')
    if last_ts and (time.time() - last_ts) < 600 and not (not html_path.exists()):
        st.info("Using cached dashboard (cooldown 10 min to respect fair use limits).")
    else:
        with st.spinner("Running model..."):
            run_model()
            st.session_state['last_run_ts'] = time.time()

filter_text = st.text_input("Filter by team (e.g., TOR, BOS)", value="")
filter_rec = st.selectbox("Recommendation filter", ["All", "OVER", "UNDER", "No Bet"], index=0)

def filter_html(html: str, ftxt: str, frec: str) -> str:
    try:
        # very lightweight filtering by data attributes; falls back to cell text
        import re
        # Extract tbody
        m = re.search(r"(<tbody>)([\s\S]*?)(</tbody>)", html, flags=re.I)
        if not m:
            return html
        head, body, tail = m.group(1), m.group(2), m.group(3)
        rows = re.findall(r"(<tr[\s\S]*?</tr>)", body, flags=re.I)
        ftxt_u = (ftxt or "").strip().upper()
        frec_u = (frec or "").strip().upper()
        out_rows = []
        for r in rows:
            # data-matchup
            dm = re.search(r"data-matchup=\"([^\"]*)\"", r)
            matchup = (dm.group(1) if dm else "").upper()
            if not matchup:
                # fallback to first cell text
                tdm = re.search(r"<td[^>]*>([\s\S]*?)</td>", r)
                matchup = (re.sub(r"<[^>]+>", " ", tdm.group(1)) if tdm else "").upper()
            dr = re.search(r"data-rec=\"([^\"]*)\"", r)
            rec = (dr.group(1) if dr else "").upper()
            ok_txt = (not ftxt_u) or (matchup and ftxt_u in matchup)
            ok_rec = (not frec_u) or (frec_u == "ALL") or (rec == frec_u)
            if ok_txt and ok_rec:
                out_rows.append(r)
        new_body = "\n".join(out_rows)
        return html.replace(body, new_body)
    except Exception:
        return html

def parse_rows_for_display(html_text: str):
    import re
    rows_out = []
    m = re.search(r"(<tbody>)([\s\S]*?)(</tbody>)", html_text, flags=re.I)
    if not m:
        return rows_out
    body = m.group(2)
    rows = re.findall(r"<tr[^>]*>([\s\S]*?)</tr>", body, flags=re.I)
    for r in rows:
        tds = re.findall(r"<td[^>]*>([\s\S]*?)</td>", r, flags=re.I)
        # Expect columns: Matchup, Line, Pred, Edge, Over%, Under%, Confidence, Env, Lineup, Rec, Kelly
        def clean(x: str) -> str:
            x = re.sub(r"<[^>]+>", " ", x)
            return " ".join(x.replace("\u00a0"," ").split())
        vals = [clean(x) for x in tds]
        # Extract data-rec if present
        rec_m = re.search(r"data-rec=\"([^\"]*)\"", r)
        rec = rec_m.group(1) if rec_m else (vals[10] if len(vals) > 10 else '')
        ref_crew_m = re.search(r"data-ref-crew=\"([^\"]*)\"", r)
        ref_avg_m = re.search(r"data-ref-avg=\"([^\"]*)\"", r)
        ref_bias_m = re.search(r"data-ref-bias=\"([^\"]*)\"", r)
        ref_source_m = re.search(r"data-ref-source=\"([^\"]*)\"", r)
        ref_adjust_m = re.search(r"data-ref-adjust=\"([^\"]*)\"", r)
        ref_crew_raw = html.unescape(ref_crew_m.group(1)) if ref_crew_m else ''
        ref_avg_goals = html.unescape(ref_avg_m.group(1)) if ref_avg_m else ''
        ref_home_bias = html.unescape(ref_bias_m.group(1)) if ref_bias_m else ''
        ref_source = html.unescape(ref_source_m.group(1)) if ref_source_m else ''
        ref_goal_adjustment = html.unescape(ref_adjust_m.group(1)) if ref_adjust_m else ''
        rows_out.append({
            'matchup': vals[0] if len(vals) > 0 else '',
            'line': vals[1] if len(vals) > 1 else '',
            'pred': vals[2] if len(vals) > 2 else '',
            'edge': vals[3] if len(vals) > 3 else '',
            'over': vals[4] if len(vals) > 4 else '',
            'under': vals[5] if len(vals) > 5 else '',
            'conf': vals[6] if len(vals) > 6 else '',
            'refs': vals[7] if len(vals) > 7 else '',
            'env': vals[8] if len(vals) > 8 else '',
            'lineup': vals[9] if len(vals) > 9 else '',
            'rec': rec,
            'kelly': vals[11] if len(vals) > 11 else '',
            'ref_crew_raw': ref_crew_raw,
            'ref_avg_goals': ref_avg_goals,
            'ref_home_bias': ref_home_bias,
            'ref_source': ref_source,
            'ref_goal_adjustment': ref_goal_adjustment
        })
    return rows_out

if html_path.exists():
    raw_html = html_path.read_text(encoding="utf-8")
    filtered_html = filter_html(raw_html, filter_text, filter_rec)

    # Top controls (server-side working buttons)
    rows = parse_rows_for_display(filtered_html)
    if rows:
        # CSV export
        import csv
        from io import StringIO
        csv_buf = StringIO()
        w = csv.writer(csv_buf)
        w.writerow([
            "Matchup","Line","Predicted","Edge","Over%","Under%","Confidence","Refs",
            "Referee Crew","Ref Avg Goals","Ref Home Bias","Ref Source","Ref Goal Adjustment",
            "Env","Lineup","Recommendation","Kelly%"
        ])
        for r in rows:
            w.writerow([
                r.get('matchup',''), r.get('line',''), r.get('pred',''), r.get('edge',''),
                r.get('over',''), r.get('under',''), r.get('conf',''), r.get('refs',''),
                r.get('ref_crew_raw',''), r.get('ref_avg_goals',''), r.get('ref_home_bias',''),
                r.get('ref_source',''), r.get('ref_goal_adjustment',''), r.get('env',''),
                r.get('lineup',''), r.get('rec',''), r.get('kelly','')
            ])
        st.download_button("Export CSV", data=csv_buf.getvalue().encode("utf-8"), file_name="predictions.csv", mime="text/csv")
        # Copy best bets (use a small component to access clipboard)
        best_lines = []
        for r in rows:
            if r['rec'] and r['rec'].upper() != 'NO BET':
                best_lines.append(f"{r['matchup']}: {r['rec']} {r['line']} (Pred {r['pred']}, {r['edge']})")
        best_text = "\n".join(best_lines)
        if best_text:
            st.components.v1.html(f"""
                <button onclick=\"(function(){{
                  const text = {best_text!r};
                  function fb(){{
                    const ta = document.createElement('textarea'); ta.value=text; ta.style.position='fixed'; ta.style.top='-1000px';
                    document.body.appendChild(ta); ta.focus(); ta.select();
                    try{{document.execCommand('copy'); alert('Best bets copied');}}catch(e){{}}
                    document.body.removeChild(ta);
                  }}
                  if(navigator.clipboard && navigator.clipboard.writeText){{navigator.clipboard.writeText(text).then(()=>alert('Best bets copied')).catch(fb);}} else {{fb();}}
                }})()\">Copy Best Bets</button>
            """, height=50)

    st.components.v1.html(filtered_html, height=1200, scrolling=True)
else:
    st.error("Dashboard not found yet.")


