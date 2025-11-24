"""Social media publishing utilities."""

import argparse
import json
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import requests

from nhl_model.common import OverUnderPrediction, format_matchup_display

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


class SocialMediaPoster:
    """Handles posting predictions to X (Twitter) and Discord."""

    def __init__(self, image_renderer: Optional[Callable[..., Optional[str]]] = None) -> None:
        self.image_renderer = image_renderer
        self.twitter_api = None
        self.discord_webhook_url = None
        self.discord_verify = True
        self.setup_credentials()
        if TWITTER_AVAILABLE:
            self.setup_twitter()

    def setup_credentials(self) -> None:
        """Setup API credentials from environment variables."""
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.twitter_consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
        self.twitter_consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
        self.twitter_access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.twitter_access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        api_key = os.getenv('TWITTER_API_KEY')
        api_secret = os.getenv('TWITTER_API_SECRET')
        if not self.twitter_consumer_key and api_key:
            self.twitter_consumer_key = api_key
        if not self.twitter_consumer_secret and api_secret:
            self.twitter_consumer_secret = api_secret
        self.discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL')

        try:
            with open('social_config.json', 'r') as file:
                config = json.load(file)

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

            insecure_env = os.getenv('DISCORD_INSECURE')
            if isinstance(insecure_env, str) and insecure_env.strip().lower() in ('1', 'true', 'yes', 'y'):
                self.discord_verify = False
            deploy_cfg = config.get('deploy') or {}
            insecure_cfg = str((deploy_cfg.get('discord_insecure') or '')).strip().lower()
            if insecure_cfg in ('1', 'true', 'yes', 'y'):
                self.discord_verify = False
        except FileNotFoundError:
            print("⚠️  No social_config.json found. Creating template...")
            self.create_config_template()
        try:
            dbg = {
                'discord': 'yes' if bool(self.discord_webhook_url) else 'no',
                'twitter': 'yes' if bool(self.twitter_bearer_token) else 'no'
            }
            print(f"🔧 Social config -> Discord? {dbg['discord']} | Twitter? {dbg['twitter']}")
        except Exception:
            pass

    def create_config_template(self) -> None:
        """Create a template configuration file."""
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
            "deploy": {
                "method": "http",
                "http": {
                    "url": "https://www.thepointou.com/nhl_real_data_dashboard.html",
                    "http_method": "PUT",
                    "auth": {
                        "type": "bearer",
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

        with open('social_config.json', 'w', encoding='utf-8') as file:
            json.dump(template, file, indent=4)

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
            ctype = 'application/octet-stream'
            fn_l = filename.lower()
            if fn_l.endswith('.xlsx'):
                ctype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            elif fn_l.endswith('.xls'):
                ctype = 'application/vnd.ms-excel'
            elif fn_l.endswith('.csv'):
                ctype = 'text/csv'
            elif fn_l.endswith(('.html', '.htm')):
                ctype = 'text/html'
            elif fn_l.endswith('.png'):
                ctype = 'image/png'
            with open(file_path, 'rb') as file:
                files = {'file': (filename, file, ctype)}
                data = {'content': message or '📎 Predictions export'}
                resp = requests.post(self.discord_webhook_url, data=data, files=files, timeout=30, verify=self.discord_verify)
                resp.raise_for_status()
                print(f"✅ Uploaded file to Discord: {filename}")
                return True
        except Exception as exc:
            print(f"❌ Discord file upload failed: {exc}")
            return False

    def post_inline_predictions(self, predictions: List[OverUnderPrediction], top_n: int = 10, title: str = "NHL Predictions (Top)") -> bool:
        """Post a compact inline summary of top predictions to Discord as embeds."""
        if not self.discord_webhook_url:
            print("⚠️  Discord webhook URL not available")
            return False
        try:
            preds = list(predictions or [])
            preds.sort(key=lambda pred: (pred.recommendation == 'No Bet', -abs(float(getattr(pred, 'edge', 0.0) or 0.0))), reverse=False)
            preds = preds[:max(1, int(top_n))]
            chunk_size = 8
            posted_any = False
            for i in range(0, len(preds), chunk_size):
                chunk = preds[i:i + chunk_size]
                embed = {
                    "title": f"{title} {i + 1}-{i + len(chunk)}",
                    "color": 0x2ecc71,
                    "fields": []
                }
                for pred in chunk:
                    name = format_matchup_display(pred.away_team, pred.home_team)
                    val_bits = []
                    try:
                        val_bits.append(f"Line {pred.betting_line:.1f} | Pred {pred.predicted_total:.2f}")
                    except Exception:
                        val_bits.append(f"Line {pred.betting_line} | Pred {pred.predicted_total}")
                    val_bits.append(f"Rec {pred.recommendation} {pred.edge:+.2f}")
                    try:
                        val_bits.append(f"Kelly {pred.kelly_bet_size:.1f}%")
                    except Exception:
                        pass
                    try:
                        val_bits.append(f"O {pred.over_probability:.0%} / U {pred.under_probability:.0%}")
                    except Exception:
                        pass
                    ref_bits = getattr(pred, 'referee_info', None)
                    if not ref_bits:
                        crew_list = getattr(pred, 'referee_crew', []) or []
                        ref_bits = ", ".join([str(name_part) for name_part in crew_list if str(name_part).strip()])
                    ref_metrics: List[str] = []
                    if isinstance(pred.referee_avg_goals, (int, float)) and np.isfinite(pred.referee_avg_goals):
                        ref_metrics.append(f"{pred.referee_avg_goals:.2f} G/G")
                    if isinstance(pred.referee_home_bias, (int, float)) and np.isfinite(pred.referee_home_bias):
                        ref_metrics.append(f"HB {pred.referee_home_bias:+.2f}")
                    if isinstance(pred.ref_goals_gm, (int, float)) and np.isfinite(pred.ref_goals_gm) and not ref_metrics:
                        ref_metrics.append(f"Feature {pred.ref_goals_gm:.2f} G/G")
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
        except Exception as exc:
            print(f"❌ Inline Discord post failed: {exc}")
            return False

    def deploy_dashboard_html(self, html_path: str, cli_args: Optional[argparse.Namespace] = None) -> bool:
        """Deploy the dashboard HTML to www.thepointou.com using configured method."""
        try:
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
                with open(html_path, 'rb') as html_file:
                    data = html_file.read()
                try:
                    if http_method == 'PUT':
                        resp = requests.put(url, data=data, headers=headers, auth=auth, timeout=30)
                    else:
                        files = {'file': ('nhl_real_data_dashboard.html', data, 'text/html')}
                        resp = requests.post(url, headers={k: v for k, v in headers.items() if k.lower() != 'content-type'}, files=files, auth=auth, timeout=30)
                    if 200 <= resp.status_code < 300:
                        print("✅ Deployed dashboard via HTTP")
                        return True
                    print(f"⚠️  HTTP deploy failed: {resp.status_code} {resp.text[:200]}")
                except Exception as exc:
                    print(f"⚠️  HTTP deploy error: {exc}")
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
                    with open(html_path, 'rb') as html_file:
                        s3.put_object(Bucket=bucket, Key=key, Body=html_file, ContentType='text/html', ACL=acl)
                    print("✅ Deployed dashboard to S3")
                    return True
                except Exception as exc:
                    print(f"⚠️  S3 deploy error: {exc}")
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
                except Exception as exc:
                    print(f"⚠️  SFTP deploy error: {exc}")
                    return False

            print(f"⚠️  Unknown deploy method: {method}")
            return False
        except Exception as exc:
            print(f"⚠️  Deploy failed: {exc}")
            return False

    def setup_twitter(self) -> None:
        """Setup Twitter API client."""
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

        except Exception as exc:
            print(f"❌ Twitter API setup failed: {exc}")
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
            pick_txt = pick_raw if pick_raw != 'NO BET' else 'NO BET'
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

        separator = ['-' * width for width in widths]
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
        """Post predictions to Twitter."""
        if not self.twitter_api or not TWITTER_AVAILABLE:
            print("⚠️  Twitter API not available")
            return False

        try:
            img_path = None
            if self.image_renderer:
                try:
                    img_path = self.image_renderer(
                        predictions,
                        training_results=training_results,
                        html_path='predictions_table.html',
                        image_path='predictions.png'
                    )
                except Exception as exc:
                    print(f"⚠️  Rendering predictions image failed: {exc}")

            tweet_caption = "Predictor picks for tonight's NHL games."

            if img_path and os.path.exists(img_path):
                return self.post_image_to_twitter(img_path, caption=tweet_caption)

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
                    matchup_display = format_matchup_display(top.away_team, top.home_team)
                    tweet_text = (
                        f"{tweet_caption} Top: {matchup_display} — "
                        f"{top.recommendation} {top.betting_line} (edge {float(getattr(top, 'edge', 0.0)):+.1f})."
                    )

            try:
                response = self.twitter_api.create_tweet(text=tweet_text)
            except tweepy.TooManyRequests:
                print("⏳ Twitter rate limit hit while posting text tweet. Skipping for now.")
                return False
            print(f"✅ Posted to Twitter: {response.data['id']}")
            return True

        except Exception as exc:
            print(f"❌ Twitter posting failed: {exc}")
            return False

    def post_to_discord_webhook(self, predictions: List[OverUnderPrediction], training_results: Dict) -> bool:
        """Post predictions to Discord via webhook."""
        if not self.discord_webhook_url:
            print("⚠️  Discord webhook URL not available")
            return False

        try:
            betting_preds = [p for p in predictions if p.recommendation != 'No Bet']

            embed = {
                "title": "🏒 NHL Over/Under Predictions",
                "description": "Daily predictions powered by machine learning",
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
                matchup_display = format_matchup_display(pred.away_team, pred.home_team)
                field_name = f"🏒 {matchup_display}"
                field_value = f"**Line:** {pred.betting_line} | **Pred:** {pred.predicted_total:.1f}\n"
                field_value += f"**Rec:** {pred.recommendation} ({pred.edge:+.2f})\n"
                field_value += f"**Conf:** {pred.confidence:.0%}"
                ref_bits = getattr(pred, 'referee_info', None)
                if not ref_bits:
                    crew_list = getattr(pred, 'referee_crew', []) or []
                    ref_bits = ", ".join([str(name_part) for name_part in crew_list if str(name_part).strip()])
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

        except Exception as exc:
            print(f"❌ Discord webhook posting failed: {exc}")
            return False

    def post_predictions(self, predictions: List[OverUnderPrediction], training_results: Dict) -> Dict[str, bool]:
        """Post predictions to all configured social media platforms."""
        results = {}

        print("\n📱 Posting predictions to social media...")

        if TWITTER_AVAILABLE:
            print("🐦 Posting to X (Twitter)...")
            results['twitter'] = self.post_to_twitter(predictions, training_results)
        else:
            results['twitter'] = False

        print("💬 Skipping Discord summary (minimal mode)")
        results['discord'] = False

        return results

    def render_dashboard_image(self, html_path: str, image_path: str = 'dashboard.png') -> Optional[str]:
        """Render the local HTML dashboard to an image suitable for posting."""
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception:
            pass

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
                with open(image_path, 'rb') as file:
                    files = {'file': (os.path.basename(image_path), file, 'image/png')}
                    data = {'content': '🏒 NHL Over/Under Dashboard'}
                    resp = requests.post(self.discord_webhook_url, data=data, files=files, verify=self.discord_verify)
                    resp.raise_for_status()
                    print("✅ Dashboard image posted to Discord")
                    return True
            print("ℹ️ Image render unavailable; posting local file path to Discord…")
            payload = {'content': f"Dashboard saved locally: {os.path.abspath(html_path)}"}
            resp = requests.post(self.discord_webhook_url, json=payload, verify=self.discord_verify)
            resp.raise_for_status()
            print("✅ Dashboard link posted to Discord")
            return True
        except Exception as exc:
            print(f"❌ Discord dashboard posting failed: {exc}")
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

            with open(image_path, 'rb') as file:
                media_bytes = file.read()
                _ = media_bytes  # placeholder to keep lint quiet if unused elsewhere

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
            except Exception as exc:
                print(f"❌ Twitter media upload failed: {exc}")
                return False
        except Exception as exc:
            print(f"❌ Twitter dashboard posting failed: {exc}")
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
            except Exception as exc:
                print(f"❌ Twitter media upload failed: {exc}")
                return False
        except Exception as exc:
            print(f"❌ Twitter image posting failed: {exc}")
            return False
