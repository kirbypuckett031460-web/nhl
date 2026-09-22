import hashlib
import hmac
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse

import streamlit as st
import streamlit.components.v1 as components

from streamlit_public import _latest_record


APP_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = APP_ROOT / "data" / "runtime"


def _read_streamlit_secret(name: str) -> str:
    """Safely read a Streamlit secret value without raising on missing keys."""
    try:
        val = st.secrets.get(name, "")
        return str(val).strip() if val is not None else ""
    except Exception:
        try:
            return str(st.secrets[name]).strip()
        except Exception:
            return ""


def _read_first_secret_or_env(keys: List[str]) -> str:
    for key in keys:
        val = _read_streamlit_secret(key)
        if val:
            return val
        env_val = str(os.getenv(key, "")).strip()
        if env_val:
            return env_val
    return ""


def _secret_plain_passphrase() -> str:
    return _read_streamlit_secret("ADMIN_PASSPHRASE") or str(os.getenv("ADMIN_PASSPHRASE", "")).strip()


def _secret_hashed_passphrase() -> str:
    return _read_streamlit_secret("ADMIN_PASSPHRASE_SHA256") or str(os.getenv("ADMIN_PASSPHRASE_SHA256", "")).strip().lower()


def _verify_passphrase(user_input: str) -> bool:
    entered = str(user_input or "")
    plain_secret = _secret_plain_passphrase()
    if plain_secret:
        return bool(hmac.compare_digest(entered, plain_secret))
    hash_secret = _secret_hashed_passphrase()
    if hash_secret:
        digest = hashlib.sha256(entered.encode("utf-8")).hexdigest().lower()
        return bool(hmac.compare_digest(digest, hash_secret))
    return False


def _require_admin_login() -> bool:
    if st.session_state.get("admin_authenticated"):
        return True
    with st.form("admin_login_form", clear_on_submit=False):
        st.subheader("Admin Login")
        passphrase = st.text_input("Passphrase", type="password")
        submitted = st.form_submit_button("Unlock Admin")
    if submitted:
        if _verify_passphrase(passphrase):
            st.session_state["admin_authenticated"] = True
            st.rerun()
        st.error("Invalid passphrase.")
    return False


def _save_uploaded_file(uploaded_file, target_name: str) -> Optional[Path]:
    if uploaded_file is None:
        return None
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNTIME_DIR / target_name
    out_path.write_bytes(uploaded_file.getvalue())
    return out_path


def _run_command(command: List[str], env_overrides: Dict[str, str]) -> Tuple[int, str]:
    env = os.environ.copy()
    env.update(env_overrides)
    proc = subprocess.Popen(
        command,
        cwd=str(APP_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    placeholder = st.empty()
    output_lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        output_lines.append(line.rstrip())
        if len(output_lines) > 800:
            output_lines = output_lines[-800:]
        placeholder.code("\n".join(output_lines), language="bash")
    exit_code = proc.wait()
    return exit_code, "\n".join(output_lines)


def _infer_github_repo_from_git_remote() -> str:
    try:
        proc = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=str(APP_ROOT),
            text=True,
            capture_output=True,
        )
        raw = (proc.stdout or "").strip()
        if not raw:
            return ""

        # git@github.com:owner/repo.git
        if raw.startswith("git@") and ":" in raw:
            rhs = raw.split(":", 1)[1].strip()
            rhs = rhs[:-4] if rhs.endswith(".git") else rhs
            return rhs.strip("/")

        # https://github.com/owner/repo.git (or with credentials)
        parsed = urlparse(raw)
        host = (parsed.netloc or "").lower()
        if "github.com" in host:
            path = (parsed.path or "").strip("/")
            path = path[:-4] if path.endswith(".git") else path
            return path
    except Exception:
        return ""
    return ""


def _publish_outputs_to_github(
    repo: str,
    branch: str,
    token: str,
    files_to_publish: List[str],
    commit_message: str,
) -> Tuple[bool, str, Dict[str, object]]:
    details: Dict[str, object] = {
        "repo": "",
        "branch": "",
        "published_files": [],
        "commit_sha": "",
        "commit_url": "",
        "committed_new_changes": False,
        "pushed": False,
    }
    repo = str(repo or "").strip().strip("/")
    branch = str(branch or "").strip() or "main"
    token = str(token or "").strip()
    details["repo"] = repo
    details["branch"] = branch
    if not repo:
        return False, "GitHub repo is required (example: owner/name).", details
    if not token:
        return False, "GitHub push token is required to publish outputs.", details

    rel_files: List[str] = []
    for item in files_to_publish:
        rel = str(item or "").strip().lstrip("/")
        if not rel:
            continue
        abs_path = APP_ROOT / rel
        if abs_path.exists():
            rel_files.append(rel)
    details["published_files"] = rel_files
    if not rel_files:
        return False, "No output files found to publish.", details

    def _run_git(args: List[str]) -> Tuple[int, str]:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(APP_ROOT),
            text=True,
            capture_output=True,
        )
        text = "\n".join(part for part in [proc.stdout, proc.stderr] if part).strip()
        return proc.returncode, text

    rc, out = _run_git(["add", "--", *rel_files])
    if rc != 0:
        return False, f"git add failed:\n{out or '(no output)'}", details

    rc, _ = _run_git(["diff", "--cached", "--quiet"])
    head_rc, head_out = _run_git(["rev-parse", "HEAD"])
    if head_rc == 0:
        head_sha = str(head_out).splitlines()[-1].strip()
        details["commit_sha"] = head_sha
        if repo and head_sha:
            details["commit_url"] = f"https://github.com/{repo}/commit/{head_sha}"
    if rc == 0:
        return True, "No new output changes to publish.", details

    commit_msg = str(commit_message or "").strip()
    if not commit_msg:
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        commit_msg = f"chore(admin): refresh public outputs ({now_utc})"

    rc, out = _run_git(["commit", "-m", commit_msg])
    if rc != 0:
        return False, f"git commit failed:\n{out or '(no output)'}", details
    details["committed_new_changes"] = True

    head_rc, head_out = _run_git(["rev-parse", "HEAD"])
    if head_rc == 0:
        head_sha = str(head_out).splitlines()[-1].strip()
        details["commit_sha"] = head_sha
        if repo and head_sha:
            details["commit_url"] = f"https://github.com/{repo}/commit/{head_sha}"

    safe_token = quote(token, safe="")
    remote_url = f"https://x-access-token:{safe_token}@github.com/{repo}.git"
    push = subprocess.run(
        ["git", "push", remote_url, f"HEAD:{branch}"],
        cwd=str(APP_ROOT),
        text=True,
        capture_output=True,
    )
    push_out = "\n".join(part for part in [push.stdout, push.stderr] if part).strip()
    if token:
        push_out = push_out.replace(token, "[REDACTED]")
    if safe_token:
        push_out = push_out.replace(safe_token, "[REDACTED]")
    if push.returncode != 0:
        return False, f"git push failed:\n{push_out or '(no output)'}", details
    details["pushed"] = True
    return True, f"Published {len(rel_files)} file(s) to {repo}@{branch}.", details


def render_admin_app() -> None:
    st.set_page_config(page_title="NHL O/U Admin Runner", layout="wide")
    st.title("NHL Over/Under Admin")
    st.caption("Protected admin interface for training and generating predictions.")

    if not (_secret_plain_passphrase() or _secret_hashed_passphrase()):
        st.error(
            "Admin passphrase is not configured. "
            "Set `ADMIN_PASSPHRASE` (or `ADMIN_PASSPHRASE_SHA256`) in Streamlit secrets."
        )
        st.code('ADMIN_PASSPHRASE = "your_passphrase_here"', language="toml")
        st.stop()

    if not _require_admin_login():
        st.stop()

    with st.sidebar:
        st.success("Admin access enabled")
        if st.button("Log out"):
            st.session_state["admin_authenticated"] = False
            st.rerun()

        st.header("Run Settings")
        run_date = st.date_input("Prediction date", value=date.today())
        train_speed = st.selectbox("Training speed", options=["fast", "balanced", "full"], index=1)
        train_target = st.selectbox("Train target", options=["auto", "edge", "total"], index=0)
        historical_days = st.number_input("Historical days", min_value=30, max_value=2000, value=365, step=30)
        use_saved_model = st.checkbox("Use saved model artifact", value=True)
        save_trained_model = st.checkbox("Save trained model artifact", value=True)
        model_path = st.text_input("Model path", value="data/cache/trained_model.joblib")
        max_train_samples = st.number_input("Max train samples (0 = no cap)", min_value=0, max_value=20000, value=0, step=100)

        st.subheader("Odds + I/O")
        realtime_odds = st.checkbox("Use realtime odds API", value=False)
        odds_path_input = st.text_input("Odds JSON path", value="odds.json")
        odds_regions = st.text_input("Odds regions", value="us")

        secret_odds_api_key = _read_streamlit_secret("ODDS_API_KEY")
        env_odds_api_key = str(os.getenv("ODDS_API_KEY", "")).strip()
        default_odds_api_key = secret_odds_api_key or env_odds_api_key
        odds_api_key_override = st.text_input("ODDS_API_KEY override (optional)", value="", type="password")
        if default_odds_api_key:
            st.caption("Default ODDS_API_KEY loaded from Streamlit secrets/env. Leave override blank to use it.")
        else:
            st.caption("No default ODDS_API_KEY found in secrets/env. Provide override to use realtime odds.")
        odds_upload = st.file_uploader("Upload odds JSON (optional)", type=["json"])

        today_games_upload = st.file_uploader("Upload today_games JSON (optional)", type=["json"])
        environment_upload = st.file_uploader("Upload environment JSON (optional)", type=["json"])

        log_bets = st.checkbox("Log bets", value=True)
        log_path = st.text_input("Bets log path", value="bets_log.csv")

        st.subheader("Publish")
        publish_to_github = st.checkbox("Publish outputs to GitHub after successful run", value=True)
        inferred_repo = _infer_github_repo_from_git_remote()
        default_repo = _read_streamlit_secret("GITHUB_REPO") or str(os.getenv("GITHUB_REPOSITORY", "")).strip() or inferred_repo
        default_branch = _read_streamlit_secret("GITHUB_BRANCH") or str(os.getenv("GITHUB_BRANCH", "main")).strip()
        token_keys = ["GITHUB_PUSH_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"]
        default_push_token = _read_first_secret_or_env(token_keys)
        github_repo = st.text_input("GitHub repo (owner/name)", value=default_repo, help="Example: kirbypuckett031460-web/nhl")
        github_branch = st.text_input("GitHub branch", value=default_branch or "main")
        push_token_override = st.text_input("GitHub push token override (optional)", value="", type="password")
        publish_commit_message = st.text_input(
            "Publish commit message",
            value="chore(admin): refresh public app outputs [skip ci]",
        )
        if default_push_token:
            st.caption("Default GitHub push token detected from secrets/env.")
        else:
            st.caption("Set `GITHUB_PUSH_TOKEN` (or `GITHUB_TOKEN` / `GH_TOKEN`) in secrets/env, or use override, to enable publishing.")

    run_clicked = st.button("Run Model", type="primary")

    if not run_clicked:
        return

    uploaded_odds_path = _save_uploaded_file(odds_upload, "odds_uploaded.json")
    uploaded_today_games_path = _save_uploaded_file(today_games_upload, "today_games_uploaded.json")
    uploaded_environment_path = _save_uploaded_file(environment_upload, "environment_uploaded.json")

    command = [
        sys.executable,
        "-u",
        "nhl_model3.py",
        "--no-open-browser",
        "--date",
        run_date.isoformat(),
        "--train-speed",
        train_speed,
        "--train-target",
        train_target,
        "--historical-days",
        str(int(historical_days)),
        "--model-path",
        model_path.strip() or "data/cache/trained_model.joblib",
        "--odds-regions",
        odds_regions.strip() or "us",
    ]
    if max_train_samples > 0:
        command.extend(["--max-train-samples", str(int(max_train_samples))])
    if use_saved_model:
        command.append("--use-saved-model")
    if save_trained_model:
        command.append("--save-trained-model")
    if log_bets:
        command.extend(["--log-bets", "--log-path", log_path.strip() or "bets_log.csv"])
    else:
        command.extend(["--log-path", log_path.strip() or "bets_log.csv"])
    if realtime_odds:
        command.append("--realtime-odds")
    else:
        odds_path = str(uploaded_odds_path) if uploaded_odds_path else (odds_path_input.strip() or "odds.json")
        command.extend(["--odds-path", odds_path])
    if uploaded_today_games_path:
        command.extend(["--today-games-path", str(uploaded_today_games_path), "--offline"])
    if uploaded_environment_path:
        command.extend(["--environment-path", str(uploaded_environment_path)])

    env_overrides: Dict[str, str] = {}
    effective_odds_api_key = odds_api_key_override.strip() or default_odds_api_key
    if effective_odds_api_key:
        env_overrides["ODDS_API_KEY"] = effective_odds_api_key
    if realtime_odds and not effective_odds_api_key:
        st.warning("Realtime odds enabled but no ODDS_API_KEY is configured (secrets/env/override).")

    st.subheader("Live Run Output")
    with st.spinner("Running model... this may take a while depending on training mode."):
        rc, output = _run_command(command, env_overrides)

    if rc == 0:
        st.success("Model run completed successfully.")
    else:
        st.error(f"Model run failed with exit code {rc}.")

    if rc == 0 and publish_to_github:
        effective_push_token = push_token_override.strip() or default_push_token
        effective_repo = (github_repo or "").strip()
        effective_branch = (github_branch or "").strip() or "main"
        publish_files = [
            "public_predictions.json",
            log_path.strip() or "bets_log.csv",
            "predictions.png",
            "nhl_real_data_dashboard.html",
        ]
        publish_details: Dict[str, object] = {
            "repo": effective_repo,
            "branch": effective_branch,
            "published_files": publish_files,
            "commit_sha": "",
            "commit_url": "",
            "committed_new_changes": False,
            "pushed": False,
        }
        if not effective_repo:
            ok = False
            msg = "Publish skipped: GitHub repo is not configured. Set GITHUB_REPO or fill the field."
            st.warning(msg)
        elif not effective_push_token:
            ok = False
            msg = "Publish skipped: GitHub push token is not configured. Set GITHUB_PUSH_TOKEN or provide override."
            st.warning(msg)
        else:
            with st.spinner("Publishing outputs to GitHub..."):
                ok, msg, publish_details = _publish_outputs_to_github(
                    repo=effective_repo,
                    branch=effective_branch,
                    token=effective_push_token,
                    files_to_publish=publish_files,
                    commit_message=publish_commit_message,
                )
            if ok:
                st.success(msg)
            else:
                st.error(msg)
        repo_used = str(publish_details.get("repo") or github_repo or "").strip()
        branch_used = str(publish_details.get("branch") or github_branch or "main").strip()
        commit_sha = str(publish_details.get("commit_sha") or "").strip()
        commit_url = str(publish_details.get("commit_url") or "").strip()
        published_files = publish_details.get("published_files") or []
        if not isinstance(published_files, list):
            published_files = []
        st.markdown("#### Publish status")
        st.caption(f"Target: `{repo_used}@{branch_used}`")
        if commit_sha:
            st.code(commit_sha, language="text")
            if commit_url:
                st.markdown(f"[View commit on GitHub]({commit_url})")
        if published_files:
            st.caption("Files considered for publish:")
            st.code("\n".join(str(v) for v in published_files), language="text")

    st.subheader("Artifacts")
    predictions_image = APP_ROOT / "predictions.png"
    dashboard_html = APP_ROOT / "nhl_real_data_dashboard.html"
    effective_log_path = APP_ROOT / (log_path.strip() or "bets_log.csv")

    if predictions_image.exists():
        st.image(str(predictions_image), caption="predictions.png", use_container_width=True)
        st.download_button(
            "Download predictions.png",
            data=predictions_image.read_bytes(),
            file_name="predictions.png",
            mime="image/png",
        )
    else:
        st.info("predictions.png not found for this run.")

    if dashboard_html.exists():
        dashboard_text = dashboard_html.read_text(encoding="utf-8", errors="ignore")
        with st.expander("Preview dashboard HTML", expanded=False):
            components.html(dashboard_text, height=900, scrolling=True)
        st.download_button(
            "Download dashboard HTML",
            data=dashboard_text,
            file_name="nhl_real_data_dashboard.html",
            mime="text/html",
        )
    else:
        st.info("nhl_real_data_dashboard.html not found for this run.")

    if effective_log_path.exists():
        st.download_button(
            "Download bets log CSV",
            data=effective_log_path.read_bytes(),
            file_name=effective_log_path.name,
            mime="text/csv",
        )
        record = _latest_record(effective_log_path)
        if record:
            st.metric(
                "Latest per-game O/U record",
                f"{int(record['wins'])}-{int(record['losses'])}",
                f"{record['win_rate'] * 100:.1f}% win rate",
            )
    else:
        st.info("No bets log found yet.")

    with st.expander("Final command used", expanded=False):
        st.code(" ".join(command), language="bash")
    with st.expander("Run output (last 800 lines)", expanded=False):
        st.code(output or "(no output)", language="bash")


if __name__ == "__main__":
    render_admin_app()

