#!/usr/bin/env python3
"""
Automated MoneyPuck ETL, anomaly detection, and dataset versioning.

This module downloads the latest team and goalie metrics directly from
MoneyPuck, normalizes them into the columns expected by the NHL model,
detects data drift or stale snapshots, and versions every refresh so
regressions can be investigated later.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np
import pandas as pd
import requests

MONEYPUCK_BASE_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/{stage}/{dataset}.csv"
DEFAULT_HISTORY_DIR = Path("data/history")
DEFAULT_ANOMALY_DIRNAME = "anomalies"
WINDOWS_DRIVE_PATTERN = re.compile(r"^[a-zA-Z]:[\\/]")

TEAM_NAME_MAP: Dict[str, str] = {
    "ANAHEIM DUCKS": "ANA",
    "ARIZONA COYOTES": "ARI",
    "UTAH HOCKEY CLUB": "UTA",
    "ATLANTA THRASHERS": "WPG",
    "BOSTON BRUINS": "BOS",
    "BUFFALO SABRES": "BUF",
    "CALGARY FLAMES": "CGY",
    "CAROLINA HURRICANES": "CAR",
    "CHICAGO BLACKHAWKS": "CHI",
    "COLORADO AVALANCHE": "COL",
    "COLUMBUS BLUE JACKETS": "CBJ",
    "DALLAS STARS": "DAL",
    "DETROIT RED WINGS": "DET",
    "EDMONTON OILERS": "EDM",
    "FLORIDA PANTHERS": "FLA",
    "LOS ANGELES KINGS": "LAK",
    "MINNESOTA WILD": "MIN",
    "MONTREAL CANADIENS": "MTL",
    "MONTRÉAL CANADIENS": "MTL",
    "NEW JERSEY DEVILS": "NJD",
    "NASHVILLE PREDATORS": "NSH",
    "NEW YORK ISLANDERS": "NYI",
    "NEW YORK RANGERS": "NYR",
    "OTTAWA SENATORS": "OTT",
    "PHILADELPHIA FLYERS": "PHI",
    "PITTSBURGH PENGUINS": "PIT",
    "SAN JOSE SHARKS": "SJS",
    "ST LOUIS BLUES": "STL",
    "ST. LOUIS BLUES": "STL",
    "SEATTLE KRAKEN": "SEA",
    "TAMPA BAY LIGHTNING": "TBL",
    "TORONTO MAPLE LEAFS": "TOR",
    "VANCOUVER CANUCKS": "VAN",
    "VEGAS GOLDEN KNIGHTS": "VGK",
    "WASHINGTON CAPITALS": "WSH",
    "WINNIPEG JETS": "WPG",
}

# Include direct abbreviation lookups
for abbr in [
    "ANA", "ARI", "UTA", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL",
    "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
    "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "VAN", "VGK", "WSH", "WPG"
]:
    TEAM_NAME_MAP.setdefault(abbr, abbr)


def _strip_accents(value: str) -> str:
    """Remove accents so comparisons are case-insensitive ASCII."""

    try:
        return (
            unicodedata.normalize("NFKD", str(value))
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    except Exception:
        return str(value)


def to_team_abbr(value: str) -> str:
    """Map a team name to a tri-code, falling back to 3-character slices."""

    cleaned = _strip_accents(value or "").strip().upper()
    if not cleaned:
        return ""
    if cleaned in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[cleaned]
    for key, abbr in TEAM_NAME_MAP.items():
        if key in cleaned:
            return abbr
    if len(cleaned) <= 4 and cleaned.isalpha():
        return cleaned
    return cleaned[:3]


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame columns are a flat Index of strings."""

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(piece) for piece in col]).strip() for col in df.columns
        ]
    else:
        df.columns = [str(col).strip() for col in df.columns]
    return df


def _maybe_windows_path(value: str) -> Optional[Path]:
    """Return a Path for Windows drive-style inputs."""

    if not value:
        return None
    if WINDOWS_DRIVE_PATTERN.match(value):
        normalized = value.replace("\\", "/")
        return Path(normalized)
    if value.startswith("\\\\"):
        return Path(value)
    return None


def _find_column(df: pd.DataFrame, patterns: Sequence[str]) -> Optional[str]:
    """Return the first column whose lowercase name matches any regex pattern."""

    lower_cols = {col.lower(): col for col in df.columns}
    for pattern in patterns:
        compiled = re.compile(pattern)
        for lower, original in lower_cols.items():
            if compiled.search(lower):
                return original
    return None


def normalize_team_rates(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce MoneyPuck teams CSV to the columns used by the model."""

    df = flatten_columns(raw_df.copy())
    team_col = None
    for alt in [
        "team",
        "team code",
        "teamcode",
        "team_name",
        "name",
        "teamname",
        "team name",
        "abbr",
        "abbreviation",
    ]:
        if alt in df.columns:
            team_col = alt
            break
    if team_col is None and df.index.name and "team" in str(df.index.name).lower():
        df = df.reset_index()
        team_col = df.columns[0]
    if team_col is None:
        raise ValueError("No team column found in MoneyPuck teams dataset")
    df = df.rename(columns={team_col: "team"})
    df["team"] = df["team"].apply(to_team_abbr)
    df = df[df["team"].astype(str).str.len() > 0].copy()

    # Prefer 5v5 situation rows when available
    filtered = df.copy()
    for col in df.columns:
        lc = col.lower()
        if "situation" in lc or "strength" in lc:
            mask = df[col].astype(str).str.contains("5", case=False) & df[col].astype(
                str
            ).str.contains("v", case=False)
            if mask.any():
                filtered = df[mask].copy()
            break
    df = filtered

    lower_cols = {col.lower(): col for col in df.columns}

    def find_col(patterns: Sequence[str]) -> Optional[str]:
        for pattern in patterns:
            compiled = re.compile(pattern)
            for lower, original in lower_cols.items():
                if compiled.search(lower):
                    return original
        return None

    out = pd.DataFrame({"team": df["team"]})

    c_xgf60 = find_col(
        [
            r"xg[f]?[a-z_]*60.*5.*v.*5",
            r"5.*v.*5.*xg[f]?.*60",
            r"even.*xg[f]?.*60",
        ]
    )
    if c_xgf60 is None:
        c_xgf = find_col([r"xg[f]?\b.*5.*v.*5", r"5.*v.*5.*xg[f]?"])
        c_toi = find_col([r"toi.*5.*v.*5", r"minutes.*5.*v.*5", r"time.*5.*v.*5"])
        if c_xgf is not None and c_toi is not None:
            toi = pd.to_numeric(df[c_toi], errors="coerce").replace(0, np.nan)
            out["xgf60_5v5"] = (
                pd.to_numeric(df[c_xgf], errors="coerce") / (toi / 60.0)
            )
    else:
        out["xgf60_5v5"] = pd.to_numeric(df[c_xgf60], errors="coerce")

    c_hdcf60 = find_col(
        [
            r"hdcf.*60.*5.*v.*5",
            r"5.*v.*5.*hdcf.*60",
            r"danger.*chances.*60",
        ]
    )
    if c_hdcf60 is not None:
        out["hdcf60_5v5"] = pd.to_numeric(df[c_hdcf60], errors="coerce")

    c_ppxgf60 = find_col([r"pp.*xg[f]?.*60", r"power.*play.*xg[f]?.*60"])
    if c_ppxgf60 is not None:
        out["pp_xgf60"] = pd.to_numeric(df[c_ppxgf60], errors="coerce")

    c_pkxga60 = find_col(
        [
            r"pk.*xg[a]?.*60",
            r"short.*hand.*xg[a]?.*60",
            r"penalty.*kill.*xg[a]?.*60",
        ]
    )
    if c_pkxga60 is not None:
        out["pk_xga60"] = pd.to_numeric(df[c_pkxga60], errors="coerce")

    grouped = out.groupby("team", as_index=False).first()
    return grouped


def normalize_goalie_rates(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce MoneyPuck goalies CSV to the columns used by the model."""

    df = flatten_columns(raw_df.copy())

    def _norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    goalie_col = None
    for alt in ["goalie", "goaliename", "player", "playername", "name"]:
        if alt in df.columns:
            goalie_col = alt
            break
    if goalie_col is None and df.index.name and "goalie" in str(df.index.name).lower():
        df = df.reset_index()
        goalie_col = df.columns[0]
    if goalie_col is None:
        raise ValueError("No goalie column found in MoneyPuck goalies dataset")

    team_col = None
    for alt in [
        "team",
        "team code",
        "teamcode",
        "team_name",
        "abbr",
        "abbreviation",
        "teamabbr",
    ]:
        if alt in df.columns:
            team_col = alt
            break
    if team_col is None and df.index.name and "team" in str(df.index.name).lower():
        df = df.reset_index()
        team_col = df.columns[0]
    if team_col is None:
        raise ValueError("No team column found in MoneyPuck goalies dataset")

    df = df.rename(columns={goalie_col: "goalie", team_col: "team"})
    df["goalie"] = df["goalie"].astype(str).str.strip()
    df["team"] = df["team"].apply(to_team_abbr)
    df = df[(df["goalie"] != "") & (df["team"] != "")]

    situation_col = None
    for col in df.columns:
        lc = str(col).lower()
        if any(keyword in lc for keyword in ("situation", "strength", "state")):
            situation_col = col
            break
    if situation_col:
        situation = df[situation_col].astype(str).str.strip().str.lower()
        mask = situation == "all"
        if not mask.any():
            mask = situation.str.contains("all", case=False, na=False)
        if not mask.any():
            mask = situation.str.contains("overall", case=False, na=False)
        if mask.any():
            df = df[mask].copy()

    normalized_cols: Dict[str, str] = {}
    for col in df.columns:
        normalized_cols.setdefault(_norm(col), col)

    def find_exact(candidates: Sequence[str]) -> Optional[str]:
        for candidate in candidates:
            key = _norm(candidate)
            if key in normalized_cols:
                return normalized_cols[key]
        return None

    def find_by_terms(*terms: str) -> Optional[str]:
        lowered = [_norm(term) for term in terms if term]
        for norm_name, original in normalized_cols.items():
            if all(term in norm_name for term in lowered):
                return original
        return None

    gsax_series: Optional[pd.Series] = None
    gsax_col = find_exact(
        [
            "gsax_rolling",
            "gsax",
            "goals_saved_above_expected",
            "goals saved above expected",
            "goalssavedaboveexpected",
            "goals_saved_above_expected_all",
        ]
    ) or find_by_terms("saved", "expected")

    if gsax_col is not None:
        gsax_series = pd.to_numeric(df[gsax_col], errors="coerce")
    else:
        xg_col = find_exact(
            [
                "xga",
                "x_goals_against",
                "xgoalsagainst",
                "expected_goals_against",
                "expectedgoalsagainst",
                "xGoals",
                "x_goals",
            ]
        ) or find_by_terms("x", "goal")
        goals_col = find_exact(
            ["ga", "goals_against", "goalsagainst", "goals_allowed", "goals"]
        ) or find_by_terms("goal", "against")

        if xg_col and goals_col:
            expected = pd.to_numeric(df[xg_col], errors="coerce")
            actual = pd.to_numeric(df[goals_col], errors="coerce")
            gsax_series = expected - actual

    if gsax_series is None:
        raise ValueError("Could not locate or derive a GSAx column in MoneyPuck goalies dataset")

    df["gsax_rolling"] = gsax_series.fillna(0.0)

    if "prob_start" not in df.columns:
        starts_col = None
        games_col = None
        for col in df.columns:
            lc = col.lower()
            if starts_col is None and "start" in lc and "last" not in lc:
                starts_col = col
            if (
                games_col is None
                and "game" in lc
                and "last" not in lc
                and "start" not in lc
            ):
                games_col = col
            if starts_col and games_col:
                break
        if starts_col and games_col:
            starts = pd.to_numeric(df[starts_col], errors="coerce").clip(lower=0)
            games = pd.to_numeric(df[games_col], errors="coerce").replace(0, np.nan)
            ratio = (starts / games).clip(0.0, 1.0).fillna(0.5)
            df["prob_start"] = ratio
        else:
            usage_col = find_exact(
                [
                    "icetime",
                    "ice_time",
                    "time_on_ice",
                    "toi",
                    "minutes_played",
                    "minutes",
                    "games_played",
                    "games",
                ]
            )
            if usage_col:
                usage = pd.to_numeric(df[usage_col], errors="coerce").clip(lower=0.0)
                team_totals = usage.groupby(df["team"]).transform("sum").replace(0, np.nan)
                share = (usage / team_totals).clip(0.0, 1.0).fillna(0.0)
                df["prob_start"] = share
            else:
                df["prob_start"] = 0.5

    if "prob_start" in df.columns:
        df["prob_start"] = (
            pd.to_numeric(df["prob_start"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
        )
    else:
        df["prob_start"] = 0.5

    grouped = (
        df[["goalie", "team", "gsax_rolling", "prob_start"]]
        .groupby(["goalie", "team"], as_index=False)
        .mean()
    )
    return grouped


def estimate_current_season(reference: Optional[datetime] = None) -> int:
    """Return the NHL season identifier MoneyPuck uses (year of season end)."""

    ref = reference or datetime.now(timezone.utc)
    if ref.month >= 9:
        return ref.year + 1
    return ref.year


def default_season_candidates(
    explicit: Optional[Sequence[int]] = None, reference: Optional[datetime] = None
) -> List[int]:
    """Determine which seasons to poll, preferring the newest first."""

    if explicit:
        unique = sorted({int(season) for season in explicit}, reverse=True)
        return unique
    current = estimate_current_season(reference)
    return [current, current - 1]


@dataclass
class DatasetTarget:
    name: str
    slug: str
    output_path: Path
    entity_column: str
    numeric_columns: List[str]
    min_rows: int
    expected_unique: Optional[int] = None


@dataclass
class DatasetSummary:
    dataset: str
    rows: int
    output_path: Optional[Path]
    history_path: Optional[Path]
    source_url: Optional[str]
    season: Optional[int]
    stage: Optional[str]
    anomalies: List[str] = field(default_factory=list)


class MoneyPuckDownloader:
    """Download helper that tries multiple season/stage combinations."""

    def __init__(self, timeout: float = 25.0) -> None:
        self.session = requests.Session()
        self.timeout = timeout

    def fetch_first_available(
        self,
        dataset: str,
        slug: str,
        seasons: Sequence[int],
        stages: Sequence[str],
        override_url: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
        candidates: List[Tuple[Optional[int], Optional[str], str]] = []
        if override_url:
            candidates.append((None, None, override_url))
        for season in seasons:
            for stage in stages:
                url = MONEYPUCK_BASE_URL.format(season=season, stage=stage, dataset=slug)
                candidates.append((season, stage, url))

        errors: List[str] = []
        for season, stage, url in candidates:
            try:
                local_path: Optional[Path] = None
                parsed = urlparse(url)

                windows_path = _maybe_windows_path(url)
                if windows_path is not None:
                    local_path = windows_path
                if parsed.scheme == "file":
                    local_path = Path(url2pathname(parsed.path))
                    if parsed.netloc and parsed.netloc not in ("", "localhost"):
                        local_path = Path(f"//{parsed.netloc}") / local_path
                elif (
                    local_path is None
                    and parsed.scheme == ""
                    and Path(url).expanduser().exists()
                ):
                    local_path = Path(url).expanduser()

                if local_path is not None:
                    if local_path.exists():
                        df = pd.read_csv(local_path)
                        if df.empty:
                            continue
                        return df, {
                            "season": season,
                            "stage": stage,
                            "source_url": str(local_path),
                            "dataset": dataset,
                        }
                    errors.append(
                        f"{url}: Local path {local_path} does not exist on this host."
                    )
                    continue

                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                text = response.text.strip()
                if not text or "\n" not in text:
                    continue
                df = pd.read_csv(io.StringIO(response.text))
                if df.empty:
                    continue
                return df, {
                    "season": season,
                    "stage": stage,
                    "source_url": url,
                    "dataset": dataset,
                }
            except Exception as exc:  # pragma: no cover - defensive logging
                errors.append(f"{url}: {exc}")
        raise RuntimeError(
            f"MoneyPuck download failed for {dataset}. Tried {len(candidates)} urls. "
            f"Sample errors: {errors[:3]}"
        )


class AnomalyDetector:
    """Surface data drift and staleness issues."""

    def __init__(self, target_season: int, rel_tolerance: float = 0.35) -> None:
        self.target_season = target_season
        self.rel_tolerance = rel_tolerance

    def detect(
        self,
        target: DatasetTarget,
        new_df: pd.DataFrame,
        previous_df: Optional[pd.DataFrame],
        metadata: Dict[str, Optional[str]],
    ) -> List[str]:
        issues: List[str] = []
        rows = len(new_df)

        if rows < target.min_rows:
            issues.append(
                f"Row count {rows} is below minimum requirement ({target.min_rows})."
            )

        unique_entities = new_df[target.entity_column].nunique(dropna=True)
        if (
            target.expected_unique is not None
            and unique_entities < target.expected_unique
        ):
            issues.append(
                f"{target.entity_column} unique count {unique_entities} is below "
                f"expected {target.expected_unique}."
            )

        season_val = metadata.get("season")
        if season_val is not None:
            try:
                season_int = int(season_val)
                if season_int < self.target_season - 1:
                    issues.append(
                        f"Latest MoneyPuck season {season_int} lags target season "
                        f"{self.target_season}."
                    )
            except Exception:
                issues.append(
                    f"Could not interpret MoneyPuck season value: {season_val}"
                )

        if previous_df is not None and not previous_df.empty:
            prev_rows = len(previous_df)
            if rows < prev_rows * 0.75:
                issues.append(
                    f"Row count dropped from {prev_rows} to {rows} (>25% reduction)."
                )
            prev_unique = previous_df[target.entity_column].nunique(dropna=True)
            if unique_entities < prev_unique * 0.8:
                issues.append(
                    f"{target.entity_column} unique count dropped from "
                    f"{prev_unique} to {unique_entities}."
                )
            for col in target.numeric_columns:
                if col not in new_df.columns or col not in previous_df.columns:
                    continue
                new_series = pd.to_numeric(new_df[col], errors="coerce")
                prev_series = pd.to_numeric(previous_df[col], errors="coerce")
                if new_series.isna().mean() > 0.25:
                    issues.append(f"{col} has >25% missing values after ETL.")
                    continue
                prev_mean = prev_series.mean()
                new_mean = new_series.mean()
                if pd.isna(prev_mean) or pd.isna(new_mean):
                    continue
                baseline = abs(prev_mean) if abs(prev_mean) > 1e-6 else 1.0
                rel_change = abs(new_mean - prev_mean) / baseline
                if rel_change > self.rel_tolerance and abs(new_mean - prev_mean) > 0.05:
                    issues.append(
                        f"{col} mean changed from {prev_mean:.3f} to {new_mean:.3f} "
                        f"({rel_change:.0%} change)."
                    )

        return issues


class VersionManager:
    """Persist history snapshots and metadata."""

    def __init__(self, history_root: Path) -> None:
        self.history_root = history_root
        self.history_root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.history_root / "versions.json"
        self.anomaly_dir = self.history_root / DEFAULT_ANOMALY_DIRNAME
        self.anomaly_dir.mkdir(parents=True, exist_ok=True)

    def _load_meta(self) -> Dict[str, List[Dict[str, object]]]:
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                return {}
        return {}

    def _write_meta(self, payload: Dict[str, List[Dict[str, object]]]) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def write(
        self,
        target: DatasetTarget,
        df: pd.DataFrame,
        metadata: Dict[str, Optional[str]],
        output_path: Path,
        anomalies: List[str],
    ) -> Tuple[Path, Path]:
        timestamp = datetime.now(timezone.utc)
        stamp_tag = timestamp.strftime("%Y%m%dT%H%M%SZ")
        history_dir = self.history_root / target.name
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / f"{stamp_tag}_{target.name}.csv"

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        sha = hashlib.sha256(csv_bytes).hexdigest()

        with open(history_path, "wb") as handle:
            handle.write(csv_bytes)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as handle:
            handle.write(csv_bytes)

        meta = self._load_meta()
        entry = {
            "timestamp": timestamp.isoformat(),
            "history_path": str(history_path),
            "output_path": str(output_path),
            "rows": len(df),
            "hash": sha,
            "source_url": metadata.get("source_url"),
            "season": metadata.get("season"),
            "stage": metadata.get("stage"),
            "anomalies": anomalies,
        }
        meta.setdefault(target.name, []).append(entry)
        self._write_meta(meta)

        anomaly_report = {
            "dataset": target.name,
            "generated_at": timestamp.isoformat(),
            "issues": anomalies,
            "rows": len(df),
        }
        with open(
            self.anomaly_dir / f"{target.name}.json", "w", encoding="utf-8"
        ) as handle:
            json.dump(anomaly_report, handle, indent=2)

        return history_path, output_path


class MoneyPuckETLPipeline:
    """Coordinates download, normalization, anomaly detection, and versioning."""

    def __init__(
        self,
        team_output_path: str = "team_rates.csv",
        goalie_output_path: str = "goalie_gsax.csv",
        history_dir: Path = DEFAULT_HISTORY_DIR,
        stages: Optional[Sequence[str]] = None,
        seasons: Optional[Sequence[int]] = None,
        team_override_url: Optional[str] = None,
        goalie_override_url: Optional[str] = None,
        dry_run: bool = False,
        fail_on_anomaly: bool = False,
        request_timeout: float = 25.0,
    ) -> None:
        self.team_override_url = team_override_url
        self.goalie_override_url = goalie_override_url
        self.stages = list(stages) if stages else ["regular"]
        self.seasons = default_season_candidates(seasons)
        self.detector = AnomalyDetector(target_season=max(self.seasons))
        self.downloader = MoneyPuckDownloader(timeout=request_timeout)
        self.version_manager = VersionManager(Path(history_dir))
        self.dry_run = dry_run
        self.fail_on_anomaly = fail_on_anomaly
        self.targets = [
            DatasetTarget(
                name="team_rates",
                slug="teams",
                output_path=Path(team_output_path),
                entity_column="team",
                numeric_columns=["xgf60_5v5", "hdcf60_5v5", "pp_xgf60", "pk_xga60"],
                min_rows=28,
                expected_unique=30,
            ),
            DatasetTarget(
                name="goalie_gsax",
                slug="goalies",
                output_path=Path(goalie_output_path),
                entity_column="goalie",
                numeric_columns=["gsax_rolling", "prob_start"],
                min_rows=40,
                expected_unique=None,
            ),
        ]

    def _load_previous(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    def _normalize(self, target: DatasetTarget, raw_df: pd.DataFrame) -> pd.DataFrame:
        if target.name == "team_rates":
            return normalize_team_rates(raw_df)
        if target.name == "goalie_gsax":
            return normalize_goalie_rates(raw_df)
        raise ValueError(f"No normalizer configured for {target.name}")

    def _maybe_version(
        self,
        target: DatasetTarget,
        df: pd.DataFrame,
        metadata: Dict[str, Optional[str]],
        anomalies: List[str],
    ) -> Tuple[Optional[Path], Optional[Path]]:
        if self.dry_run:
            return None, None
        return self.version_manager.write(target, df, metadata, target.output_path, anomalies)

    def run(self) -> List[DatasetSummary]:
        summaries: List[DatasetSummary] = []
        for target in self.targets:
            override = (
                self.team_override_url if target.name == "team_rates" else self.goalie_override_url
            )
            print(
                f"\n📥 Refreshing {target.name} "
                f"(seasons={self.seasons}, stages={self.stages})..."
            )
            raw_df, metadata = self.downloader.fetch_first_available(
                dataset=target.name,
                slug=target.slug,
                seasons=self.seasons,
                stages=self.stages,
                override_url=override,
            )
            normalized = self._normalize(target, raw_df)
            if normalized.empty:
                raise RuntimeError(f"{target.name} normalization produced an empty frame.")

            previous = self._load_previous(target.output_path)
            anomalies = self.detector.detect(target, normalized, previous, metadata)
            if anomalies:
                for msg in anomalies:
                    print(f"⚠️  {target.name}: {msg}")
                if self.fail_on_anomaly:
                    raise RuntimeError(
                        f"{target.name} refresh aborted because anomalies were detected."
                    )
            else:
                print(f"✅ {target.name}: no anomalies detected.")

            history_path, output_path = self._maybe_version(
                target, normalized, metadata, anomalies
            )

            if self.dry_run:
                print(
                    f"🧪 Dry-run complete for {target.name}: would write "
                    f"{len(normalized)} rows to {target.output_path}."
                )
            else:
                print(
                    f"💾 Wrote {len(normalized)} rows to {target.output_path} "
                    f"(history snapshot: {history_path})."
                )

            summaries.append(
                DatasetSummary(
                    dataset=target.name,
                    rows=len(normalized),
                    output_path=output_path,
                    history_path=history_path,
                    source_url=metadata.get("source_url"),
                    season=metadata.get("season"),
                    stage=metadata.get("stage"),
                    anomalies=anomalies,
                )
            )
        print("\n🎯 MoneyPuck ETL complete.")
        return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh MoneyPuck team and goalie data with anomaly detection."
    )
    parser.add_argument(
        "--team-output",
        type=str,
        default=os.getenv("TEAM_RATES_PATH", "team_rates.csv"),
        help="Where to write the normalized team rates CSV.",
    )
    parser.add_argument(
        "--goalie-output",
        type=str,
        default=os.getenv("GOALIE_GSAX_PATH", "goalie_gsax.csv"),
        help="Where to write the normalized goalie GSAx CSV.",
    )
    parser.add_argument(
        "--history-dir",
        type=str,
        default=str(DEFAULT_HISTORY_DIR),
        help="Directory for dataset history snapshots and metadata.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=os.getenv("MONEYPUCK_STAGES", "regular").split(","),
        help="MoneyPuck stages to consider (default: regular).",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        help="Explicit MoneyPuck season identifiers (e.g., 2024 2023).",
    )
    parser.add_argument(
        "--team-url",
        type=str,
        help="Override URL for the team CSV (skips MoneyPuck discovery).",
    )
    parser.add_argument(
        "--goalie-url",
        type=str,
        help="Override URL for the goalie CSV (skips MoneyPuck discovery).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline without writing files.",
    )
    parser.add_argument(
        "--fail-on-anomaly",
        action="store_true",
        help="Raise an error when anomalies are detected.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=25.0,
        help="HTTP timeout when contacting MoneyPuck.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    pipeline = MoneyPuckETLPipeline(
        team_output_path=args.team_output,
        goalie_output_path=args.goalie_output,
        history_dir=Path(args.history_dir),
        stages=args.stages,
        seasons=args.seasons,
        team_override_url=args.team_url,
        goalie_override_url=args.goalie_url,
        dry_run=args.dry_run,
        fail_on_anomaly=args.fail_on_anomaly,
        request_timeout=args.request_timeout,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
