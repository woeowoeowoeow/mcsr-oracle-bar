"""
Where the models and the data live.

  models/                         frozen coefficients (prematch.json, live.json), shipped with the code
  data/                           everything pulled from the API, never committed; MCSR_ORACLE_DATA overrides it
    state.json.gz                 player state the app predicts from (python -m mcsr_oracle.build_state)
    seasons/s<N>/                 one folder per season (python -m mcsr_oracle.collect <N>)
      qualified.json              UUIDs of the season's 2000+ Elo players
      all_matches/<uuid>.parquet  every ranked match of each qualified player, any opponent (match lists)
      forfeits.parquet            forfeited qualified-vs-qualified matches, draws included
      match_cache/<id>.json       raw match details of the decisive qualified-vs-qualified matches
      matches_raw.parquet         played-out decisive qualified-vs-qualified matches, one row each with split times
      events.csv.gz               timelines of the matches in match_cache: match_id, uuid, event, time (ms)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"


def data_dir():
    return Path(os.environ.get("MCSR_ORACLE_DATA", ROOT / "data"))


def season_dir(season):
    return data_dir() / "seasons" / f"s{int(season)}"


def state_file():
    return data_dir() / "state.json.gz"


def seasons():
    """seasons present in the data folder, in order"""
    d = data_dir() / "seasons"
    return sorted(int(p.name[1:]) for p in d.glob("s*") if p.is_dir() and p.name[1:].isdigit()) if d.exists() else []
