"""
Load the three match streams from the data folder and replay them through model.State.

  target  decisive qualified-vs-qualified matches: played-out (matches_raw) + forfeit wins (forfeits), both Elos
          known, sorted by date (stable: matches with the same date keep their file order)
  full    every decisive match in all_matches/ (any opponent), both Elos known, sorted by (date, match_id)
  live    target matches with a timeline (events.csv.gz), in target order

Players only interact with their own earlier matches, so the streams can be merged by date: at each date, the
pre-match features of captured matches are taken before any update at that date.
"""
import gzip, json
import numpy as np, pandas as pd
from . import paths
from .model import Oracle, State, SPLIT_COLS

TARGET_COLS = (["match_id", "season", "date", "p0_uuid", "p1_uuid", "p0_won", "is_ff", "forfeit_s", "p0_elo", "p1_elo",
                "p0_elo_change", "p1_elo_change", "seed_overworld", "seed_nether"]
               + [f"p{k}_{s}" for k in (0, 1) for s in SPLIT_COLS])
FULL_COLS = ["match_id", "date", "p0_uuid", "p1_uuid", "winner_uuid", "forfeited", "p0_elo", "p1_elo", "p0_elo_change",
             "p1_elo_change"]


def target_stream(seasons):
    df = pd.concat([pd.read_parquet(paths.season_dir(s) / "matches_raw.parquet").sort_values("date", kind="mergesort")
                    .reset_index(drop=True) for s in seasons], ignore_index=True)
    assert df.date.is_monotonic_increasing, "season files overlap in time"
    df = df.dropna(subset=["p0_elo", "p1_elo"]).reset_index(drop=True)
    ff = pd.concat([pd.read_parquet(paths.season_dir(s) / "forfeits.parquet") for s in seasons], ignore_index=True)
    ff = ff[ff.winner_uuid.notna() & ~ff.match_id.isin(df.match_id)].dropna(subset=["p0_elo", "p1_elo"]).drop_duplicates("match_id")
    ff = ff.assign(p0_won=(ff.winner_uuid == ff.p0_uuid).astype(int), forfeit_s=ff.forfeit_time / 1000, is_ff=1)
    S = pd.concat([df.assign(is_ff=0, forfeit_s=np.nan), ff], ignore_index=True)
    S = S.sort_values("date", kind="mergesort").reset_index(drop=True)
    for c in TARGET_COLS:
        if c not in S:
            S[c] = np.nan
    return S


def full_stream(seasons):
    fs = [f for s in seasons for f in sorted((paths.season_dir(s) / "all_matches").glob("*.parquet"))]
    FS = pd.concat([pd.read_parquet(f) for f in fs]).drop_duplicates("match_id")
    FS = FS[FS.winner_uuid.notna()].dropna(subset=["p0_elo", "p1_elo"]).sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    return FS


def timelines(seasons, ids):
    """event rows (match_id, uuid, event, time) of the matches in ids"""
    ev = pd.concat([pd.read_csv(paths.season_dir(s) / "events.csv.gz") for s in seasons]).drop_duplicates()
    return ev[ev.match_id.isin(ids)]


def events_by_match(ev, S):
    """{match_id: [(side, event, time_ms), ...]} (side 0 = p0)"""
    p0 = S.set_index("match_id").p0_uuid
    side = np.where(ev.uuid.to_numpy() == p0.reindex(ev.match_id).to_numpy(), 0, 1)
    out = {}
    for mid, s, e, tm in zip(ev.match_id.to_numpy(), side, ev.event.to_numpy(), ev.time.to_numpy()):
        out.setdefault(mid, []).append((int(s), e, float(tm)))
    return out


def records(df, cols):
    out = []
    for r in df[cols].itertuples(index=False):
        d = r._asdict()
        out.append({k: (None if (isinstance(v, float) and v != v and k.startswith("seed")) else v) for k, v in d.items()})
    return out


def replay(oracle, S, FS, live_ids, evm, capture=None, on_capture=None):
    """feed both streams to oracle.state in date order. For each target match whose id is in `capture`, call
    on_capture(match_row_dict, prematch) before any update at its date. Target matches in live_ids also update the
    personal segment speeds from evm[match_id]."""
    capture = capture or set()
    tr = records(S, TARGET_COLS); fr = records(FS, FULL_COLS)
    ops = [(r["date"], 1, i, "t") for i, r in enumerate(tr)] + [(r["date"], 1, i, "f") for i, r in enumerate(fr)]
    ops += [(r["date"], 0, i, "c") for i, r in enumerate(tr) if r["match_id"] in capture]
    ops.sort(key=lambda x: (x[0], x[1], x[3] != "f", x[2]))          # captures first at each date; then full, target
    for date, _, i, kind in ops:
        if kind == "c":
            r = tr[i]
            pm = oracle.prematch(r["p0_uuid"], r["p1_uuid"], r["date"], r["p0_elo"], r["p1_elo"],
                                 r["seed_overworld"], r["seed_nether"])
            on_capture(r, pm)
        elif kind == "f":
            oracle.state.update_full(fr[i])
        else:
            r = tr[i]
            oracle.update_target(r, evm.get(r["match_id"]) if r["match_id"] in live_ids else None)
    return oracle


def load_models(folder=None):
    f = paths.MODELS if folder is None else folder
    return json.loads((f / "prematch.json").read_text()), json.loads((f / "live.json").read_text())


def load_oracle(state=None, models=None):
    """frozen models + the player state in data/state.json.gz (an empty state if there is none yet)"""
    pm, lm = load_models(models)
    sf = paths.state_file() if state is None else state
    if not sf.exists():
        return Oracle(pm, lm, State())
    with gzip.open(sf, "rt") as fh:
        return Oracle(pm, lm, State.from_dump(json.load(fh)))


def build_state(seasons, oracle=None, capture=None, on_capture=None):
    """replay every stream of the given seasons into oracle (default: the frozen models with an empty state).
    Returns (oracle, target stream)."""
    if oracle is None:
        oracle = Oracle(*load_models(), State())
    S, FS = target_stream(seasons), full_stream(seasons)
    ev = timelines(seasons, set(S.match_id))
    live_ids = set(ev.match_id)
    return replay(oracle, S, FS, live_ids, events_by_match(ev, S), capture, on_capture), S
