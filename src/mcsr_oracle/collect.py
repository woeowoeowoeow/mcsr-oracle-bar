"""
Collect one season from the MCSR Ranked API into data/seasons/s<N>/ (layout in paths.py).

    python -m mcsr_oracle.collect 11

Steps (each resumable; stop and rerun at any time):
  1. qualified.json        season leaderboard, eloRate >= 2000. For a finished season that is the season-end
                           result; for the current season it is whoever is 2000+ at collection time.
  2. all_matches/          every ranked match of each qualified player, any opponent (match lists, cheap)
  3. forfeits.parquet      forfeited qualified-vs-qualified matches, draws included
  4. match_cache/          details of the decisive qualified-vs-qualified matches (one request each, the slow step:
                           about 2,900 an hour at the rate limit, so roughly 10 hours for a full season)
  5. matches_raw.parquet   parsed played-out matches
  6. events.csv.gz         timelines of the cached matches
"""
import csv, gzip, json, sys
import pandas as pd
from . import paths
from .api import get
from .events import RAW_COLS, SPLIT_COLS, parse_match, timeline_rows

MIN_ELO = 2000


def qualified(season):
    f = paths.season_dir(season) / "qualified.json"
    if not f.exists():
        lb = get("/leaderboard", {"season": season})
        q = [u["uuid"] for u in lb["users"] if (u.get("seasonResult") or {}).get("eloRate", 0) >= MIN_ELO]
        f.write_text(json.dumps(q))
    q = set(json.loads(f.read_text()))
    print(f"season {season}: {len(q)} qualified players", flush=True)
    return q


def scan(season, q):
    out = paths.season_dir(season) / "all_matches"
    out.mkdir(exist_ok=True)
    for i, uuid in enumerate(sorted(q)):
        f = out / f"{uuid}.parquet"
        if f.exists():
            continue
        rows, before = [], None
        while True:
            params = {"season": season, "type": 2, "count": 100, "excludedecay": "true"}
            if before: params["before"] = before
            ms = get(f"/users/{uuid}/matches", params)
            if not ms: break
            for m in ms:
                ps = [p["uuid"] for p in m.get("players", [])]
                if len(ps) != 2:
                    continue
                ch = {c["uuid"]: c for c in m.get("changes", [])}
                res = m.get("result") or {}
                rows.append({"match_id": m["id"], "season": season, "date": m.get("date"),
                             "p0_uuid": ps[0], "p1_uuid": ps[1], "winner_uuid": res.get("uuid"),
                             "result_time": res.get("time"), "forfeited": bool(m.get("forfeited")),
                             "p0_elo": ch.get(ps[0], {}).get("eloRate"), "p1_elo": ch.get(ps[1], {}).get("eloRate"),
                             "p0_elo_change": ch.get(ps[0], {}).get("change"), "p1_elo_change": ch.get(ps[1], {}).get("change"),
                             "seed_overworld": (m.get("seed") or {}).get("overworld") or m.get("seedType"),
                             "seed_nether": (m.get("seed") or {}).get("nether") or m.get("bastionType")})
            if len(ms) < 100: break
            before = ms[-1]["id"]
        pd.DataFrame(rows).to_parquet(f, index=False)
        print(f"s{season} [{i + 1}/{len(q)}] {uuid}: {len(rows)} matches", flush=True)


def qvq(season, q):
    d = paths.season_dir(season) / "all_matches"
    A = pd.concat([pd.read_parquet(d / f"{u}.parquet") for u in sorted(q)]).drop_duplicates("match_id")
    return A[A.p0_uuid.isin(q) & A.p1_uuid.isin(q)]


def forfeits(season, Q):
    F = Q[Q.forfeited].rename(columns={"result_time": "forfeit_time"})
    F = F[["match_id", "season", "date", "p0_uuid", "p1_uuid", "winner_uuid", "forfeit_time", "p0_elo", "p1_elo",
           "p0_elo_change", "p1_elo_change", "seed_overworld", "seed_nether"]]
    F.to_parquet(paths.season_dir(season) / "forfeits.parquet", index=False)
    print(f"{len(F)} forfeited q-vs-q matches ({F.winner_uuid.notna().sum()} forfeit wins)", flush=True)


def fetch(season, ids):
    cache = paths.season_dir(season) / "match_cache"
    cache.mkdir(exist_ok=True)
    todo = [i for i in ids if not (cache / f"{i}.json").exists()]
    print(f"{len(todo)} of {len(ids)} match details still needed", flush=True)
    for k, mid in enumerate(todo):
        d = get(f"/matches/{mid}")
        if d is not None:
            (cache / f"{mid}.json").write_text(json.dumps(d))
        if k % 200 == 0:
            print(f"  {k}/{len(todo)}", flush=True)


def build_raw(season, ids):
    cache = paths.season_dir(season) / "match_cache"
    recs = [r for i in ids if (cache / f"{i}.json").exists()
            for r in [parse_match(json.loads((cache / f"{i}.json").read_text()), season)] if r]
    raw = pd.DataFrame(recs, columns=RAW_COLS)
    for c in RAW_COLS:
        if c.split("_", 1)[-1] in SPLIT_COLS or c.endswith("_elo") or c.endswith("_elo_change"):
            raw[c] = raw[c].astype(float)
    raw.to_parquet(paths.season_dir(season) / "matches_raw.parquet", index=False)
    print(f"{len(raw)} of {len(ids)} played-out matches parsed", flush=True)


def build_events(season, ids):
    cache = paths.season_dir(season) / "match_cache"
    n = 0
    with gzip.open(paths.season_dir(season) / "events.csv.gz", "wt", newline="") as out:
        w = csv.writer(out)
        w.writerow(["match_id", "uuid", "event", "time"])
        for i in ids:
            f = cache / f"{i}.json"
            if f.exists():
                w.writerows(timeline_rows(json.loads(f.read_text()))); n += 1
    print(f"{n} timelines -> events.csv.gz", flush=True)


def collect(season):
    paths.season_dir(season).mkdir(parents=True, exist_ok=True)
    q = qualified(season)
    scan(season, q)
    Q = qvq(season, q)
    forfeits(season, Q)
    played = Q[~Q.forfeited & Q.winner_uuid.notna()].sort_values("date").match_id.tolist()
    ff_wins = Q[Q.forfeited & Q.winner_uuid.notna()].sort_values("date").match_id.tolist()
    print(f"{len(played)} played-out and {len(ff_wins)} forfeit-win q-vs-q matches", flush=True)
    fetch(season, played + ff_wins)
    build_raw(season, played)
    build_events(season, played + ff_wins)
    print(f"season {season} collection done", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m mcsr_oracle.collect <season>")
    collect(int(sys.argv[1]))
