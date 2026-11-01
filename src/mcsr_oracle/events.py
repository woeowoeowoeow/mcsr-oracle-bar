"""
Match JSON from the API -> the rows and event lists the models use.

  timeline_rows(detail)       [(match_id, uuid, event, time_ms), ...] for events.csv.gz
  side_events(detail, p0)     [(side, event, time_ms), ...] for Oracle.live (side 0 = p0)
  parse_match(detail, season) one matches_raw row: Elos, Elo changes, winner and the 11 split times per player
"""
from .model import API_EVENTS, SPLIT_COLS

SPLIT_TYPES = {"nether_enter": "story.enter_the_nether", "bastion_enter": "nether.find_bastion",
               "fortress_enter": "nether.find_fortress", "bastion_loot": "nether.loot_bastion",
               "blaze_rod": "nether.obtain_blaze_rod", "blind_travel": "projectelo.timeline.blind_travel",
               "stronghold": "story.follow_ender_eye", "end_enter": "story.enter_the_end",
               "finish": "projectelo.timeline.dragon_death", "iron_tools": "story.iron_tools",
               "lava_bucket": "story.lava_bucket"}
RAW_COLS = (["match_id", "season", "date", "seed_overworld", "seed_nether", "finish_time"]
            + [f"p{k}_{c}" for k in (0, 1) for c in ["uuid", "elo", "elo_change", "won"] + SPLIT_COLS])


def timeline_rows(detail):
    return [(detail["id"], t["uuid"], API_EVENTS[t["type"]], t["time"])
            for t in detail.get("timelines", []) if t["type"] in API_EVENTS]


def side_events(detail, p0_uuid):
    return [(0 if u == p0_uuid else 1, e, float(tm)) for _, u, e, tm in timeline_rows(detail)]


def splits(t):
    """{split: time_ms} from {timeline type: time_ms}; the last occurrence of each type wins (as the original parser)"""
    return {c: t.get(api) for c, api in SPLIT_TYPES.items()}


def parse_match(detail, season):
    players, changes = detail.get("players", []), {c["uuid"]: c for c in detail.get("changes", [])}
    result = detail.get("result", {}); winner = result.get("uuid")
    if len(players) != 2 or winner is None:
        return None
    p0, p1 = players
    tl = {p0["uuid"]: {}, p1["uuid"]: {}}
    for t in detail.get("timelines", []):
        if t["uuid"] in tl: tl[t["uuid"]][t["type"]] = t["time"]
    side = lambda p, k: {f"{k}_uuid": p["uuid"], f"{k}_elo": changes.get(p["uuid"], {}).get("eloRate"),
                         f"{k}_elo_change": changes.get(p["uuid"], {}).get("change"), f"{k}_won": int(p["uuid"] == winner),
                         **{f"{k}_{s}": v for s, v in splits(tl[p["uuid"]]).items()}}
    return {"match_id": detail["id"], "season": season, "date": detail.get("date"),
            "seed_overworld": (detail.get("seed") or {}).get("overworld"), "seed_nether": (detail.get("seed") or {}).get("nether"),
            "finish_time": result.get("time"), **side(p0, "p0"), **side(p1, "p1")}
