"""
Unit tests on synthetic players and events: no data folder needed.

  - pre-match and live probabilities are symmetric: swapping the players gives 1 - P
  - live predictions use no future events: P at t is unchanged when every event after t is removed
  - the competing-risks race integral (race2) agrees with the independent-race form at rho = 0 and with Monte Carlo
  - the match parser reads splits, Elos and the winner from API-shaped JSON

Run with pytest, or directly: python tests/test_model.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from mcsr_oracle.data import load_models
from mcsr_oracle.events import parse_match, side_events
from mcsr_oracle.model import Oracle, State, SPLIT_COLS, HB, race, race2, snapshot

PLAYERS = [f"{i:032x}" for i in range(1, 9)]          # fake UUIDs
OW, NT = ["BURIED_TREASURE", "VILLAGE", "SHIPWRECK"], ["BRIDGE", "STABLES", "TREASURE"]
SPLIT_MS = dict(zip(SPLIT_COLS, [110e3, 160e3, 230e3, 200e3, 290e3, 360e3, 420e3, 480e3, 560e3, 60e3, 90e3]))


def synthetic_oracle(n_target=400, n_full=800, seed=0):
    rs = np.random.default_rng(seed)
    pm, lm = load_models()
    orc = Oracle(pm, lm, State())
    skill = dict(zip(PLAYERS, rs.normal(0, 1, len(PLAYERS))))
    date = 1_745_000_000
    for i in range(n_full):
        a, b = rs.choice(PLAYERS + ["f" * 32], 2, replace=False)
        sa = skill.get(a, -1.0) - skill.get(b, -1.0)
        date += 600
        orc.update_full(dict(match_id=i, date=date, p0_uuid=a, p1_uuid=b,
                             winner_uuid=a if rs.random() < 1 / (1 + np.exp(-sa)) else b, forfeited=bool(rs.random() < 0.15),
                             p0_elo=2000 + 100 * skill.get(a, -1), p1_elo=2000 + 100 * skill.get(b, -1),
                             p0_elo_change=rs.integers(-20, 20), p1_elo_change=rs.integers(-20, 20)))
        if i % 2 == 0 and i // 2 < n_target and a in skill and b in skill:
            isf = rs.random() < 0.15
            r = dict(match_id=10_000 + i, date=date, p0_uuid=a, p1_uuid=b, p0_won=int(rs.random() < 1 / (1 + np.exp(-sa))),
                     is_ff=int(isf), forfeit_s=rs.uniform(30, 600) if isf else np.nan,
                     p0_elo_change=rs.integers(-20, 20), p1_elo_change=rs.integers(-20, 20),
                     seed_overworld=rs.choice(OW), seed_nether=rs.choice(NT))
            for k in (0, 1):
                for c, ms in SPLIT_MS.items():
                    r[f"p{k}_{c}"] = np.nan if isf else ms * rs.lognormal(0, 0.2)
            orc.update_target(r, None if isf else synthetic_events(rs))
    return orc, date + 3600


def synthetic_events(rs, deaths=True):
    """both players' (side, event, time_ms) for one match"""
    ev = []
    for side in (0, 1):
        t = 0.0
        for e in ["nether_enter", "bastion_enter", "fortress_enter", "blind_travel", "stronghold", "end_enter", "finish"]:
            t += rs.uniform(40e3, 120e3)
            ev.append((side, e, t))
            if deaths and rs.random() < 0.15:
                ev.append((side, "death", t + rs.uniform(1e3, 20e3)))
    return sorted(ev, key=lambda x: x[2])


def test_prematch_symmetry():
    orc, date = synthetic_oracle()
    for a in PLAYERS:
        for b in PLAYERS:
            if a != b:
                p, q = orc.prematch(a, b, date, seed_overworld="VILLAGE").p, orc.prematch(b, a, date, seed_overworld="VILLAGE").p
                assert 0 < p < 1 and abs(p + q - 1) < 1e-12


def test_live_symmetry_and_no_lookahead():
    orc, date = synthetic_oracle()
    rs = np.random.default_rng(1)
    for _ in range(20):
        a, b = rs.choice(PLAYERS, 2, replace=False)
        pm = orc.prematch(a, b, date)
        ev = synthetic_events(rs)
        ts = np.arange(0, 600, 15.0)
        p = orc.live(pm, ev, ts)
        assert np.all((p > 0) & (p < 1))
        c = pm.context
        sw = dict(c, prior=-c["prior"], ffl=-c["ffl"], pers0=c["pers1"], pers1=c["pers0"])
        assert np.abs(orc.live(sw, [(1 - s, e, tm) for s, e, tm in ev], ts) - (1 - p)).max() < 1e-12
        for k, t in enumerate(ts):                    # events after t must not matter
            seen = [x for x in ev if x[2] <= t * 1000]
            one = orc.live(pm, ev, t)
            assert orc.live(pm, seen, t) == one and abs(one - p[k]) < 1e-12   # batching only changes rounding
            full, cut = snapshot(ev, [t]), snapshot(seen, [t])
            assert all(np.array_equal(full[col], cut[col]) for col in full)


def test_race_integral():
    rs = np.random.default_rng(7); m = 6; nh = len(HB) - 1
    muA, muB = rs.normal(5.2, 0.6, m), rs.normal(5.4, 0.6, m)
    sA, sB = rs.uniform(0.2, 1.0, m), rs.uniform(0.2, 1.0, m)
    lA, lB = rs.uniform(0, 0.004, m), rs.uniform(0, 0.004, m)
    LA, LB = np.repeat(lA[:, None], nh, 1), np.repeat(lB[:, None], nh, 1)
    assert np.abs(race2(muA, sA, LA, muB, sB, LB, np.zeros(m)) - race(muA, sA, lA, muB, sB, lB)).max() < 2e-3
    LA = LA * np.array([3, 2, 1.5, 1, 0.7, 0.5]); n_mc = 200_000
    fine = np.r_[0, np.geomspace(0.01, 3 * 3600, 4000)]
    for rho in (0.0, 0.5, -0.3):
        p = race2(muA, sA, LA, muB, sB, LB, np.full(m, rho))
        for i in range(m):
            z = rs.standard_normal((n_mc, 2)); z[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho * rho) * z[:, 1]
            TA, TB = np.exp(muA[i] + sA[i] * z[:, 0]), np.exp(muB[i] + sB[i] * z[:, 1])
            cum = np.r_[0, np.cumsum(np.diff(fine) * LA[i, np.minimum(np.searchsorted(HB, fine[:-1], "right") - 1, nh - 1)])]
            FA = np.interp(rs.exponential(size=n_mc), cum, fine, right=np.inf)     # A's forfeit time (path hazard)
            FB = rs.exponential(size=n_mc) / lB[i]
            a_, b_ = np.minimum(TA, FB), np.minimum(TB, FA)                          # A wins at a_, B at b_
            mc = np.where(np.minimum(a_, b_) < 3 * 3600, a_ < b_, 0.5).mean()
            assert abs(p[i] - mc) < 0.01, (rho, i, p[i], mc)


def test_parse_match():
    a, b = "a" * 32, "b" * 32
    d = {"id": 42, "date": 1_750_000_000, "seed": {"overworld": "VILLAGE", "nether": "BRIDGE"},
         "players": [{"uuid": a}, {"uuid": b}], "result": {"uuid": b, "time": 600_000},
         "changes": [{"uuid": a, "change": -15, "eloRate": 2100}, {"uuid": b, "change": 15, "eloRate": 2050}],
         "timelines": [{"uuid": a, "time": 100_000, "type": "story.enter_the_nether"},
                       {"uuid": b, "time": 95_000, "type": "story.enter_the_nether"},
                       {"uuid": a, "time": 150_000, "type": "projectelo.timeline.death"},
                       {"uuid": b, "time": 600_000, "type": "projectelo.timeline.dragon_death"},
                       {"uuid": a, "time": 5_000, "type": "some.unknown.event"}]}
    r = parse_match(d, 12)
    assert (r["p0_elo"], r["p1_elo"], r["p0_won"], r["p1_won"]) == (2100, 2050, 0, 1)
    assert (r["p0_nether_enter"], r["p1_nether_enter"], r["p1_finish"], r["p0_finish"]) == (100_000, 95_000, 600_000, None)
    assert side_events(d, a) == [(0, "nether_enter", 100_000.0), (1, "nether_enter", 95_000.0), (0, "death", 150_000.0),
                                 (1, "finish", 600_000.0)]


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("passed", name)
