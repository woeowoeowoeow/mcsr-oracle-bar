"""
Integration test: replaying the collected Season 8-11 data reproduces the frozen models' predictions and the
player state they were frozen with. It needs data/seasons/s8 ... s11 and the frozen reference outputs in
data/reference/ (state.json.gz, reference_prematch.parquet, reference_live.parquet), and is skipped without them.

  1. Pre-match: for 300 random valid matches, P and `valid` equal the reference (P to 1e-9).
  2. Live: for the 300 matches in reference_live.parquet, P at every 15 s snapshot equals the reference (1e-9).
  3. Symmetry: swapping the players gives 1 - P, pre-match and live (1e-12).
  4. The replayed state equals reference/state.json.gz and survives a dump / load round trip.

About 1 minute. Run with pytest, or directly: python tests/test_reproduce.py
"""
import gzip, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from mcsr_oracle import paths
from mcsr_oracle.data import build_state, events_by_match, load_models, timelines
from mcsr_oracle.model import Oracle, State

SEASONS, N_PRE, TOL = [8, 9, 10, 11], 300, 1e-9
REF = paths.data_dir() / "reference"


def missing():
    need = [paths.season_dir(s) / "matches_raw.parquet" for s in SEASONS] + \
        [REF / f for f in ("state.json.gz", "reference_prematch.parquet", "reference_live.parquet")]
    return [str(p) for p in need if not p.exists()]


def run():
    T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)
    refp = pd.read_parquet(REF / "reference_prematch.parquet").set_index("match_id")
    refl = pd.read_parquet(REF / "reference_live.parquet")
    rs = np.random.default_rng(29)
    pre_ids = set(rs.choice(refp.index[refp.valid].to_numpy(), N_PRE, replace=False).tolist())
    live_ids = set(refl.match_id.unique().tolist())
    orc = Oracle(*load_models(), State())
    got = {}

    def on_capture(r, pm):                  # the swapped call must see the same state, so it is made here
        sw = orc.prematch(r["p1_uuid"], r["p0_uuid"], r["date"], r["p1_elo"], r["p0_elo"], r["seed_overworld"], r["seed_nether"])
        got[r["match_id"]] = (r, pm, sw)

    _, S = build_state(SEASONS, orc, capture=pre_ids | live_ids, on_capture=on_capture)
    m = orc.state.meta
    log(f"replayed {m['n_target']:,} target, {m['n_full']:,} full-history and {m['n_live']:,} timeline matches; "
        f"captured {len(got)}")
    assert len(S) == len(refp) and (S.match_id.to_numpy() == refp.index.to_numpy()).all(), "target stream differs"

    worst = 0.0
    for mid in pre_ids:
        r, pm, sw = got[mid]
        assert pm.valid == bool(refp.at[mid, "valid"]), f"valid flag differs for match {mid}"
        worst = max(worst, abs(pm.p - refp.at[mid, "p"]))
        assert abs(sw.p - (1 - pm.p)) < 1e-12, "pre-match symmetry"
    log(f"pre-match, {len(pre_ids)} matches: max |P diff| {worst:.1e}")
    assert worst < TOL

    evm = events_by_match(timelines(SEASONS, live_ids), S)
    worst, nsn = 0.0, 0
    for mid, g in refl.groupby("match_id"):
        r, pm, _ = got[mid]
        ts = g.t.to_numpy(float); evs = evm.get(mid, [])
        p = orc.live(pm, evs, ts)
        both = ~np.isnan(g.p.to_numpy())
        assert (np.isnan(p) == ~both).all()
        worst = max(worst, np.abs(p[both] - g.p.to_numpy()[both]).max(initial=0)); nsn += len(ts)
        c = pm.context
        ctx = dict(c, prior=-c["prior"], ffl=-c["ffl"], pers0=c["pers1"], pers1=c["pers0"])
        ps = orc.live(ctx, [(1 - s, e, tm) for s, e, tm in evs], ts)
        assert np.abs(ps[both] - (1 - p[both])).max(initial=0) < 1e-12, "live symmetry"
    log(f"live, {len(live_ids)} matches / {nsn:,} snapshots: max |P diff| {worst:.1e}")
    assert worst < TOL

    with gzip.open(REF / "state.json.gz", "rt") as fh:
        saved = json.load(fh)
    mine = json.loads(json.dumps(orc.state.dump()))
    for k in ("target", "full", "pers", "elo"):
        assert mine[k] == saved[k], f"state part {k} differs from the reference state"
    o2 = Oracle(orc.pm, orc.lm, State.from_dump(mine))
    later = m["last_full_date"] + 86400
    for mid in list(pre_ids)[:50]:
        r = got[mid][0]
        assert orc.prematch(r["p0_uuid"], r["p1_uuid"], later).p == o2.prematch(r["p0_uuid"], r["p1_uuid"], later).p
    log("replayed state equals the reference state; dump / load round trip exact")
    log("ALL CHECKS PASSED")


def test_reproduce():
    if missing():
        import pytest
        pytest.skip("needs the collected data and reference files: " + ", ".join(missing()))
    run()


if __name__ == "__main__":
    if missing():
        sys.exit("missing: " + ", ".join(missing()))
    run()
