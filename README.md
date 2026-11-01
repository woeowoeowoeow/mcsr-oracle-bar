# MCSR Oracle Bar

Win probabilities for MCSR Ranked (Minecraft speedrun 1v1) matches, made to drive a live broadcast bar: like a
chess evaluation bar, it shows who's ahead and each player's chance of winning as the match unfolds. A pre-match
prediction is also available.

**Status:** the models, data collection and player state are in place and tested. The broadcast overlay is not
built yet.

## How it works

- **Live model** (the bar): every 15 s, P(player A wins) from both players' events so far: stage reached, time
  lead, deaths, resets. A survival model of each player's remaining time to finish, with correlated finish times and
  a forfeit hazard, is stacked into a logistic model of the match state. It starts from the pre-match rating gap.
  Walk-forward Brier score 0.2006 (Oct 2025 – Apr 2026) and 0.1997 (Season 11), against 0.2377 for ratings alone.
- **Pre-match model:** logistic regression on 30 player-difference features (TrueSkill over each player's full
  ranked history, win and forfeit rates, split-time averages, Elo). Brier 0.2369 / 0.2384, 58.8% / 58.6% accuracy.
- Both were fit on ranked matches between two 2000+ Elo players, Seasons 8–11, and have only been validated on that
  population.

The math and the case for the methodology are in `docs/` (in progress).

## Setup

```
pip install -e .            # or: pip install -e ".[dev]" for pytest
```

The data comes from the public MCSR Ranked API (https://docs.mcsrranked.com) and is not part of this repository.
Collect each season, then build the player state:

```
python -m mcsr_oracle.collect 8           # repeat for each season; resumable
python -m mcsr_oracle.build_state         # replays every collected season -> data/state.json.gz
```

Collection respects the API's rate limit, so it is slow: about 2,900 requests an hour, and a full season needs one
request per qualified-vs-qualified match. Data goes to `data/` (or `MCSR_ORACLE_DATA`); the layout is in
`src/mcsr_oracle/paths.py`.

## Using the models

```python
from mcsr_oracle import load_oracle
o = load_oracle()                              # models/ + data/state.json.gz
pm = o.prematch(p0_uuid, p1_uuid, date)        # pm.p = P(p0 wins); pm.valid = both players have enough history
p = o.live(pm, events, t)                      # events: [(side 0/1, event, time_ms), ...] so far; t in seconds
```

`live()` takes the API's timeline types (`story.enter_the_nether`, `projectelo.timeline.death`, …) directly;
`events.side_events(match_json, p0_uuid)` converts a match's JSON.

## Tests

```
python tests/test_model.py        # no data needed: symmetry, no look-ahead, race integral vs Monte Carlo, parser
python tests/test_reproduce.py    # needs Seasons 8-11 and data/reference/: reproduces the frozen predictions
```

## Layout

```
models/              frozen coefficients (prematch.json, live.json)
src/mcsr_oracle/
  model.py           the models and the player state
  data.py            match streams, replay, loading
  events.py          API match JSON -> rows and event lists
  collect.py         season collection from the API
  api.py             rate-limited API client
  build_state.py     replay -> data/state.json.gz
  paths.py           data layout
tests/
docs/                write-up
```
