"""
The frozen pre-match and live win-probability models, and the player state they need.

    from mcsr_oracle import load_oracle
    o = load_oracle()                                  # models/*.json + data/state.json.gz
    pm = o.prematch(p0_uuid, p1_uuid, date)            # pm.p = P(p0 wins) before the match; pm.valid, pm.features
    o.live(pm, events, t)                              # P(p0 wins) at t seconds, from both players' events so far
    o.update_target(record, events)                    # add a finished qualified-vs-qualified match to the state
    o.update_full(record)                              # add a finished match vs any opponent to the state
    o.save(data_dir)                                   # write the updated state back (state.json.gz)

Models (coefficients in models/, fit on every valid Season 8-11 match):
  pre-match  mirrored, no-intercept logistic regression (C = 0.1) on 30 player-difference features
             (walk-forward 0.2369 Brier / 58.8% Oct 2025-Apr 2026, 0.2384 / 58.6% Season 11)
  live       a logistic model of the in-match state (41 features: stage lead, time lead, deaths, prior, each by
             elapsed-time bucket) stacked with a survival model's win probability x 7 elapsed buckets. The survival
             model is a censored log-normal AFT model of each player's remaining time, with the two times correlated
             (rho by leader's segment) and a forfeit hazard that changes over the horizon after t, combined as
             competing risks (walk-forward 0.2006 Oct 2025-Apr 2026, 0.1997 Season 11). live_predict() also runs the
             older constant-hazard, independent-race form if live.json has "forfeit_hazard" instead.

Player state comes from three streams, each kept in its own order (data.py builds them):
  target   decisive qualified-vs-qualified matches (played-out + forfeit wins): EWMs of win rate, completion rate,
           split times, Elo change, forfeit rates; seed-type affinity; TrueSkill on played-out matches and on all
           target matches; date of the last played-out match
  full     every decisive match of a tracked player vs any opponent: TrueSkill (tau 0.042) and span-600 EWMs of
           wins, forfeit losses, forfeit wins and Elo residual; latest Elo
  live     target matches with a timeline: EWM (span 200) of each player's log segment times
A feature uses only matches added before it.

Events for live(): an iterable of (side, event, time_ms), side 0 = p0 and 1 = p1. Event names are those of
events.py ("nether_enter", "death", "reset", ...) or the raw API timeline types
("story.enter_the_nether", "projectelo.timeline.death", ...). Other events are ignored.

Below this docstring the code is unchanged from the research version that produced the reported numbers;
tests/test_reproduce.py checks the frozen models' predictions and the replayed state against it.
"""
import gzip, json, math
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import trueskill
from scipy.special import ndtr

# ---------------------------------------------------------------- constants (identical to the research scripts)
SPLIT_COLS = ["nether_enter", "bastion_enter", "fortress_enter", "bastion_loot", "blaze_rod",
              "blind_travel", "stronghold", "end_enter", "finish", "iron_tools", "lava_bucket"]
EARLY_S, SHRINK_K, SPAN, TREND_SPAN, MINP = 300, 10, 200, 10, 10
TAU, ELO_SLOPE, FULL_SPAN = 25 / 600, 0.99, 600
STEP, RECENT = 15, 30
STAGE = {"nether_enter": 1, "bastion_enter": 2, "fortress_enter": 3, "blind_travel": 4, "stronghold": 5, "end_enter": 6}
SEG = np.array([0, 1, 1, 1, 2, 2, 3])
OW = ["BURIED_TREASURE", "DESERT_TEMPLE", "RUINED_PORTAL", "SHIPWRECK", "VILLAGE"]
NT = ["BRIDGE", "HOUSING", "STABLES", "TREASURE"]
GRID = np.r_[0.0, np.geomspace(3, 3 * 3600, 79)]
TB_EDGES = [0, 60, 120, 180, 300, 420, 600, np.inf]
P_SPAN, P_SHRINK, P_NCAP = 200, 5, 50
API_EVENTS = {"story.enter_the_nether": "nether_enter", "nether.find_bastion": "bastion_enter",
              "nether.loot_bastion": "bastion_loot", "nether.find_fortress": "fortress_enter",
              "nether.obtain_blaze_rod": "blaze_rod", "projectelo.timeline.blind_travel": "blind_travel",
              "story.follow_ender_eye": "stronghold", "story.enter_the_end": "end_enter",
              "projectelo.timeline.dragon_death": "finish", "story.iron_tools": "iron_tools",
              "story.lava_bucket": "lava_bucket", "projectelo.timeline.death": "death",
              "projectelo.timeline.death_spawnpoint": "death_sp", "projectelo.timeline.reset": "reset",
              "nether.charge_respawn_anchor": "anchor", "adventure.sleep_in_bed": "bed",
              "projectelo.timeline.forfeit": "forfeit"}
nan = float("nan")


class Ewm:
    """pandas x.shift(1).ewm(span, adjust=True, ignore_na=True).mean(), kept online: mean() is the value before
    the next observation; missing observations are skipped"""
    __slots__ = ("a", "num", "den", "n")

    def __init__(self, span, num=0.0, den=0.0, n=0):
        self.a, self.num, self.den, self.n = 2 / (span + 1), num, den, n

    def add(self, x):
        if x is None or x != x:
            return
        self.num = (1 - self.a) * self.num + x
        self.den = (1 - self.a) * self.den + 1
        self.n += 1

    def mean(self, minp=1):
        return self.num / self.den if self.n >= max(minp, 1) else nan

    def dump(self):
        return [self.num, self.den, self.n]


def _isnan(x):
    return x is None or (isinstance(x, float) and x != x)


def _logit_p(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


# ---------------------------------------------------------------- player state
TARGET_EWM = {**{c: SPAN for c in ["won_nf", "fin_nf", "won", "ff_loss", "early_ff_loss", "late_ff_loss", "ff_win"]},
              **{s: SPAN for s in SPLIT_COLS}, "elo_change_nf": TREND_SPAN, "elo_change": TREND_SPAN}
FULL_EWM = ["won", "ff_loss", "ff_win", "elo_resid"]


class State:
    def __init__(self):
        self.env = trueskill.TrueSkill(tau=TAU, draw_probability=0.0)
        self.tgt, self.full, self.pers, self.elo = {}, {}, {}, {}
        self.meta = dict(n_target=0, n_full=0, n_live=0, last_target_date=None, last_full_date=None)

    # ------------------------------------------------------------ per-player records
    def _t(self, u):
        s = self.tgt.get(u)
        if s is None:
            s = self.tgt[u] = dict(ew={c: Ewm(sp) for c, sp in TARGET_EWM.items()}, last_nf=None, aff_ow={}, aff_nt={},
                                   ts_nf=self.env.create_rating(), ts_all=self.env.create_rating())
        return s

    def _f(self, u):
        s = self.full.get(u)
        if s is None:
            s = self.full[u] = dict(ew={c: Ewm(FULL_SPAN) for c in FULL_EWM}, ts=self.env.create_rating())
        return s

    def _p(self, u):
        s = self.pers.get(u)
        if s is None:
            s = self.pers[u] = [Ewm(P_SPAN) for _ in range(4)]
        return s

    # ------------------------------------------------------------ features of one player before a match
    def target_player(self, u, date, ow, nt):
        s = self.tgt.get(u)
        if s is None:
            s = dict(ew={c: Ewm(sp) for c, sp in TARGET_EWM.items()}, last_nf=None, aff_ow={}, aff_nt={})
        e = s["ew"]
        f = {"win_rate": e["won_nf"].mean(MINP), "completion_rate": e["fin_nf"].mean(MINP)}
        for c in SPLIT_COLS:
            f[f"{c}_mean"] = e[c].mean(MINP)
            f[f"{c}_n"] = e[c].n
        f["elo_trend"] = e["elo_change_nf"].mean(MINP)
        f["log_days_idle"] = math.log1p(max(date - s["last_nf"], 0) / 86400) if s["last_nf"] is not None else nan
        ov = 0.5 if _isnan(f["win_rate"]) else f["win_rate"]
        for key, val in (("ow", ow), ("nt", nt)):
            if _isnan(val):
                f[f"{key}_aff"] = nan
            else:
                w_, n_ = s[f"aff_{key}"].get(val, (0.0, 0.0))
                f[f"{key}_aff"] = (w_ + SHRINK_K * ov) / (n_ + SHRINK_K) - ov
        f["win_rate_all"] = e["won"].mean(MINP)
        f["elo_trend_all"] = e["elo_change"].mean(MINP)
        for c in ["early_ff_loss", "late_ff_loss", "ff_win"]:
            f[f"{c}_rate"] = e[c].mean(MINP)
        return f

    def full_player(self, u):
        s = self.full.get(u)
        if s is None:
            return {f"{c}_full_600": nan for c in FULL_EWM}
        return {f"{c}_full_600": s["ew"][c].mean(MINP) for c in FULL_EWM}

    def pers_player(self, u, pop):
        out = []
        for j, e in enumerate(self.pers.get(u, [Ewm(P_SPAN) for _ in range(4)])):
            k = min(e.n, P_NCAP)
            out.append((k * e.mean() + P_SHRINK * pop[j]) / (k + P_SHRINK) if e.n > 0 else pop[j])
        return out

    def ts_logit(self, a, b, which):
        """pre-match TrueSkill logit of a vs b; which = 'nf' / 'all' (target stream) or 'full'"""
        get = (lambda u: self.full[u]["ts"] if u in self.full else self.env.create_rating()) if which == "full" else \
            (lambda u: self.tgt[u][f"ts_{which}"] if u in self.tgt else self.env.create_rating())
        ra, rb = get(a), get(b)
        p = self.env.cdf((ra.mu - rb.mu) / math.sqrt(2 * self.env.beta ** 2 + ra.sigma ** 2 + rb.sigma ** 2))
        return _logit_p(p)

    # ------------------------------------------------------------ updates
    def update_target(self, r):
        """r: dict with match_id, date, p0_uuid, p1_uuid, p0_won (1/0), is_ff (1/0), forfeit_s (s, forfeits only),
        p{k}_elo_change, seed_overworld, seed_nether and the 11 p{k}_<split> times in ms (played-out only)"""
        isf = bool(r.get("is_ff", 0)); o = int(r["p0_won"])
        for k in (0, 1):
            u = r[f"p{k}_uuid"]; s = self._t(u); e = s["ew"]
            won = float(o if k == 0 else 1 - o)
            ffl = float(isf and won == 0)
            fs = r.get("forfeit_s", nan)
            vals = {"won_nf": nan if isf else won, "fin_nf": nan if isf else float(not _isnan(r.get(f"p{k}_finish"))),
                    "elo_change_nf": nan if isf else r.get(f"p{k}_elo_change", nan), "elo_change": r.get(f"p{k}_elo_change", nan),
                    "won": won, "ff_loss": ffl, "ff_win": float(isf and won == 1),
                    "early_ff_loss": ffl * float(not _isnan(fs) and fs < EARLY_S),
                    "late_ff_loss": ffl * float(not _isnan(fs) and fs >= EARLY_S)}
            for c in SPLIT_COLS:
                vals[c] = nan if isf else r.get(f"p{k}_{c}", nan)
            for c, v in vals.items():
                e[c].add(v)
            for key, col in (("ow", "seed_overworld"), ("nt", "seed_nether")):
                val = r.get(col)
                if not _isnan(val):
                    w_, n_ = s[f"aff_{key}"].get(val, (0.0, 0.0))
                    s[f"aff_{key}"][val] = (w_ + (0.0 if isf else won), n_ + (0.0 if isf else 1.0))
            if not isf:
                s["last_nf"] = r["date"]
        a, b = self._t(r["p0_uuid"]), self._t(r["p1_uuid"])
        for which, do in (("ts_nf", not isf), ("ts_all", True)):
            if do:
                (a[which],), (b[which],) = self.env.rate([(a[which],), (b[which],)], ranks=[1 - o, o])
        self.meta["n_target"] += 1; self.meta["last_target_date"] = int(r["date"])

    def update_full(self, r):
        """r: dict with match_id, date, p0_uuid, p1_uuid, winner_uuid, forfeited, p0_elo, p1_elo, p{k}_elo_change"""
        o = int(r["winner_uuid"] == r["p0_uuid"])
        p_elo = 1 / (1 + np.exp(-ELO_SLOPE * (r["p0_elo"] - r["p1_elo"]) / 400))
        for k, p_me in ((0, p_elo), (1, 1 - p_elo)):
            u = r[f"p{k}_uuid"]; s = self._f(u)
            won = float(o if k == 0 else 1 - o); isf = bool(r["forfeited"])
            for c, v in (("won", won), ("ff_loss", float(isf and won == 0)), ("ff_win", float(isf and won == 1)),
                         ("elo_resid", won - p_me)):
                s["ew"][c].add(float(v))
            ch = r.get(f"p{k}_elo_change", nan)
            if not _isnan(ch) and (u not in self.elo or r["date"] >= self.elo[u][0]):
                self.elo[u] = (int(r["date"]), float(r[f"p{k}_elo"] + ch))
        a, b = self._f(r["p0_uuid"]), self._f(r["p1_uuid"])
        (a["ts"],), (b["ts"],) = self.env.rate([(a["ts"],), (b["ts"],)], ranks=[1 - o, o])
        self.meta["n_full"] += 1; self.meta["last_full_date"] = int(r["date"])

    def update_live(self, p0_uuid, p1_uuid, events):
        """personal segment speeds from a finished target match's timeline. events: (side, event, time_ms)"""
        for k, u in ((0, p0_uuid), (1, p1_uuid)):
            for j, v in enumerate(segment_log_times([(e, tm) for sd, e, tm in _norm_events(events) if sd == k])):
                self._p(u)[j].add(v)
        self.meta["n_live"] += 1

    # ------------------------------------------------------------ persistence
    def dump(self):
        rt = lambda r: [r.mu, r.sigma]
        return dict(meta=self.meta,
                    target={u: dict(ew={c: e.dump() for c, e in s["ew"].items()}, last_nf=s["last_nf"],
                                    aff_ow={k: list(v) for k, v in s["aff_ow"].items()}, aff_nt={k: list(v) for k, v in s["aff_nt"].items()},
                                    ts_nf=rt(s["ts_nf"]), ts_all=rt(s["ts_all"])) for u, s in self.tgt.items()},
                    full={u: dict(ew={c: e.dump() for c, e in s["ew"].items()}, ts=rt(s["ts"])) for u, s in self.full.items()},
                    pers={u: [e.dump() for e in s] for u, s in self.pers.items()},
                    elo={u: list(v) for u, v in self.elo.items()})

    @classmethod
    def from_dump(cls, d):
        st = cls(); R = lambda x: st.env.create_rating(mu=x[0], sigma=x[1])
        st.meta = d["meta"]
        for u, s in d["target"].items():
            st.tgt[u] = dict(ew={c: Ewm(TARGET_EWM[c], *v) for c, v in s["ew"].items()}, last_nf=s["last_nf"],
                             aff_ow={k: tuple(v) for k, v in s["aff_ow"].items()}, aff_nt={k: tuple(v) for k, v in s["aff_nt"].items()},
                             ts_nf=R(s["ts_nf"]), ts_all=R(s["ts_all"]))
        for u, s in d["full"].items():
            st.full[u] = dict(ew={c: Ewm(FULL_SPAN, *v) for c, v in s["ew"].items()}, ts=R(s["ts"]))
        st.pers = {u: [Ewm(P_SPAN, *v) for v in s] for u, s in d["pers"].items()}
        st.elo = {u: tuple(v) for u, v in d["elo"].items()}
        return st


# ---------------------------------------------------------------- timelines
def _norm_events(events):
    out = []
    for sd, e, tm in events:
        e = API_EVENTS.get(e, e)
        out.append((int(sd), e, float(tm)))
    return out


def segment_log_times(evs):
    """one player's (event, time_ms) list for a whole match -> log seconds of the 4 segments of the last attempt
    (after the final reset): overworld, nether (-> blind travel), blind travel + stronghold (-> end), end (-> finish).
    NaN where the segment wasn't completed. (live_round2.py)"""
    H = [(e, tm) for e, tm in evs if e in STAGE or e in ("reset", "finish")]
    resets = [tm for e, tm in H if e == "reset"]
    lr = max(resets) if resets else 0.0
    H = [(e, tm) for e, tm in H if tm >= lr]
    r = {0: lr}
    for s in (1, 4, 6):
        ts = [tm for e, tm in H if STAGE.get(e, 0) >= s]
        r[s] = min(ts) if ts else nan
    fin = [tm for e, tm in H if e == "finish"]
    r["fin"] = min(fin) if fin else nan
    out = []
    for a, b in ((0, 1), (1, 4), (4, 6), (6, "fin")):
        d = (r[b] - r[a]) / 1000
        out.append(math.log(d) if d == d and d > 0 else nan)
    return out


def player_state_at(evs, t_ms):
    """one player's (event, time_ms) list -> state using only events with time <= t_ms (live_snapshots.py):
    stage, reach times r[0..6] (ms), real deaths, spawnpoint deaths, resets, last real death time"""
    stage, r, deaths, sp, resets, last_death = 0, [0.0] + [nan] * 6, 0, 0, 0, nan
    for tm, e in sorted([(tm, e) for e, tm in evs if tm <= t_ms], key=lambda x: (x[0], x[1] != "reset", x[1])):
        if e == "reset":
            stage, r, resets = 0, [float(tm)] + [nan] * 6, resets + 1
        elif e in STAGE and STAGE[e] > stage:
            for k in range(stage + 1, STAGE[e] + 1):
                r[k] = float(tm)
            stage = STAGE[e]
        elif e == "death":
            deaths, last_death = deaths + 1, float(tm)
        elif e == "death_sp":
            sp += 1
    return stage, r, deaths, sp, resets, last_death


def snapshot(events, ts):
    """both players' events -> snapshot columns of live_snapshots.parquet at each t in ts (seconds)"""
    ev = _norm_events(events)
    per = [[(e, tm) for sd, e, tm in ev if sd == k] for k in (0, 1)]
    cols = {c: [] for c in ["stage0", "stage1", "tis0", "tis1", "lead", "ahead", "deaths0", "deaths1", "recent0", "recent1",
                            "sp0", "sp1", "resets0", "resets1"]}
    for t in ts:
        tm = t * 1000.0
        (st0, R0, d0, sp0, rs0, ld0), (st1, R1, d1, sp1, rs1, ld1) = (player_state_at(p, tm) for p in per)
        c = min(st0, st1)
        ahead = (tm - R0[min(st1 + 1, 6)]) / 1000 if st0 > st1 else (-(tm - R1[min(st0 + 1, 6)]) / 1000 if st1 > st0 else 0.0)
        for k, v in [("stage0", st0), ("stage1", st1), ("tis0", np.float32((tm - R0[st0]) / 1000)),
                     ("tis1", np.float32((tm - R1[st1]) / 1000)), ("lead", np.float32((R1[c] - R0[c]) / 1000)),
                     ("ahead", np.float32(ahead)), ("deaths0", d0), ("deaths1", d1),
                     ("recent0", int(tm - (-1e12 if ld0 != ld0 else ld0) <= RECENT * 1000)),
                     ("recent1", int(tm - (-1e12 if ld1 != ld1 else ld1) <= RECENT * 1000)),
                     ("sp0", sp0), ("sp1", sp1), ("resets0", rs0), ("resets1", rs1)]:
            cols[k].append(v)
    out = {k: np.array(v) for k, v in cols.items()}
    for k in ("tis0", "tis1", "lead", "ahead"):
        out[k] = out[k].astype(np.float32)
    out["t"] = np.asarray(ts)
    return out


# ---------------------------------------------------------------- live model math (live_round1.py / live_round2.py)
sig = lambda z: 1 / (1 + np.exp(-z))
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
cl = lambda v, c: np.clip(np.asarray(v, float), -c, c)


def live_arrays(sn, prior, ffl, pers0, pers1, ow, nt):
    """snapshot columns + match context -> (F3 [n x 41], per-side dicts, elapsed-bucket one-hot)"""
    t = np.asarray(sn["t"]); n = len(t)
    prior = np.full(n, prior, float)
    tb = np.searchsorted(TB_EDGES, t, side="right") - 1
    TB1 = (tb[:, None] == np.arange(7)).astype(np.float32)
    sd = np.clip(sn["stage0"].astype(int) - sn["stage1"].astype(int), -3, 3)
    lead = cl(sn["lead"], 180)
    cols = []
    for b in range(7):
        cols += [((sd == k).astype(float) - (sd == -k)) * TB1[:, b] for k in (1, 2, 3)]
    stage_cols = cols
    F = {}
    for b in range(7):
        for j, k in enumerate((1, 2, 3)):
            F[("stage", b, k)] = stage_cols[3 * b + j]
        F[("lead", b)] = lead * TB1[:, b]
        F[("prior", b)] = prior * TB1[:, b]
    death = [cl(np.asarray(sn[a], float) - np.asarray(sn[b_], float), 3) for a, b_ in
             [("deaths0", "deaths1"), ("recent0", "recent1"), ("sp0", "sp1"), ("resets0", "resets1")]]
    F3 = np.column_stack([F[("stage", b, k)] for b in range(7) for k in (1, 2, 3)] + [F[("lead", b)] for b in range(7)]
                         + [F[("prior", b)] for b in range(7)]
                         + [cl(sn["ahead"], 180), cl(np.asarray(sn["tis0"], float) - np.asarray(sn["tis1"], float), 300)]
                         + death).astype(np.float32)
    ffl = float(np.nan_to_num(ffl))
    owc = np.full(n, OW.index(ow) if ow in OW else -1); ntc = np.full(n, NT.index(nt) if nt in NT else -1)
    SIDE = []
    for k in (0, 1):
        o = 1 - k; sg = 1 if k == 0 else -1
        SIDE.append(dict(stage=np.asarray(sn[f"stage{k}"], int), tis=np.asarray(sn[f"tis{k}"], float),
                         deaths=np.asarray(sn[f"deaths{k}"], float), rec=np.asarray(sn[f"recent{k}"], float),
                         sp=np.asarray(sn[f"sp{k}"], float), resets=np.asarray(sn[f"resets{k}"], float),
                         opp_stage=np.asarray(sn[f"stage{o}"], int),
                         pers=np.tile(np.asarray(pers0 if k == 0 else pers1, np.float32), (n, 1)),
                         lead=sg * lead, ahead=sg * cl(sn["ahead"], 180), prior=sg * prior, ffl=sg * np.full(n, ffl),
                         ow=owc, nt=ntc))
    return F3, SIDE, TB1, t


def aft_design(S, t):
    st = S["stage"]; g = SEG[st]; tis = S["tis"]; tr_ = t - tis
    oh = (st[:, None] == np.arange(7)).astype(np.float32)
    rec = S["rec"].astype(np.float32)
    P_ = S["pers"]
    dc = np.minimum(S["deaths"], 3).astype(np.float32)
    B = [oh, oh * np.log1p(tis)[:, None], oh * np.log1p(tr_)[:, None],
         P_ * (g[:, None] <= np.arange(4)), P_ * (g[:, None] == np.arange(4)),
         (S["ow"][:, None] == np.arange(1, 5)) * (g == 0)[:, None], (S["nt"][:, None] == np.arange(1, 4)) * (g <= 1)[:, None],
         oh * rec[:, None], (g[:, None] == np.arange(4)) * dc[:, None],
         np.minimum(S["sp"], 3)[:, None], np.minimum(S["resets"], 3)[:, None]]
    Z = np.column_stack([oh, np.log1p(tis), rec]).astype(np.float32)
    return np.column_stack(B).astype(np.float32), Z


def ff_design(S, TB1):
    st = S["stage"]; d = np.clip(st - S["opp_stage"], -3, 3)
    B = [TB1, (st[:, None] == np.arange(7)), (d[:, None] == np.array([-3, -2, -1, 1, 2, 3])),
         S["lead"][:, None] / 60, S["ahead"][:, None] / 60, (S["opp_stage"] == 6)[:, None],
         S["rec"][:, None], np.minimum(S["deaths"], 3)[:, None], np.minimum(S["resets"], 3)[:, None],
         np.log1p(S["tis"])[:, None], S["prior"][:, None], S["ffl"][:, None]]
    return np.column_stack(B).astype(np.float64)


def race(muA, sA, lA, muB, sB, lB):
    """P(A wins): independent log-normal finish times, constant forfeit rates (live_round2.py)"""
    lg = np.log(GRID[1:])
    FA = np.c_[np.zeros(len(muA)), ndtr((lg - muA[:, None]) / sA[:, None])]
    FB = np.c_[np.zeros(len(muB)), ndtr((lg - muB[:, None]) / sB[:, None])]
    GA = np.exp(-lA[:, None] * GRID); GB = np.exp(-lB[:, None] * GRID)
    dFA, dFB, dGA, dGB = np.diff(FA, 1), np.diff(FB, 1), -np.diff(GA, 1), -np.diff(GB, 1)
    mFA, mFB = 1 - (FA[:, 1:] + FA[:, :-1]) / 2, 1 - (FB[:, 1:] + FB[:, :-1]) / 2
    mGA, mGB = (GA[:, 1:] + GA[:, :-1]) / 2, (GB[:, 1:] + GB[:, :-1]) / 2
    wA = (dFA * mGA * mFB * mGB + dGB * mFB * mFA * mGA).sum(1)
    wB = (dFB * mGB * mFA * mGA + dGA * mFA * mFB * mGB).sum(1)
    rem = (1 - FA[:, -1]) * (1 - FB[:, -1]) * GA[:, -1] * GB[:, -1]
    return (wA + rem / 2) / (wA + wB + rem)


HB = np.array([0, 30, 60, 120, 240, 480, np.inf])               # path-hazard horizon intervals (live_round3.py)
OV = np.clip(GRID[:, None] - HB[None, :-1], 0, np.diff(HB)[None, :])


def race2(muA, sA, LA, muB, sB, LB, rho):
    """P(A wins): bivariate log-normal finish times with correlation rho, forfeit rates piecewise constant over the
    horizon intervals HB (LA, LB: n x 6, per second) (live_round3.py)"""
    lg = np.log(GRID[1:]); m_ = len(muA)
    zA = (lg - muA[:, None]) / sA[:, None]; zB = (lg - muB[:, None]) / sB[:, None]
    r = rho[:, None]; q = np.sqrt(1 - r * r)
    one, zero = np.ones((m_, 1)), np.zeros((m_, 1))
    FA, FB = np.c_[zero, ndtr(zA)], np.c_[zero, ndtr(zB)]
    cB = np.c_[one, ndtr(-(zB - r * zA) / q)]
    cA = np.c_[one, ndtr(-(zA - r * zB) / q)]
    wAf = np.diff(FA, 1) * (cB[:, 1:] + cB[:, :-1]) / 2
    wBf = np.diff(FB, 1) * (cA[:, 1:] + cA[:, :-1]) / 2
    Sj = np.clip(1 - np.c_[zero, np.cumsum(wAf + wBf, 1)], 0, 1)
    mS = (Sj[:, 1:] + Sj[:, :-1]) / 2
    GA = np.exp(-LA @ OV.T); GB = np.exp(-LB @ OV.T)
    dGA, dGB = -np.diff(GA, 1), -np.diff(GB, 1)
    mGA, mGB = (GA[:, 1:] + GA[:, :-1]) / 2, (GB[:, 1:] + GB[:, :-1]) / 2
    wA = (wAf * mGA * mGB + dGB * mS * mGA).sum(1)
    wB = (wBf * mGA * mGB + dGA * mS * mGB).sum(1)
    rem = Sj[:, -1] * GA[:, -1] * GB[:, -1]
    return (wA + rem / 2) / (wA + wB + rem)


def ph_V(S):
    return np.column_stack([np.ones(len(S["stage"])), S["rec"], S["lead"] / 60, np.clip(S["stage"] - S["opp_stage"], -3, 3),
                            S["opp_stage"] == 6])


def live_predict(L, sn, prior, ffl, pers0, pers1, ow, nt):
    """frozen live model L (models/live.json) on snapshot columns sn -> P(p0 wins) per snapshot"""
    F3, SIDE, TB1, t = live_arrays(sn, prior, ffl, pers0, pers1, ow, nt)
    b, g = np.asarray(L["aft_location"]), np.asarray(L["aft_log_scale"])
    mu, sg, lam = [], [], []
    for S in SIDE:
        Xa, Za = aft_design(S, t)
        mu.append(Xa.astype(np.float64) @ b); sg.append(np.exp(Za.astype(np.float64) @ g))
        if "path_hazard_state" in L:              # live_round3.py H: forfeit rate per horizon interval
            eta = (ff_design(S, TB1) @ np.asarray(L["path_hazard_state"]))[:, None] + \
                np.c_[np.zeros(len(t)), ph_V(S) @ np.asarray(L["path_hazard_interval"]).T]
            lam.append(np.exp(np.minimum(eta, 30)) / STEP)
        else:
            h = sig(ff_design(S, TB1) @ np.asarray(L["forfeit_hazard"]))
            lam.append(-np.log1p(-np.minimum(h, 0.999)) / STEP)
    if "path_hazard_state" in L or "rho" in L:
        if lam[0].ndim == 1:
            lam = [np.repeat(l_[:, None], len(HB) - 1, 1) for l_ in lam]
        rho = np.zeros(len(t))
        if "rho" in L:
            gA, gB = SEG[SIDE[0]["stage"]], SEG[SIDE[1]["stage"]]
            rho = np.asarray(L["rho"])[np.array([0, 1, 3, 5])[np.maximum(gA, gB)] + (gA != gB)]
        ls2 = logit(race2(mu[0], sg[0], lam[0], mu[1], sg[1], lam[1], rho))
    else:
        ls2 = logit(race(mu[0], sg[0], lam[0], mu[1], sg[1], lam[1]))
    D = np.column_stack([F3, ls2[:, None] * TB1])
    return sig(D @ np.asarray(L["stacking"]))


# ---------------------------------------------------------------- the oracle
@dataclass
class Prematch:
    p: float                       # P(p0 wins) before the match
    valid: bool                    # both players have the history the model was trained on (round 7 `valid`)
    features: dict                 # the 30 model inputs (A - B); NaN entered the model as 0
    context: dict = field(default_factory=dict)   # what live() needs: prior, forfeit-loss edge, personal speeds, seeds


class Oracle:
    def __init__(self, prematch_model, live_model, state):
        self.pm, self.lm, self.state = prematch_model, live_model, state

    @classmethod
    def load(cls, folder="models"):
        f = Path(folder)
        pm = json.loads((f / "prematch.json").read_text()); lm = json.loads((f / "live.json").read_text())
        with gzip.open(f / "state.json.gz", "rt") as fh:
            st = State.from_dump(json.load(fh))
        return cls(pm, lm, st)

    def save(self, folder="models"):
        with gzip.open(Path(folder) / "state.json.gz", "wt") as fh:
            json.dump(self.state.dump(), fh)

    def prematch_features(self, p0, p1, date, elo0=None, elo1=None, seed_overworld=None, seed_nether=None):
        st = self.state
        if elo0 is None or elo1 is None:
            elo0 = st.elo.get(p0, (0, nan))[1] if elo0 is None else elo0
            elo1 = st.elo.get(p1, (0, nan))[1] if elo1 is None else elo1
        a = st.target_player(p0, date, seed_overworld, seed_nether)
        b = st.target_player(p1, date, seed_overworld, seed_nether)
        d = {f"d_{c}": a[c] - b[c] for c in a if not c.endswith("_n")}
        fa, fb = st.full_player(p0), st.full_player(p1)
        d.update({f"d_{c}": fa[c] - fb[c] for c in fa})
        d["d_elo"] = (elo0 - elo1) / 400
        d["d_trueskill"] = st.ts_logit(p0, p1, "nf")
        d["d_trueskill_all"] = st.ts_logit(p0, p1, "all")
        d["d_ts_full"] = st.ts_logit(p0, p1, "full")
        # round 7 `valid`: win rate, completion rate and every split mean/std defined for both players
        valid = all(not _isnan(x[c]) for x in (a, b) for c in ("win_rate", "completion_rate")) and \
            all(x[f"{c}_n"] >= MINP for x in (a, b) for c in SPLIT_COLS)
        return d, valid

    def prematch(self, p0, p1, date, elo0=None, elo1=None, seed_overworld=None, seed_nether=None):
        """P(p0 wins) before a match at `date` (unix s). Elo defaults to each player's latest Elo in the state.
        Seed types only enter the seed affinity features (NaN -> 0 when unknown)."""
        d, valid = self.prematch_features(p0, p1, date, elo0, elo1, seed_overworld, seed_nether)
        x = np.array([0.0 if _isnan(d[c]) else d[c] for c in self.pm["features"]])
        p = float(sig(x @ np.asarray(self.pm["coef_raw"])))
        ctx = dict(p0=p0, p1=p1, prior=d["d_ts_full"], ffl=d["d_ff_loss_full_600"],
                   pers0=self.state.pers_player(p0, self.lm["pop_segment_log_s"]),
                   pers1=self.state.pers_player(p1, self.lm["pop_segment_log_s"]),
                   seed_overworld=seed_overworld, seed_nether=seed_nether)
        return Prematch(p, valid, {c: d[c] for c in self.pm["features"]}, ctx)

    def live(self, pm, events, t):
        """P(p0 wins) at t seconds (scalar or array) from both players' events so far, (side, event, time_ms)"""
        ts = np.atleast_1d(np.asarray(t, float))
        c = pm.context if isinstance(pm, Prematch) else pm
        p = live_predict(self.lm, snapshot(events, ts), c["prior"], c["ffl"], c["pers0"], c["pers1"],
                         c["seed_overworld"], c["seed_nether"])
        return p if np.ndim(t) else float(p[0])

    def update_target(self, record, events=None):
        """add a finished decisive qualified-vs-qualified match (and its timeline, if any) to the state"""
        self.state.update_target(record)
        if events is not None:
            self.state.update_live(record["p0_uuid"], record["p1_uuid"], events)

    def update_full(self, record):
        """add a finished decisive match of a tracked player vs any opponent to the state"""
        self.state.update_full(record)
