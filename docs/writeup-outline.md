# Write-up outline (draft)

Two documents, written from the code in `src/mcsr_oracle/model.py` and the frozen coefficients in `models/`.

## 1. The math

- Data and population: ranked 1v1s between 2000+ Elo players, Seasons 8–11; forfeits counted as outcomes.
- Player ratings: TrueSkill (τ = 25/600, no draws) over each player's full ranked history; win probability
  Φ(Δμ / √(2β² + σ²ₐ + σ²ᵦ)).
- Pre-match model: mirrored logistic regression without an intercept, so P(A beats B) + P(B beats A) = 1; the 30
  difference features; exponentially weighted player stats (spans 200 / 600), seed-type affinity with shrinkage.
- Live model:
  - match state every 15 s: stage, time lead, deaths, resets
  - survival model: censored log-normal remaining time per player (AFT), correlation ρ between the two players'
    times by segment, forfeit hazard piecewise constant over the horizon after t (Poisson landmark model)
  - the competing-risks race integral (`race2`)
  - stacking: logistic model of the state plus the survival logit, by elapsed-time bucket

## 2. Methodology and its defense

- Walk-forward monthly evaluation; Season 11 as a holdout; Season 12 as the prospective test.
- No look-ahead: features use only earlier matches; live predictions use only events up to t (tested).
- Symmetry by construction (tested).
- Baselines: coin flip, Elo, TrueSkill.
- Limitations and open validation work (from the Phase 1 audit): hindsight-defined population, Elo slope fit on
  data overlapping the test months, selection on the Oct–Apr folds, clustered confidence intervals, pre-match
  calibration, live model fit diagnostics.
