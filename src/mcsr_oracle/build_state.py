"""
Build the player state the app predicts from by replaying every collected season, and write data/state.json.gz.

    python -m mcsr_oracle.build_state            # every season in data/seasons/
    python -m mcsr_oracle.build_state 8-11       # a range
"""
import sys, time
from . import paths
from .data import build_state


def parse_seasons(arg):
    a, b = (int(x) for x in arg.split("-")) if "-" in arg else (int(arg), int(arg))
    return list(range(a, b + 1))


if __name__ == "__main__":
    seasons = parse_seasons(sys.argv[1]) if len(sys.argv) > 1 else paths.seasons()
    if not seasons:
        sys.exit(f"no seasons in {paths.data_dir() / 'seasons'}; collect one with python -m mcsr_oracle.collect <N>")
    t0 = time.time()
    orc, _ = build_state(seasons)
    orc.save(paths.data_dir())
    m = orc.state.meta
    print(f"seasons {seasons}: {m['n_target']:,} target, {m['n_full']:,} full-history and {m['n_live']:,} timeline "
          f"matches -> {paths.state_file()} ({time.time() - t0:.0f} s)")
