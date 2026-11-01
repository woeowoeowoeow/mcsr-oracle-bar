"""
Rate-limited client for the public MCSR Ranked API (https://docs.mcsrranked.com). The documented limit is 500
requests per 10 minutes; this stays at 480 to leave room for anything else running against the same address.
"""
import time
from threading import Lock
import requests

BASE_URL = "https://api.mcsrranked.com"
RATE_LIMIT, PERIOD, RETRY_SLEEP = 480, 600, 61


class RateLimiter:
    def __init__(self):
        self.calls, self.lock = [], Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < PERIOD]
            if len(self.calls) >= RATE_LIMIT:
                time.sleep(PERIOD - (now - self.calls[0]))
            self.calls.append(time.time())


rl, session = RateLimiter(), requests.Session()


def get(path, params=None):
    """the response's `data`, or None on an error status"""
    while True:
        rl.wait()
        r = session.get(BASE_URL + path, params=params, timeout=15)
        if r.status_code == 429:
            print("  rate limited, sleeping"); time.sleep(RETRY_SLEEP); continue
        if r.status_code != 200:
            return None
        d = r.json()
        return d["data"] if d.get("status") == "success" else None
