#!/usr/bin/env python3
"""Sporty Hero ground-truth seed collector (backfill, resumable, rate-limited).

Walks round IDs downward from the current latest completed round and pulls the
revealed seed material from the public, unauthenticated endpoint:

    GET /api/ng/games/sporty-hero/v1/round/{roundId}/seeds

For every round we store the operator `serverSeed` (48-bit / 12-hex commitment,
revealed after the round), the per-bettor `clientSeeds`, the `generatedHash`,
`decimal`, `houseCoefficient`, and timestamps. Each row is integrity-checked by
recomputing SHA512 -> decimal -> houseCoefficient and asserting it matches the
value the API returned (this re-confirms the Bustabit formula at scale and flags
any anomaly).

Storage: SQLite (data/sportyhero_seeds.db). Resumable: on restart it continues
from one below the lowest round_id already stored. Polite: single keep-alive
connection, configurable delay, exponential backoff on HTTP 429.

Usage:
    python scripts/sh_seed_collect.py --target 10000 --delay 0.35
    python scripts/sh_seed_collect.py --export-jsonl data/sh_seeds.jsonl
    python scripts/sh_seed_collect.py --status
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HOST = "www.sportybet.com"
BASE = "/api/ng/games/sporty-hero/v1"
HDRS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "x-platform": "WEB",
    "Accept": "application/json",
    "Connection": "keep-alive",
}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(REPO, "data", "sportyhero_seeds.db")

TWO32 = 2 ** 32
HOUSE = 0.97  # 3% edge constant baked into the formula


# --------------------------------------------------------------------------- #
# Formula / integrity                                                         #
# --------------------------------------------------------------------------- #
def predict_coefficient(server_seed: str, client_seeds: list[str]):
    """Recompute (decimal, houseCoefficient, hash) from seeds via the Bustabit formula.

    The raw value is 0.97*2^32/(2^32-decimal). When decimal < 0.03*2^32 the raw
    value is < 1.00 and the game clamps the displayed multiplier to 1.00x (the
    "instant bust" point mass, which occurs with probability exactly 3%).
    """
    msg = "-".join([server_seed, *client_seeds]).encode()
    h = hashlib.sha512(msg).hexdigest()
    dec = int(h[:8], 16)
    raw = HOUSE * TWO32 / (TWO32 - dec)
    coef = math.floor(raw * 100 + 0.5) / 100  # half-up to 2dp
    coef = max(1.0, coef)                      # clamp: instant-bust floor at 1.00x
    return dec, coef, h


# --------------------------------------------------------------------------- #
# Storage                                                                     #
# --------------------------------------------------------------------------- #
def init_db(path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS rounds (
            round_id          INTEGER PRIMARY KEY,
            created_at        TEXT,
            start_time        TEXT,
            server_seed       TEXT,
            server_seed_int   INTEGER,
            client_seeds      TEXT,
            client_seed_count INTEGER,
            decimal           INTEGER,
            house_coefficient REAL,
            generated_hash    TEXT,
            verified          INTEGER
        )
        """
    )
    con.commit()
    return con


def store_round(con: sqlite3.Connection, rid: int, data: dict) -> int:
    """Insert one round. Returns 1 if verified, 0 if mismatch, -1 if unusable."""
    server_seeds = data.get("serverSeeds") or []
    client_objs = data.get("clientSeeds") or []
    if not server_seeds:
        return -1
    ss = server_seeds[0]
    cs = [c.get("clientSeed") for c in client_objs]
    api_dec = data.get("decimal")
    api_hc = data.get("houseCoefficient")
    if api_hc is None:
        return -1

    dec, coef, h = predict_coefficient(ss, cs)
    verified = int(
        (api_dec is None or dec == api_dec) and abs(coef - float(api_hc)) < 1e-9
    )

    con.execute(
        """INSERT OR IGNORE INTO rounds
           (round_id, created_at, start_time, server_seed, server_seed_int,
            client_seeds, client_seed_count, decimal, house_coefficient,
            generated_hash, verified)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            rid,
            data.get("createdAt"),
            data.get("startTime"),
            ss,
            int(ss, 16),
            json.dumps(cs),
            data.get("clientSeedCount", len(cs)),
            api_dec if api_dec is not None else dec,
            float(api_hc),
            data.get("generatedHash", h),
            verified,
        ),
    )
    return verified


def db_bounds(con: sqlite3.Connection) -> tuple[int | None, int | None, int]:
    row = con.execute(
        "SELECT MIN(round_id), MAX(round_id), COUNT(*) FROM rounds"
    ).fetchone()
    return row[0], row[1], row[2]


# --------------------------------------------------------------------------- #
# HTTP (single keep-alive connection)                                         #
# --------------------------------------------------------------------------- #
class Client:
    def __init__(self, delay: float = 0.35):
        self.delay = delay
        self.conn: http.client.HTTPSConnection | None = None
        self._connect()

    def _connect(self):
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
        self.conn = http.client.HTTPSConnection(HOST, timeout=15)

    def get_json(self, path: str, retries: int = 4):
        for attempt in range(retries):
            try:
                self.conn.request("GET", BASE + path, headers=HDRS)
                resp = self.conn.getresponse()
                body = resp.read()
                if resp.status == 200:
                    return json.loads(body)
                if resp.status == 429:
                    wait = 3 * (attempt + 1)
                    print(f"  429 rate-limited, sleeping {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                if resp.status in (502, 503, 504):
                    time.sleep(2 * (attempt + 1))
                    continue
                # other status: brief retry
                time.sleep(1)
            except Exception as e:
                print(f"  conn error ({type(e).__name__}: {e}); reconnecting", flush=True)
                self._connect()
                time.sleep(1)
        return None


# --------------------------------------------------------------------------- #
# Collection                                                                  #
# --------------------------------------------------------------------------- #
def latest_round_id(client: Client) -> int | None:
    j = client.get_json("/round/previous-multipliers")
    if not j:
        return None
    coeffs = (j.get("data") or {}).get("coefficients") or []
    if not coeffs:
        return None
    return max(int(c["id"]) for c in coeffs)


def collect(target: int, delay: float, floor: int | None, workers: int = 1):
    con = init_db()
    client = Client(delay=delay)

    mn, mx, have = db_bounds(con)
    if have >= target:
        print(f"Already have {have} >= target {target}; nothing to do.", flush=True)
        return
    if mn is not None:
        cursor = mn - 1
        print(f"Resuming: {have} rounds stored (ids {mn}..{mx}); continuing below {mn} "
              f"with {workers} worker(s)", flush=True)
    else:
        latest = latest_round_id(client)
        if latest is None:
            print("Could not determine latest round id; aborting.", flush=True)
            return
        cursor = latest
        print(f"Fresh start: latest completed round id = {latest}; {workers} worker(s)",
              flush=True)

    tls = threading.local()

    def fetch(rid):
        if getattr(tls, "c", None) is None:
            tls.c = http.client.HTTPSConnection(HOST, timeout=15)
        for attempt in range(4):
            try:
                tls.c.request("GET", BASE + f"/round/{rid}/seeds", headers=HDRS)
                resp = tls.c.getresponse()
                body = resp.read()
                if resp.status == 200:
                    if delay:
                        time.sleep(delay)
                    j = json.loads(body)
                    return rid, (j.get("data") or {})
                if resp.status == 429:
                    time.sleep(3 * (attempt + 1))
                    continue
                if resp.status in (502, 503, 504):
                    time.sleep(2 * (attempt + 1))
                    continue
                time.sleep(1)
            except Exception:
                try:
                    tls.c.close()
                except Exception:
                    pass
                tls.c = http.client.HTTPSConnection(HOST, timeout=15)
                time.sleep(1)
        return rid, None

    collected = 0
    mismatches = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        while have + collected < target and cursor > 0 and (floor is None or cursor >= floor):
            batch_n = min(target - have - collected, 400)
            if floor is not None:
                batch_n = min(batch_n, cursor - floor + 1)
            if batch_n <= 0:
                break
            ids = [cursor - i for i in range(batch_n)]
            cursor -= batch_n
            results = ex.map(fetch, ids) if workers > 1 else map(fetch, ids)
            for rid, data in results:
                if data is None:
                    continue
                res = store_round(con, rid, data)
                if res == 1:
                    collected += 1
                elif res == 0:
                    collected += 1
                    mismatches += 1
                    print(f"  !! INTEGRITY MISMATCH at round {rid}", flush=True)
            con.commit()
            rate = collected / max(time.time() - t0, 1e-9)
            eta = (target - have - collected) / max(rate, 1e-9)
            print(f"  total {have + collected}/{target} (cursor={cursor}) "
                  f"verified_ok={collected - mismatches} mism={mismatches} "
                  f"{rate:.1f}/s eta={eta/60:.1f}m", flush=True)

    con.commit()
    mn, mx, have = db_bounds(con)
    dt = time.time() - t0
    print(f"DONE: stored {have} rounds (ids {mn}..{mx}); "
          f"this run +{collected} in {dt/60:.1f}m, mismatches={mismatches}", flush=True)
    con.close()


def export_jsonl(path: str):
    con = init_db()
    rows = con.execute(
        "SELECT round_id, created_at, start_time, server_seed, server_seed_int, "
        "client_seeds, client_seed_count, decimal, house_coefficient, "
        "generated_hash, verified FROM rounds ORDER BY round_id ASC"
    ).fetchall()
    cols = [
        "round_id", "created_at", "start_time", "server_seed", "server_seed_int",
        "client_seeds", "client_seed_count", "decimal", "house_coefficient",
        "generated_hash", "verified",
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            d = dict(zip(cols, r))
            d["client_seeds"] = json.loads(d["client_seeds"])
            f.write(json.dumps(d) + "\n")
    print(f"Exported {len(rows)} rows -> {path}", flush=True)
    con.close()


def status():
    con = init_db()
    mn, mx, have = db_bounds(con)
    ver = con.execute("SELECT COUNT(*) FROM rounds WHERE verified=1").fetchone()[0]
    mism = con.execute("SELECT COUNT(*) FROM rounds WHERE verified=0").fetchone()[0]
    print(f"stored={have} ids={mn}..{mx} verified={ver} mismatches={mism}", flush=True)
    con.close()


def main():
    ap = argparse.ArgumentParser(description="Sporty Hero seed backfill collector")
    ap.add_argument("--target", type=int, default=10000, help="total rounds to accumulate")
    ap.add_argument("--delay", type=float, default=0.35, help="seconds between requests (per worker)")
    ap.add_argument("--workers", type=int, default=1, help="concurrent fetch workers (be polite)")
    ap.add_argument("--floor", type=int, default=None, help="lowest round id to fetch")
    ap.add_argument("--export-jsonl", metavar="PATH", help="export DB to JSONL and exit")
    ap.add_argument("--status", action="store_true", help="print DB status and exit")
    args = ap.parse_args()

    if args.status:
        status()
    elif args.export_jsonl:
        export_jsonl(args.export_jsonl)
    else:
        collect(args.target, args.delay, args.floor, args.workers)


if __name__ == "__main__":
    main()
