#!/usr/bin/env python3
"""Sporty Hero data collector.
Collects: previous-multipliers list, full per-round seeds, and the LIVE next-round/seeds at each iteration.
Goal: 
  1. Fit the multiplier formula given (decimal -> houseCoefficient)
  2. Test if the published next-round serverSeed predicts the actual round serverSeed
"""
import time, json, urllib.request, urllib.error, hashlib, os, sys

BASE = "https://www.sportybet.com/api/ng/games/sporty-hero/v1"
HDRS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "x-platform": "WEB",
    "Accept": "application/json",
}
OUT_PRED = os.path.join(os.path.dirname(__file__), "sh_predictions.jsonl")
OUT_ROUND = os.path.join(os.path.dirname(__file__), "sh_rounds.jsonl")

def req(path):
    r = urllib.request.Request(BASE + path, headers=HDRS)
    with urllib.request.urlopen(r, timeout=8) as resp:
        return json.loads(resp.read())

def get_next_seed():
    return req("/round/next-round/seeds").get("data", {})

def get_round_seeds(rid):
    return req(f"/round/{rid}/seeds").get("data", {})

def get_history():
    return req("/round/previous-multipliers").get("data", {}).get("coefficients", [])

def get_round_status():
    return req("/round").get("data", {})

# Phase 1: bulk-collect historical rounds for formula fitting
def bulk_history():
    print("Bulk collecting last 45 rounds...", flush=True)
    rows = get_history()
    print(f"Got {len(rows)} from previous-multipliers", flush=True)
    for r in rows:
        rid = r["id"]
        try:
            time.sleep(0.5)
            seed_info = get_round_seeds(rid)
            entry = {"round_id": rid, "houseCoefficient": r["houseCoefficient"], "createdAt": r["createdAt"], **seed_info}
            with open(OUT_ROUND, "a") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"  {rid}: hc={r['houseCoefficient']} dec={seed_info.get('decimal')} hex={seed_info.get('hex')} ssLen={len(seed_info.get('serverSeeds',[''])[0]) if seed_info.get('serverSeeds') else 0} csCount={seed_info.get('clientSeedCount')}", flush=True)
        except Exception as e:
            print(f"  err {rid}: {e}", flush=True)

# Phase 2: continuously poll next-round/seeds and round to detect when next-round seed becomes the active round's seed
def predict_loop(duration_s=3600):
    print("Starting prediction loop...", flush=True)
    last_predicted = None
    end = time.time() + duration_s
    last_rid_seen = None
    while time.time() < end:
        try:
            ns = get_next_seed()
            published_ss = ns.get("serverSeed")
            rs = get_round_status()
            ongoing = rs.get("ongoingRound") or {}
            waiting = rs.get("waitingRound") or {}
            ongoing_id = ongoing.get("id") if ongoing else None
            waiting_id = waiting.get("id") if waiting else None
            now = time.time()
            entry = {
                "ts": now,
                "published_serverSeed": published_ss,
                "ongoing_round_id": ongoing_id,
                "waiting_round_id": waiting_id,
            }
            if published_ss != last_predicted:
                last_predicted = published_ss
                with open(OUT_PRED, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                print(f"{now:.2f} new published seed {published_ss[:16]}... waiting={waiting_id} ongoing={ongoing_id}", flush=True)
            # If a round just finished (ongoing went away), capture it
            if ongoing_id and ongoing_id != last_rid_seen:
                last_rid_seen = ongoing_id
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(3)
            else:
                print(f"http {e.code}", flush=True)
                time.sleep(2)
        except Exception as e:
            print(f"err: {type(e).__name__} {e}", flush=True)
            time.sleep(2)
        time.sleep(0.8)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bulk":
        bulk_history()
    elif len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict_loop(int(sys.argv[2]) if len(sys.argv) > 2 else 3600)
    else:
        # Both
        bulk_history()
        predict_loop()
