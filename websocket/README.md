# Sporty Hero — WebSocket / STOMP probe

> **Date:** 2026-04-17
> **Goal:** determine whether the live game WebSocket leaks any information
> useful for predicting the crash multiplier in real time.
> **Result:** **No exploitable leak.** The public topics broadcast nothing
> beyond what an in-game player already sees on screen.

---

## Endpoint

```
wss://www.sportybet.com/ws/ng/games/sporty-hero/v1/game
```

* Sub-protocols accepted: `v12.stomp`, `v11.stomp`, `v10.stomp`
* Auth model: requires a valid session cookie. Just visiting
  `https://www.sportybet.com/ng/m/games/sporty-hero` once is enough — the
  server auto-issues a guest identity (`id: null`, `puid: null`,
  `userIp: <client>`).
* Without cookies the server returns a STOMP `ERROR` frame:
  `{"bizCode":401,"message":"Access denied"}` and closes the socket.

---

## Public topics observed (346 frames captured)

| Topic | messageType(s) | Schema | Useful? |
|---|---|---|---|
| `/topic/ng-multiplier-wap` | `ROUND_WAITING`, `ROUND_PRE_START`, `ROUND_ONGOING`, `ROUND_END_WAIT` | `{roundId, currentMultiplier?, millisLeft?, totalMillis?, messageType}` | streams the live multiplier 1.00 → … → CRASH; **no advance leak** |
| `/topic/ng-round-wap` | `ROUND_NEXT` | `{roundId, messageType}` | only announces the next round id |
| `/topic/ng-lastRoundMultiplier-wap` | (binary, gzip JSON) | past-multiplier history | redundant with REST |

**Per-user topics** (`/topic/{country}-user-{userId}-info-wap`,
`/topic/user-{userId}-country-{country}-biz-exception-wap`) are addressed by
authenticated user id and were not probed.

### Speculative topics that are silent

I subscribed to plausible-but-undocumented topics and got SUBSCRIBE-ACK but
zero messages over a full round cycle:

```
/topic/ng-roundBet-wap
/topic/ng-bet-wap
/topic/ng-cashout-wap
/topic/ng-seed-wap
/topic/ng-roomInfo-wap
/topic/ng-info-wap
```

So there is **no public bus broadcasting other players' bets, clientSeeds,
or any pre-reveal seed material**.

---

## Frame inspector

Every captured frame was parsed and scanned for any field whose name
contained `seed`, `hash`, `decimal`, `coef`. Result: **0 hits across all
346 frames** (`data/sh_ws_capture.jsonl`).

```python
import json
suspicious = ("seed", "hash", "decimal", "coef")
with open("data/sh_ws_capture.jsonl") as f:
    leaks = 0
    for line in f:
        d = json.loads(line)
        if d.get("ev") != "stomp": continue
        body = d["body"]
        if isinstance(body, dict):
            for k in body:
                if any(s in k.lower() for s in suspicious):
                    leaks += 1
                    print("LEAK", d["sub"], k, body[k])
print("total leaks:", leaks)   # -> 0
```

---

## What about `/round/next-round/seeds`?

This REST endpoint sounds like the operator's pre-round commitment but is
**not**. It returns a 256-bit value that is **session-bound**:

* Different IP → different value for the same upcoming round id
* Same IP, different User-Agent → identical value
* The value never matches the actual `serverSeeds[0]` revealed when the
  round closes (tried sha256, sha512, first/last 12 hex, etc.)

Conclusion: that field is the **user's own pseudo-clientSeed** for the
session — it is misnamed. The operator's true 48-bit `serverSeed` is
properly committed-then-revealed; it is never visible until after the
round ends.

---

## Reproducing the probe

```bash
pip install websocket-client requests
python scripts/sh_ws_cookied.py    # one-shot CONNECT + SUBSCRIBE
python scripts/sh_ws_capture.py    # 5-minute full capture into JSONL
```

Output goes to `tmp/sh_ws_capture.jsonl` (one frame per line).
