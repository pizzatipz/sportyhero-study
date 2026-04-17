# Sporty Hero — Algorithm Reverse-Engineering

> **Date:** 2026-04-17
> **Sample:** 45 consecutive rounds (round IDs 6428647–6428691, market `ng`)
> **Result:** Algorithm fully recovered. **45/45 rounds match the formula exactly.**

This document complements the existing 1,008-round behavioural study
(`FINDINGS.md`) by recovering the **actual formula** SportyBet uses to
compute each round's crash multiplier, rather than inferring its
distribution empirically.

---

## TL;DR — the formula

```
generatedHash    = SHA-512( serverSeed + "-" + clientSeed1 + "-" + ... + "-" + clientSeedN )
                   |        ^                ^
                   |        12-hex (48 bit)  12-hex (48 bit), in API-returned order
                   v
                   128 hex chars (64 bytes)

decimal          = int(generatedHash[:8], 16)               # 32 bits (0 .. 2^32 - 1)

houseCoefficient = round(0.97 * 2^32 / (2^32 - decimal), 2) # half-up rounding
```

This is the textbook **Bustabit** construction. The constant `0.97` is the
**3 % house edge** (theoretical RTP = 97 %).

**Verified to 45/45 rounds collected** between 2026-04-17 10:51 UTC and
2026-04-17 11:00 UTC — see `data/sh_rounds_45.jsonl`.

---

## How the input lives in the API

`GET /api/ng/games/sporty-hero/v1/round/{roundId}/seeds` returns
(unauthenticated):

```jsonc
{
  "round_id": 6428647,
  "houseCoefficient": 1.51,
  "serverSeeds": ["ab69e06d5c71"],            // 12 hex = 48-bit operator commitment, revealed AFTER round
  "clientSeeds": [
    {"uid": 4598118, "clientSeed": "c4d6976bb3aa", "name": "7***5"},
    {"uid": 8656679, "clientSeed": "090c9a83cb60", "name": "8***5"},
    {"uid": 6933633, "clientSeed": "29958768c3d7", "name": "9***5"}
  ],
  "clientSeedCount": 3,
  "generatedHash": "5b52efdeb92594de747bef466cb5abe616392b14b99cba6761449cace876f2783da41b9c38e92eb2d8d3ac91bae08ea9e8bfb8c3c7635a19b798e855551013c8",
  "hex": "5b52efde",
  "decimal": 1532162014,
  ...
}
```

The hash input is built by joining (with `"-"`):

```
serverSeed + "-" + clientSeed_1 + "-" + clientSeed_2 + "-" + clientSeed_3
```

…in the **exact order** the API returns. Different orderings or different
separators do not match.

### Verification one-liner

```python
import hashlib, json, math

def predict(server_seed, client_seeds):
    msg = "-".join([server_seed, *client_seeds]).encode()
    h   = hashlib.sha512(msg).hexdigest()
    dec = int(h[:8], 16)
    raw = 0.97 * (2**32) / ((2**32) - dec)
    # half-up rounding to 2 dp
    return math.floor(raw * 100 + 0.5) / 100

with open("data/sh_rounds_45.jsonl") as f:
    matches = total = 0
    for line in f:
        r = json.loads(line)
        ss = r["serverSeeds"][0]
        cs = [c["clientSeed"] for c in r["clientSeeds"]]
        total += 1
        if predict(ss, cs) == r["houseCoefficient"]:
            matches += 1
    print(matches, "/", total)   # -> 45 / 45
```

See `scripts/sh_stats.py` for a runnable equivalent.

---

## House edge — derived from the formula

Because `decimal` is uniform on `[0, 2^32)`:

```
P(crash ≥ X)  =  P( 0.97·2^32 / (2^32 − D) ≥ X )
              =  P( D ≥ 2^32 · (1 − 0.97/X) )
              =  0.97 / X       (for X ≥ 1)
```

So **for any cashout target X, P(win) · X = 0.97**, and the **theoretical
house edge is exactly 3 %**. This is the analytical truth.

> **Discrepancy with `FINDINGS.md` (~10.7 %):** the empirical scrape
> probably included a non-trivial fraction of `1.00x` "instant-bust" rounds
> as a point mass, which inflates the apparent edge. Worth re-checking with
> the seed-API ground truth.

### Empirical fairness on 45 rounds

| Threshold | Observed P(≥X) | Theoretical | Δ |
|----|----|----|----|
| ≥ 1.5x | 64.4 % | 64.7 % | −0.2 pp |
| ≥ 2x   | 42.2 % | 48.5 % | −6.3 pp |
| ≥ 5x   | 13.3 % | 19.4 % | −6.1 pp |
| ≥ 10x  |  6.7 % |  9.7 % | −3.0 pp |
| ≥ 20x  |  4.4 % |  4.9 % | −0.4 pp |

Within sampling noise at n=45 (1σ ≈ 7 pp).

---

## PRNG entropy of the 48-bit `serverSeed` (45 samples)

| Metric | Observed | Theoretical |
|----|----|----|
| Min in 2⁴⁸ space | 0.65 % | ~ 1 / n = 2.2 % |
| Max in 2⁴⁸ space | 97.91 % | ~ 1 − 1/n = 97.8 % |
| Mean | 1.41 × 10¹⁴ | 1.41 × 10¹⁴ |
| Std dev | 7.97 × 10¹³ | 8.13 × 10¹³ |
| Mean sequential diff | 51.2 % of range | 50 % |
| Hex char freq | uniform by eye | 33.75 each |

No detectable bias. A confident PRNG audit needs ≥10 000 seeds and the
NIST SP 800-22 / dieharder batteries.

---

## What this enables (and doesn't)

✅ Independently verify any past round — no need to trust SportyBet.
✅ Compute the exact theoretical house edge.
❌ **Predict a future round** — the operator's `serverSeed` is committed
   then revealed *after* the round ends. It does not leak via REST or
   WebSocket (see `websocket/README.md`).

---

## Data

- `data/sh_rounds_45.jsonl` — 45 consecutive rounds with full seeds + hash + houseCoefficient
- `scripts/sh_collect.py` — collector (`bulk` and `predict` modes)
- `scripts/sh_stats.py` — fairness / entropy analysis
