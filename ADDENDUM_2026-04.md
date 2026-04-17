# Addendum (April 2026) — Algorithm reverse-engineering & live-WS exploit hunt

The original study (`FINDINGS.md`, March 2026) determined the **shape** of
Sporty Hero's crash distribution by scraping ~1 000 rounds and concluded
no exploitable structure exists. This addendum complements that work by
attacking the same question from a different angle:

> **"Can we exploit Sporty Hero in real time, given everything that's
> network-accessible to an unauthenticated observer?"**

Investigation date: **2026-04-17**.

## What's new

| Question | Answer | See |
|---|---|---|
| What's the **exact formula**? | `houseCoefficient = round(0.97·2³² / (2³² − int(SHA512(serverSeed-cs1-…-csN)[:8],16)), 2)` — verified 45/45 rounds | [`algorithm/`](algorithm/) |
| What's the **theoretical** house edge? | **3 % exactly** (not 10.7 %) — derived analytically, observed within sampling noise on n=45 | [`algorithm/`](algorithm/) |
| Does the WebSocket leak seeds? | **No.** Captured 346 STOMP frames across multiple rounds — zero seed/hash/coefficient fields anywhere | [`websocket/`](websocket/) |
| Does `/round/next-round/seeds` leak the next operator seed? | **No.** That field is session-bound (per-IP), not the operator commitment | [`websocket/`](websocket/) |
| Is the 48-bit `serverSeed` PRNG biased? | **No** in 45 samples (uniform hex, full-range coverage, mean/stddev match theory). Confident audit needs ≥10 000 seeds + NIST SP 800-22 | [`algorithm/`](algorithm/) |

## Discrepancy with original `FINDINGS.md`

The original study reports a **~10.7 % house edge**. The algorithm derives
**exactly 3 %**. The most likely explanation is that the scraper's data
contained a high proportion of `1.00x` "instant-bust" rounds (which the
provably-fair algorithm does not produce — the minimum value is
`0.97·2³² / (2³² − 0) = 0.97`, displayed as `0.97x`/`1.00x` after rounding,
but only when `decimal` is very small).

Re-running the original analysis on data sourced from
`/round/{id}/seeds` (ground truth) instead of the on-screen scraper
should resolve this. PRs welcome.

## Files added

```
algorithm/README.md         — formula, derivation, verification one-liner
websocket/README.md         — STOMP probe, topic map, leak checks
data/sh_rounds_45.jsonl     — 45 consecutive rounds, full seed material
data/sh_ws_capture.jsonl    — 346 STOMP frames (multiple complete rounds)
scripts/sh_collect.py       — REST collector (bulk + live)
scripts/sh_stats.py         — fairness + entropy analyser
scripts/sh_ws_cookied.py    — minimal one-shot WebSocket prober
scripts/sh_ws_capture.py    — long-running WebSocket capture + leak scanner
```

## Bottom line

Combined with the original behavioural study, **two independent angles of
attack agree**: Sporty Hero is fair-as-advertised and offers **no
real-time exploit** to an unauthenticated observer. The only remaining
theoretical attack surface is a hidden PRNG flaw in the operator
`serverSeed` generator — which would require ≥10 000 captured seeds plus
formal randomness testing to either prove or rule out.
