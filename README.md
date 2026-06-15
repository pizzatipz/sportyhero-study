# Sporty Hero — Crash Game RNG Study

> **Status**: ✅ Study Complete (three rounds of investigation)  
> **Rounds analyzed**: 1,008 behavioural + **10,000 ground-truth seeds**  
> **Key finding**: provably-fair **3% house edge**, no exploitable patterns — confirmed down to the seed generator  

Automated data collection and statistical analysis system for SportyBet's **Sporty Hero** crash/multiplier game. This study empirically determines whether the crash multiplier distribution exhibits any detectable, exploitable statistical structure. The answer, across three independent angles of attack, is **no**.

📄 **[Read the full findings →](FINDINGS.md)** (March 2026 — behavioural study, 1,008 rounds)

📄 **[April 2026 addendum: algorithm reverse-engineering & WebSocket probe →](ADDENDUM_2026-04.md)**

📄 **[June 2026 addendum II: 10k-seed PRNG audit & the "hourly pattern" thesis →](SEED_AUDIT_2026-06.md)**

> **House-edge correction:** the original behavioural study reported ~10.7%. Ground-truth seeds reveal that was a scraper artifact from over-counting the 1.00x instant-bust point mass. The true rule is `houseCoefficient = max(1.00, round(0.97·2³²/(2³²−decimal), 2))` — a **3% edge** with a 3.48% instant-bust point mass. See the [June 2026 addendum](SEED_AUDIT_2026-06.md).

## Key Results

| Analysis | Result |
|----------|--------|
| RNG model | `crash = 0.89 / U`, U ~ Uniform(0,1) — confirmed via KS test (p=0.49) |
| Distribution | Log-normal (p=0.49) — not exponential (p≈0) |
| House edge | **~10.7%** — house keeps ~11 cents per dollar wagered |
| Independence | Fully confirmed: zero autocorrelation, Ljung-Box p>0.08 |
| Streak patterns | None — all binomial tests non-significant |
| Exploitable edge | **None** — Kelly criterion fraction = 0 for all targets |

### Crash Value Distribution

![Distribution](docs/distribution.png)

### House Edge

![House Edge](docs/house_edge.png)

### Autocorrelation (Independence Proof)

![ACF](docs/acf.png)

## Architecture

```
Browser (Chromium) ──▸ Playwright Bot ──▸ SQLite DB ──▸ Analysis Pipeline ──▸ Reports
                           │                  │                │
                      iframe scraper     WAL mode       15 statistical tests
                      (coefficient-row)  deduplication   5 distribution fits
                                                        Monte Carlo sims
```

## Components

| File | Purpose |
|------|---------|
| `src/bot.py` | Playwright-based crash value scraper (iframe-aware, auto-refresh) |
| `src/db.py` | SQLite storage with WAL mode, bulk insert, stats queries |
| `src/analyze.py` | Statistical analysis + pattern detection (5 analyses) |
| `src/deep_analyze.py` | Comprehensive 15-dimension deep analysis |
| `src/strategies.py` | Cashout strategy backtesting (5 strategies) |
| `scripts/sh_seed_collect.py` | Resumable, integrity-checking 48-bit seed backfiller (threaded) |
| `scripts/sh_seed_audit.py` | serverSeed PRNG audit: uniformity, NIST, time-seeding, LCG/LFSR recovery |
| `scripts/sh_temporal.py` | Hourly-pattern thesis: dispersion, Fano, duration coupling, exploitability |
| `scripts/nist.py` | NIST SP 800-22 core randomness battery (validated) |
| `FINDINGS.md` | Full research findings with charts and conclusions |
| `SEED_AUDIT_2026-06.md` | June 2026 addendum: 10k-seed PRNG audit & hourly-pattern resolution |

## Quick Start

```bash
# Install dependencies
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
python -m playwright install chromium

# Collect data (observe mode — scrapes crash values)
python -m src bot --rounds 1000

# Quick analysis
python -m src analyze

# Deep analysis (15 statistical tests, charts, Monte Carlo)
python -m src deep

# Strategy backtesting
python -m src strategies
```

## Commands

| Command | Description |
|---------|-------------|
| `python -m src bot --rounds N` | Scrape N crash values from the game |
| `python -m src bot --auto-inspect --wait 60` | DOM inspection mode for selector discovery |
| `python -m src analyze` | Basic statistical analysis + distribution charts |
| `python -m src analyze --no-plot` | Text-only analysis (no matplotlib) |
| `python -m src deep` | Full deep analysis (15 dimensions, all charts) |
| `python -m src strategies` | Backtest all cashout strategies |
| `python -m src strategies --stake 50` | Backtest with custom stake amount |

## Analyses Performed

### Basic (`analyze`)
1. Descriptive statistics (mean, median, percentiles)
2. Distribution fitting (exponential, log-normal, Pareto)
3. Streak / run analysis with Wald-Wolfowitz test
4. Autocorrelation (ACF with confidence intervals)
5. 4-panel visualization

### Deep (`deep`)
1. Descriptive statistics with robust measures
2. Empirical survival function S(x) = P(crash ≥ x)
3. House edge & RTP at every cashout target
4. Optimal cashout (EV-maximizing target)
5. Multi-distribution fitting with Anderson-Darling
6. Independence tests (ACF, Ljung-Box, Spearman, Mutual Information)
7. Conditional probability analysis
8. Streak conditional analysis with binomial tests
9. Run-length distribution vs geometric
10. Volatility clustering (ARCH/GARCH effects)
11. Digit & rounding analysis
12. Segmented stability analysis (drift detection)
13. Extreme value analysis (GEV, GPD, return levels)
14. Monte Carlo bankroll simulations
15. Definitive house edge calculation

### Strategies (`strategies`)
- Fixed multiplier (1.5x through 10x)
- Martingale (double after loss)
- Anti-Martingale (double after win)
- Streak-based (boost after N lows)
- Kelly criterion (optimal sizing)

## Tech Stack

- **Python 3.12** + **Playwright** (browser automation)
- **SQLite** with WAL mode (concurrent-safe data storage)
- **pandas**, **scipy**, **numpy**, **statsmodels** (statistical analysis)
- **matplotlib**, **seaborn** (visualization)

## Key Questions Answered

| # | Question | Answer |
|---|----------|--------|
| 1 | What is the empirical distribution? | Log-normal on (x−1), with point mass at 1.00x |
| 2 | Are there streaks? | Yes, but consistent with pure randomness |
| 3 | Is there autocorrelation? | No — zero significant lags |
| 4 | Does history influence the next crash? | No — conditional = unconditional |
| 5 | Optimal fixed cashout for EV? | 1.1–1.3x (lowest edge, still negative) |
| 6 | Can adaptive strategies win? | No — Kelly fraction = 0 |

## License

MIT